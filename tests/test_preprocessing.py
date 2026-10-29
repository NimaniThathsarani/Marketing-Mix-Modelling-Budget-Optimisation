"""Unit tests. Run from the project root:  python -m pytest tests -v"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import (CHANNEL_COLS, TARGET_COL, TOTAL_SPEND_COL, add_outlier_count, detect_outliers,
                          drop_constant_columns, enforce_dtypes, fix_dtypes, handle_missing, reconcile_total_spend,
                          remove_duplicates, standardise_columns)
from src.preprocessing import (build_features, chronological_split, expanding_window_folds, feature_dictionary,
                               geometric_adstock, hill_saturation)
from src.validation import validate, validate_features


def make_df(n=12, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"month": pd.date_range("2023-01-01", periods=n, freq="MS")})
    for c in CHANNEL_COLS:
        df[c] = rng.integers(1_000_000, 5_000_000, n)
    df[TOTAL_SPEND_COL] = df[CHANNEL_COLS].sum(axis=1)
    df[TARGET_COL] = rng.integers(60_000_000, 100_000_000, n)
    df["units_sold"] = rng.integers(100_000, 200_000, n)
    df["impressions"] = rng.integers(8 * 10**8, 2 * 10**9, n)
    df["clicks"] = rng.integers(10**7, 5 * 10**7, n)
    df["conversions"] = rng.integers(3 * 10**5, 2 * 10**6, n)
    df["average_price_index"] = np.linspace(1.0, 1.2, n)
    df["promotion_discount_pct"] = rng.uniform(2, 25, n)
    df["competitor_pressure_index_0_100"] = rng.uniform(30, 65, n)
    df["economic_sentiment_index_0_100"] = rng.uniform(30, 70, n)
    df["lkr_usd_exchange_rate"] = rng.uniform(320, 350, n)
    df["inflation_proxy_index"] = np.linspace(1.0, 1.5, n)
    df["seasonal_demand_index"] = rng.uniform(0.95, 1.2, n)
    df["public_holiday_count"] = rng.integers(1, 8, n)
    return df


def test_standardise_columns():
    out = standardise_columns(pd.DataFrame(columns=[" Sales Revenue (LKR) ", "TV-Spend"]))
    assert list(out.columns) == ["sales_revenue_lkr", "tv_spend"]


def test_remove_duplicates_rows_and_months():
    df = make_df(5)
    dup = pd.concat([df, df.iloc[[2]]], ignore_index=True)
    out, n = remove_duplicates(dup)
    assert n == 1 and len(out) == 5


def test_handle_missing_interpolates_but_drops_missing_target():
    df = make_df(6)
    df.loc[2, "tv_spend_lkr"] = np.nan
    df.loc[4, TARGET_COL] = np.nan
    out, report = handle_missing(df)
    assert report["tv_spend_lkr"] == 1
    assert len(out) == 5 and out.isna().sum().sum() == 0


def test_reconcile_total_spend_fixes_rounding_and_rejects_large_gap():
    df = make_df(4)
    df.loc[0, TOTAL_SPEND_COL] += 2
    out, n = reconcile_total_spend(df)
    assert n == 1 and (out[CHANNEL_COLS].sum(axis=1) == out[TOTAL_SPEND_COL]).all()
    df.loc[1, TOTAL_SPEND_COL] += 1000
    try:
        reconcile_total_spend(df)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_detect_outliers_flags_extreme_value_only():
    df = make_df(20)
    df.loc[7, TARGET_COL] = 10**10
    flags = detect_outliers(df, [TARGET_COL])
    assert list(flags["month"]) == [df.loc[7, "month"]]


def test_chronological_split_has_no_overlap_and_keeps_order():
    feats = build_features(make_df(36))
    train, test = chronological_split(feats, test_months=6)
    assert len(train) == 30 and len(test) == 6
    assert train["month"].max() < test["month"].min()
    assert train["month"].is_monotonic_increasing and test["month"].is_monotonic_increasing


def test_expanding_window_folds_never_train_on_future():
    folds = expanding_window_folds(30, n_splits=4, test_size=3, min_train=18)
    assert len(folds) == 4
    for tr, va in folds:
        assert tr.max() < va.min() and len(va) == 3
    assert folds[-1][1].max() == 29


def test_lag_features_use_past_values_only():
    feats = build_features(make_df(12))
    assert pd.isna(feats.loc[0, "tv_spend_lag1"]) and pd.isna(feats.loc[1, "tv_spend_lag2"])
    assert feats.loc[5, "tv_spend_lag1"] == feats.loc[4, "tv_spend_lkr"]
    assert feats.loc[5, "tv_spend_lag2"] == feats.loc[3, "tv_spend_lkr"]


def test_channel_shares_sum_to_one():
    feats = build_features(make_df(12))
    share_cols = [c for c in feats.columns if c.endswith("_share")]
    assert len(share_cols) == 6
    assert np.allclose(feats[share_cols].sum(axis=1), 1.0)


def test_flags_use_fixed_thresholds():
    feats = build_features(make_df(12))
    assert ((feats["promotion_discount_pct"] >= 15).astype(int) == feats["high_promo_flag"]).all()


def test_geometric_adstock_matches_recursion():
    out = geometric_adstock(np.array([10.0, 0.0, 0.0]), 0.5)
    assert np.allclose(out, [10.0, 5.0, 2.5])


def test_validate_passes_on_clean_data_and_catches_bad_data():
    df = make_df(12)
    assert all(c["passed"] for c in validate(df).values())
    bad = df.copy(); bad.loc[3, "clicks"] = bad.loc[3, "impressions"] + 1
    assert not validate(bad)["clicks_not_above_impressions"]["passed"]


def test_enforce_dtypes_makes_counts_integers():
    df = make_df(6)
    df["units_sold"] = df["units_sold"].astype(float)
    out = enforce_dtypes(df)
    assert out["units_sold"].dtype == "int64" and out["tv_spend_lkr"].dtype == "int64"
    assert out["promotion_discount_pct"].dtype == "float64"


def test_fix_dtypes_parses_month_and_numbers():
    df = pd.DataFrame({"month": ["2023-01-01", "2023-02-01"], "tv_spend_lkr": ["100", "200"], "country": [" LK ", "LK"]})
    out = fix_dtypes(df)
    assert str(out["month"].dtype).startswith("datetime64")
    assert out["tv_spend_lkr"].tolist() == [100, 200]
    assert out["country"].tolist() == ["LK", "LK"]


def test_drop_constant_columns():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    out, dropped = drop_constant_columns(df)
    assert dropped == ["a"] and list(out.columns) == ["b"]


def test_add_outlier_count_counts_flags_per_month():
    df = make_df(6)
    flags = pd.DataFrame({"month": [df.loc[2, "month"], df.loc[2, "month"], df.loc[4, "month"]]})
    out = add_outlier_count(df, flags)
    assert out["n_outlier_flags"].tolist() == [0, 0, 2, 0, 1, 0]


def test_hill_saturation_is_between_0_and_1_and_half_at_half_saturation():
    x = np.array([0.0, 5.0, 1000.0])
    out = hill_saturation(x, half_saturation=5.0, shape=2.0)
    assert out[0] == 0 and abs(out[1] - 0.5) < 1e-12 and 0 < out[2] < 1


def test_feature_dictionary_documents_every_column():
    feats = build_features(make_df(12).assign(n_outlier_flags=0))
    doc = feature_dictionary(feats)
    assert list(doc["column"]) == list(feats.columns)
    assert (doc["description"].str.len() > 0).all()


def test_validate_features_passes_and_detects_overlap():
    feats = build_features(make_df(36))
    train, test = chronological_split(feats)
    assert all(c["passed"] for c in validate_features(feats, train, test).values())
    overlapping_test = feats.iloc[20:]
    assert not validate_features(feats, train, overlapping_test)["train_before_test"]["passed"]


def test_gap_in_counts_is_filled_and_stays_integer():
    df = make_df(8)
    df.loc[3, "units_sold"] = np.nan
    df, report = handle_missing(df)
    df = enforce_dtypes(df)
    assert report == {"units_sold": 1}
    assert df["units_sold"].dtype == "int64" and df["units_sold"].notna().all()


def test_clean_pipeline_on_provided_workbook():
    from src.cleaning import clean_pipeline
    workbook = Path(__file__).resolve().parents[1] / "data" / "raw" / "Dataset.xlsx"
    if not workbook.exists():
        return
    result = clean_pipeline(str(workbook))
    clean = result["clean"]
    assert len(clean) == 36 and clean["month"].is_monotonic_increasing
    assert all(c["passed"] for c in validate(clean).values())
    assert result["log"]["total_spend_rows_adjusted"] == 16
