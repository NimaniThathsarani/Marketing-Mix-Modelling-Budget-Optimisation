"""Validation checks for the cleaned data and for the engineered features."""
import json

import numpy as np
import pandas as pd

from .cleaning import CHANNEL_COLS, TARGET_COL, TOTAL_SPEND_COL
from .preprocessing import CHANNEL_NAMES, LAGS


def _result(passed, detail=""):
    return {"passed": bool(passed), "detail": detail}


def validate(df):
    """Checks on the cleaned data. Returns {check name: {passed, detail}}."""
    checks = {}
    n_missing = int(df.isna().sum().sum())
    checks["no_missing_values"] = _result(n_missing == 0, f"{n_missing} missing cells")
    n_dup = int(df.duplicated().sum())
    checks["no_duplicate_rows"] = _result(n_dup == 0, f"{n_dup} duplicate rows")
    checks["unique_months"] = _result(df["month"].is_unique)
    checks["sorted_by_month"] = _result(df["month"].is_monotonic_increasing)
    checks["month_start_dates"] = _result((df["month"].dt.day == 1).all())
    expected = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
    checks["no_missing_months"] = _result(len(expected) == len(df), f"expected {len(expected)}, found {len(df)}")
    checks["spend_non_negative"] = _result((df[CHANNEL_COLS] >= 0).all().all())
    checks["revenue_positive"] = _result((df[TARGET_COL] > 0).all())
    checks["total_equals_channel_sum"] = _result((df[CHANNEL_COLS].sum(axis=1) == df[TOTAL_SPEND_COL]).all())
    checks["clicks_not_above_impressions"] = _result((df["clicks"] <= df["impressions"]).all())
    checks["conversions_not_above_clicks"] = _result((df["conversions"] <= df["clicks"]).all())
    checks["promotion_pct_in_0_100"] = _result(df["promotion_discount_pct"].between(0, 100).all())
    for c in ["competitor_pressure_index_0_100", "economic_sentiment_index_0_100"]:
        checks[f"{c}_in_0_100"] = _result(df[c].between(0, 100).all())
    checks["holiday_count_0_to_31"] = _result(df["public_holiday_count"].between(0, 31).all())
    checks["finite_numeric_values"] = _result(np.isfinite(df.select_dtypes("number")).all().all())
    return checks


def validate_features(feats, train, test):
    """Checks on the engineered features and the train/test split."""
    checks = {}
    share_cols = [f"{n}_share" for n in CHANNEL_NAMES.values()]
    checks["channel_shares_sum_to_one"] = _result(np.allclose(feats[share_cols].sum(axis=1), 1.0))

    lag_ok = True
    for lag in LAGS:
        for col, name in CHANNEL_NAMES.items():
            expected = feats[col].shift(lag)
            lag_ok &= feats[f"{name}_spend_lag{lag}"].equals(expected)
    checks["lags_use_earlier_months_only"] = _result(lag_ok)

    flags_ok = set(feats["high_promo_flag"].unique()) <= {0, 1} and set(feats["high_holiday_flag"].unique()) <= {0, 1}
    checks["flags_are_binary"] = _result(flags_ok)
    checks["no_infinite_values"] = _result(~np.isinf(feats.select_dtypes("number")).any().any())

    nan_cols = feats.columns[feats.isna().any()]
    allowed = {f"{n}_spend_lag{lag}" for n in CHANNEL_NAMES.values() for lag in LAGS}
    allowed |= {"spend_rolling3_mean", "fx_mom_change_pct", "revenue_mom_growth_pct"}
    checks["missing_values_only_where_expected"] = _result(set(nan_cols) <= allowed, ", ".join(nan_cols))

    checks["train_before_test"] = _result(train["month"].max() < test["month"].min())
    checks["train_and_test_cover_all_rows"] = _result(len(train) + len(test) == len(feats))
    checks["train_test_no_shared_months"] = _result(not set(train["month"]) & set(test["month"]))
    return checks


def build_quality_summary(raw, clean, log, checks):
    """Dictionary saved as data_quality_summary.json."""
    return {
        "dataset": "Dataset.xlsx, sheet MMM_Data",
        "period": [str(clean["month"].min().date()), str(clean["month"].max().date())],
        "frequency": "monthly",
        "rows_raw": log["rows_raw"],
        "rows_clean": log["rows_clean"],
        "columns_raw": int(raw.shape[1]),
        "columns_clean": int(clean.shape[1]),
        "duplicates_removed": log["duplicates_removed"],
        "missing_values_found": log["missing_values_found"],
        "total_spend_rows_adjusted": log["total_spend_rows_adjusted"],
        "outlier_flags": log["outlier_flags"],
        "months_with_outlier_flag": log["months_with_outlier_flag"],
        "constant_columns_dropped": log["constant_columns_dropped"],
        "checks_passed": int(sum(c["passed"] for c in checks.values())),
        "checks_total": len(checks),
        "checks": checks,
    }


def save_summary(summary, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
