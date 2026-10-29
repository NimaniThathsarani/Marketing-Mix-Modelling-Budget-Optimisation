"""Feature engineering, feature documentation and the chronological split."""
import numpy as np
import pandas as pd

from .cleaning import CHANNEL_COLS, TARGET_COL, TOTAL_SPEND_COL

CHANNEL_NAMES = {c: c.replace("_spend_lkr", "") for c in CHANNEL_COLS}
HIGH_PROMO_PCT = 15.0     # fixed business threshold, not taken from the data
HIGH_HOLIDAY_COUNT = 6    # months with at least this many public holidays
TEST_MONTHS = 6           # last 6 months are held out
LAGS = (1, 2)


def add_time_features(df):
    out = df.copy()
    out["year"] = out["month"].dt.year
    out["month_num"] = out["month"].dt.month
    out["quarter"] = out["month"].dt.quarter
    out["time_index"] = np.arange(len(out))
    out["month_sin"] = np.sin(2 * np.pi * out["month_num"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month_num"] / 12)
    return out


def add_spend_features(df):
    out = df.copy()
    for col, name in CHANNEL_NAMES.items():
        out[f"{name}_share"] = out[col] / out[TOTAL_SPEND_COL]
        out[f"log_{name}_spend"] = np.log1p(out[col])
    out["log_total_ad_spend"] = np.log1p(out[TOTAL_SPEND_COL])
    out["spend_rolling3_mean"] = out[TOTAL_SPEND_COL].rolling(3, min_periods=3).mean()
    return out


def add_lag_features(df, lags=LAGS):
    """Spend of earlier months only, so nothing from the future is used."""
    out = df.copy()
    for lag in lags:
        for col, name in CHANNEL_NAMES.items():
            out[f"{name}_spend_lag{lag}"] = out[col].shift(lag)
    return out


def add_efficiency_features(df):
    out = df.copy()
    out["ctr"] = out["clicks"] / out["impressions"]
    out["cvr"] = out["conversions"] / out["clicks"]
    out["cpm_lkr"] = out[TOTAL_SPEND_COL] / out["impressions"] * 1000
    out["cpc_lkr"] = out[TOTAL_SPEND_COL] / out["clicks"]
    out["cost_per_conversion_lkr"] = out[TOTAL_SPEND_COL] / out["conversions"]
    out["avg_unit_revenue_lkr"] = out[TARGET_COL] / out["units_sold"]
    out["roas_blended"] = out[TARGET_COL] / out[TOTAL_SPEND_COL]
    return out


def add_external_features(df):
    out = df.copy()
    out["high_promo_flag"] = (out["promotion_discount_pct"] >= HIGH_PROMO_PCT).astype(int)
    out["high_holiday_flag"] = (out["public_holiday_count"] >= HIGH_HOLIDAY_COUNT).astype(int)
    out["revenue_real_lkr"] = out[TARGET_COL] / out["inflation_proxy_index"]
    out["fx_mom_change_pct"] = out["lkr_usd_exchange_rate"].pct_change() * 100
    out["revenue_mom_growth_pct"] = out[TARGET_COL].pct_change() * 100
    return out


def build_features(clean):
    """Apply all feature steps to the cleaned data (sorted by month)."""
    df = clean.sort_values("month").reset_index(drop=True)
    for step in (add_time_features, add_spend_features, add_lag_features,
                 add_efficiency_features, add_external_features):
        df = step(df)
    return df


def geometric_adstock(x, decay):
    """Carry-over effect: a[t] = x[t] + decay * a[t-1]. Helper for the model."""
    out = np.zeros(len(x), dtype=float)
    carry = 0.0
    for i, value in enumerate(np.asarray(x, dtype=float)):
        carry = value + decay * carry
        out[i] = carry
    return out


def hill_saturation(x, half_saturation, shape=1.0):
    """Saturation curve x^s / (x^s + k^s), between 0 and 1. Helper for the model."""
    x = np.asarray(x, dtype=float)
    return x ** shape / (x ** shape + half_saturation ** shape)


def chronological_split(df, test_months=TEST_MONTHS):
    """The last `test_months` rows are the test set. No shuffling."""
    df = df.sort_values("month").reset_index(drop=True)
    cut = len(df) - test_months
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    if train["month"].max() >= test["month"].min():
        raise ValueError("train and test periods overlap")
    return train, test


def expanding_window_folds(n_rows, n_splits=4, test_size=3, min_train=18):
    """Walk-forward folds (train_idx, validation_idx) for use inside the train set."""
    first_val_start = n_rows - n_splits * test_size
    if first_val_start < min_train:
        raise ValueError("not enough rows for the requested folds")
    folds = []
    for k in range(n_splits):
        start = first_val_start + k * test_size
        folds.append((np.arange(0, start), np.arange(start, start + test_size)))
    return folds


def split_info(train, test, folds):
    """Description of the split, saved as split_info.json."""
    return {
        "split_type": "chronological, no shuffling",
        "train_rows": int(len(train)),
        "train_start": str(train["month"].min().date()),
        "train_end": str(train["month"].max().date()),
        "test_rows": int(len(test)),
        "test_start": str(test["month"].min().date()),
        "test_end": str(test["month"].max().date()),
        "walk_forward_folds": [
            {
                "fold": i + 1,
                "train_rows": int(len(a)),
                "train_end": str(train["month"].iloc[a[-1]].date()),
                "validation_start": str(train["month"].iloc[b[0]].date()),
                "validation_end": str(train["month"].iloc[b[-1]].date()),
            }
            for i, (a, b) in enumerate(folds)
        ],
    }


# ---------------------------------------------------------------- documentation
_BASE_DOCS = {
    "month": ("Month start date", "index"),
    "tv_spend_lkr": ("TV advertising spend, LKR", "media"),
    "facebook_instagram_spend_lkr": ("Facebook / Instagram advertising spend, LKR", "media"),
    "google_search_spend_lkr": ("Google Search advertising spend, LKR", "media"),
    "youtube_spend_lkr": ("YouTube advertising spend, LKR", "media"),
    "radio_spend_lkr": ("Radio advertising spend, LKR", "media"),
    "outdoor_spend_lkr": ("Outdoor advertising spend, LKR", "media"),
    "total_ad_spend_lkr": ("Sum of the six channel spends", "exclude: sum of media columns"),
    "sales_revenue_lkr": ("Monthly sales revenue, LKR", "target"),
    "units_sold": ("Monthly units sold", "alternative target"),
    "impressions": ("Total ad impressions", "exclude: result of spend"),
    "clicks": ("Total ad clicks", "exclude: result of spend"),
    "conversions": ("Total ad conversions", "exclude: result of spend"),
    "average_price_index": ("Relative price index, about 1.0 at the start", "control, collinear with inflation and time"),
    "promotion_discount_pct": ("Average promotion discount, percent", "control"),
    "competitor_pressure_index_0_100": ("Competitor pressure index, 0 to 100", "control"),
    "economic_sentiment_index_0_100": ("Economic sentiment index, 0 to 100", "control"),
    "lkr_usd_exchange_rate": ("LKR per USD", "control"),
    "inflation_proxy_index": ("Inflation proxy index, rises almost linearly with time", "control, collinear with price and time"),
    "seasonal_demand_index": ("Seasonal demand multiplier, includes festive peaks", "control"),
    "public_holiday_count": ("Number of public holidays in the month", "control"),
    "n_outlier_flags": ("Number of values flagged as outliers in the month (see outlier_flags.csv)", "diagnostic only"),
    "year": ("Calendar year", "time control"),
    "month_num": ("Calendar month, 1 to 12", "time control"),
    "quarter": ("Calendar quarter, 1 to 4", "time control"),
    "time_index": ("Month counter starting at 0", "time control, collinear with inflation"),
    "month_sin": ("sin(2 * pi * month / 12), smooth seasonal term", "time control"),
    "month_cos": ("cos(2 * pi * month / 12), smooth seasonal term", "time control"),
    "log_total_ad_spend": ("log(1 + total ad spend)", "exclude: sum of media columns"),
    "spend_rolling3_mean": ("Trailing 3-month mean of total ad spend", "exclude: sum of media columns"),
    "ctr": ("Clicks divided by impressions", "diagnostic only"),
    "cvr": ("Conversions divided by clicks", "diagnostic only"),
    "cpm_lkr": ("Ad spend per 1,000 impressions, LKR", "diagnostic only"),
    "cpc_lkr": ("Ad spend per click, LKR", "diagnostic only"),
    "cost_per_conversion_lkr": ("Ad spend per conversion, LKR", "diagnostic only"),
    "avg_unit_revenue_lkr": ("Revenue divided by units sold", "exclude: built from the target"),
    "roas_blended": ("Revenue divided by total ad spend", "exclude: built from the target"),
    "high_promo_flag": (f"1 if promotion discount is at least {HIGH_PROMO_PCT:g} percent", "control"),
    "high_holiday_flag": (f"1 if the month has at least {HIGH_HOLIDAY_COUNT} public holidays", "control"),
    "revenue_real_lkr": ("Revenue divided by the inflation proxy index", "exclude: built from the target"),
    "fx_mom_change_pct": ("Month-on-month change in the exchange rate, percent", "control"),
    "revenue_mom_growth_pct": ("Month-on-month revenue growth, percent", "exclude: built from the target"),
}


def feature_docs():
    """Description and modelling role of every column in the analysis-ready data."""
    docs = dict(_BASE_DOCS)
    for name in CHANNEL_NAMES.values():
        docs[f"{name}_share"] = (f"Share of total ad spend going to {name}; the six shares add up to 1", "media mix, drop one share if all are used")
        docs[f"log_{name}_spend"] = (f"log(1 + {name} spend), concave transform", "media, transformed")
        for lag in LAGS:
            docs[f"{name}_spend_lag{lag}"] = (f"{name} spend {lag} month(s) earlier", f"media, lag {lag}")
    return docs


def feature_dictionary(df):
    """One row per column of df with description, role, dtype and missing count."""
    docs = feature_docs()
    unknown = [c for c in df.columns if c not in docs]
    if unknown:
        raise KeyError(f"no documentation for columns: {unknown}")
    return pd.DataFrame([
        {"column": c, "description": docs[c][0], "modelling_role": docs[c][1],
         "dtype": str(df[c].dtype), "n_missing": int(df[c].isna().sum())}
        for c in df.columns
    ])


def write_data_dictionary(df, path):
    """Write reports/data_dictionary.md for the analysis-ready data."""
    doc = feature_dictionary(df)
    lines = [
        "# Data Dictionary: mmm_analysis_ready.csv",
        "",
        f"{len(doc)} columns, {len(df)} monthly rows ({df['month'].min().date()} to {df['month'].max().date()}). "
        "Generated by `src/preprocessing.py` (function `write_data_dictionary`).",
        "",
        "The modelling_role column is guidance for the modelling team. Columns marked `exclude` "
        "should not be used as predictors of revenue because they are a result of spend, are built "
        "from the target, or are a sum of other columns.",
        "",
        "| column | description | modelling_role | dtype | n_missing |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| `{r.column}` | {r.description} | {r.modelling_role} | {r.dtype} | {r.n_missing} |"
              for r in doc.itertuples()]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return doc
