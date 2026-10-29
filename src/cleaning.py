"""Loading and cleaning of the MMM dataset.

The dataset has only 36 monthly rows, so rows are never dropped because
of outliers. Outliers are flagged and left in the data.
"""
import numpy as np
import pandas as pd

CHANNEL_COLS = [
    "tv_spend_lkr",
    "facebook_instagram_spend_lkr",
    "google_search_spend_lkr",
    "youtube_spend_lkr",
    "radio_spend_lkr",
    "outdoor_spend_lkr",
]
TOTAL_SPEND_COL = "total_ad_spend_lkr"
TARGET_COL = "sales_revenue_lkr"
EXTERNAL_COLS = [
    "average_price_index",
    "promotion_discount_pct",
    "competitor_pressure_index_0_100",
    "economic_sentiment_index_0_100",
    "lkr_usd_exchange_rate",
    "inflation_proxy_index",
    "seasonal_demand_index",
    "public_holiday_count",
]
META_COLS = ["country", "brand_category", "region_scope", "data_type"]
COUNT_COLS = CHANNEL_COLS + [TOTAL_SPEND_COL, TARGET_COL, "units_sold", "impressions", "clicks",
                             "conversions", "public_holiday_count"]
SPEND_TOLERANCE_LKR = 5


def load_raw(path, sheet="MMM_Data"):
    """Read the data sheet from the Excel workbook."""
    return pd.read_excel(path, sheet_name=sheet)


def standardise_columns(df):
    """Strip, lower-case and snake_case the column names."""
    out = df.copy()
    out.columns = (
        out.columns.str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True)
        .str.strip("_")
    )
    return out


def fix_dtypes(df):
    """Parse the month column as a date and make all other measures numeric."""
    out = df.copy()
    out["month"] = pd.to_datetime(out["month"], errors="coerce")
    for col in out.columns:
        if col == "month" or col in META_COLS:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in META_COLS:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
    return out


def remove_duplicates(df):
    """Drop exact duplicate rows, then duplicate months (first one is kept).

    Returns the cleaned frame and the number of rows removed.
    """
    before = len(df)
    out = df.drop_duplicates().drop_duplicates(subset="month", keep="first")
    return out.reset_index(drop=True), before - len(out)


def handle_missing(df):
    """Handle missing values in time order.

    Rows without a month or without the target are dropped. Other numeric
    gaps are filled by linear interpolation. Returns the frame and a dict
    with the number of missing cells found per column.
    """
    out = df.sort_values("month").reset_index(drop=True)
    report = {c: int(n) for c, n in out.isna().sum().items() if n > 0}
    out = out.dropna(subset=["month", TARGET_COL]).reset_index(drop=True)
    numeric = out.select_dtypes("number").columns
    out[numeric] = out[numeric].interpolate(method="linear", limit_direction="both")
    return out, report


def enforce_dtypes(df):
    """Store money amounts and counts as whole numbers (int64).

    Interpolating a gap would turn these columns into decimals, so they are
    rounded and converted back. Indices and percentages stay as floats.
    """
    out = df.copy()
    for col in COUNT_COLS:
        if col in out.columns:
            out[col] = out[col].round().astype("int64")
    return out


def reconcile_total_spend(df, tol=SPEND_TOLERANCE_LKR):
    """Make total_ad_spend_lkr equal to the sum of the six channels.

    Differences up to `tol` LKR are treated as rounding. A larger
    difference raises an error because it points to a real data problem.
    Returns the frame and the number of rows that differed.
    """
    out = df.copy()
    diff = (out[CHANNEL_COLS].sum(axis=1) - out[TOTAL_SPEND_COL]).abs()
    if (diff > tol).any():
        raise ValueError("total_ad_spend_lkr differs from the channel sum by more than the tolerance")
    n_changed = int((diff > 0).sum())
    out[TOTAL_SPEND_COL] = out[CHANNEL_COLS].sum(axis=1)
    return out, n_changed


def detect_outliers(df, cols=None, iqr_k=1.5, z_thresh=3.0):
    """Flag values outside the IQR fences or with |z| above z_thresh.

    Returns one row per flagged value.
    """
    cols = cols or list(df.select_dtypes("number").columns)
    rows = []
    for c in cols:
        s = df[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        z = (s - s.mean()) / s.std(ddof=0)
        flagged = (s < lower) | (s > upper) | (z.abs() > z_thresh)
        for idx in s.index[flagged]:
            outside_fence = s[idx] < lower or s[idx] > upper
            rows.append({
                "month": df.loc[idx, "month"],
                "column": c,
                "value": s[idx],
                "iqr_lower": lower,
                "iqr_upper": upper,
                "z_score": float(z[idx]),
                "rule": "IQR" if outside_fence else "z-score",
            })
    return pd.DataFrame(rows, columns=["month", "column", "value", "iqr_lower", "iqr_upper", "z_score", "rule"])


def add_outlier_count(df, outliers):
    """Add n_outlier_flags: number of flagged values in each month.

    Lets the modelling team rerun a model without the flagged months.
    """
    out = df.copy()
    counts = outliers.groupby("month").size()
    out["n_outlier_flags"] = out["month"].map(counts).fillna(0).astype(int)
    return out


def drop_constant_columns(df):
    """Remove columns that have a single value in every row."""
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    return df.drop(columns=constant), constant


def clean_pipeline(path):
    """Run every cleaning step and return the cleaned data with an audit log."""
    raw = load_raw(path)
    df = fix_dtypes(standardise_columns(raw))
    df, n_duplicates = remove_duplicates(df)
    df, missing = handle_missing(df)
    df = enforce_dtypes(df)
    df, n_total_changed = reconcile_total_spend(df)
    scan_cols = CHANNEL_COLS + [TARGET_COL, "units_sold"] + EXTERNAL_COLS
    outliers = detect_outliers(df, scan_cols)
    df = add_outlier_count(df, outliers)
    df, dropped = drop_constant_columns(df)
    log = {
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(df)),
        "duplicates_removed": int(n_duplicates),
        "missing_values_found": missing,
        "total_spend_rows_adjusted": n_total_changed,
        "outlier_flags": int(len(outliers)),
        "months_with_outlier_flag": int((df["n_outlier_flags"] > 0).sum()),
        "constant_columns_dropped": dropped,
    }
    return {"clean": df, "outliers": outliers, "log": log, "raw": raw}
