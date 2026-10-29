# Cleaning Report

Input: `data/raw/Dataset.xlsx`, sheet `MMM_Data` (36 rows, 25 columns).
Output: `data/processed/mmm_cleaned.csv` (36 rows, 22 columns).
Code: `src/cleaning.py`. Notebook: `notebooks/02_data_cleaning.ipynb`.

## Steps and decisions

| Step | Finding | Action |
|---|---|---|
| Column names | Already in snake_case | A standardising function is applied, nothing changed |
| Data types | `month` is read as text | Converted to a date (first day of the month). All other measures are already numeric |
| Missing values | None | Nothing to fill. For new data the pipeline drops rows with a missing target and fills other numeric gaps by linear interpolation in time order |
| Duplicates | No duplicate rows and no duplicate months | Nothing removed |
| Total spend | `total_ad_spend_lkr` differs from the sum of the six channels in 16 rows, by at most 2 LKR | Treated as rounding. The total is replaced by the exact channel sum. A difference above 5 LKR stops the pipeline with an error |
| Outliers | 6 flagged values in 6 different months (IQR rule or absolute z-score above 3) | All kept, see below |
| Constant columns | `country`, `brand_category`, `region_scope`, `data_type` have one value each | Dropped from the cleaned data |
| Outlier count | Needed for sensitivity checks | New column `n_outlier_flags` (flagged values per month) |

## Flagged values

| Month | Variable | Value | Other information |
|---|---|---|---|
| 2023-01 | sales_revenue_lkr | 68.7 m (low) | promotion 7.3%, seasonal index 1.08 |
| 2023-04 | sales_revenue_lkr | 100.7 m (high) | promotion 21.5%, 6 public holidays, seasonal index 1.19 |
| 2024-08 | sales_revenue_lkr | 69.9 m (low) | promotion 16.7%, seasonal index 0.99 |
| 2023-12, 2024-12, 2025-12 | seasonal_demand_index | 1.24 | festive peak every December |

The flagged values are kept for three reasons. With 36 rows every month matters. Lag features need unbroken months. And the absolute z-scores are only 2.0 to 2.2, with no sign of a data entry error. The full list is in `data/processed/outlier_flags.csv`.

The raw workbook is not modified.
