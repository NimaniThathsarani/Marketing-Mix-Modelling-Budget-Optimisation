# Data Quality Report

Machine-readable version: `reports/data_quality_summary.json`.
Code: `src/validation.py`. Notebooks: `03_data_validation.ipynb` (data) and `05_feature_engineering.ipynb` (features and split).

## Result

16 of 16 checks on the cleaned data pass, and 8 of 8 checks on the engineered features and the split pass.

| Check | Result |
|---|---|
| No missing values, no duplicate rows, unique months | pass |
| Sorted by month, month-start dates, no missing months (36 found, 36 expected) | pass |
| Channel spend not negative, revenue positive | pass |
| Total spend equals the sum of the six channels | pass |
| Clicks not above impressions, conversions not above clicks | pass |
| Promotion between 0 and 100 percent, competitor and sentiment indices between 0 and 100, holidays between 0 and 31 | pass |
| All numeric values finite | pass |
| Spend shares add up to 1, lag columns equal the shifted spend, flags are 0 or 1, no infinite values | pass |
| Empty values only in lag, rolling-mean and growth columns | pass |
| Train period ends before test period starts, no shared months, train plus test equals all rows | pass |

## Problems found in the raw data

| Problem | Size | Handling |
|---|---|---|
| Total spend off by up to 2 LKR in 16 rows | negligible | replaced by the exact channel sum |
| Four columns with a single value | none | dropped |
| Six flagged values (absolute z-score 2.0 to 2.2) | low | kept and flagged |

## Limitations to keep in mind

1. The data is synthetic. The workbook states that exchange rate and inflation values are not real history.
2. There are only 36 monthly rows for 6 media and 8 external variables.
3. `inflation_proxy_index`, `average_price_index` and the month counter correlate 0.99 to 1.00. The exchange rate correlates 0.87 with them.
4. Impressions, clicks and conversions are results of spend and should not predict revenue.
5. Only `outdoor_spend_lkr` (Shapiro-Wilk p = 0.008) and `seasonal_demand_index` (p < 0.001) are clearly not normal. Revenue is (p = 0.89).
6. Revenue has almost no lag-1 autocorrelation (0.08). Price (0.99), inflation (1.00), sentiment (0.79) and exchange rate (0.70) have a high one.
