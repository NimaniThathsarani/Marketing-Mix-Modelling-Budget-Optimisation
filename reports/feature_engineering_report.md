# Feature Engineering Report and Train/Test Split

Code: `src/preprocessing.py`. Notebook: `notebooks/05_feature_engineering.ipynb`.
Column-by-column documentation: `reports/data_dictionary.md`.
Output: `data/processed/mmm_analysis_ready.csv` (36 rows, 66 columns: 22 cleaned columns and 44 new ones).

The notebook runs after the EDA (notebook 04), and the features follow from its findings.

## New features

| Group | Columns | Purpose |
|---|---|---|
| Time (6) | `year`, `month_num`, `quarter`, `time_index`, `month_sin`, `month_cos` | trend and smooth seasonality |
| Spend shape (14) | `<channel>_share` (6), `log_<channel>_spend` (6), `log_total_ad_spend`, `spend_rolling3_mean` | channel mix, concave transform, smoothing |
| Lags (12) | `<channel>_spend_lag1`, `<channel>_spend_lag2` | delayed effect, earlier months only |
| Efficiency (7) | `ctr`, `cvr`, `cpm_lkr`, `cpc_lkr`, `cost_per_conversion_lkr`, `avg_unit_revenue_lkr`, `roas_blended` | dashboard and diagnostics |
| External (5) | `high_promo_flag`, `high_holiday_flag`, `revenue_real_lkr`, `fx_mom_change_pct`, `revenue_mom_growth_pct` | regimes and real-terms view |

Notes:

* The two flags use fixed thresholds (promotion of at least 15%, at least 6 public holidays), not percentiles of the data, so the test period does not influence them.
* Empty values exist only because a lag, rolling mean or growth rate has no earlier month to use: 1 row for lag 1, growth and exchange-rate change, 2 rows for lag 2 and the rolling mean. They are not filled, because filling from later months would use future information. Lag-2 features are available from 2023-03.
* Scaling, adstock and saturation are not applied. Their parameters have to be estimated by the modelling team, and any scaler must be fitted on the training data only. The helpers `geometric_adstock` and `hill_saturation` are provided.
* Columns built from revenue (`roas_blended`, `avg_unit_revenue_lkr`, `revenue_real_lkr`, `revenue_mom_growth_pct`) and the funnel counts are marked `exclude` in the data dictionary so that they are not used as predictors.

## Chronological split

| Set | Rows | Period | File |
|---|---|---|---|
| Train | 30 | 2023-01 to 2025-06 | `data/processed/train.csv` |
| Test | 6 | 2025-07 to 2025-12 | `data/processed/test.csv` |

The data is not shuffled. The code stops with an error if the train period does not end before the test period starts, and unit tests cover this. The split is also saved in `data/processed/split_info.json`.

Walk-forward folds inside the training set (function `expanding_window_folds`):

| Fold | Training rows | Training ends | Validation period |
|---|---|---|---|
| 1 | 18 | 2024-06 | 2024-07 to 2024-09 |
| 2 | 21 | 2024-09 | 2024-10 to 2024-12 |
| 3 | 24 | 2024-12 | 2025-01 to 2025-03 |
| 4 | 27 | 2025-03 | 2025-04 to 2025-06 |

## Difference between train and test

| Variable | Train mean | Test mean | Change |
|---|---|---|---|
| Revenue (LKR) | 84.6 m | 87.6 m | +3.6% |
| Total ad spend (LKR) | 18.3 m | 19.9 m | +8.7% |
| Promotion discount (%) | 11.8 | 11.0 | -6.6% |
| Seasonal demand index | 1.047 | 1.047 | 0.0% |
| Inflation proxy index | 1.197 | 1.475 | +23.2% |
| LKR per USD | 334.0 | 344.1 | +3.0% |

The test period has a much higher inflation index and contains the December peak. A model that relies on the trending variables will have to predict outside the range it was trained on, so a single trend term and a regularised model are safer.
