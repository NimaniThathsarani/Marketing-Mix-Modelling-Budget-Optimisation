# Marketing Mix Modelling Report

Code: `src/mmm_core.py`, `src/mmm_response.py`.

Notebooks:
`notebooks/06_mmm_baseline_and_controls.ipynb`,
`notebooks/07_mmm_training_and_evaluation.ipynb`,
`notebooks/08_mmm_contributions_and_response_curves.ipynb`,
`notebooks/09_mmm_alternative_model.ipynb`.

Input:
`data/processed/mmm_analysis_ready.csv`
and
`data/processed/split_info.json`.

Outputs:
`models/mmm/`,
`outputs/mmm/`,
figures
`reports/figures/06_*`,
`07_*`,
`08_*`,
`09_*`.

Tests:
`tests/test_mmm_core.py`.

Target:
`sales_revenue_lkr`
(monthly revenue, LKR).

Train:
2023-01 to 2025-06 (30 months)

Test:
2025-07 to 2025-12 (6 months)

as defined in `split_info.json`.

---

# Model

For each channel *c* and month *t*:

### 1. Adstock (carryover)

`a_t = x_t + λ_c · a_(t−1)`

The adstock series is multiplied by `(1 − λ_c)` so that steady-state adstock remains in spend units. A proportion of each month's advertising effect is carried forward into future months. The first observation assumes prior spending was at the same level as the first month of data.

### 2. Saturation (diminishing returns)

Hill curve:

`h = s / (s + K_c)`

where

`s = adstock / mean spend`

and `K_c` represents the spend level, expressed as a multiple of average spend, at which the channel reaches half of its maximum effect.

### 3. Regression

`revenue_t = base + Σ β_c · h_ct + Σ γ_k · control_kt + ε`

with:

- β_c ≥ 0 (advertising should not reduce revenue),
- Ridge regularisation,
- external controls,
- non-negative media coefficients.

The final control set contains:

- Promotion discount %
- Seasonal demand index
- Competitor pressure index
- Economic sentiment index
- Average price index

---

| Choice | Reason |
|---|---|
| Price index kept, inflation and trend excluded | Inflation, trend and price index were almost perfectly correlated (≈0.99). Including all of them created severe multicollinearity. The price index produced the best cross-validation performance and was retained as the representative macroeconomic control. |
| Holidays excluded | Adding holiday counts increased validation error once the seasonal demand index was included. Seasonal demand already captures holiday-driven demand patterns. |
| Funnel metrics excluded | Impressions, clicks and conversions were not used because they are outcomes of advertising spend rather than independent drivers of revenue. Including them would introduce target leakage and distort media attribution. |
| λ bounded by channel type | TV 0.3–0.8, Outdoor 0.2–0.7, YouTube 0.1–0.6, Radio 0.1–0.5, Facebook/Instagram 0–0.4, Search 0–0.3 |
| Hill shape fixed at 1 | A more flexible S-curve could not be reliably estimated from only 36 observations and failed to improve cross-validation performance. |

The final specification balances predictive accuracy, stability and business interpretability while remaining appropriate for the small sample size available.

---

λ, K and ridge strength were selected using SciPy Differential Evolution and optimised against rolling-origin cross-validation RMSE.

Cross-validation setup:

- Minimum training window: 18 months
- Forecast horizon: 3 months
- Rolling step: 1 month
- Total folds: 10

The test data (2025-07 → 2025-12) was used only for final evaluation.

The final model was refit on the full dataset using the selected hyperparameters.

---

# Model Comparison

| Model | CV RMSE | Test RMSE | Test MAE | Test MAPE | Test R² |
|---|---|---|---|---|---|
| M0 Baseline OLS (all controls) | 6.38 m | 3.48 m | 2.97 m | 3.4% | 0.55 |
| M1 Linear, constrained + ridge | 3.21 m | 2.51 m | 2.43 m | 2.8% | 0.77 |
| M2 Adstock | 3.09 m | 1.98 m | 1.85 m | 2.1% | 0.86 |
| M3 Adstock + saturation | 3.08 m | 2.04 m | 1.90 m | 2.1% | 0.85 |
| **M3a + price index (final)** | **2.96 m** | **1.75 m** | **1.56 m** | **1.8%** | **0.89** |
| M3b all 6 controls | 2.97 m | 1.99 m | 1.89 m | 2.2% | 0.85 |
| M3c S-shaped curve | 3.01 m | 2.13 m | 1.97 m | 2.2% | 0.83 |

The final model selection rule was defined before test evaluation.

A candidate model had to:

1. Produce the lowest rolling-origin CV RMSE.
2. Improve upon standard M3 by at least 2%.

Under these criteria, M3a was selected as the final MMM.

The baseline OLS assigned Radio an implausible negative coefficient (approximately −9 LKR revenue per LKR spent), which motivated the use of non-negative coefficient constraints in the final specification.

---

Several validation procedures were performed to evaluate model robustness.

Rather than relying solely on holdout test performance, the model was examined using:

- Rolling-origin cross-validation
- Residual diagnostics
- Hyperparameter search stability checks
- Bootstrap resampling

These procedures help ensure that the reported channel effects are not dependent on a particular train/test split or optimisation run.

### Reliability Checks

- **Residual diagnostics:** Durbin-Watson 1.76, Ljung-Box p = 0.29, Shapiro-Wilk p = 0.13, Breusch-Pagan p = 0.23.
- **Search stability:** Multiple random seeds produced nearly identical solutions and similar ROAS estimates.
- **Bootstrap:** 1,000 moving-block bootstrap samples (3-month blocks) were used to estimate uncertainty intervals.

---

# Alternative Model (Notebook 09)

To test whether the conclusions of the final MMM remain consistent under different modelling assumptions, two independent benchmark models were developed.

## Model A: Ridge Regression with Log Spend

Features:

- log_tv_spend
- log_facebook_instagram_spend
- log_google_search_spend
- log_youtube_spend
- log_radio_spend
- log_outdoor_spend

plus the same five controls used by M3a.

The logarithmic transformation approximates diminishing returns without explicitly modelling saturation.

### Model A Results

| Metric | Value |
|---|---|
| Train R² | 0.747 |
| Test RMSE | 2.15 m |
| Test MAE | 1.82 m |
| Test MAPE | 2.06% |
| Test R² | 0.83 |

---

## Model B: Ridge Regression with Lagged Spend

Features:

- Current spend
- Spend lagged one month
- Final control variables

Lagged spending provides a simple approximation of carry-over effects.

### Model B Results

| Metric | Value |
|---|---|
| Train R² | 0.847 |
| Test RMSE | 1.16 m |
| Test MAE | 1.00 m |
| Test MAPE | 1.13% |
| Test R² | 0.95 |

---

## Alternative Model Findings

Both alternative specifications broadly support the conclusions of the final MMM.

Consistent findings:

- Google Search remains a high-performing channel.
- YouTube remains one of the strongest digital channels.
- Outdoor consistently shows strong effectiveness.
- Radio remains the weakest channel.

Model A channel ranking:

1. Google Search
2. YouTube
3. Outdoor
4. TV
5. Facebook/Instagram
6. Radio

Model B channel ranking:

1. Google Search
2. Facebook/Instagram
3. YouTube
4. Outdoor
5. TV
6. Radio

The final MMM ROAS ranking was:

1. Outdoor
2. YouTube
3. Google Search
4. TV
5. Facebook/Instagram
6. Radio

Although exact rankings differ across methods, the strongest and weakest channels remain broadly consistent.

Both benchmark models assigned negative coefficients to Radio.

This behaviour is a limitation of unconstrained regression models and reinforces the final MMM design choice of enforcing non-negative media coefficients.

Overall, the alternative modelling exercise increases confidence that the main strategic conclusions are robust across modelling approaches.

---

# Channel Results (Final Model, 2023–2025)

| Channel | Spend share | Effect share | ROAS | 90% interval | P(ROAS > 1) | Marginal ROAS at 2025 spend | Carryover λ |
|---|---|---|---|---|---|---|---|
| Outdoor | 14.1% | 34.6% | 6.12 | 1.91 – 7.30 | 0.99 | 1.91 | 0.38 |
| YouTube | 12.0% | 20.0% | 4.17 | 0.10 – 6.70 | 0.90 | 1.16 | 0.11 |
| Google Search | 17.9% | 29.3% | 4.08 | 0.15 – 4.78 | 0.85 | 1.27 | 0.00 |
| TV | 23.5% | 9.3% | 0.98 | 0.00 – 2.22 | 0.46 | 0.48 | 0.30 |
| Facebook/Instagram | 23.7% | 6.8% | 0.72 | 0.00 – 2.56 | 0.60 | 0.41 | 0.08 |
| Radio | 9.0% | 0.0% | 0.00 | 0.00 – 7.34 | 0.55 | 0.00 | 0.22 |

ROAS = incremental revenue ÷ spend.

Marginal ROAS = revenue generated by the next LKR spent at the current average monthly spend level.

Media contributes approximately 54% of total revenue, producing an overall media ROAS of 2.49.

---

# Limitations

- The dataset contains only 36 monthly observations, which limits statistical power and increases uncertainty around channel-level estimates.
- ROAS is calculated using revenue rather than profit. A ROAS above one does not guarantee profitability.
- The bootstrap intervals condition on fixed adstock and saturation parameters and therefore underestimate total parameter uncertainty.
- Baseline revenue without media is estimated by extrapolation because no zero-spend periods exist in the dataset.
- Ridge regularisation shrinks coefficients toward zero and may produce conservative effect estimates.
- The dataset is synthetic and the model identifies statistical associations rather than experimentally verified causal effects.
- Both alternative Ridge benchmark models produced negative Radio coefficients. Such behaviour can occur in unconstrained regression due to noise and multicollinearity and supports the use of non-negative media constraints in the final MMM.

---

# Hand-off

| File | For | Content |
|---|---|---|
| `src/mmm_response.py` + `models/mmm/final_mmm.json` | Budget Optimisation | ResponseModel functions |
| `outputs/mmm/channel_summary.csv` | ROI, Dashboard, Report | Contribution, ROAS, marginal ROAS, λ, K and saturation |
| `outputs/mmm/roas_bootstrap.csv` | ROI, Report | ROAS intervals |
| `outputs/mmm/response_curves.csv` | ROI, Dashboard | Spend-response curves |
| `outputs/mmm/contributions_monthly.csv`, `contributions_long.csv` | ROI, Dashboard | Monthly decomposition |
| `outputs/mmm/adstock_decay.csv` | Dashboard, Report | Adstock decay profiles |
| `outputs/mmm/07_model_comparison.csv` | Dashboard, Report | Model comparison metrics |
| `outputs/mmm/09_alternative_model_comparison.csv` | Dashboard, Report | Alternative model evaluation results |

---

Using the response functions:

```python
from src.mmm_response import ResponseModel

rm = ResponseModel()

rm.current_spend
rm.total_revenue(spend)
rm.gradient(spend)