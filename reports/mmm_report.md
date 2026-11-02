# Marketing Mix Modelling Report

Code: `src/mmm_core.py`, `src/mmm_response.py`. Notebooks: `notebooks/06_mmm_baseline_and_controls.ipynb`, `notebooks/07_mmm_training_and_evaluation.ipynb`, `notebooks/08_mmm_contributions_and_response_curves.ipynb`, `notebooks/09_mmm_alternative_model.ipynb`.
Input: `data/processed/mmm_analysis_ready.csv` and `data/processed/split_info.json` (Team 1).
Outputs: `models/mmm/`, `outputs/mmm/`, figures `reports/figures/06_*` to `09_*`. Tests: `tests/test_mmm_core.py`.

Target: `sales_revenue_lkr` (monthly revenue, LKR). Train 2023-01 to 2025-06 (30 months), test 2025-07 to 2025-12 (6 months), as defined in `split_info.json`.

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

- **Residual diagnostics:** Durbin-Watson 1.76, Ljung-Box p = 0.29, Shapiro-Wilk p = 0.13, Breusch-Pagan p = 0.23. There is no evidence of autocorrelation, non-normality or heteroscedasticity in the residuals.
- **Search stability:** Three random seeds gave the same CV error (2.96 m) and similar channel ROAS.
- **Reproducibility:** Notebook 07 was re-run on a second machine (Windows) and reproduced the same model; all parameters matched to floating-point precision.
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

### Why M3a remains the final model

Model B has the best test score of all models (test R² 0.95, MAPE 1.13%), but it is not selected:

- **The model was chosen by cross-validation, not by the test set.** Model B has the worst CV RMSE of the non-baseline models (3.85 m vs 2.96 m for M3a), so its validation error across 10 time-ordered folds is about 30% higher than M3a's.
- **The test set is only 6 months.** A single good 6-month window can happen by chance; choosing the model on it would be overfitting to the test set.
- **Model B is less reliable for attribution.** It estimates 17 coefficients from 29 months without constraints and gives Radio a negative effect, which is not plausible.
- **M3a provides what the next teams need:** non-negative channel effects and smooth response curves for ROI analysis and budget optimisation.

**M3a stays the final model for all hand-offs.**

### Channel ranking across methods

| Rank | M3a (ROAS) | Model A (log Ridge) | Model B (lag Ridge) |
|---|---|---|---|
| 1 | Outdoor | Google Search | Google Search |
| 2 | YouTube | YouTube | Facebook/Instagram |
| 3 | Google Search | Outdoor | YouTube |
| 4 | TV | TV | Outdoor |
| 5 | Facebook/Instagram | Facebook/Instagram | TV |
| 6 | Radio | Radio | Radio |

- **Consistent across all three methods:** Google Search and YouTube are strong channels (top 3 in every model), and Radio is the weakest.
- **Method-dependent:** Outdoor ranks 1st in M3a but 3rd–4th in the Ridge models, and Facebook/Instagram ranks 2nd in Model B but 5th in the other two. Conclusions about these two channels should be treated with more caution.
- Both Ridge models gave Radio a negative coefficient. Unconstrained regression can produce implausible negative effects from noise and correlated channels, which supports M3a's non-negative constraint.

Overall, the alternative models confirm the robust findings (Search and YouTube strong, Radio weak) and show which channel conclusions depend on the modelling method.

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

Key findings:

- Outdoor, YouTube and Google Search receive 44% of the spend but deliver 84% of the media effect. TV and Facebook/Instagram receive 47% of the spend and deliver 16%.
- Carryover is short for every channel (half-life under one month), which fits monthly FMCG data.
- Every channel is already past the steep part of its response curve, so extra spend still adds revenue, but less each time. Budget should move from channels with low marginal ROAS to channels with high marginal ROAS until the marginal ROAS values are about equal. This is Team 4's optimisation.
- Illustration: moving 1.83 m LKR a month from Radio to Outdoor, with the total budget unchanged, adds about 2.4 m LKR of revenue a month.

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

Using the response functions (from the project root):

```python
from src.mmm_response import ResponseModel

rm = ResponseModel()
rm.current_spend          # 2025 average monthly spend per channel
rm.total_revenue(spend)   # incremental revenue per month for {channel: spend}
rm.gradient(spend)        # marginal ROAS per channel (jacobian for SLSQP)
```

The response curves are concave, so `scipy.optimize.minimize(lambda x: -total, method="SLSQP")` with a budget equality constraint and per-channel bounds has a single optimum.

Requirements: `pandas numpy scipy statsmodels scikit-learn matplotlib joblib`. Notebook 07 takes about 10–15 minutes (hyperparameter search); notebook 09 runs in under a minute. `build_mmm_notebooks.py` regenerates notebooks 06–08 from source.
