# Marketing Mix Modelling Report

Code: `src/mmm_core.py`, `src/mmm_response.py`. Notebooks: `notebooks/06_mmm_baseline_and_controls.ipynb`,
`notebooks/07_mmm_training_and_evaluation.ipynb`, `notebooks/08_mmm_contributions_and_response_curves.ipynb`.
Input: `data/processed/mmm_analysis_ready.csv` and `data/processed/split_info.json` (Team 1).
Outputs: `models/mmm/`, `outputs/mmm/`, figures `reports/figures/06_*`, `07_*`, `08_*`. Tests: `tests/test_mmm_core.py`.

Target: `sales_revenue_lkr` (monthly, LKR). Train 2023-01 to 2025-06 (30 months), test 2025-07 to 2025-12 (6 months), as in `split_info.json`.

## Model

For each channel *c* and month *t*:

1. **Adstock (carryover):** `a_t = x_t + λ_c · a_{t-1}`, multiplied by `(1 − λ_c)`. A share λ of the effect carries into the next month. Normalising keeps adstock in spend units, so with constant spend the adstock equals the spend. The first month assumes earlier spend was at the same level.
2. **Saturation (diminishing returns):** Hill curve `h = s / (s + K_c)`, where `s` = adstock ÷ the channel's mean monthly spend. `K_c` is the spend level, as a multiple of average spend, at which the channel reaches half of its maximum effect.
3. **Regression:** `revenue_t = base + Σ β_c · h_ct + Σ γ_k · control_kt + ε`, with **β_c ≥ 0** (advertising cannot reduce sales) and a **ridge penalty** (36 rows, correlated channels).

Controls in the final model: promotion %, seasonal demand index, competitor pressure, economic sentiment and the average price index.

| Choice | Reason |
|---|---|
| Price index kept, inflation and trend left out | Price, inflation and time are ~0.99 correlated, so only one can be used. Revenue has no trend (2023: 85.7 m, 2024: 82.8 m, 2025: 86.8 m a month). The price index reduced CV error by 4%. It acts as a price-level control, and its coefficient is not a price elasticity. |
| Holidays left out | Higher CV error once the seasonal index (which includes festive months) is in |
| Funnel counts (impressions, clicks, conversions) not used | They are results of spend, so using them would leak signal and take credit away from the channels |
| λ bounded per channel type | TV 0.3–0.8, Outdoor 0.2–0.7, YouTube 0.1–0.6, Radio 0.1–0.5, Facebook/Instagram 0–0.4, Search 0–0.3 |
| Hill shape fixed at 1 (concave) | An S-curve cannot be identified from 36 points and did not improve CV (model M3c). Concave curves give the optimiser a single optimum. |

λ, K and the ridge strength were chosen by SciPy differential evolution to minimise **rolling-origin CV RMSE** on the training months. Every validation window starts after its training window ends: 18 months minimum training, a 3-month horizon, and the origin moved 1 month at a time (10 folds). These are the same origin and horizon as Team 1's folds, but with more of them. The test months were used only once, at the end. The final coefficients were then refit on all 36 months with those settings.

## Model comparison

| Model | CV RMSE | Test RMSE | Test MAE | Test MAPE | Test R² |
|---|---|---|---|---|---|
| M0 Baseline OLS (all controls) | 6.38 m | 3.48 m | 2.97 m | 3.4% | 0.55 |
| M1 Linear, constrained + ridge | 3.21 m | 2.51 m | 2.43 m | 2.8% | 0.77 |
| M2 Adstock | 3.09 m | 1.98 m | 1.85 m | 2.1% | 0.86 |
| M3 Adstock + saturation | 3.08 m | 2.04 m | 1.90 m | 2.1% | 0.85 |
| **M3a + price index (final)** | **2.96 m** | **1.75 m** | **1.56 m** | **1.8%** | **0.89** |
| M3b all 6 controls | 2.97 m | 1.99 m | 1.89 m | 2.2% | 0.85 |
| M3c S-shaped curve | 3.01 m | 2.13 m | 1.97 m | 2.2% | 0.83 |

The rule for choosing the final model was fixed before looking at test results: lowest CV RMSE among the M3 variants, and a variant must beat plain M3 by at least 2% to be chosen.

The baseline OLS gives Radio a coefficient of about −9 LKR per LKR spent (p ≈ 0.06). That is not plausible, and it is the main reason for the non-negative, regularised model.

Reliability checks:

* **Residual diagnostics:** Durbin-Watson 1.76, Ljung-Box p = 0.29, Shapiro-Wilk p = 0.13, Breusch-Pagan p = 0.23. There is no evidence of autocorrelation, non-normality or heteroscedasticity.
* **Search stability:** three random seeds give the same CV error (2.96 m) and similar channel ROAS.
* **Bootstrap:** a moving-block bootstrap (1,000 resamples, 3-month blocks) gives the 90% intervals below.

## Channel results (final model, 2023–2025)

| Channel | Spend share | Effect share | ROAS | 90% interval | P(ROAS > 1) | Marginal ROAS at 2025 spend | Carryover λ |
|---|---|---|---|---|---|---|---|
| Outdoor | 14.1% | 34.6% | 6.12 | 1.91 – 7.30 | 0.99 | 1.91 | 0.38 |
| YouTube | 12.0% | 20.0% | 4.17 | 0.10 – 6.70 | 0.90 | 1.16 | 0.11 |
| Google Search | 17.9% | 29.3% | 4.08 | 0.15 – 4.78 | 0.85 | 1.27 | 0.00 |
| TV | 23.5% | 9.3% | 0.98 | 0.00 – 2.22 | 0.46 | 0.48 | 0.30 |
| Facebook/Instagram | 23.7% | 6.8% | 0.72 | 0.00 – 2.56 | 0.60 | 0.41 | 0.08 |
| Radio | 9.0% | 0.0% | 0.00 | 0.00 – 7.34 | 0.55 | 0.00 | 0.22 |

ROAS = incremental revenue ÷ spend. Marginal ROAS = revenue from the *next* LKR at the current (2025 average) monthly spend.

* In the model, media drives about 54% of revenue, giving an overall media ROAS of 2.49.
* Outdoor, YouTube and Search receive 44% of the spend but deliver 84% of the media effect. TV and Facebook/Instagram receive 47% of the spend and deliver 16%.
* Carryover is short for every channel (half-life under one month), which fits monthly FMCG data.
* Every channel is already past the steep part of its curve, so extra spend still adds revenue, but less each time. Budget should move from low to high marginal ROAS until the marginal ROAS values are about equal. This is Team 4's optimisation.
* Illustration: moving 1.83 m LKR a month from Radio to Outdoor, with the total budget unchanged, adds about 2.4 m LKR of revenue a month.

## Limitations

* With 36 monthly points, channel-level effects are uncertain. Only Outdoor's ROAS is above 1 with more than 95% probability. Radio's zero effect means "no measurable effect", not "proven useless".
* ROAS is based on revenue because there is no margin or cost data. A ROAS above 1 does not mean the spend is profitable.
* The bootstrap keeps λ and K fixed, so the real uncertainty is wider.
* Revenue "without media" extrapolates to zero spend, which never happens in the data. Marginal ROAS near current spend is the more reliable figure for decisions.
* The ridge penalty shrinks effects towards zero, so ROAS is slightly conservative.
* The data is synthetic, and the model shows associations rather than proven causal effects.

## Hand-off

| File | For | Content |
|---|---|---|
| `src/mmm_response.py` + `models/mmm/final_mmm.json` | Budget Optimisation | `ResponseModel`: `response`, `marginal_roas`, `total_revenue(spend)`, `gradient(spend)`, `current_spend` |
| `outputs/mmm/channel_summary.csv` | ROI, Dashboard, Report | spend, contribution, shares, ROAS, marginal ROAS, λ, half-life, K, saturation |
| `outputs/mmm/roas_bootstrap.csv` | ROI, Report | ROAS 90% intervals, P(ROAS > 1) |
| `outputs/mmm/response_curves.csv` | ROI, Dashboard | revenue, marginal and average ROAS on a 0–3× spend grid |
| `outputs/mmm/contributions_monthly.csv`, `contributions_long.csv` | ROI, Dashboard | monthly decomposition (base, external factors, channels) with spend |
| `outputs/mmm/adstock_decay.csv` | Dashboard, Report | share of effect by month after spend |
| `outputs/mmm/07_model_comparison.csv` | Dashboard, Report | metrics for every model |

Using the response functions (from the project root):

```python
from src.mmm_response import ResponseModel
rm = ResponseModel()
rm.current_spend                                 # 2025 average monthly spend per channel
rm.total_revenue(spend)                          # incremental revenue per month for {channel: spend}
rm.gradient(spend)                               # marginal ROAS per channel (jacobian for SLSQP)
```

The curves are concave, so `scipy.optimize.minimize(lambda x: -total, method="SLSQP")` with a budget equality constraint and per-channel bounds has one optimum.

Requirements: `pandas numpy scipy statsmodels matplotlib joblib`. Notebook 07 takes about 10 minutes (hyperparameter search). `build_mmm_notebooks.py` regenerates notebooks 06–08 from source.
