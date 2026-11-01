"""Builds the MMM notebooks from cell lists (keeps notebooks diff-friendly). Run: python build_nb.py"""
import nbformat as nbf
from pathlib import Path

NB = Path(__file__).parent / "notebooks"
NB.mkdir(exist_ok=True)


def save(name, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c.strip("\n")) for c in cells]
    nbf.write(nb, NB / name)


SETUP = r'''
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from src.mmm_core import *
from src import mmm_plot_style as ps; ps.apply()
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
FIG, TAB, MOD = ROOT/"reports/figures", ROOT/"outputs/mmm", ROOT/"models/mmm"
for d in (FIG, TAB, MOD): d.mkdir(parents=True, exist_ok=True)

# ---- Inputs from Team 1 (Data Engineering & EDA) ----
DATA_PATH = ROOT / "data" / "processed" / "mmm_analysis_ready.csv"
SPLIT = json.load(open(ROOT / "data" / "processed" / "split_info.json"))
TRAIN_END = SPLIT["train_end"]   # documented chronological split (train 2023-01..2025-06, test 2025-07..2025-12)
LAB = CHANNEL_LABELS
df = load_data(DATA_PATH)
train, test = df[df.month <= TRAIN_END].reset_index(drop=True), df[df.month > TRAIN_END].reset_index(drop=True)
print(f"{len(df)} months | train {train.month.min():%Y-%m} → {train.month.max():%Y-%m} ({len(train)}) | test {test.month.min():%Y-%m} → {test.month.max():%Y-%m} ({len(test)})")
'''

# =====================================================================================
nb1 = [
"""md:# 06 · Baseline Marketing Mix Model & control-variable selection
**Team 2 – Marketing Mix Modelling** · Target: `sales_revenue_lkr` (monthly, LKR)

This notebook:
1. Re-checks the data received from Team 1 (`data/processed/mmm_analysis_ready.csv`). These are sanity checks only; cleaning is Team 1's job.
2. Confirms that the chronological train/test split matches Team 1's `split_info.json`.
3. Selects the external factors (controls) for the model, using correlation, VIF and a trend check.
4. Fits the **baseline**: an ordinary least-squares regression on raw spend + controls, without carryover or saturation.

> The baseline is the reference point. Every later model must beat it on hold-out accuracy *and* give more credible channel effects.
""",
SETUP,
"""md:## 1. Sanity checks on the input data""",
r'''
checks = {
    "rows": len(df),
    "missing values": int(df.isna().sum().sum()),
    "duplicate months": int(df.month.duplicated().sum()),
    "months contiguous": bool((df.month.diff().dropna().dt.days.between(28, 31)).all()),
    "negative spend": int((df[CHANNELS] < 0).sum().sum()),
    "total_ad_spend = sum of channels": bool(np.allclose(df[CHANNELS].sum(1), df.total_ad_spend_lkr, atol=5)),
}
pd.Series(checks, name="result").to_frame()
''',
"""md:## 2. Chronological train / test split
Time-series data must not be shuffled. The **last 6 months (Jul–Dec 2025)** are held out as a test set and are never used to choose any model setting. Inside the 30 training months, model settings are chosen with **rolling-origin (expanding-window) cross-validation**: train on months 1…k, predict months k+1…k+3, then move k forward by one month.""",
r'''
assert (len(train), len(test)) == (SPLIT["train_rows"], SPLIT["test_rows"]), "split does not match split_info.json"
assert f"{test.month.min():%Y-%m-%d}" == SPLIT["test_start"]
print("Split matches Team 1's split_info.json ✓")
print("Team 1 walk-forward folds (step 3):", [(f["train_end"], f["validation_start"], f["validation_end"]) for f in SPLIT["walk_forward_folds"]])
print("Model tuning below uses the same origin (18 months) and horizon (3 months), but moves the origin 1 month at a time (10 folds) for a more stable signal.")
folds = pd.DataFrame([{"fold": i+1, "train_months": f"{df.month[tr[0]]:%Y-%m} → {df.month[tr[-1]]:%Y-%m}", "n_train": len(tr),
                       "validate_months": f"{df.month[va[0]]:%Y-%m} → {df.month[va[-1]]:%Y-%m}"}
                      for i, (tr, va) in enumerate(rolling_origin_splits(len(train), 18, 3, 1))])
folds
''',
r'''
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.plot(train.month, train[TARGET], color=ps.ACTUAL_COLOR, label="train")
ax.plot(test.month, test[TARGET], color=ps.PRED_COLOR, label="test (hold-out)")
ax.axvline(pd.Timestamp(TRAIN_END) + pd.Timedelta(days=15), color=ps.INK2, ls="--", lw=1)
ps.millions(ax); ax.set_title("Monthly sales revenue (LKR) and the train/test split"); ax.legend(loc="upper left", ncol=2)
fig.savefig(FIG / "06_train_test_split.png"); plt.show()
''',
"""md:## 3. Spend levels and raw relationships
Raw correlations are a first look only. They mix up the effect of a channel with everything else that moved at the same time (seasonality, promotions, other channels).""",
r'''
summary = pd.DataFrame({
    "avg monthly spend (LKR M)": df[CHANNELS].mean() / 1e6,
    "spend share %": df[CHANNELS].sum() / df[CHANNELS].sum().sum() * 100,
    "coef. of variation": df[CHANNELS].std() / df[CHANNELS].mean(),
    "corr with revenue": df[CHANNELS].corrwith(df[TARGET]),
}).rename(index=LAB)
summary.round(3)
''',
"""md:## 4. Selecting external factors (controls)
Candidates from the brief: promotions, holidays, inflation, seasonal demand, economic conditions, competition, price.

Collinear predictors make coefficients unstable, so we check **VIF** (variance inflation factor; > 10 is a problem) and whether there is a **trend** in revenue for trend-like variables to explain.""",
r'''
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
cand = CHANNELS + ALL_CONTROLS + ["inflation_proxy_index", "lkr_usd_exchange_rate"]
Xv = sm.add_constant(df[cand].assign(time_trend=np.arange(len(df))))
vif = pd.Series([variance_inflation_factor(Xv.values, i) for i in range(1, Xv.shape[1])], index=Xv.columns[1:], name="VIF")
vif.rename(index=LAB).sort_values(ascending=False).round(1).to_frame()
''',
r'''
yr = df.assign(year=df.month.dt.year).groupby("year").agg(
    revenue_M=(TARGET, lambda s: s.mean()/1e6), units_k=("units_sold", lambda s: s.mean()/1e3),
    price_index=("average_price_index", "mean"), inflation_index=("inflation_proxy_index", "mean"),
    ad_spend_M=("total_ad_spend_lkr", lambda s: s.mean()/1e6))
print("Correlation of price index with inflation index and time:",
      round(df.average_price_index.corr(df.inflation_proxy_index), 3), round(df.average_price_index.corr(pd.Series(np.arange(36))), 3))
yr.round(3)
''',
r'''
from matplotlib.colors import LinearSegmentedColormap
div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
ctrl_cand = ALL_CONTROLS + ["inflation_proxy_index", "lkr_usd_exchange_rate"]
corr = df[[TARGET] + ctrl_cand].corr()
short = {TARGET: "revenue", "promotion_discount_pct": "promotion %", "seasonal_demand_index": "seasonality",
         "public_holiday_count": "holidays", "competitor_pressure_index_0_100": "competitor pressure",
         "economic_sentiment_index_0_100": "econ. sentiment", "average_price_index": "price index",
         "inflation_proxy_index": "inflation", "lkr_usd_exchange_rate": "LKR/USD"}
fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(corr.values, cmap=div, vmin=-1, vmax=1)
lbl = [short[c] for c in corr.columns]
ax.set_xticks(range(len(lbl)), lbl, rotation=45, ha="right"); ax.set_yticks(range(len(lbl)), lbl); ax.grid(False)
for i in range(len(lbl)):
    for j in range(len(lbl)):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=8, color=ps.INK)
fig.colorbar(im, ax=ax, shrink=0.75); ax.set_title("Correlation: revenue vs candidate external factors")
fig.savefig(FIG / "06_controls_correlation.png"); plt.show()
''',
r'''
# Test the decision with time-aware CV: does adding the trend-like / holiday variables help forecasts?
def ols_cv(controls):
    m = MMM(use_adstock=False, use_saturation=False, nonneg_media=False, controls=controls)
    return cv_rmse(m, train) / 1e6
sets = {"final 4 controls": CONTROLS,
        "+ holidays": CONTROLS + ["public_holiday_count"],
        "+ price index": CONTROLS + ["average_price_index"],
        "+ inflation": CONTROLS + ["inflation_proxy_index"],
        "all 6 candidate controls": ALL_CONTROLS}
pd.Series({k: ols_cv(v) for k, v in sets.items()}, name="CV RMSE (LKR M), lower is better").round(3).to_frame()
''',
"""md:**Interim decision on controls** (confirmed with the full model in notebook 07):

* **Price index / inflation / exchange rate are kept out of the final model's starting set.** Price and inflation are ~0.99 correlated with each other and with time (VIF in the hundreds), so together they act as a single time trend. Revenue itself has **no trend** (2023: 85.7M, 2024: 82.8M, 2025: 86.8M a month), so whatever the model learns from that "trend" is not interpretable. Adding it lowers plain-OLS CV error a little, but OLS is unconstrained and overfits. Notebook 02 re-tests the price index inside the final constrained model and keeps it only if CV error improves there.
* **Public holiday count is excluded.** Adding it makes CV error worse. The seasonal demand index already includes festive periods.
* **Final starting set:** promotion %, seasonal demand index, competitor pressure, economic sentiment.
""",
"""md:## 5. Baseline model — OLS on raw spend + controls
Plain `statsmodels` OLS on the training months. No carryover, no diminishing returns, and coefficients may be negative. A media coefficient here reads as *extra revenue (LKR) per extra LKR spent in the same month*.""",
r'''
def fit_ols(controls):
    X = sm.add_constant(train[CHANNELS + controls]); return sm.OLS(train[TARGET], X).fit()
base_all = fit_ols(ALL_CONTROLS)
base_fin = fit_ols(CONTROLS)
coef = pd.DataFrame({"coef": base_all.params, "std err": base_all.bse, "p-value": base_all.pvalues,
                     "95% low": base_all.conf_int()[0], "95% high": base_all.conf_int()[1]}).rename(index=LAB)
print(f"Baseline OLS (all controls): train R² = {base_all.rsquared:.3f}, adj. R² = {base_all.rsquared_adj:.3f}")
coef.loc[[LAB[c] for c in CHANNELS]].round(3)
''',
r'''
rows = []
for name, res, ctr in [("Baseline OLS – all controls", base_all, ALL_CONTROLS), ("Baseline OLS – final controls", base_fin, CONTROLS)]:
    p_tr = res.predict(sm.add_constant(train[CHANNELS + ctr]))
    p_te = res.predict(sm.add_constant(test[CHANNELS + ctr], has_constant="add"))
    r = {"model": name, "train R2": res.rsquared, **{f"test {k}": v for k, v in metrics(test[TARGET], p_te).items()}}
    rows.append(r)
base_metrics = pd.DataFrame(rows).set_index("model")
base_metrics.to_csv(TAB / "06_baseline_metrics.csv"); base_metrics.round(3)
''',
r'''
ci = coef.loc[[LAB[c] for c in CHANNELS]]
fig, ax = plt.subplots(figsize=(8, 3.6))
y = np.arange(len(ci))[::-1]
for yi, (name, r) in zip(y, ci.iterrows()):
    col = ps.CHANNEL_COLORS[name]
    ax.plot([r["95% low"], r["95% high"]], [yi, yi], color=col, lw=2, solid_capstyle="round")
    ax.scatter(r["coef"], yi, s=60, color=col, edgecolor=ps.SURFACE, linewidth=2, zorder=3)
ax.axvline(0, color=ps.INK2, lw=1)
ax.set_yticks(y, ci.index); ax.set_xlabel("extra revenue per extra LKR spent (same month)")
ax.set_title("Baseline OLS: channel effects with 95% confidence intervals")
fig.savefig(FIG / "06_baseline_coefficients.png"); plt.show()
''',
"""md:### What the baseline tells us
* The model explains revenue well **in-sample** but most channel intervals are very wide and cross zero, because 30 months is not enough to separate six correlated channels.
* **Radio gets a large negative coefficient** (about −9 LKR per LKR, p ≈ 0.06). Advertising cannot plausibly *reduce* sales, so this is a sign of overfitting or confounding, not a real effect. This motivates the final model's design:
  1. **non-negative media coefficients**
  2. **adstock**, so the effect of spend can carry into later months
  3. **saturation**, so returns diminish as spend grows
  4. **ridge regularisation**, so effects are not split unstably across correlated channels
""",
r'''
import joblib
joblib.dump({"all_controls": base_all, "final_controls": base_fin}, MOD / "baseline_ols.pkl")
coef.to_csv(TAB / "06_baseline_coefficients.csv")
print("saved: baseline_ols.pkl, 01_baseline_coefficients.csv, 01_baseline_metrics.csv, train_test_split.csv")
''',
]
save("06_mmm_baseline_and_controls.ipynb", nb1)
print("ok")

# =====================================================================================
nb2 = [
"""md:# 07 · Adstock + saturation MMM: training, comparison & evaluation
**Team 2 – Marketing Mix Modelling**

The model ladder (each step adds one idea to the previous one):

| # | Model | Carryover (adstock) | Diminishing returns (saturation) | Media effects ≥ 0 | Ridge |
|---|---|---|---|---|---|
| M0 | Baseline OLS (all controls) | – | – | – | – |
| M1 | Linear, constrained | – | – | ✓ | ✓ |
| M2 | Adstock | ✓ | – | ✓ | ✓ |
| M3 | **Adstock + saturation (final)** | ✓ | ✓ | ✓ | ✓ |

**How the final model works**, for each channel *c* and month *t*:
1. **Adstock:** `a_t = x_t + λ_c · a_{t-1}`, rescaled by `(1-λ_c)`. A share λ of the effect carries into the next month. λ is limited to a plausible range for each channel type (e.g. TV 0.3–0.8, Search 0–0.3).
2. **Saturation (Hill curve):** `h_t = s/(s + K_c)` with `s = a_t / mean spend`. Revenue grows with spend but each extra LKR adds less. `K_c` is the spend level (as a multiple of average spend) where the channel reaches half of its maximum effect.
3. **Regression:** `revenue_t = base + Σ β_c·h_ct + Σ γ_k·control_kt + ε`, with **β_c ≥ 0** and a ridge penalty.

λ, K and the ridge strength are **hyperparameters**. They are chosen by differential evolution (SciPy) to minimise **rolling-origin CV error on the 30 training months only**. The 6 test months are used once, at the end.
""",
SETUP,
r'''
import time, joblib
SEARCH = dict(maxiter=40, popsize=10)   # differential-evolution budget (raise for a final run if time allows)
results, models = {}, {}

def evaluate(name, model, info=None):
    tr_pred, te_pred = model.predict(train), model.predict(test)
    r = {"train R2": metrics(train[TARGET], tr_pred)["R2"],
         "CV RMSE": (info or {}).get("cv_rmse", cv_rmse(model, train)),
         **{f"test {k}": v for k, v in metrics(test[TARGET], te_pred).items()}}
    model.fit(train)  # cv_rmse refits on folds; restore the full-train fit
    results[name], models[name] = r, model
    return r
''',
"""md:## 1. Train the model ladder""",
r'''
t0 = time.time()
evaluate("M0 Baseline OLS (all controls)", MMM(use_adstock=False, use_saturation=False, nonneg_media=False, controls=ALL_CONTROLS).fit(train))
m, info = tune_mmm(train, use_adstock=False, use_saturation=False, **SEARCH);  evaluate("M1 Linear, constrained + ridge", m, info)
m, info = tune_mmm(train, use_adstock=True,  use_saturation=False, **SEARCH);  evaluate("M2 Adstock", m, info)
m, info = tune_mmm(train, use_adstock=True,  use_saturation=True,  **SEARCH);  evaluate("M3 Adstock + saturation", m, info); final_info = info
print(f"trained in {time.time()-t0:.0f}s")
''',
"""md:## 2. Sensitivity variants of M3
Same structure, one change each. These confirm (or overturn) the choices of controls and curve shape.""",
r'''
t0 = time.time()
m, info = tune_mmm(train, controls=CONTROLS + ["average_price_index"], **SEARCH); evaluate("M3a + price index", m, info)
m, info = tune_mmm(train, controls=ALL_CONTROLS, **SEARCH);                       evaluate("M3b all 6 controls", m, info)
m, info = tune_mmm(train, alpha=None, **SEARCH);                                  evaluate("M3c S-shaped curve (alpha tuned)", m, info)
print(f"trained in {time.time()-t0:.0f}s")
''',
r'''
comp = pd.DataFrame(results).T
show = comp.copy()
for c in ["CV RMSE", "test RMSE", "test MAE"]: show[c] = show[c] / 1e6
show = show.rename(columns={"CV RMSE": "CV RMSE (M)", "test RMSE": "test RMSE (M)", "test MAE": "test MAE (M)"})
comp.to_csv(TAB / "07_model_comparison.csv")
show.round(3)
''',
r'''
fig, ax = plt.subplots(figsize=(9, 3.8))
order = list(comp.index)[::-1]
y = np.arange(len(order))
ax.barh(y + 0.18, comp.loc[order, "CV RMSE"], height=0.34, color="#86b6ef", label="CV RMSE (train period)")
ax.barh(y - 0.18, comp.loc[order, "test RMSE"], height=0.34, color="#2a78d6", label="Test RMSE (6-month hold-out)")
ax.set_yticks(y, order); ps.millions(ax, "x"); ax.set_xlabel("RMSE (LKR) — lower is better")
ax.legend(loc="lower right"); ax.set_title("Model comparison: forecast error"); ax.grid(axis="y", visible=False)
fig.savefig(FIG / "07_model_comparison.png"); plt.show()
''',
"""md:## 3. Choose the final model
The rule set before seeing test results: **lowest CV RMSE among the M3 family**. If a variant beats M3 by less than 2%, keep the simpler M3, because its effects are easier to interpret and it has fewer things to go wrong.""",
r'''
fam = comp.loc[[i for i in comp.index if i.startswith("M3")], "CV RMSE"]
best = fam.idxmin()
FINAL_NAME = best if fam[best] < 0.98 * fam["M3 Adstock + saturation"] else "M3 Adstock + saturation"
final = models[FINAL_NAME]
print("Final model:", FINAL_NAME)
print("controls:", final.controls)
hp = pd.DataFrame({"adstock λ": final.lam, "half-life (months)": {c: (np.log(.5)/np.log(l) if l > 0.01 else 0) for c, l in final.lam.items()},
                   "Hill K (× mean spend)": final.K, "Hill alpha": final.alpha}).rename(index=LAB)
print(f"ridge = {final.ridge:.3f}")
hp.round(3)
''',
"""md:**Result:** adding the price index lowers CV error by about 4% (more than the 2% threshold) and also gives the best hold-out accuracy, so it is **kept as a control** in the final model. It is there to absorb price-level and inflation effects (it is collinear with inflation and time). Its coefficient should not be read as a causal price elasticity. Its job is to stop media from being credited for price-driven revenue changes. Holidays (M3b) and an S-shaped curve (M3c) do not beat the threshold, so the simpler choices stay. Carryover is short for every channel (half-life under one month), which is plausible for an FMCG category with monthly data.
""",
"""md:## 4. Hold-out fit: actual vs predicted""",
r'''
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(df.month, df[TARGET], color=ps.ACTUAL_COLOR, label="actual")
ax.plot(train.month, final.predict(train), color=ps.PRED_COLOR, lw=1.6, label="final model – fitted (train)")
ax.plot(test.month, final.predict(test), color=ps.PRED_COLOR, lw=2, ls="--", marker="o", ms=5, label="final model – forecast (test)")
ax.plot(test.month, models["M0 Baseline OLS (all controls)"].predict(test), color="#eb6834", lw=1.5, ls=":", marker="s", ms=4, label="baseline OLS – forecast")
ax.axvspan(test.month.min() - pd.Timedelta(days=15), test.month.max() + pd.Timedelta(days=15), color="#f0efec", zorder=0)
ps.millions(ax); ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=4, fontsize=9)
r = results[FINAL_NAME]
ax.set_title(f"Actual vs predicted revenue — test R² {r['test R2']:.2f}, MAPE {r['test MAPE_%']:.1f}%")
fig.savefig(FIG / "07_actual_vs_predicted.png"); plt.show()
''',
"""md:## 5. Residual diagnostics (training period)
Regression assumes errors are unpatterned. We check autocorrelation (Durbin-Watson ≈ 2 is ideal, and the Ljung-Box p-value should be > 0.05), normality (Shapiro-Wilk p > 0.05) and heteroscedasticity (Breusch-Pagan p > 0.05).""",
r'''
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from scipy.stats import shapiro
res = train[TARGET].values - final.predict(train)
import statsmodels.api as sm
bp = het_breuschpagan(res, sm.add_constant(final.predict(train)))
diag = pd.Series({"Durbin-Watson": durbin_watson(res),
                  "Ljung-Box p (lag 3)": acorr_ljungbox(res, lags=[3]).lb_pvalue.iloc[0],
                  "Shapiro-Wilk p": shapiro(res).pvalue,
                  "Breusch-Pagan p": bp[1],
                  "residual mean (LKR)": res.mean(), "residual std (LKR)": res.std()}, name="value")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
axs[0].bar(train.month, res, width=20, color=["#2a78d6" if v >= 0 else "#eb6834" for v in res]); ps.millions(axs[0]); axs[0].set_title("Residuals over time")
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(res, lags=10, ax=axs[1], color=ps.PRED_COLOR, vlines_kwargs={"colors": ps.PRED_COLOR}); axs[1].set_title("Residual autocorrelation")
fig.tight_layout(); fig.savefig(FIG / "07_residuals.png"); plt.show()
diag.to_frame().round(3)
''',
"""md:## 6. Reliability check 1: does the search land in the same place with a different random seed?
Differential evolution is stochastic. If two seeds give very different channel effects, those effects are not well identified by the data.""",
r'''
seed_rows = {}
for seed in [1, 2]:
    m, info = tune_mmm(train, controls=final.controls, alpha=final.alpha[CHANNELS[0]] if FINAL_NAME != "M3c S-shaped curve (alpha tuned)" else None, seed=seed, **SEARCH)
    seed_rows[f"seed {seed}"] = m
seed_rows["seed 0 (final)"] = final
def roas(m, d):
    c = m.contributions(d)[CHANNELS].sum(); return (c / d[CHANNELS].sum()).rename(index=LAB)
stab = pd.DataFrame({k: roas(m, train) for k, m in seed_rows.items()})
stab["CV RMSE seed (M)"] = np.nan
print("CV RMSE by seed:", {k: round(cv_rmse(m, train)/1e6, 3) for k, m in seed_rows.items()}); final.fit(train)
stab.drop(columns="CV RMSE seed (M)").round(2).rename_axis("training-period ROAS")
''',
"""md:## 7. Refit the final model on all 36 months
Hyperparameters (λ, K, ridge) stay fixed at the values chosen on the training period. Only the regression coefficients are re-estimated using all 36 months. The test set has done its job of validating the *method*. The final coefficients should use all the data, and these are the ones handed to the ROI, Optimisation and Dashboard teams.""",
r'''
final_full = MMM(lam=final.lam, K=final.K, alpha=final.alpha, ridge=final.ridge, controls=final.controls).fit(df)
full_fit = metrics(df[TARGET], final_full.predict(df))
print("Full-data fit:", {k: round(v, 3) for k, v in full_fit.items()})

params = final_full.to_dict()
params.update({
    "model_name": FINAL_NAME, "fit_period": f"{df.month.min():%Y-%m} to {df.month.max():%Y-%m}",
    "hyperparameters_tuned_on": f"{train.month.min():%Y-%m} to {train.month.max():%Y-%m} (rolling-origin CV)",
    "holdout_metrics": {k: float(v) for k, v in results[FINAL_NAME].items()},
    "current_monthly_spend": {c: float(df[df.month.dt.year == df.month.dt.year.max()][c].mean()) for c in CHANNELS},
    "current_spend_definition": "average monthly spend in the latest year (2025)",
})
import json
json.dump(params, open(MOD / "final_mmm.json", "w"), indent=2)
final.save_json(MOD / "final_mmm_train_only.json")
joblib.dump({"final_full": final_full, "final_train": final, "ladder": models}, MOD / "mmm_models.pkl")
print("saved: final_mmm.json, final_mmm_train_only.json, mmm_models.pkl, 02_model_comparison.csv")
''',
]
save("07_mmm_training_and_evaluation.ipynb", nb2)
print("ok2")

# =====================================================================================
nb3 = [
"""md:# 08 · Channel contributions, adstock & saturation, response functions
**Team 2 – Marketing Mix Modelling** · uses the final model from notebook 07 (refit on all 36 months)

Outputs for the other teams:
| File | For | What it is |
|---|---|---|
| `models/mmm/final_mmm.json` | Optimisation, ROI | every fitted parameter |
| `src/mmm_response.py` | Optimisation | `ResponseModel` — revenue and marginal ROAS for any spend |
| `outputs/mmm/contributions_monthly.csv` / `_long.csv` | ROI, Dashboard | monthly revenue decomposition |
| `outputs/mmm/channel_summary.csv` | ROI, Dashboard, Report | contribution, ROAS, mROAS, carryover per channel |
| `outputs/mmm/roas_bootstrap.csv` | ROI, Report | ROAS uncertainty (90% intervals) |
| `outputs/mmm/response_curves.csv` | Dashboard, ROI | curves on a spend grid (diminishing returns) |
| `outputs/mmm/adstock_decay.csv` | Dashboard, Report | carryover profile by month |

**ROAS here = incremental *revenue* per LKR of spend.** The dataset has no margin or cost data, so profit ROI cannot be computed. A ROAS above 1 does not by itself mean the spend is profitable.
""",
SETUP,
r'''
from src.mmm_response import ResponseModel
mmm = MMM.from_json(MOD / "final_mmm.json")
rm = ResponseModel(MOD / "final_mmm.json")
P = json.load(open(MOD / "final_mmm.json"))
print("model:", P["model_name"], "| fit period:", P["fit_period"], "| controls:", mmm.controls)
pred = mmm.predict(df); print("full-period fit:", {k: round(v, 3) for k, v in metrics(df[TARGET], pred).items()})
''',
"""md:## 1. Revenue decomposition
`base` is the revenue expected with zero media and the external factors at their average level. Control contributions are deviations from average, so they move above and below zero.""",
r'''
contrib = mmm.contributions(df)
contrib.columns = ["base"] + [LAB.get(c, c) for c in contrib.columns[1:]]
contrib.insert(0, "month", df.month)
contrib["predicted_revenue"] = pred; contrib["actual_revenue"] = df[TARGET].values
contrib.to_csv(TAB / "contributions_monthly.csv", index=False)
ctrl_names = mmm.controls
long = contrib.melt(id_vars="month", value_vars=["base"] + [LAB[c] for c in CHANNELS] + ctrl_names, var_name="component", value_name="revenue_lkr")
long["component_type"] = np.select([long.component.eq("base"), long.component.isin(LAB.values())], ["base", "media"], "external factor")
long.merge(df[["month"] + CHANNELS].melt(id_vars="month", var_name="ch", value_name="spend_lkr").assign(component=lambda d: d.ch.map(LAB)).drop(columns="ch"),
           on=["month", "component"], how="left").to_csv(TAB / "contributions_long.csv", index=False)
contrib.head(3)
''',
r'''
media = contrib[[LAB[c] for c in CHANNELS]]
fig, axs = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
ax = axs[0]
ax.plot(df.month, df[TARGET], color=ps.ACTUAL_COLOR, label="actual revenue")
ax.plot(df.month, pred, color=ps.PRED_COLOR, ls="--", lw=1.6, label="model")
ax.plot(df.month, contrib["base"] + contrib[ctrl_names].sum(1), color=ps.CONTROL_COLOR, lw=1.6, label="base + external factors (no media)")
ps.millions(ax); ax.set_ylim(0, None); ax.legend(loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.0)); ax.set_title("Revenue: actual, model, and what it would be without media")
ax = axs[1]
bottom = np.zeros(len(df))
for col in media.columns:
    ax.bar(df.month, media[col], bottom=bottom, width=24, color=ps.CHANNEL_COLORS[col], label=col, edgecolor=ps.SURFACE, linewidth=0.8)
    bottom += media[col].values
ps.millions(ax); ax.set_title("Incremental revenue from media, by channel"); ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(); fig.savefig(FIG / "08_decomposition.png"); plt.show()
''',
"""md:## 2. Channel contribution & ROAS summary""",
r'''
def channel_table(d, c):
    sp = d[CHANNELS].sum(); con = c[[LAB[x] for x in CHANNELS]].sum(); con.index = CHANNELS
    cur = pd.Series(P["current_monthly_spend"])
    t = pd.DataFrame({
        "total_spend_lkr": sp, "incremental_revenue_lkr": con,
        "spend_share_pct": sp / sp.sum() * 100, "contribution_share_pct": con / con.sum() * 100 if con.sum() > 0 else 0,
        "share_of_total_revenue_pct": con / d[TARGET].sum() * 100,
        "roas": con / sp,
        "current_monthly_spend_lkr": cur,
        "marginal_roas_at_current": [float(rm.marginal_roas(x, cur[x])) for x in CHANNELS],
        "adstock_lambda": pd.Series(mmm.lam), "carryover_half_life_months": [rm.carryover_half_life(x) for x in CHANNELS],
        "hill_K_x_mean_spend": pd.Series(mmm.K), "saturation_at_current_pct": [float(rm.response(x, cur[x]) / rm.beta[x] * 100) if rm.beta[x] > 0 else np.nan for x in CHANNELS],
    })
    t.index = [LAB[x] for x in CHANNELS]; t.index.name = "channel"
    return t
summary = channel_table(df, contrib)
summary["roas_rank"] = summary.roas.rank(ascending=False, method="min").astype(int)
summary.to_csv(TAB / "channel_summary.csv")
tot_media = summary.incremental_revenue_lkr.sum()
print(f"Media-driven revenue: {tot_media/1e6:,.1f}M LKR = {tot_media/df[TARGET].sum()*100:.1f}% of revenue | overall media ROAS {tot_media/df[CHANNELS].sum().sum():.2f}")
summary.drop(columns=["total_spend_lkr", "incremental_revenue_lkr", "current_monthly_spend_lkr"]).round(2)
''',
r'''
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
s = summary.sort_values("roas")
axs[0].barh(s.index, s.roas, color=[ps.CHANNEL_COLORS[i] for i in s.index], height=0.6)
for y, v in enumerate(s.roas): axs[0].text(v + 0.05, y, f"{v:.2f}", va="center", fontsize=9, color=ps.INK2)
axs[0].axvline(1, color=ps.INK2, lw=1, ls="--"); axs[0].set_title("ROAS: revenue per LKR (2023–2025)"); axs[0].grid(axis="y", visible=False)
y = np.arange(len(summary))
axs[1].barh(y + 0.2, summary.spend_share_pct, height=0.38, color="#c9c8c3", label="share of spend")
axs[1].barh(y - 0.2, summary.contribution_share_pct, height=0.38, color=[ps.CHANNEL_COLORS[i] for i in summary.index], label="share of media effect")
axs[1].set_yticks(y, summary.index); axs[1].invert_yaxis(); axs[1].set_xlabel("%"); axs[1].legend(loc="lower right", fontsize=9)
axs[1].set_title("Spend share vs effect share"); axs[1].grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(FIG / "08_roas_and_shares.png"); plt.show()
''',
"""md:## 3. Adstock (carryover) profiles
The share of one month's spend effect that is felt in that month and in each later month. A long tail means the channel keeps paying back after the spend stops.""",
r'''
months = np.arange(7)
decay = pd.DataFrame({LAB[c]: adstock_weights(mmm.lam[c], len(months)) for c in CHANNELS}, index=months)
decay.rename_axis("months_after_spend").to_csv(TAB / "adstock_decay.csv")
fig, ax = plt.subplots(figsize=(8, 3.8))
for col in decay:
    ax.plot(months, decay[col] * 100, color=ps.CHANNEL_COLORS[col], marker="o", ms=5, label=f"{col} (λ={mmm.lam[[k for k,v in LAB.items() if v==col][0]]:.2f})")
ax.set_xlabel("months after spend"); ax.set_ylabel("% of total effect"); ax.legend(fontsize=8, ncol=2)
ax.set_title("Adstock: how the effect of one month's spend is spread over time")
fig.savefig(FIG / "08_adstock_decay.png"); plt.show()
''',
"""md:## 4. Saturation / response curves (diminishing returns)
Each curve gives the incremental monthly revenue the channel generates at a sustained monthly spend level. The dot marks current spend (2025 monthly average). The dashed line is *marginal ROAS*: what the **next** LKR returns. Budget should move from channels with low marginal ROAS to channels with high marginal ROAS.""",
r'''
rows = []
for c in CHANNELS:
    grid = np.linspace(0, 3 * mmm.scale_[c], 121)
    rows.append(pd.DataFrame({"channel": LAB[c], "monthly_spend_lkr": grid, "incremental_revenue_lkr": rm.response(c, grid),
                              "marginal_roas": rm.marginal_roas(c, grid), "average_roas": rm.average_roas(c, grid),
                              "spend_vs_current": grid / P["current_monthly_spend"][c]}))
curves = pd.concat(rows); curves.to_csv(TAB / "response_curves.csv", index=False)

fig, axs = plt.subplots(2, 3, figsize=(12, 6.4))
for ax, c in zip(axs.flat, CHANNELS):
    d = curves[curves.channel == LAB[c]]; col = ps.CHANNEL_COLORS[LAB[c]]; cur = P["current_monthly_spend"][c]
    ax.plot(d.monthly_spend_lkr, d.incremental_revenue_lkr, color=col)
    ax.scatter([cur], [rm.response(c, cur)], s=60, color=col, edgecolor=ps.SURFACE, linewidth=2, zorder=3)
    ax2 = ax.twinx(); ax2.plot(d.monthly_spend_lkr, d.marginal_roas, color=ps.INK2, ls="--", lw=1.2); ax2.set_ylim(bottom=0)
    ax2.spines["right"].set_visible(True); ax2.grid(False); ax2.tick_params(labelsize=8)
    ps.millions(ax); ps.millions(ax, "x"); ax.tick_params(labelsize=8)
    ax.set_title(f"{LAB[c]} · mROAS now {float(rm.marginal_roas(c, cur)):.2f}", fontsize=10)
    if rm.beta[c] == 0:
        ax.set_ylim(-1e5, 1e6); ax2.set_ylim(0, 1); ax.text(0.5, 0.5, "no measurable effect\n(coefficient = 0)", transform=ax.transAxes, ha="center", va="center", color=ps.INK2)
fig.supxlabel("monthly spend (LKR)", color=ps.INK2); fig.supylabel("incremental monthly revenue (solid)", color=ps.INK2)
fig.text(0.995, 0.5, "marginal ROAS (dashed)", rotation=90, va="center", ha="right", color=ps.INK2)
fig.tight_layout(rect=(0, 0, 0.985, 1)); fig.savefig(FIG / "08_response_curves.png"); plt.show()
''',
"""md:## 5. Reliability check 2: bootstrap uncertainty on ROAS
**Moving-block bootstrap** (blocks of 3 months, so autocorrelation is kept): resample the model residuals, rebuild a synthetic revenue series, refit the coefficients, and recompute ROAS. Repeat 1,000 times. The intervals show how far each channel's ROAS could move from noise alone. They are conditional on the chosen adstock/saturation settings, so the true uncertainty is somewhat larger.""",
r'''
rng = np.random.default_rng(42)
resid = df[TARGET].values - pred; n, B, L = len(df), 1000, 3
boot = []
for b in range(B):
    starts = rng.integers(0, n - L + 1, size=int(np.ceil(n / L)))
    e = np.concatenate([resid[s:s + L] for s in starts])[:n]
    db = df.copy(); db[TARGET] = pred + e
    mb = MMM(lam=mmm.lam, K=mmm.K, alpha=mmm.alpha, ridge=mmm.ridge, controls=mmm.controls).fit(db)
    cb = mb.contributions(db)[CHANNELS].sum()
    boot.append((cb / df[CHANNELS].sum()).values)
boot = pd.DataFrame(boot, columns=[LAB[c] for c in CHANNELS])
ci = pd.DataFrame({"roas_point": summary.roas, "roas_p05": boot.quantile(.05), "roas_median": boot.median(), "roas_p95": boot.quantile(.95),
                   "prob_roas_gt_1": (boot > 1).mean(), "prob_effect_zero": (boot < 1e-6).mean()})
ci.index.name = "channel"; ci.to_csv(TAB / "roas_bootstrap.csv")
ci.round(2)
''',
r'''
fig, ax = plt.subplots(figsize=(8, 3.6))
c2 = ci.sort_values("roas_point")
for y, (name, r) in enumerate(c2.iterrows()):
    col = ps.CHANNEL_COLORS[name]
    ax.plot([r.roas_p05, r.roas_p95], [y, y], color=col, lw=3, solid_capstyle="round", alpha=.55)
    ax.scatter(r.roas_point, y, s=70, color=col, edgecolor=ps.SURFACE, linewidth=2, zorder=3)
ax.axvline(1, color=ps.INK2, ls="--", lw=1); ax.set_yticks(range(len(c2)), c2.index)
ax.set_xlabel("ROAS (revenue per LKR)"); ax.set_title("ROAS with 90% bootstrap intervals"); ax.grid(axis="y", visible=False)
fig.savefig(FIG / "08_roas_uncertainty.png"); plt.show()
''',
"""md:## 6. Hand-off check: using the response functions
How the Budget Optimisation team will call the model. Below is a quick sanity check: revenue at current spend, and the effect of moving 10% of the total budget from the lowest to the highest marginal-ROAS channel.""",
r'''
cur = P["current_monthly_spend"]
base_rev = rm.total_revenue(cur)
mr = pd.Series({c: float(rm.marginal_roas(c, cur[c])) for c in CHANNELS})
lo, hi = mr.idxmin(), mr.idxmax(); shift = 0.10 * sum(cur.values())
shift = min(shift, cur[lo])
new = dict(cur); new[lo] -= shift; new[hi] += shift
print(f"Current total monthly budget: {sum(cur.values())/1e6:.2f}M LKR -> media-driven revenue {base_rev/1e6:.2f}M / month")
print(f"Move {shift/1e6:.2f}M from {LAB[lo]} (mROAS {mr[lo]:.2f}) to {LAB[hi]} (mROAS {mr[hi]:.2f}) -> {rm.total_revenue(new)/1e6:.2f}M / month "
      f"(+{(rm.total_revenue(new)-base_rev)/1e6:.2f}M, same budget)")
print("\nThis is only an illustration; the full constrained optimisation (min/max limits per channel) is Team 4's job.")
''',
]
save("08_mmm_contributions_and_response_curves.ipynb", nb3)
print("ok3")
