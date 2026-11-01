"""
mmm_core.py — Marketing Mix Model core library
================================================
Reusable building blocks for the MMM team (and for the ROI / Budget Optimisation
teams downstream):

  * geometric_adstock()   carryover effect of advertising
  * hill()                diminishing returns (saturation) curve
  * MMM                   adstock + saturation + controls regression with
                          non-negative media coefficients
  * metrics(), rolling_origin_splits()   time-aware evaluation helpers

Conventions
-----------
* Adstock is *normalised*: a_t = x_t + lam * a_{t-1}, then multiplied by (1-lam).
  With constant spend x, the steady-state adstock equals x, so adstock keeps the
  same unit (LKR/month) as spend. The first month is warm-started at steady state.
* Saturation input is spend expressed relative to the channel's mean training
  spend:  s = adstock / scale.  hill(s) = s^a / (s^a + K^a), which is 0.5 at s = K.
  This differs on purpose from the simple helpers src.preprocessing.geometric_adstock /
  hill_saturation (unnormalised, cold start): normalising keeps response curves in
  spend units so they can be summed in the budget optimiser, and the warm start stops
  January 2023 from looking like a month with no past advertising.
* Media coefficients beta_c >= 0 (LKR of revenue at full saturation).
  Monthly media contribution of channel c = beta_c * hill(s_c,t).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

# ----------------------------------------------------------------------------
# Column definitions
# ----------------------------------------------------------------------------
CHANNELS = [
    "tv_spend_lkr",
    "facebook_instagram_spend_lkr",
    "google_search_spend_lkr",
    "youtube_spend_lkr",
    "radio_spend_lkr",
    "outdoor_spend_lkr",
]
CHANNEL_LABELS = {
    "tv_spend_lkr": "TV",
    "facebook_instagram_spend_lkr": "Facebook/Instagram",
    "google_search_spend_lkr": "Google Search",
    "youtube_spend_lkr": "YouTube",
    "radio_spend_lkr": "Radio",
    "outdoor_spend_lkr": "Outdoor",
}
# Candidate external factors tested. average_price_index, inflation_proxy_index and a
# time trend are ~0.99 correlated with each other while revenue itself has no trend,
# so they act as a spurious trend and hurt out-of-sample accuracy. public_holiday_count
# is ~0 once seasonal_demand_index (which already includes festive periods) is in.
ALL_CONTROLS = [
    "promotion_discount_pct",
    "seasonal_demand_index",
    "public_holiday_count",
    "competitor_pressure_index_0_100",
    "economic_sentiment_index_0_100",
    "average_price_index",
]
CONTROLS = [  # final model
    "promotion_discount_pct",
    "seasonal_demand_index",
    "competitor_pressure_index_0_100",
    "economic_sentiment_index_0_100",
]
TARGET = "sales_revenue_lkr"

# Plausible carryover (adstock decay) ranges by channel type, monthly data.
# Broad-reach brand media carry over longer; search/social act mostly in-month.
LAMBDA_BOUNDS = {
    "tv_spend_lkr": (0.3, 0.8),
    "facebook_instagram_spend_lkr": (0.0, 0.4),
    "google_search_spend_lkr": (0.0, 0.3),
    "youtube_spend_lkr": (0.1, 0.6),
    "radio_spend_lkr": (0.1, 0.5),
    "outdoor_spend_lkr": (0.2, 0.7),
}
K_BOUNDS = (0.5, 2.0)        # half-saturation, in multiples of mean monthly spend
ALPHA_BOUNDS = (0.8, 3.0)    # Hill shape (shared) - only if alpha is tuned
ALPHA_FIXED = 1.0            # final model: concave Hill (pure diminishing returns)
RIDGE_BOUNDS = (0.0, 1.0)


# ----------------------------------------------------------------------------
# Transformations
# ----------------------------------------------------------------------------
def geometric_adstock(x: np.ndarray, lam: float, normalise: bool = True) -> np.ndarray:
    """Geometric carryover. lam in [0,1): share of last month's effect that persists."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    prev = x[0] / (1 - lam)  # warm start: assume pre-sample spend = first month
    for t, xt in enumerate(x):
        prev = xt + lam * prev
        out[t] = prev
    return out * (1 - lam) if normalise else out


def hill(s: np.ndarray, K: float, alpha: float) -> np.ndarray:
    """Hill saturation curve. s = relative spend (>=0); K = half-saturation point."""
    s = np.maximum(np.asarray(s, dtype=float), 0.0)
    return s**alpha / (s**alpha + K**alpha)


def adstock_weights(lam: float, n: int = 12, normalise: bool = True) -> np.ndarray:
    """Share of one month's spend effect felt in months 0..n-1 (for plots)."""
    w = lam ** np.arange(n)
    return w * (1 - lam) if normalise else w


# ----------------------------------------------------------------------------
# Metrics and time-aware splits
# ----------------------------------------------------------------------------
def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_true - y_pred
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE_%": float(np.mean(np.abs(err / y_true)) * 100),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def rolling_origin_splits(n: int, min_train: int = 18, horizon: int = 3, step: int = 3):
    """Expanding-window CV: train on [0, k), validate on [k, k+horizon)."""
    k = min_train
    while k + horizon <= n:
        yield np.arange(k), np.arange(k, k + horizon)
        k += step


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@dataclass
class MMM:
    """Adstock + Hill saturation + controls, media betas constrained >= 0.

    use_adstock / use_saturation switch the transformations on/off so the same
    class gives the whole model ladder (baseline -> adstock -> adstock+saturation).
    ridge: small L2 penalty on standardised coefficients (stabilises 36-row fit).
    """

    lam: dict = field(default_factory=dict)       # channel -> decay
    K: dict = field(default_factory=dict)         # channel -> half-saturation (relative)
    alpha: dict = field(default_factory=dict)     # channel -> Hill shape
    use_adstock: bool = True
    use_saturation: bool = True
    nonneg_media: bool = True
    ridge: float = 0.0
    channels: list = field(default_factory=lambda: list(CHANNELS))
    controls: list = field(default_factory=lambda: list(CONTROLS))

    # fitted
    scale_: dict = field(default_factory=dict)
    ctrl_mean_: dict = field(default_factory=dict)
    ctrl_std_: dict = field(default_factory=dict)
    coef_: dict = field(default_factory=dict)
    intercept_: float = 0.0

    # ---------------- feature construction ----------------
    def _media_feature(self, c: str, x: np.ndarray) -> np.ndarray:
        a = geometric_adstock(x, self.lam.get(c, 0.0)) if self.use_adstock else np.asarray(x, float)
        if self.use_saturation:
            return hill(a / self.scale_[c], self.K.get(c, 1.0), self.alpha.get(c, 1.0))
        return a / self.scale_[c]  # linear: coef = revenue per "1x mean spend"

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        cols = [self._media_feature(c, df[c].values) for c in self.channels]
        cols += [(df[k].values - self.ctrl_mean_[k]) / self.ctrl_std_[k] for k in self.controls]
        return np.column_stack(cols)

    # ---------------- fit / predict ----------------
    def fit(self, df: pd.DataFrame, y=None):
        y = np.asarray(df[TARGET] if y is None else y, float)
        self.scale_ = {c: float(df[c].mean()) for c in self.channels}
        self.ctrl_mean_ = {k: float(df[k].mean()) for k in self.controls}
        self.ctrl_std_ = {k: float(df[k].std(ddof=0)) or 1.0 for k in self.controls}
        X = self._design(df)
        n, p = X.shape
        # centre so the intercept is handled exactly, then solve bounded LS
        Xm, ym = X.mean(0), y.mean()
        Xc, yc = X - Xm, y - ym
        sd = Xc.std(0)
        sd[sd == 0] = 1.0
        Xs = Xc / sd
        if self.ridge > 0:
            Xs = np.vstack([Xs, np.sqrt(self.ridge * n) * np.eye(p)])
            yc = np.concatenate([yc, np.zeros(p)])
        nm = len(self.channels)
        lb = np.r_[np.zeros(nm) if self.nonneg_media else -np.inf * np.ones(nm), -np.inf * np.ones(p - nm)]
        res = lsq_linear(Xs, yc, bounds=(lb, np.inf * np.ones(p)), method="bvls")
        b = res.x / sd
        names = self.channels + self.controls
        self.coef_ = dict(zip(names, b))
        self.intercept_ = float(ym - Xm @ b)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.intercept_ + self._design(df) @ np.array([self.coef_[k] for k in self.channels + self.controls])

    # ---------------- decomposition ----------------
    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Monthly revenue decomposition: base (intercept + controls at mean) + controls + media."""
        X = self._design(df)
        names = self.channels + self.controls
        out = pd.DataFrame(X * np.array([self.coef_[k] for k in names]), columns=names, index=df.index)
        out.insert(0, "base", self.intercept_)
        return out

    # ---------------- response functions (for ROI / optimisation) ----------------
    def response(self, channel: str, monthly_spend) -> np.ndarray:
        """Steady-state incremental monthly revenue if `monthly_spend` is sustained.
        (Normalised adstock => steady-state adstock = spend.)"""
        s = np.asarray(monthly_spend, float) / self.scale_[channel]
        f = hill(s, self.K.get(channel, 1.0), self.alpha.get(channel, 1.0)) if self.use_saturation else s
        return self.coef_[channel] * f

    def marginal_roas(self, channel: str, monthly_spend, eps: float = 1e3) -> np.ndarray:
        x = np.asarray(monthly_spend, float)
        return (self.response(channel, x + eps) - self.response(channel, x - eps)) / (2 * eps)

    # ---------------- persistence ----------------
    def to_dict(self) -> dict:
        return {
            "model": "adstock + hill saturation MMM",
            "target": TARGET,
            "use_adstock": self.use_adstock,
            "use_saturation": self.use_saturation,
            "channels": self.channels,
            "controls": self.controls,
            "intercept": self.intercept_,
            "coef": {k: float(v) for k, v in self.coef_.items()},
            "lam": {c: float(self.lam.get(c, 0.0)) for c in self.channels},
            "K": {c: float(self.K.get(c, 1.0)) for c in self.channels},
            "alpha": {c: float(self.alpha.get(c, 1.0)) for c in self.channels},
            "scale": self.scale_,
            "ctrl_mean": self.ctrl_mean_,
            "ctrl_std": self.ctrl_std_,
            "ridge": self.ridge,
        }

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "MMM":
        d = json.load(open(path))
        m = cls(lam=d["lam"], K=d["K"], alpha=d["alpha"], use_adstock=d["use_adstock"],
                use_saturation=d["use_saturation"], ridge=d["ridge"],
                channels=d["channels"], controls=d["controls"])
        m.scale_, m.ctrl_mean_, m.ctrl_std_ = d["scale"], d["ctrl_mean"], d["ctrl_std"]
        m.coef_, m.intercept_ = d["coef"], d["intercept"]
        return m


def load_data(path) -> pd.DataFrame:
    """Load Team 1's processed CSV (or the raw Excel file) sorted by month."""
    df = pd.read_excel(path, sheet_name="MMM_Data") if str(path).endswith("xlsx") else pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"])
    return df.sort_values("month").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Hyperparameter tuning (time-aware)
# ----------------------------------------------------------------------------
def cv_rmse(model: MMM, df: pd.DataFrame, min_train: int = 18, horizon: int = 3, step: int = 1) -> float:
    """Mean RMSE over expanding-window, rolling-origin folds inside `df`."""
    errs = []
    for tr_idx, va_idx in rolling_origin_splits(len(df), min_train, horizon, step):
        model.fit(df.iloc[tr_idx])
        errs.append(metrics(df.iloc[va_idx][TARGET], model.predict(df.iloc[va_idx]))["RMSE"])
    return float(np.mean(errs))


def tune_mmm(train: pd.DataFrame, use_adstock=True, use_saturation=True, controls=None,
             tune_ridge=True, ridge=0.0, alpha=ALPHA_FIXED, seed=0, maxiter=60, popsize=12, **cv_kw):
    """Differential-evolution search over adstock decay (per channel, bounded by channel
    type), Hill K (per channel), optionally a shared Hill alpha (alpha=None) and the ridge strength, minimising
    rolling-origin CV RMSE on the training period only. Returns (model fitted on train, info)."""
    from scipy.optimize import differential_evolution

    controls = list(CONTROLS if controls is None else controls)
    bounds, names = [], []
    if use_adstock:
        bounds += [LAMBDA_BOUNDS[c] for c in CHANNELS]; names += [f"lam:{c}" for c in CHANNELS]
    if use_saturation:
        bounds += [K_BOUNDS] * len(CHANNELS); names += [f"K:{c}" for c in CHANNELS]
        if alpha is None:
            bounds += [ALPHA_BOUNDS]; names += ["alpha"]
    if tune_ridge:
        bounds += [RIDGE_BOUNDS]; names += ["ridge"]

    def build(p):
        d = dict(zip(names, p))
        return MMM(
            lam={c: d.get(f"lam:{c}", 0.0) for c in CHANNELS},
            K={c: d.get(f"K:{c}", 1.0) for c in CHANNELS},
            alpha={c: d.get("alpha", alpha or 1.0) for c in CHANNELS},
            use_adstock=use_adstock, use_saturation=use_saturation,
            ridge=d.get("ridge", ridge), controls=controls,
        )

    if not bounds:  # nothing to tune
        m = build([])
        return m.fit(train), {"cv_rmse": cv_rmse(m, train, **cv_kw), "params": {}}
    res = differential_evolution(lambda p: cv_rmse(build(p), train, **cv_kw), bounds,
                                 seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-7, polish=False)
    m = build(res.x)
    cv = cv_rmse(m, train, **cv_kw)
    return m.fit(train), {"cv_rmse": cv, "params": dict(zip(names, map(float, res.x)))}
