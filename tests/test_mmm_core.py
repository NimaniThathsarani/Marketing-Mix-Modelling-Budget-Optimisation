"""Unit tests for the MMM code. Run from the project root:  python -m pytest tests -v"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mmm_core import (CHANNELS, CONTROLS, MMM, TARGET, geometric_adstock, hill, metrics,
                          rolling_origin_splits)
from src.mmm_response import ResponseModel


def make_df(n=36, seed=0):
    """Synthetic data with a known structure: only the first two channels drive revenue."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"month": pd.date_range("2023-01-01", periods=n, freq="MS")})
    for c in CHANNELS:
        df[c] = rng.uniform(1e6, 5e6, n)
    for k in CONTROLS:
        df[k] = rng.uniform(0, 1, n)
    df[TARGET] = 5e7 + 3 * df[CHANNELS[0]] + 2 * df[CHANNELS[1]] + 1e6 * df[CONTROLS[0]] + rng.normal(0, 2e5, n)
    return df


def test_adstock_no_decay_is_identity():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(geometric_adstock(x, 0.0), x)


def test_adstock_constant_spend_is_steady_state():
    x = np.full(24, 5.0)
    assert np.allclose(geometric_adstock(x, 0.6), 5.0)  # normalised + warm start


def test_adstock_carries_effect_forward():
    a = geometric_adstock(np.array([0.0, 10.0, 0.0, 0.0]), 0.5)
    assert a[2] > 0 and a[3] > 0 and a[3] < a[2]


def test_hill_half_saturation_and_bounds():
    assert np.isclose(hill(np.array([2.0]), K=2.0, alpha=1.0)[0], 0.5)
    h = hill(np.linspace(0, 100, 50), K=1.0, alpha=1.5)
    assert h.min() >= 0 and h.max() < 1 and np.all(np.diff(h) >= 0)


def test_media_coefficients_non_negative():
    df = make_df()
    m = MMM(use_adstock=False, use_saturation=False).fit(df)
    assert all(m.coef_[c] >= 0 for c in CHANNELS)


def test_linear_model_recovers_known_effects():
    df = make_df()
    m = MMM(use_adstock=False, use_saturation=False).fit(df)
    per_lkr = {c: m.coef_[c] / m.scale_[c] for c in CHANNELS}   # coef is per "1x mean spend"
    assert abs(per_lkr[CHANNELS[0]] - 3) < 0.3 and abs(per_lkr[CHANNELS[1]] - 2) < 0.3
    assert metrics(df[TARGET], m.predict(df))["R2"] > 0.95


def test_contributions_add_up_to_prediction():
    df = make_df()
    m = MMM(lam={c: 0.3 for c in CHANNELS}, K={c: 1.0 for c in CHANNELS}, ridge=0.1).fit(df)
    assert np.allclose(m.contributions(df).sum(axis=1).values, m.predict(df))


def test_rolling_origin_splits_never_look_ahead():
    for tr, va in rolling_origin_splits(30, 18, 3, 1):
        assert tr.max() < va.min() and len(va) == 3


def test_json_round_trip_and_response_model(tmp_path):
    df = make_df()
    m = MMM(lam={c: 0.2 for c in CHANNELS}, K={c: 1.0 for c in CHANNELS}).fit(df)
    path = tmp_path / "m.json"
    m.save_json(path)
    m2 = MMM.from_json(path)
    assert np.allclose(m.predict(df), m2.predict(df))
    rm = ResponseModel(path)
    c = CHANNELS[0]
    x = m.scale_[c]
    assert np.isclose(rm.response(c, x), m.response(c, x))
    num = (rm.response(c, x + 1e3) - rm.response(c, x - 1e3)) / 2e3
    assert np.isclose(rm.marginal_roas(c, x), num, rtol=1e-3)
    assert rm.marginal_roas(c, 2 * x) < rm.marginal_roas(c, x)   # diminishing returns
