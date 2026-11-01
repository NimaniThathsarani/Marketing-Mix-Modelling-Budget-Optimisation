"""
mmm_response.py — hand-off module for the ROI and Budget Optimisation teams
================================================================================
Loads the final fitted MMM (models/mmm/final_mmm.json) and exposes the channel
response curves as plain functions. No model training is needed to use this file.

Meaning of the curves
---------------------
response(channel, x) = incremental monthly sales revenue (LKR) if the channel's
monthly spend is held at x LKR long enough for carryover to settle (steady state).
Because adstock is normalised, carryover changes *when* revenue arrives, not the
steady-state total, so these curves can be summed across channels and used directly
as the objective of a budget optimiser.

    from src.mmm_response import ResponseModel      # run with the project root on sys.path
    rm = ResponseModel()                              # loads models/mmm/final_mmm.json
    rm.total_revenue({"tv_spend_lkr": 4e6, ...})       # incremental LKR / month
    rm.marginal_roas("google_search_spend_lkr", 3.3e6) # extra LKR revenue per extra LKR

All curves are concave (Hill alpha = 1), so maximising total response under a budget
constraint is a well-behaved (convex) problem with a unique optimum.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_DEFAULT = Path(__file__).resolve().parents[1] / "models" / "mmm" / "final_mmm.json"

LABELS = {
    "tv_spend_lkr": "TV",
    "facebook_instagram_spend_lkr": "Facebook/Instagram",
    "google_search_spend_lkr": "Google Search",
    "youtube_spend_lkr": "YouTube",
    "radio_spend_lkr": "Radio",
    "outdoor_spend_lkr": "Outdoor",
}


class ResponseModel:
    def __init__(self, path: str | Path = _DEFAULT):
        p = json.load(open(path))
        self.channels: list[str] = p["channels"]
        self.beta = {c: p["coef"][c] for c in self.channels}      # revenue at full saturation
        self.K = p["K"]                                           # half-saturation (x mean spend)
        self.alpha = p["alpha"]
        self.scale = p["scale"]                                   # mean monthly spend used for scaling
        self.lam = p["lam"]                                       # adstock decay
        self.current_spend = p.get("current_monthly_spend", self.scale)

    # --- single channel -------------------------------------------------------
    def response(self, channel: str, monthly_spend):
        s = np.maximum(np.asarray(monthly_spend, float), 0) / self.scale[channel]
        a, K = self.alpha[channel], self.K[channel]
        return self.beta[channel] * s**a / (s**a + K**a)

    def marginal_roas(self, channel: str, monthly_spend):
        """d(revenue)/d(spend): extra revenue from one more LKR at this spend level."""
        x = np.maximum(np.asarray(monthly_spend, float), 1.0)
        s, a, K = x / self.scale[channel], self.alpha[channel], self.K[channel]
        return self.beta[channel] * a * K**a * s ** (a - 1) / (s**a + K**a) ** 2 / self.scale[channel]

    def average_roas(self, channel: str, monthly_spend):
        x = np.maximum(np.asarray(monthly_spend, float), 1.0)
        return self.response(channel, x) / x

    # --- whole portfolio --------------------------------------------------------
    def total_revenue(self, spend: dict) -> float:
        """Total incremental monthly revenue for a {channel: monthly spend} allocation."""
        return float(sum(self.response(c, spend.get(c, 0.0)) for c in self.channels))

    def gradient(self, spend: dict) -> np.ndarray:
        """Marginal ROAS vector in self.channels order (handy as an optimiser jacobian)."""
        return np.array([self.marginal_roas(c, spend.get(c, 0.0)) for c in self.channels], float)

    def carryover_half_life(self, channel: str) -> float:
        """Months until a burst of spend has lost half its effect."""
        lam = self.lam[channel]
        return float(np.log(0.5) / np.log(lam)) if 0 < lam < 1 else 0.0
