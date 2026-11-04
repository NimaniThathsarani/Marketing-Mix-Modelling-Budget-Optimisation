"""Budget optimisation from the fitted MMM's *steady-state* revenue response curves.

Run from project root: python -m src.budget_optimization

This maximises predicted INCREMENTAL REVENUE (not profit); a profit objective
requires a defensible contribution margin. Do not interpret the model as a
causal guarantee or a forecast of the first month after reallocation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.mmm_response import LABELS, ResponseModel

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "mmm_analysis_ready.csv"
OUTPUT = ROOT / "outputs" / "budget_optimization"
FIGURES = ROOT / "reports" / "figures"


@dataclass
class OptimizationResult:
    budget_lkr: float
    allocation: dict[str, float]
    incremental_revenue_lkr: float
    portfolio_roas: float
    method: str
    message: str


def build_bounds(
    model: ResponseModel,
    history: pd.DataFrame,
    lower_ratio: float = 0.70,
    upper_ratio: float = 1.30,
    cap_to_history: bool = True,
) -> dict[str, tuple[float, float]]:
    """Illustrative bounds: +/-30% of current spend AND historical spend range.

    Historical constraints limit extrapolation, but should ultimately be replaced
    or validated using commercial feasibility, contracts and channel capacity.
    """
    if not (0 <= lower_ratio <= 1 <= upper_ratio):
        raise ValueError("Require 0 <= lower_ratio <= 1 <= upper_ratio")
    bounds = {}
    for channel in model.channels:
        if channel not in history.columns:
            raise ValueError(f"Missing historical column: {channel}")
        cur = float(model.current_spend[channel])
        low, high = cur * lower_ratio, cur * upper_ratio
        if cap_to_history:
            observed = pd.to_numeric(history[channel], errors="raise")
            low = max(low, float(observed.min()))
            high = min(high, float(observed.max()))
        if low > high or not (low - 1e-6 <= cur <= high + 1e-6):
            raise ValueError(f"Invalid bounds for {channel}: {low} to {high}, current {cur}")
        bounds[channel] = (low, high)
    return bounds


def optimize_budget(
    model: ResponseModel,
    bounds: dict[str, tuple[float, float]],
    budget_lkr: float | None = None,
) -> OptimizationResult:
    """SLSQP with analytic gradient and exact (within tolerance) budget constraint.

    Optimisation is performed in LKR *millions* to avoid numerical scaling issues.
    Since Hill alpha=1 and beta>=0, summed channel responses are concave.
    """
    ch = model.channels
    budget = float(sum(model.current_spend.values()) if budget_lkr is None else budget_lkr)
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("Budget must be a finite nonnegative amount")
    if set(bounds) != set(ch):
        raise ValueError("Bounds must be provided for exactly the MMM channels")
    lower = np.array([bounds[c][0] for c in ch], float) / 1e6
    upper = np.array([bounds[c][1] for c in ch], float) / 1e6
    B = budget / 1e6
    if np.any(lower < 0) or np.any(upper < lower) or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Bounds must be finite, nonnegative and ordered")
    if B < lower.sum() - 1e-8 or B > upper.sum() + 1e-8:
        raise ValueError(
            f"Infeasible budget LKR {budget:,.0f}; under supplied bounds feasible "
            f"range is LKR {lower.sum()*1e6:,.0f} to {upper.sum()*1e6:,.0f}"
        )

    # Feasible starting point based on box capacity, independent of the budget.
    capacity = upper - lower
    if np.isclose(capacity.sum(), 0.0):
        x0 = lower.copy()
    else:
        x0 = lower + (B - lower.sum()) / capacity.sum() * capacity

    def to_allocation(x_million):
        return {c: float(x_million[i] * 1e6) for i, c in enumerate(ch)}

    def objective(x_million):
        return -model.total_revenue(to_allocation(x_million)) / 1e6

    def jacobian(x_million):
        return -model.gradient(to_allocation(x_million))

    result = minimize(
        objective,
        x0=x0,
        jac=jacobian,
        bounds=list(zip(lower, upper)),
        constraints=[{"type": "eq", "fun": lambda x: float(np.sum(x) - B),
                      "jac": lambda x: np.ones(len(ch))}],
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-11, "disp": False},
    )
    x = np.asarray(result.x, float)
    if (not result.success or abs(x.sum() - B) > 1e-6 or
            (x < lower - 1e-7).any() or (x > upper + 1e-7).any()):
        raise RuntimeError(f"SLSQP failed or produced infeasible allocation: {result.message}")
    allocation = to_allocation(x)
    incremental_revenue = model.total_revenue(allocation)
    return OptimizationResult(
        budget_lkr=budget, allocation=allocation,
        incremental_revenue_lkr=incremental_revenue,
        portfolio_roas=incremental_revenue / budget if budget > 0 else np.nan,
        method="SLSQP", message=str(result.message),
    )


def allocation_comparison(model: ResponseModel, result: OptimizationResult) -> pd.DataFrame:
    rows = []
    for channel in model.channels:
        current = float(model.current_spend[channel])
        optimized = result.allocation[channel]
        rows.append({
            "channel": LABELS[channel], "channel_key": channel,
            "current_spend_lkr": current, "optimized_spend_lkr": optimized,
            "spend_change_lkr": optimized - current,
            "spend_change_pct": 100 * (optimized / current - 1) if current else np.nan,
            "current_incremental_revenue_lkr": float(model.response(channel, current)),
            "optimized_incremental_revenue_lkr": float(model.response(channel, optimized)),
            "current_marginal_roas": float(model.marginal_roas(channel, current)),
            "optimized_marginal_roas": float(model.marginal_roas(channel, optimized)),
        })
    return pd.DataFrame(rows)


def evaluate_scenarios(
    model: ResponseModel,
    bounds: dict[str, tuple[float, float]],
    multipliers: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_budget = float(sum(model.current_spend.values()))
    base_revenue = model.total_revenue(model.current_spend)
    summary, long = [], []
    for multiplier in multipliers:
        result = optimize_budget(model, bounds, base_budget * multiplier)
        summary.append({
            "budget_multiplier": multiplier,
            "budget_change_pct": 100 * (multiplier - 1),
            "budget_lkr": result.budget_lkr,
            "predicted_incremental_revenue_lkr": result.incremental_revenue_lkr,
            "predicted_portfolio_roas": result.portfolio_roas,
            "revenue_change_vs_current_lkr": result.incremental_revenue_lkr - base_revenue,
            "revenue_change_vs_current_pct": 100 * (result.incremental_revenue_lkr / base_revenue - 1),
        })
        for c in model.channels:
            long.append({"budget_multiplier": multiplier, "channel": LABELS[c],
                         "channel_key": c, "optimized_spend_lkr": result.allocation[c]})
    return pd.DataFrame(summary), pd.DataFrame(long)


def plot_outputs(model: ResponseModel, comparison: pd.DataFrame, scenarios: pd.DataFrame,
                 figures_dir: Path = FIGURES) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(comparison))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.18, comparison.current_spend_lkr / 1e6, width=0.36, label="Current")
    ax.bar(x + 0.18, comparison.optimized_spend_lkr / 1e6, width=0.36, label="Optimized")
    ax.set_xticks(x, comparison.channel, rotation=15, ha="right")
    ax.set_ylabel("Monthly budget (LKR million)")
    ax.set_title("Current vs optimized allocation (unchanged budget)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / "10_current_vs_optimized.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scenarios.budget_lkr / 1e6,
            scenarios.predicted_incremental_revenue_lkr / 1e6, marker="o")
    ax.set_xlabel("Total monthly advertising budget (LKR million)")
    ax.set_ylabel("Optimized incremental revenue (LKR million / month)")
    ax.set_title("Budget scenarios: modelled steady-state response")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / "10_budget_scenarios.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, channel in zip(axes.flat, model.channels):
        item = comparison.loc[comparison.channel_key == channel].iloc[0]
        current, new = item.current_spend_lkr, item.optimized_spend_lkr
        xs = np.linspace(0, max(current, new) * 1.55, 120)
        ax.plot(xs / 1e6, model.response(channel, xs) / 1e6)
        ax.scatter([current / 1e6, new / 1e6],
                   [model.response(channel, current) / 1e6,
                    model.response(channel, new) / 1e6], zorder=3)
        ax.set_title(LABELS[channel]); ax.set_xlabel("Spend (LKR M)")
        ax.set_ylabel("Incremental revenue (LKR M)")
        ax.grid(alpha=0.15)
    fig.suptitle("MMM response curves: current and optimized spend points")
    fig.tight_layout()
    fig.savefig(figures_dir / "10_response_curve_allocations.png", dpi=170)
    plt.close(fig)



def sensitivity_analysis(model: ResponseModel, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check whether improvement survives tighter/looser limits and train-only fit.

    The train-only refit uses the SAME current-budget baseline and SAME moderate
    bounds, so its outcomes are comparable at a fixed real-world spend allocation.
    """
    current = model.current_spend
    budget = sum(current.values())
    ranges = [("conservative", 0.85, 1.15), ("moderate", 0.70, 1.30),
              ("flexible", 0.60, 1.50)]
    rows = []
    for name, low, high in ranges:
        bounds = build_bounds(model, history, low, high, cap_to_history=True)
        result = optimize_budget(model, bounds, budget)
        rows.append({
            "constraint_scenario": name, "lower_ratio": low, "upper_ratio": high,
            "budget_lkr": budget,
            "predicted_incremental_revenue_lkr": result.incremental_revenue_lkr,
            "predicted_revenue_lift_pct": 100 * (result.incremental_revenue_lkr /
                                                  model.total_revenue(current) - 1),
        })

    train_path = ROOT / "models" / "mmm" / "final_mmm_train_only.json"
    test_models = [("full_data_refit", model)]
    if train_path.exists():
        train_only = ResponseModel(train_path)
        train_only.current_spend = current.copy()
        test_models.append(("train_only_fit", train_only))

    comparison = []
    common_bounds = build_bounds(model, history, 0.70, 1.30)
    # Find proposed allocation using final fitted model. Evaluate its uplift under
    # an alternative fitted model as a useful sensitivity check (not a confidence interval).
    final_result = optimize_budget(model, common_bounds, budget)
    for name, alt_model in test_models:
        alt_result = optimize_budget(alt_model, common_bounds, budget)
        base = alt_model.total_revenue(current)
        proposed = alt_model.total_revenue(final_result.allocation)
        comparison.append({
            "model_variant": name, "current_revenue_lkr": base,
            "revenue_using_full_model_allocation_lkr": proposed,
            "lift_full_model_allocation_pct": 100 * (proposed / base - 1),
            "revenue_using_own_optimum_lkr": alt_result.incremental_revenue_lkr,
            "lift_own_optimum_pct": 100 * (alt_result.incremental_revenue_lkr / base - 1),
        })
    return pd.DataFrame(rows), pd.DataFrame(comparison)

def run_pipeline(output_dir: Path = OUTPUT, figures_dir: Path = FIGURES,
                 lower_ratio: float = 0.70, upper_ratio: float = 1.30,
                 cap_to_history: bool = True,
                 model_path: Path | None = None) -> tuple[OptimizationResult, pd.DataFrame]:
    model = ResponseModel(model_path) if model_path is not None else ResponseModel()
    history = pd.read_csv(PROCESSED)
    bounds = build_bounds(model, history, lower_ratio, upper_ratio, cap_to_history)
    result = optimize_budget(model, bounds)
    comparison = allocation_comparison(model, result)
    scenarios, scenario_allocations = evaluate_scenarios(model, bounds)
    sensitivity, model_sensitivity = sensitivity_analysis(model, history)
    base_budget = sum(model.current_spend.values())
    base_revenue = model.total_revenue(model.current_spend)
    outputs = {
        "channel_allocation_comparison.csv": comparison,
        "constraint_sensitivity.csv": sensitivity,
        "model_sensitivity.csv": model_sensitivity,
        "budget_scenarios.csv": scenarios,
        "scenario_channel_allocations.csv": scenario_allocations,
        "channel_bounds.csv": pd.DataFrame([
            {"channel": LABELS[c], "channel_key": c,
             "historical_min_lkr": float(history[c].min()),
             "historical_max_lkr": float(history[c].max()),
             "min_spend_lkr": bounds[c][0], "max_spend_lkr": bounds[c][1]}
            for c in model.channels]),
        "optimization_summary.csv": pd.DataFrame([{
            "budget_lkr": result.budget_lkr,
            "current_predicted_incremental_revenue_lkr": base_revenue,
            "optimized_predicted_incremental_revenue_lkr": result.incremental_revenue_lkr,
            "predicted_revenue_lift_lkr": result.incremental_revenue_lkr - base_revenue,
            "predicted_revenue_lift_pct": (result.incremental_revenue_lkr / base_revenue - 1) * 100,
            "current_portfolio_roas": base_revenue / base_budget,
            "optimized_portfolio_roas": result.portfolio_roas,
        }]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        df.to_csv(output_dir / name, index=False, float_format="%.8f")
    plot_outputs(model, comparison, scenarios, figures_dir)
    return result, outputs["optimization_summary.csv"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize six MMM channel budgets")
    parser.add_argument("--lower-ratio", type=float, default=0.7)
    parser.add_argument("--upper-ratio", type=float, default=1.3)
    parser.add_argument("--no-historical-cap", action="store_true")
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    result, summary = run_pipeline(
        lower_ratio=args.lower_ratio,
        upper_ratio=args.upper_ratio,
        cap_to_history=not args.no_historical_cap,
        model_path=args.model_path,
    )
    print("\nFixed-budget MMM optimisation (monthly LKR; steady-state estimates)")
    print(summary.T.to_string(header=False))
    print(f"\nMethod: {result.method}; {result.message}")
    print(f"Files: {OUTPUT}; figures: {FIGURES}/10_*.png")


if __name__ == "__main__":
    main()
