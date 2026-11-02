"""Run from repo root: python -m pytest tests/test_budget_optimization.py -v"""
import numpy as np
import pandas as pd
import pytest

from src.mmm_response import ResponseModel
from src.budget_optimization import (
    PROCESSED, build_bounds, optimize_budget, evaluate_scenarios,
    allocation_comparison, sensitivity_analysis, run_pipeline,
)


@pytest.fixture(scope="module")
def sample():
    model = ResponseModel()
    history = pd.read_csv(PROCESSED)
    bounds = build_bounds(model, history)
    return model, history, bounds


def test_optimizer_keeps_fixed_budget(sample):
    model, _, bounds = sample
    result = optimize_budget(model, bounds)
    assert result.budget_lkr == pytest.approx(sum(model.current_spend.values()), abs=0.01)
    assert sum(result.allocation.values()) == pytest.approx(result.budget_lkr, abs=0.1)


def test_all_spends_satisfy_bounds(sample):
    model, _, bounds = sample
    result = optimize_budget(model, bounds)
    for channel, (lo, hi) in bounds.items():
        assert lo - 0.1 <= result.allocation[channel] <= hi + 0.1


def test_optimized_revenue_no_worse_than_current(sample):
    model, _, bounds = sample
    result = optimize_budget(model, bounds)
    assert result.incremental_revenue_lkr >= model.total_revenue(model.current_spend) - 0.1


def test_gradient_matches_finite_difference(sample):
    model, _, _ = sample
    for channel in model.channels:
        current = model.current_spend[channel]
        h = 1000.0
        numeric = (float(model.response(channel, current + h)) -
                   float(model.response(channel, current - h))) / (2 * h)
        assert model.marginal_roas(channel, current) == pytest.approx(numeric, rel=1e-6, abs=1e-8)


def test_unfeasible_budget_raises(sample):
    model, _, bounds = sample
    with pytest.raises(ValueError, match="Infeasible"):
        optimize_budget(model, bounds, budget_lkr=1.0)


def test_scenarios_and_comparison(sample):
    model, _, bounds = sample
    summary, long = evaluate_scenarios(model, bounds)
    assert len(summary) == 5
    assert len(long) == 5 * len(model.channels)
    for mult, subset in long.groupby("budget_multiplier"):
        assert subset.optimized_spend_lkr.sum() == pytest.approx(
            sum(model.current_spend.values()) * mult, abs=0.1)
    compare = allocation_comparison(model, optimize_budget(model, bounds))
    assert len(compare) == 6
    assert compare.spend_change_lkr.sum() == pytest.approx(0, abs=0.1)


def test_sensitivity_runs(sample):
    model, history, _ = sample
    constraints, models = sensitivity_analysis(model, history)
    assert set(constraints.constraint_scenario) == {"conservative", "moderate", "flexible"}
    assert "train_only_fit" in set(models.model_variant)
    assert (models.lift_full_model_allocation_pct >= -0.01).all()


def test_pipeline_writes_dashboard_ready_outputs(sample, tmp_path):
    output = tmp_path / "outputs"
    figures = tmp_path / "figures"
    result, summary = run_pipeline(output_dir=output, figures_dir=figures)
    assert len(list(output.glob("*.csv"))) == 7
    assert len(list(figures.glob("10_*.png"))) == 3
    assert summary.iloc[0].budget_lkr == pytest.approx(result.budget_lkr, abs=0.01)
