"""Independent holdout arithmetic, leakage checks, exports, and stale-result behavior."""

from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from utils.portfolio import portfolio_from_rows
from utils.portfolio_data import PortfolioData
from utils.portfolio_exports import csv_bytes, metric_table, portfolio_metadata
from utils.portfolio_optimization import estimate_parameters, optimize_portfolio
from utils.portfolio_evaluation import evaluate_holdout


def dataset(returns=None):
    if returns is None:
        returns = np.random.default_rng(42).normal([.001, .0005], [.02, .005], (200, 2))
    prices = pd.DataFrame(100 * np.cumprod(np.vstack([np.ones(2), 1 + returns]), axis=0),
                          index=pd.bdate_range("2024-01-02", periods=len(returns) + 1), columns=["A", "B"])
    definition = portfolio_from_rows([{"Ticker": "A", "Weight (%)": 60}, {"Ticker": "B", "Weight (%)": 40}],
                                    prices.index[0].date(), prices.index[-1].date(), "Buy and hold")
    return PortfolioData(definition, prices, prices.pct_change().iloc[1:], None, pd.DataFrame(), "USD", ())


CAPS = pd.Series({"A": .7, "B": .7})


def test_holdout_has_no_future_leakage_and_includes_first_test_return():
    data = dataset()
    result = evaluate_holdout(data, CAPS)
    changed = data.prices.copy()
    changed.iloc[141:] *= np.linspace(2, 3, 60)[:, None]
    other = evaluate_holdout(replace(data, prices=changed), CAPS)
    pd.testing.assert_series_equal(result.estimates.returns, other.estimates.returns)
    pd.testing.assert_frame_equal(result.estimates.covariance, other.estimates.covariance)
    pd.testing.assert_frame_equal(result.solution.weights, other.solution.weights)
    assert not result.metrics.equals(other.metrics)
    assert result.estimates.end == data.prices.index[140]
    assert result.growth.index[0] == data.prices.index[140]
    assert result.test_observations == 60
    for name, weights in result.solution.weights.items():
        expected = (data.prices.iloc[140:] / data.prices.iloc[140]).mul(weights).sum(axis=1) * 10000
        np.testing.assert_allclose(result.growth[name], expected)
        assert result.metrics.loc[name, "Total return"] == pytest.approx(expected.iloc[-1] / 10000 - 1)
        if name != "Current":
            assert (weights <= CAPS + 1e-7).all()


def test_negative_training_does_not_gain_a_sharpe_solution_from_positive_future():
    returns = np.tile([-.01, -.005], (200, 1))
    returns[140:] = [.02, .01]
    result = evaluate_holdout(dataset(returns), CAPS)
    assert result.solution.max_sharpe_reason
    assert "Maximum Sharpe" not in result.metrics.index
    assert (result.metrics["Total return"] > 0).all()


@pytest.mark.parametrize("count,share", [(80, .7), (100, .8), (200, .9), (200, np.nan)])
def test_insufficient_holdout_and_invalid_split_rejected(count, share):
    with pytest.raises(ValueError):
        evaluate_holdout(dataset(np.zeros((count, 2))), CAPS, share)


def test_holdout_solver_failure_is_explicit():
    with patch("utils.portfolio_evaluation.optimize_portfolio", side_effect=ValueError("Solver failed")):
        with pytest.raises(ValueError, match="Solver failed"):
            evaluate_holdout(dataset(), CAPS)


def test_holdout_benchmark_uses_same_starting_close_and_test_dates():
    data = dataset()
    benchmark = pd.Series(np.linspace(100, 200, len(data.prices)), index=data.prices.index)
    result = evaluate_holdout(replace(data, benchmark_prices=benchmark,
                                    definition=replace(data.definition, benchmark="INDEX")), CAPS)
    np.testing.assert_allclose(result.growth["Benchmark (INDEX)"], benchmark.iloc[140:] / benchmark.iloc[140] * 10000)
    assert result.metrics.loc["Benchmark (INDEX)", "Total return"] == pytest.approx(200 / benchmark.iloc[140] - 1)


def test_estimation_sensitivity_changes_weights_and_respects_same_caps():
    rng = np.random.default_rng(10)
    returns = np.vstack([rng.normal([.001, .001], [.003, .04], (200, 2)),
                         rng.normal([.001, .001], [.04, .003], (100, 2))])
    data = dataset(returns)
    current = pd.Series({"A": .6, "B": .4})
    full = optimize_portfolio(estimate_parameters(data.returns), current, CAPS, frontier_points=0)
    recent = optimize_portfolio(estimate_parameters(data.returns.iloc[150:]), current, CAPS, frontier_points=0)
    assert (full.weights["Minimum volatility"] - recent.weights["Minimum volatility"]).abs().max() > .1
    for result in (full, recent):
        assert (result.weights.drop(columns="Current").le(CAPS + 1e-7, axis=0)).all().all()


def test_csv_preserves_units_dates_context_and_undefined_values():
    data = dataset()
    frame = pd.DataFrame({"Total return": [.125, np.nan], "Sharpe ratio": [1.5, np.nan],
                          "Volatility contribution": [.03, -.01]}, index=data.prices.index[:2])
    raw = csv_bytes(metric_table(frame), dict(portfolio_metadata(data), Seed=42)).decode()
    parsed = pd.read_csv(StringIO(raw), index_col=0)
    assert parsed.index[0] == "01/02/2024"
    assert parsed["Total return (%)"].iloc[0] == 12.5
    assert parsed["Sharpe ratio"].iloc[0] == 1.5
    assert parsed["Volatility contribution (pp)"].iloc[0] == 3
    assert pd.isna(parsed["Total return (%)"].iloc[1])
    assert parsed["Context: Currency"].iloc[0] == "USD"
    assert parsed["Context: Seed"].iloc[0] == 42


def download(ticker, start):
    data = dataset()
    return pd.DataFrame({"Date": data.prices.index, "Adj Close": data.prices["A" if ticker == "SPY" else "B"].values})


def start_app():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=10).run()
    app.switch_page("pages/4_🖩_Portfolio_Analysis.py").run()
    app.button[0].click().run()
    return app


@pytest.mark.parametrize("control", ["risk_free", "start", "rebalancing", "holdings"])
def test_portfolio_edit_immediately_removes_results_and_downloads(control):
    with patch("utils.portfolio_data.load_daily_ticker_data", side_effect=download) as loader, \
         patch("utils.portfolio_data.load_ticker_currency", return_value="USD"):
        app = start_app()
        assert app.get("plotly_chart") and app.get("download_button")
        calls = loader.call_count
        prefix = "portfolio_analysis_1"
        # Use actual page-local keys so this test follows the builder's public controls.
        if control == "risk_free":
            app.number_input[0].set_value(3.).run()
        elif control == "start":
            app.date_input[0].set_value(pd.Timestamp("2023-01-01").date()).run()
        elif control == "rebalancing":
            app.selectbox[0].select("Quarterly").run()
        else:
            key = f"{prefix}_holdings"
            app.session_state[key] = {"edited_rows": {0: {"Weight (%)": 50.}}, "added_rows": [], "deleted_rows": []}
            app.run()
        assert not app.exception
        assert app.session_state["portfolio_data"] is None
        assert not app.get("plotly_chart") and not app.get("download_button")
        assert loader.call_count == calls


def test_constraint_and_simulation_edits_clear_only_dependent_results():
    with patch("utils.portfolio_data.load_daily_ticker_data", side_effect=download), \
         patch("utils.portfolio_data.load_ticker_currency", return_value="USD"):
        app = start_app()
        app.segmented_control[0].set_value("Optimize").run()
        app.button(key="optimization_1_submit").click().run()
        assert "validation" in app.session_state["portfolio_results"]["optimization"]
        app.number_input(key="optimization_1_maximum").set_value(70.).run()
        assert not app.get("plotly_chart") and not app.get("download_button")
        assert "historical" in app.session_state["portfolio_results"]
        app.button(key="optimization_1_submit").click().run()
        app.slider(key="optimization_1_training").set_value(60).run()
        assert "optimization" not in app.session_state["portfolio_results"]
        app.button(key="optimization_1_submit").click().run()
        app.session_state["optimization_1_overrides"] = {
            "edited_rows": {0: {"Individual cap (%)": 50.}}, "added_rows": [], "deleted_rows": []}
        app.run()
        assert "optimization" not in app.session_state["portfolio_results"]
        assert not app.get("plotly_chart") and not app.get("download_button")
        app.switch_page("pages/5_🎲_Portfolio_Simulation.py").run()
        revision = app.session_state["portfolio_results"].get("simulation_selection_revision", 0)
        prefix = f"simulation_1_{revision}"
        app.number_input(key=f"{prefix}_paths").set_value(100)
        app.number_input(key=f"{prefix}_years").set_value(1)
        app.button(key=f"{prefix}_run").click().run()
        assert "simulation" in app.session_state["portfolio_results"]
        for suffix, value in [("seed", 24), ("contribution", 200.)]:
            app.number_input(key=f"{prefix}_{suffix}").set_value(value).run()
            assert not app.exception
            assert "simulation" not in app.session_state["portfolio_results"]
            assert not app.get("plotly_chart") and not app.get("download_button")
            app.button(key=f"{prefix}_run").click().run()
        assert app.session_state["portfolio_results"]["simulation"]["result"].contributed.iloc[-1] == 12400


def test_optimization_invalidation_preserves_current_simulation_and_is_idempotent():
    from utils.portfolio_state import initialize_portfolio_state, invalidate_optimization_inputs
    state = {}
    initialize_portfolio_state(state)
    state["portfolio_definition"] = dataset().definition
    current_run = {"source": "Current", "result": object()}
    results = state["portfolio_results"]
    results.update(simulation=current_run, optimization=object())
    invalidate_optimization_inputs(state)
    invalidate_optimization_inputs(state)
    assert results["simulation"] is current_run
    assert "simulation_selection_revision" not in results
    assert "optimization" not in results
    results.update(simulation={"source": "Minimum volatility"}, simulation_selection="Minimum volatility",
                   simulation_selection_revision=3)
    invalidate_optimization_inputs(state)
    assert "simulation" not in results
    assert results["simulation_selection"] == "Current"
    assert results["simulation_selection_revision"] == 4
    invalidate_optimization_inputs(state)
    assert results["simulation_selection_revision"] == 4


@pytest.mark.parametrize("edits,message", [
    ({0: {"Ticker": ""}}, "enter a ticker"),
    ({1: {"Ticker": " spy "}}, "Duplicate tickers"),
    ({0: {"Weight (%)": -1}}, "non-negative"),
    ({0: {"Weight (%)": 20}}, "Weights must total 100%"),
])
def test_local_builder_feedback_before_save_without_download(edits, message):
    with patch("utils.portfolio_data.load_daily_ticker_data") as loader:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=10).run()
        app.switch_page("pages/4_🖩_Portfolio_Analysis.py").run()
        app.session_state["portfolio_analysis_0_holdings"] = {
            "edited_rows": edits, "added_rows": [], "deleted_rows": []}
        app.run()
        assert not app.exception
        assert any(message in warning.value for warning in app.warning)
        loader.assert_not_called()
