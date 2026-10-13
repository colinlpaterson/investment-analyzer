"""Known optima, feasibility boundaries, and failure handling for optimization."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult
from streamlit.testing.v1 import AppTest

from utils.portfolio_optimization import (
    Estimates, OptimizationError, estimate_parameters, optimize_portfolio, weight_caps,
)


def model(mu=(.08, .12), covariance=None):
    columns = pd.Index(["A", "B"])
    return Estimates(
        pd.Series(mu, index=columns),
        pd.DataFrame(covariance if covariance is not None else [[.04, 0], [0, .09]], index=columns, columns=columns),
        252, pd.Timestamp("2023-01-03"), pd.Timestamp("2023-12-29"), 0.0,
    )


def optimize(estimates=None, caps=(1., 1.), **kwargs):
    return optimize_portfolio(estimates or model(), pd.Series({"A": .5, "B": .5}), pd.Series(caps, index=["A", "B"]), **kwargs)


def test_known_two_asset_optima_and_frontier():
    result = optimize()
    # Independent variances .04/.09: min variance weights are .09/.13 and .04/.13.
    assert result.weights["Minimum volatility"].tolist() == pytest.approx([9 / 13, 4 / 13], abs=1e-6)
    # At rf=0, tangency weights are proportional to mu_i/variance_i: 2 and 4/3.
    assert result.weights["Maximum Sharpe"].tolist() == pytest.approx([.6, .4], abs=1e-6)
    assert result.max_sharpe_reason is None
    assert len(result.frontier) == 30
    assert result.frontier.iloc[-1]["Estimated return"] == pytest.approx(.12)
    assert result.frontier.iloc[-1]["Estimated volatility"] == pytest.approx(.3)
    assert (result.frontier["Estimated return"].diff().dropna() >= -1e-7).all()
    assert (result.frontier["Estimated volatility"].diff().dropna() >= -1e-7).all()
    # Every point is the analytic portfolio at that return, not a random sample.
    weight_b = (result.frontier["Estimated return"] - .08) / .04
    expected_vol = np.sqrt(.04 * (1 - weight_b) ** 2 + .09 * weight_b ** 2)
    np.testing.assert_allclose(result.frontier["Estimated volatility"], expected_vol, atol=1e-6)


def test_caps_hold_for_all_optimized_and_frontier_allocations():
    result = optimize(caps=(.55, .8))
    assert result.weights["Minimum volatility"].tolist() == pytest.approx([.55, .45])
    assert result.weights["Maximum Sharpe"].tolist() == pytest.approx([.55, .45], abs=1e-6)
    allocations = pd.concat([result.weights.drop(columns="Current").T, result.frontier_weights])
    assert (allocations >= -1e-7).all().all()
    assert (allocations["A"] <= .55 + 1e-7).all()
    assert (allocations["B"] <= .8 + 1e-7).all()
    np.testing.assert_allclose(allocations.sum(axis=1), 1, atol=1e-7)


def test_current_allocation_is_preserved_when_it_violates_caps():
    result = optimize(caps=(.3, .8))
    assert result.weights["Current"].tolist() == [.5, .5]
    assert not result.comparison.loc["Current", "Within caps"]
    assert result.comparison.loc["Minimum volatility", "Within caps"]


@pytest.mark.parametrize("maximum, overrides", [
    (.3, {}), (0, {}), (1.1, {}), (np.nan, {}), (.7, {"A": .8}),
    (.7, {"A": -.1}), (.7, {"A": np.inf}), (.7, {"C": .2}),
])
def test_invalid_or_infeasible_caps_are_rejected(maximum, overrides):
    with pytest.raises(ValueError):
        weight_caps(pd.Index(["A", "B"]), maximum, overrides)


def test_zero_cap_excludes_holding_and_exact_cap_sum_is_single_allocation():
    caps = weight_caps(pd.Index(["A", "B"]), 1., {"A": 0})
    result = optimize(caps=tuple(caps))
    assert result.weights["Minimum volatility"].tolist() == [0, 1]
    assert result.weights["Maximum Sharpe"].tolist() == [0, 1]
    assert len(result.frontier) == 1
    assert "only one allocation" in result.notices[0]


def test_single_asset_portfolio():
    estimates = Estimates(pd.Series({"A": .1}), pd.DataFrame([[.04]], index=["A"], columns=["A"]),
                          252, pd.Timestamp("2023-01-03"), pd.Timestamp("2023-12-29"), 0.)
    result = optimize_portfolio(estimates, pd.Series({"A": 1.}), pd.Series({"A": 1.}))
    assert result.weights.loc["A"].tolist() == [1, 1, 1]
    assert len(result.frontier) == 1


def test_equal_expected_returns_collapse_frontier():
    result = optimize(model(mu=(.1, .1)))
    assert len(result.frontier) == 1
    assert result.weights["Maximum Sharpe"].tolist() == pytest.approx(result.weights["Minimum volatility"].tolist())


@pytest.mark.parametrize("estimates,caps,rf", [
    (model(mu=(-.1, -.2)), (1., 1.), 0.),
    (model(mu=(.1, -.1)), (.4, 1.), 0.),
    (model(mu=(.01, .02)), (1., 1.), .05),
    (model(mu=(0., 0.)), (.5, .5), 0.),
])
def test_maximum_sharpe_unavailable_without_feasible_positive_excess(estimates, caps, rf):
    result = optimize(estimates, caps=caps, risk_free_rate=rf)
    assert "Maximum Sharpe" not in result.weights
    assert "positive" in result.max_sharpe_reason
    assert "Minimum volatility" in result.weights
    assert not result.frontier.empty


def test_zero_risk_positive_return_has_no_finite_maximum_sharpe():
    result = optimize(model(covariance=[[0, 0], [0, 0]]))
    assert "Maximum Sharpe" not in result.weights
    assert "zero estimated volatility" in result.max_sharpe_reason
    assert result.comparison["Estimated Sharpe ratio"].isna().all()


def test_zero_risk_positive_return_asset_is_not_reported_as_finite_sharpe():
    result = optimize(model(covariance=[[0, 0], [0, .09]]))
    assert "Maximum Sharpe" not in result.weights
    assert "zero estimated volatility" in result.max_sharpe_reason
    assert result.weights["Minimum volatility"].tolist() == pytest.approx([1., 0.], abs=1e-6)


def test_solver_exception_is_reported_as_an_optimization_error():
    with patch("utils.portfolio_optimization.minimize", side_effect=RuntimeError("backend failure")):
        with pytest.raises(OptimizationError, match="backend failure"):
            optimize()


def test_effective_risk_free_rate_uses_same_daily_conversion_as_analysis():
    result = optimize(risk_free_rate=.05)
    arithmetic_rf = 252 * (1.05 ** (1 / 252) - 1)
    expected = (.1 - arithmetic_rf) / np.sqrt(.25 * .04 + .25 * .09)
    assert result.comparison.loc["Current", "Estimated Sharpe ratio"] == pytest.approx(expected)


def test_estimates_are_arithmetic_and_covariance_is_finite_positive_semidefinite():
    values = np.tile([[.01, .02], [-.01, -.02], [.01, -.02], [-.01, .02]], (20, 1))
    returns = pd.DataFrame(values + [.001, .002], index=pd.bdate_range("2024-01-01", periods=80), columns=["A", "B"])
    estimates = estimate_parameters(returns)
    assert estimates.returns.tolist() == pytest.approx([.252, .504])
    assert estimates.observations == 80
    assert 0 <= estimates.shrinkage <= 1
    assert np.linalg.eigvalsh(estimates.covariance).min() >= 0
    # Shrinking toward equal variances preserves the trace of the MLE covariance.
    assert np.trace(estimates.covariance) == pytest.approx((.0001 + .0004) * 252)


def test_short_or_missing_return_histories_are_rejected():
    short = pd.DataFrame({"A": [.01] * 59}, index=pd.bdate_range("2024-01-01", periods=59))
    with pytest.raises(ValueError, match="at least 60"):
        estimate_parameters(short)
    missing = pd.DataFrame({"A": [.01] * 59 + [np.nan]}, index=pd.bdate_range("2024-01-01", periods=60))
    with pytest.raises(ValueError, match="finite"):
        estimate_parameters(missing)


def test_solver_failure_is_not_returned_as_a_valid_allocation():
    with patch("utils.portfolio_optimization.minimize", return_value=OptimizeResult(success=False, message="Iteration limit reached")):
        with pytest.raises(OptimizationError, match="did not converge"):
            optimize()


@pytest.mark.parametrize("candidate", [[1.2, -.2], [np.nan, .5], [.4, .4]])
def test_even_successful_solver_outputs_are_independently_validated(candidate):
    with patch("utils.portfolio_optimization.minimize", return_value=OptimizeResult(success=True, x=np.array(candidate), fun=.1)):
        with pytest.raises(OptimizationError):
            optimize()


def test_sharpe_failure_keeps_minimum_volatility_and_frontier():
    with patch("utils.portfolio_optimization._maximum_sharpe", side_effect=OptimizationError("Forced Sharpe failure")):
        result = optimize()
    assert "Forced Sharpe failure" in result.max_sharpe_reason
    assert "Maximum Sharpe" not in result.weights
    assert result.frontier.notna().all().all()


def test_failed_frontier_targets_remain_gaps():
    from utils.portfolio_optimization import _minimum_variance

    def fail_targets(cov, caps, initial, mu=None, target=None):
        if target is not None:
            raise OptimizationError("Forced frontier failure")
        return _minimum_variance(cov, caps, initial)

    with patch("utils.portfolio_optimization._minimum_variance", side_effect=fail_targets):
        result = optimize(frontier_points=4)
    assert result.frontier.iloc[0].notna().all()
    assert result.frontier.iloc[1:].isna().all().all()
    assert len(result.notices) == 3


def test_optimization_ui_runs_and_infeasible_resubmission_clears_results():
    dates = pd.bdate_range("2024-01-02", periods=261)

    def downloaded(ticker, start):
        returns = np.resize([.01, -.005, .003, -.002], 260)
        if ticker == "BND":
            returns = np.roll(returns, 1) * .5
        values = 100 * np.cumprod(np.r_[1., 1 + returns])
        return pd.DataFrame({"Date": dates, "Adj Close": values})

    with patch("utils.portfolio_data.load_daily_ticker_data", side_effect=downloaded), \
         patch("utils.portfolio_data.load_ticker_currency", return_value="USD"):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=10).run()
        app.switch_page("pages/4_🖩_Portfolio_Analysis.py").run()
        app.button[0].click().run()
        saved = app.session_state["portfolio_definition"]
        app.segmented_control[0].set_value("Optimize").run()
        app.button(key="optimization_1_submit").click().run()
        assert not app.exception and not app.error
        assert app.session_state["portfolio_definition"] == saved
        assert len(app.get("plotly_chart")) == 2  # Frontier and held-out growth.
        assert "recent" in app.session_state["portfolio_results"]["optimization"]
        app.number_input(key="optimization_1_maximum").set_value(30.)
        app.button(key="optimization_1_submit").click().run()
        assert not app.exception
        assert "Infeasible" in app.error[0].value
        assert "optimization" not in app.session_state["portfolio_results"]
        assert not app.get("plotly_chart")
