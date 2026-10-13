"""Seeded block sampling and independently calculated cash-flow examples."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from utils.portfolio import portfolio_from_rows
from utils.portfolio_simulation import SimulationConfig, sample_block_indices, simulate_portfolio
from utils.portfolio_state import initialize_portfolio_state, save_portfolio, select_simulation_allocation, simulation_allocations


def history(a=0., b=0.):
    return pd.DataFrame({"A": np.full(84, a), "B": np.full(84, b)},
                        index=pd.bdate_range("2024-01-02", periods=84))


def config(**kwargs):
    return replace(SimulationConfig(initial_value=100, months=2, paths=10), **kwargs)


WEIGHTS = pd.Series({"A": .5, "B": .5})
DOUBLE_IN_MONTH = 2 ** (1 / 21) - 1


def test_flat_returns_equal_cash_invested_and_target_equality_counts_as_success():
    result = simulate_portfolio(history(), WEIGHTS, config(months=3, monthly_contribution=10, target_value=130))
    np.testing.assert_allclose(result.monthly_values, np.tile([100, 110, 120, 130], (10, 1)))
    assert result.contributed.tolist() == [100, 110, 120, 130]
    assert result.probability_loss == 0
    assert result.probability_target == 1
    assert result.ending_values.tolist() == [130] * 10


def test_contributions_arrive_after_returns_and_compound_from_following_month():
    result = simulate_portfolio(history(DOUBLE_IN_MONTH, DOUBLE_IN_MONTH), WEIGHTS,
                                config(monthly_contribution=10))
    # 100*2+10 = 210, then 210*2+10 = 430. Only 120 was contributed.
    np.testing.assert_allclose(result.monthly_values, np.tile([100, 210, 430], (10, 1)), rtol=1e-12)
    assert result.contributed.iloc[-1] == 120
    assert result.probability_loss == 0


@pytest.mark.parametrize("schedule,expected", [
    ("Monthly", [100, 150, 225, 337.5, 506.25]),
    ("Quarterly", [100, 150, 250, 450, 675]),
    ("Annually", [100, 150, 250, 450, 850]),
    ("Buy and hold", [100, 150, 250, 450, 850]),
])
def test_rebalancing_schedule_and_drift(schedule, expected):
    result = simulate_portfolio(history(DOUBLE_IN_MONTH, 0), WEIGHTS, config(months=4, rebalancing=schedule))
    np.testing.assert_allclose(result.monthly_values[0], expected, rtol=1e-12)
    np.testing.assert_allclose(result.final_weights.sum(axis=1), 1)


def test_annual_rebalance_occurs_after_month_twelve():
    result = simulate_portfolio(history(DOUBLE_IN_MONTH, 0), WEIGHTS, config(months=13, rebalancing="Annually"))
    # After 12 months A is worth 50*4096, B is 50. Reset halves, then one doubles.
    assert result.monthly_values[0, 12] == pytest.approx(50 * 4096 + 50)
    assert result.monthly_values[0, 13] == pytest.approx((50 * 4096 + 50) * 1.5)


@pytest.mark.parametrize("schedule,ending", [("Buy and hold", 300), ("Monthly", 275)])
def test_contributions_buy_target_proportions_without_implicit_rebalancing(schedule, ending):
    result = simulate_portfolio(history(DOUBLE_IN_MONTH, 0), WEIGHTS,
                                config(monthly_contribution=20, rebalancing=schedule))
    assert result.monthly_values[0, 1] == pytest.approx(170)
    assert result.monthly_values[0, 2] == pytest.approx(ending)
    if schedule == "Buy and hold":
        # Final values: A=230, B=70. New cash buys 10 of each at each month-end.
        np.testing.assert_allclose(result.final_weights[0], [230 / 300, 70 / 300])


def test_block_indices_are_contiguous_circular_and_reproducible():
    indices = sample_block_indices(84, 21, 100, np.random.default_rng(42))
    same = sample_block_indices(84, 21, 100, np.random.default_rng(42))
    np.testing.assert_array_equal(indices, same)
    np.testing.assert_array_equal(np.diff(indices, axis=1) % 84, np.ones((100, 20)))
    assert (indices >= 0).all() and (indices < 84).all()
    assert (np.diff(indices, axis=1) < 0).any()  # At least one block crosses the end.


def test_same_seed_reproduces_paths_different_seed_changes_them_without_global_rng_changes():
    returns = history()
    returns["A"] = np.linspace(-.02, .03, len(returns))
    returns["B"] = np.linspace(.01, -.01, len(returns))
    first = simulate_portfolio(returns, WEIGHTS, config())
    np.random.seed(123)
    expected_next_random = np.random.random()
    np.random.seed(123)
    second = simulate_portfolio(returns, WEIGHTS, config())
    assert np.random.random() == expected_next_random
    np.testing.assert_array_equal(first.monthly_values, second.monthly_values)
    third = simulate_portfolio(returns, WEIGHTS, config(seed=43))
    assert not np.array_equal(first.monthly_values, third.monthly_values)


def test_assets_share_sampled_dates_instead_of_sampling_independently():
    returns = history()
    returns["A"] = np.linspace(-.02, .03, len(returns))
    returns["B"] = -returns["A"]
    settings = config(months=1, block_length=21, rebalancing="Buy and hold")
    result = simulate_portfolio(returns, WEIGHTS, settings)
    indices = sample_block_indices(len(returns), 21, 10, np.random.default_rng(settings.seed))
    # Independent fixed-share calculation uses the same index sequence for both assets.
    a = 50 * np.prod(1 + returns["A"].to_numpy()[indices], axis=1)
    b = 50 * np.prod(1 + returns["B"].to_numpy()[indices], axis=1)
    np.testing.assert_allclose(result.ending_values, a + b)


def test_zero_weight_asset_is_ignored_and_input_weights_are_not_mutated():
    weights = pd.Series({"B": 0., "A": 1.})
    original = weights.copy()
    result = simulate_portfolio(history(0, DOUBLE_IN_MONTH), weights, config())
    np.testing.assert_allclose(result.monthly_values, 100)
    pd.testing.assert_series_equal(weights, original)


def test_loss_probability_and_quantiles_are_based_on_terminal_nominal_wealth():
    result = simulate_portfolio(history(-.01, -.01), WEIGHTS, config(target_value=100))
    assert result.probability_loss == 1
    assert result.probability_target == 0
    assert (result.quantiles.diff(axis=1).iloc[:, 1:] >= -1e-10).all().all()
    np.testing.assert_allclose(result.quantiles.iloc[-1], result.ending_values[0])
    assert simulate_portfolio(history(), WEIGHTS, config()).probability_target is None


@pytest.mark.parametrize("changes", [
    {"initial_value": 0}, {"monthly_contribution": -1}, {"months": 0},
    {"months": 1.5}, {"paths": 10001}, {"seed": -1}, {"block_length": 0},
    {"inflation_rate": -1}, {"target_value": np.nan}, {"rebalancing": "Daily"},
])
def test_invalid_simulation_inputs(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_invalid_history_and_oversized_runs_are_rejected():
    with pytest.raises(ValueError, match="at least 60"):
        simulate_portfolio(history().iloc[:59], WEIGHTS, config())
    with pytest.raises(ValueError, match="Block length"):
        simulate_portfolio(history(), WEIGHTS, config(block_length=100))
    four_assets = pd.concat([history(), history().rename(columns={"A": "C", "B": "D"})], axis=1)
    with pytest.raises(ValueError, match="too large"):
        simulate_portfolio(four_assets, pd.Series({"A": .25, "B": .25, "C": .25, "D": .25}),
                            config(paths=10000, months=480))
    invalid = history()
    invalid.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        simulate_portfolio(invalid, WEIGHTS, config())


def test_handoff_changes_only_simulation_selection_and_save_invalidates_it():
    state = {}
    initialize_portfolio_state(state)
    definition = portfolio_from_rows([{"Ticker": "A", "Weight (%)": 100}],
                                     pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-12-31").date())
    save_portfolio(state, definition)
    state["portfolio_results"]["simulation"] = "stale"
    select_simulation_allocation(state, "Current")
    assert "simulation" not in state["portfolio_results"]
    assert state["portfolio_definition"] is definition
    assert simulation_allocations(state)["Current"].to_dict() == {"A": 1.}
    with pytest.raises(ValueError, match="no longer available"):
        select_simulation_allocation(state, "Maximum Sharpe")
    save_portfolio(state, definition)
    assert state["portfolio_results"] == {}


def test_optimized_handoff_runs_simulation_without_replacing_current_portfolio():
    dates = pd.bdate_range("2024-01-02", periods=127)

    def download(ticker, start):
        returns = np.resize([.01, -.005, .003, -.002], 126)
        if ticker == "BND":
            returns = np.roll(returns, 1) * .5
        return pd.DataFrame({"Date": dates, "Adj Close": 100 * np.cumprod(np.r_[1, 1 + returns])})

    with patch("utils.portfolio_data.load_daily_ticker_data", side_effect=download), \
         patch("utils.portfolio_data.load_ticker_currency", return_value="USD"):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=10).run()
        app.switch_page("pages/4_🖩_Portfolio_Analysis.py").run()
        app.button[0].click().run()
        saved = app.session_state["portfolio_definition"]
        app.segmented_control[0].set_value("Optimize").run()
        app.button(key="optimization_1_submit").click().run()
        expected = app.session_state["portfolio_results"]["optimization"]["solution"].weights["Minimum volatility"]
        app.button(key="optimization_simulate_Minimum volatility").click().run()
        assert not app.exception
        # Keep AppTest's next reruns on the destination selected by st.switch_page.
        app.switch_page("pages/5_🎲_Portfolio_Simulation.py").run()
        revision = app.session_state["portfolio_results"]["simulation_selection_revision"]
        prefix = f"simulation_1_{revision}"
        assert app.selectbox(key=f"{prefix}_source").value == "Minimum volatility"
        app.number_input(key=f"{prefix}_paths").set_value(100)
        app.number_input(key=f"{prefix}_years").set_value(1)
        app.number_input(key=f"{prefix}_contribution").set_value(100.)
        app.number_input(key=f"{prefix}_target").set_value(15000.)
        app.button(key=f"{prefix}_run").click().run()
        assert not app.exception and not app.error
        run = app.session_state["portfolio_results"]["simulation"]["result"]
        pd.testing.assert_series_equal(run.weights, expected.rename(None), check_names=False)
        assert run.contributed.iloc[-1] == 11200
        assert len(app.get("plotly_chart")) == 2
        assert app.session_state["portfolio_definition"] == saved
        with patch("utils.portfolio_simulation_ui.simulate_portfolio", side_effect=ValueError("Forced run failure")):
            app.button(key=f"{prefix}_run").click().run()
        assert not app.exception
        assert app.error and "Forced run failure" in app.error[0].value
        assert "simulation" not in app.session_state["portfolio_results"]
        assert not app.get("plotly_chart")
        app.selectbox(key=f"{prefix}_source").select("Current").run()
        assert "simulation" not in app.session_state["portfolio_results"]
        assert not app.get("plotly_chart")
