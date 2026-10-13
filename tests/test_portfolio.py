"""Offline checks for portfolio inputs, data integrity, and cross-page state."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from utils.portfolio import portfolio_from_rows
from utils.portfolio_data import prepare_portfolio_data
from utils.portfolio_state import initialize_portfolio_state, save_portfolio, store_portfolio_data


def definition(rows=None, **kwargs):
    return portfolio_from_rows(
        rows if rows is not None else [
            {"Ticker": " a ", "Weight (%)": 60}, {"Ticker": "B", "Weight (%)": 40},
        ], date(2024, 1, 1), date(2024, 1, 10), **kwargs,
    )


def frame(dates=None, prices=None):
    return pd.DataFrame({
        "Date": pd.to_datetime(dates if dates is not None else ["2024-01-02", "2024-01-03", "2024-01-04"]),
        "Close": 100.0,
        "Adj Close": prices if prices is not None else [100.0, 110.0, 99.0],
    })


def prepare(config=None, frames=None, currencies=None):
    return prepare_portfolio_data(
        config or definition(), frames or {"A": frame(), "B": frame()},
        currencies or {"A": "USD", "B": "USD"},
    )


def test_definition_normalizes_symbols_and_uses_fractional_units():
    config = definition(benchmark=" spy ", risk_free_rate_percent=4.25)
    assert config.tickers == ("A", "B")
    assert config.holdings[0].weight == .6
    assert config.benchmark == "SPY"
    assert config.risk_free_rate == .0425


@pytest.mark.parametrize("rows, message", [
    ([], "at least one"),
    ([{"Ticker": "A", "Weight (%)": 40}], "total 100%"),
    ([{"Ticker": "A", "Weight (%)": 100}, {"Ticker": " a ", "Weight (%)": 0}], "Duplicate"),
    ([{"Ticker": "", "Weight (%)": 100}], "enter a ticker"),
    ([{"Ticker": "A", "Weight (%)": None}], "numeric weight"),
    ([{"Ticker": "A", "Weight (%)": float("nan")}], "finite"),
    ([{"Ticker": "A", "Weight (%)": float("inf")}], "finite"),
    ([{"Ticker": "A", "Weight (%)": -10}], "non-negative"),
    ([{"Ticker": "A", "Weight (%)": True}], "finite"),
])
def test_invalid_holdings_are_rejected(rows, message):
    with pytest.raises(ValueError, match=message):
        definition(rows)


def test_normalization_is_explicit_and_keeps_zero_weight_candidates():
    rows = [{"Ticker": "A", "Weight (%)": 20}, {"Ticker": "B", "Weight (%)": 30}, {"Ticker": "C", "Weight (%)": 0}]
    with pytest.raises(ValueError, match="total 100%"):
        definition(rows)
    result = definition(rows, normalize=True)
    assert [h.weight for h in result.holdings] == pytest.approx([.4, .6, 0])
    with pytest.raises(ValueError, match="positive weight"):
        definition([{"Ticker": "A", "Weight (%)": 0}], normalize=True)


@pytest.mark.parametrize("kwargs", [
    {"rebalancing": "Sometimes"}, {"risk_free_rate_percent": np.nan},
    {"risk_free_rate_percent": -100},
])
def test_invalid_assumptions_are_rejected(kwargs):
    with pytest.raises(ValueError):
        definition(**kwargs)


def test_dates_reject_reversed_or_incomplete_periods():
    rows = [{"Ticker": "A", "Weight (%)": 100}]
    with pytest.raises(ValueError, match="before end"):
        portfolio_from_rows(rows, date(2024, 1, 2), date(2024, 1, 1))
    with pytest.raises(ValueError, match="before today"):
        portfolio_from_rows(rows, date(2024, 1, 1), date.today())


def test_adjusted_returns_and_common_history_are_explicit():
    result = prepare(frames={"A": frame(), "B": frame(["2024-01-03", "2024-01-04"], [50, 60])})
    assert result.returns["A"].tolist() == pytest.approx([-.1])
    assert result.returns["B"].tolist() == pytest.approx([.2])
    assert result.prices.index[0] == pd.Timestamp("2024-01-03")
    assert "01/03/2024 to 01/04/2024" in result.notices[0]
    assert len(result.coverage) == 2


def test_requested_bounds_are_inclusive_and_applied_before_returns():
    prices = frame(["2023-12-29", "2024-01-02", "2024-01-10", "2024-01-11"], [10, 100, 110, 200])
    result = prepare(frames={"A": prices, "B": prices})
    assert result.prices.index.tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-10"]))
    assert result.returns["A"].tolist() == pytest.approx([.1])


@pytest.mark.parametrize("bad_frame, message", [
    (None, "B: no price"),
    (frame().drop(columns="Adj Close"), "B: adjusted-close"),
    (frame(prices=[100, np.nan, 110]), "B: missing"),
    (frame(prices=[100, 0, 110]), "non-positive"),
    (frame(prices=[100, np.inf, 110]), "non-finite"),
    (frame(["2024-01-02", "2024-01-02", "2024-01-04"]), "duplicate"),
    (frame(["2024-01-02", "2024-01-04"], [100, 110]), "Missing dates"),
])
def test_bad_data_never_drops_a_holding_or_fills_prices(bad_frame, message):
    with pytest.raises(ValueError, match=message):
        prepare(frames={"A": frame(), "B": bad_frame})


def test_multiple_failed_tickers_are_reported_together():
    with pytest.raises(ValueError) as error:
        prepare(frames={"A": None, "B": None})
    assert "A:" in str(error.value) and "B:" in str(error.value)


@pytest.mark.parametrize("currency, message", [(None, "could not be verified"), ("EUR", "share a quote currency")])
def test_currencies_are_verified(currency, message):
    with pytest.raises(ValueError, match=message):
        prepare(currencies={"A": "USD", "B": currency})


def test_single_holding_and_benchmark_reusing_holding():
    result = prepare(definition([{"Ticker": "A", "Weight (%)": 100}], benchmark="A"))
    assert result.prices.columns.tolist() == ["A"]
    pd.testing.assert_series_equal(result.benchmark_prices, result.prices["A"])


def test_benchmark_cannot_silently_shorten_portfolio_history():
    config = definition(benchmark="C")
    with pytest.raises(ValueError, match="benchmark does not cover"):
        prepare(config, {"A": frame(), "B": frame(), "C": frame(["2024-01-03", "2024-01-04"], [100, 110])},
                {"A": "USD", "B": "USD", "C": "USD"})


def test_state_invalidates_outputs_and_rejects_data_for_previous_definition():
    state = {}
    initialize_portfolio_state(state)
    save_portfolio(state, definition())
    data = prepare()
    store_portfolio_data(state, data)
    state["portfolio_results"] = {"analysis": "old", "optimization": "old", "simulation": "old"}
    save_portfolio(state, definition(rebalancing="Quarterly"))
    assert state["portfolio_data"] is None
    assert state["portfolio_results"] == {}
    with pytest.raises(ValueError, match="different portfolio"):
        store_portfolio_data(state, data)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PAGE = "pages/4_🖩_Portfolio_Analysis.py"
SIMULATION_PAGE = "pages/5_🎲_Portfolio_Simulation.py"


def test_saved_portfolio_survives_page_switches_and_edits_invalidate_results():
    with patch("utils.portfolio_data.load_daily_ticker_data", return_value=frame()) as download, \
         patch("utils.portfolio_data.load_ticker_currency", return_value="USD"):
        app = AppTest.from_file(str(ROOT / "app.py")).run()
        app.switch_page(ANALYSIS_PAGE).run()
        assert not app.exception
        download.assert_not_called()
        app.button[0].click().run()
        assert not app.exception
        assert app.success
        saved = app.session_state["portfolio_definition"]
        app.session_state["portfolio_results"] = {"analysis": "old"}
        calls = download.call_count
        app.switch_page(SIMULATION_PAGE).run()
        assert not app.exception
        assert app.session_state["portfolio_definition"] == saved
        assert app.success
        assert download.call_count == calls
        app.selectbox[0].select("Quarterly")
        app.button[0].click().run()
        assert not app.exception
        assert app.session_state["portfolio_results"] == {}
        app.switch_page(ANALYSIS_PAGE).run()
        assert app.selectbox[0].value == "Quarterly"
        assert not app.exception


def test_failed_reload_removes_old_ready_data():
    with patch("utils.portfolio_data.load_daily_ticker_data", return_value=frame()) as download, \
         patch("utils.portfolio_data.load_ticker_currency", return_value="USD"):
        app = AppTest.from_file(str(ROOT / "app.py")).run()
        app.switch_page(SIMULATION_PAGE).run()
        app.button[0].click().run()
        assert app.success
        download.return_value = None
        app.button[0].click().run()
        assert not app.exception
        assert app.error
        assert not app.success
        assert app.session_state["portfolio_data"] is None
        assert app.session_state["portfolio_definition"] is not None


def test_normalize_action_and_invalid_submission():
    with patch("utils.portfolio_data.load_daily_ticker_data", return_value=frame()) as download, \
         patch("utils.portfolio_data.load_ticker_currency", return_value="USD"):
        app = AppTest.from_file(str(ROOT / "app.py")).run()
        app.switch_page(ANALYSIS_PAGE).run()
        # AppTest cannot replay data_editor edits from a browser; supply its edit payload on each submission.
        edits = {
            "edited_rows": {0: {"Weight (%)": 20.0}}, "added_rows": [], "deleted_rows": [],
        }
        app.session_state["portfolio_analysis_0_holdings"] = edits
        app.button[0].click().run()
        assert app.error
        assert not app.exception
        download.assert_not_called()
        app.session_state["portfolio_analysis_0_holdings"] = edits
        app.button[1].click().run()
        assert not app.exception
        weights = [h.weight for h in app.session_state["portfolio_definition"].holdings]
        assert weights == pytest.approx([1 / 3, 2 / 3])
        assert app.success
