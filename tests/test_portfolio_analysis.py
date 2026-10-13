"""Independent arithmetic examples for portfolio accounting and risk statistics."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from utils.portfolio import portfolio_from_rows
from utils.portfolio_analysis import (
    analyze_portfolio, backtest_portfolio, performance_metrics, risk_decomposition,
)
from utils.portfolio_data import prepare_portfolio_data


def prices(dates=None):
    return pd.DataFrame(
        {"A": [100., 200., 100.], "B": [100., 100., 200.]},
        index=pd.to_datetime(dates or ["2024-01-30", "2024-01-31", "2024-02-01"]),
    )


def test_buy_and_hold_matches_fixed_share_accounting():
    result = backtest_portfolio(prices(), pd.Series({"A": .5, "B": .5}), "Buy and hold", 100)
    # Start with half a share of each. Values are 50+50, 100+50, 50+100.
    assert result.values.tolist() == pytest.approx([100, 150, 150])
    assert result.returns.tolist() == pytest.approx([.5, 0])
    assert result.weights.iloc[1].tolist() == pytest.approx([2 / 3, 1 / 3])
    assert result.weights.iloc[-1].tolist() == pytest.approx([1 / 3, 2 / 3])
    assert result.rebalance_dates.empty


def test_monthly_rebalance_is_self_financing_and_applies_to_next_return():
    result = backtest_portfolio(prices(), pd.Series({"B": .5, "A": .5}), "Monthly", 100)
    # Jan 31: 150 becomes 75/75. Feb 1: 75*.5 + 75*2 = 187.5.
    assert result.values.tolist() == pytest.approx([100, 150, 187.5])
    assert result.returns.tolist() == pytest.approx([.5, .25])
    assert result.weights.iloc[1].tolist() == pytest.approx([.5, .5])
    assert result.weights.iloc[-1].tolist() == pytest.approx([.2, .8])
    assert result.rebalance_dates.tolist() == [pd.Timestamp("2024-01-31")]


@pytest.mark.parametrize("dates,schedule,expected", [
    (["2024-01-29", "2024-01-30", "2024-01-31"], "Monthly", 150),
    (["2024-02-28", "2024-02-29", "2024-03-01"], "Quarterly", 150),
    (["2024-03-27", "2024-03-28", "2024-04-01"], "Quarterly", 187.5),
    (["2023-12-28", "2023-12-29", "2024-01-02"], "Annually", 187.5),
    (["2024-03-27", "2024-03-28", "2024-04-01"], "Annually", 150),
])
def test_calendar_boundaries_include_holidays_and_do_not_rebalance_daily(dates, schedule, expected):
    result = backtest_portfolio(prices(dates), pd.Series({"A": .5, "B": .5}), schedule, 100)
    assert result.values.iloc[-1] == pytest.approx(expected)
    assert dates[-1] not in result.rebalance_dates.astype(str)


def test_single_holding_and_zero_weight_candidate():
    result = backtest_portfolio(prices(), pd.Series({"A": 1., "B": 0.}), "Monthly", 100)
    assert result.values.tolist() == pytest.approx([100, 200, 100])
    assert result.weights["B"].tolist() == [0, 0, 0]
    single = backtest_portfolio(prices()[["A"]], pd.Series({"A": 1.}), "Buy and hold", 100)
    pd.testing.assert_series_equal(result.values, single.values)


def test_metrics_match_hand_calculated_returns_and_sample_variance():
    values = pd.Series([100., 80., 120., 90.], index=pd.to_datetime([
        "2023-01-01", "2023-05-01", "2023-09-01", "2024-01-01",
    ]))
    metrics = performance_metrics(values, .05)
    # Period returns -1/5, 1/2, -1/4: mean=1/60, sample variance=211/1200.
    # This sparse synthetic series isolates formulas; real inputs are daily prices.
    assert metrics["Total return"] == pytest.approx(-.1)
    assert metrics["CAGR"] == pytest.approx(.9 ** (365.25 / 365) - 1)
    assert metrics["Volatility"] == pytest.approx(np.sqrt(211 / 1200 * 252))
    expected_sharpe = (1 / 60 - (1.05 ** (1 / 252) - 1)) / np.sqrt(211 / 1200) * np.sqrt(252)
    assert metrics["Sharpe ratio"] == pytest.approx(expected_sharpe)
    assert metrics["Max drawdown"] == pytest.approx(-.25)


def test_drawdown_includes_initial_investment_and_zero_volatility_is_undefined_sharpe():
    index = pd.date_range("2024-01-01", periods=3)
    down = performance_metrics(pd.Series([100, 90, 95], index=index))
    assert down["Max drawdown"] == pytest.approx(-.1)
    flat = performance_metrics(pd.Series([100, 100, 100], index=index))
    assert flat["Total return"] == flat["CAGR"] == flat["Max drawdown"] == flat["Volatility"] == 0
    assert np.isnan(flat["Sharpe ratio"])
    short = performance_metrics(pd.Series([100, 90], index=index[:2]))
    assert short["Total return"] == pytest.approx(-.1)
    assert np.isnan(short["Volatility"]) and np.isnan(short["Sharpe ratio"])


def test_uncorrelated_assets_have_risk_shares_proportional_to_squared_weights():
    returns = pd.DataFrame({"A": [.1, -.1, .1, -.1], "B": [.1, .1, -.1, -.1]})
    risk, volatility = risk_decomposition(returns, pd.Series({"B": .25, "A": .75}))
    # Covariance is zero, each annual variance is (.04/3)*252 = 3.36.
    assert volatility == pytest.approx(np.sqrt(3.36 * (.75 ** 2 + .25 ** 2)))
    assert risk["Risk contribution"].tolist() == pytest.approx([.9, .1])
    assert risk["Volatility contribution"].sum() == pytest.approx(volatility)


def test_negative_risk_contribution_is_preserved():
    returns = pd.DataFrame({"A": [.1, -.1, .1, -.1], "B": [-.1, .1, -.1, .1]})
    risk, volatility = risk_decomposition(returns, pd.Series({"A": .75, "B": .25}))
    assert risk["Risk contribution"].tolist() == pytest.approx([1.5, -.5])
    assert risk["Volatility contribution"].sum() == pytest.approx(volatility)
    hedged, hedged_vol = risk_decomposition(returns, pd.Series({"A": .5, "B": .5}))
    assert hedged_vol == pytest.approx(0)
    assert hedged["Risk contribution"].isna().all()


def test_risk_for_zero_weight_and_insufficient_observations():
    returns = pd.DataFrame({"A": [.1, -.1], "B": [-.1, .1]})
    risk, _ = risk_decomposition(returns, pd.Series({"A": 1., "B": 0.}))
    assert risk["Risk contribution"].tolist() == pytest.approx([1, 0])
    risk, volatility = risk_decomposition(returns.iloc[:1], pd.Series({"A": 1., "B": 0.}))
    assert risk["Risk contribution"].isna().all() and np.isnan(volatility)


@pytest.mark.parametrize("weights", [pd.Series({"A": 1.}), pd.Series({"A": .8, "B": .8}), pd.Series({"A": 1.2, "B": -.2})])
def test_accounting_rejects_mismatched_or_invalid_weights(weights):
    with pytest.raises(ValueError):
        backtest_portfolio(prices(), weights)


def test_analysis_benchmark_and_standalone_holdings_share_period():
    definition = portfolio_from_rows(
        [{"Ticker": "A", "Weight (%)": 50}, {"Ticker": "B", "Weight (%)": 50}],
        date(2024, 1, 30), date(2024, 2, 1), benchmark="A",
    )
    frames = {ticker: pd.DataFrame({"Date": prices().index, "Adj Close": prices()[ticker].values}) for ticker in ["A", "B"]}
    data = prepare_portfolio_data(definition, frames, {"A": "USD", "B": "USD"})
    analysis = analyze_portfolio(data)
    assert analysis.growth.iloc[0].tolist() == [10000, 10000]
    assert analysis.metrics.loc["Portfolio", "Total return"] == pytest.approx(.875)
    assert analysis.metrics.loc["Benchmark (A)", "Total return"] == pytest.approx(0)
    assert analysis.holdings.loc["B", "Total return"] == pytest.approx(1)
    assert analysis.drawdowns.loc[pd.Timestamp("2024-02-01"), "Benchmark (A)"] == pytest.approx(-.5)


def test_target_weight_risk_is_distinct_from_drifted_backtest_risk():
    weights = pd.Series({"A": .5, "B": .5})
    held = backtest_portfolio(prices(), weights, "Buy and hold")
    _, target_volatility = risk_decomposition(prices().pct_change().iloc[1:], weights)
    # Target returns are .5, .25; buy-and-hold returns are .5, 0.
    assert target_volatility == pytest.approx(.25 / np.sqrt(2) * np.sqrt(252))
    assert performance_metrics(held.values)["Volatility"] == pytest.approx(.5 / np.sqrt(2) * np.sqrt(252))


def test_page_renders_metrics_charts_and_invalid_submission_hides_old_results():
    def downloaded(ticker, start):
        values = [100., 200., 100.] if ticker == "SPY" else [100., 100., 200.]
        return pd.DataFrame({"Date": prices().index, "Adj Close": values})

    with patch("utils.portfolio_data.load_daily_ticker_data", side_effect=downloaded), \
         patch("utils.portfolio_data.load_ticker_currency", return_value="USD"):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        app.switch_page("pages/4_🖩_Portfolio_Analysis.py").run()
        app.text_input[0].set_value("SPY")
        app.button[0].click().run()
        assert not app.exception
        assert [metric.label for metric in app.metric] == [
            "Total return", "CAGR", "Volatility", "Sharpe ratio", "Max drawdown", "Volatility at target weights",
        ]
        assert len(app.get("plotly_chart")) == 3
        assert "historical" in app.session_state["portfolio_results"]
        # Invalid edits must not leave old analysis displayed beneath the error.
        app.date_input[0].set_value(date(2025, 1, 1))
        app.date_input[1].set_value(date(2024, 1, 1))
        app.button[0].click().run()
        assert not app.exception
        assert app.error and not app.metric and not app.get("plotly_chart")
