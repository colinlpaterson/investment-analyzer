"""Offline regressions for static correlation analysis."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from utils.correlation_analysis import (
    calculate_correlation_matrix,
    find_correlation_extremes,
    normalize_tickers,
    prepare_return_matrix,
)
from utils.data_fetcher import load_daily_ticker_data


def price_frame(dates, prices, adjusted=None):
    adjusted = prices if adjusted is None else adjusted
    return pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Close": prices,
        "Adj Close": adjusted,
    })


def prices_from_returns(returns, starting_price=100.0):
    return starting_price * np.cumprod(np.r_[1.0, 1.0 + np.asarray(returns)])


def test_normalize_tickers_preserves_order_and_removes_duplicates():
    assert normalize_tickers([" spy ", "QQQ", "spy", "", " tlt"]) == [
        "SPY",
        "QQQ",
        "TLT",
    ]


def test_return_matrix_uses_returns_and_correlation_excludes_diagonal():
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    ticker_data = {
        "A": price_frame(dates, prices_from_returns([.10, -.05, .20, .03])),
        "B": price_frame(dates, prices_from_returns([.20, -.10, .40, .06])),
        "C": price_frame(dates, prices_from_returns([-.10, .05, -.20, -.03])),
    }

    returns = prepare_return_matrix(ticker_data, "price", "daily")
    correlation = calculate_correlation_matrix(returns, min_observations=4)
    highest, lowest = find_correlation_extremes(correlation)

    assert correlation.loc["A", "B"] == pytest.approx(1.0)
    assert correlation.loc["A", "C"] == pytest.approx(-1.0)
    assert highest[:2] == ("A", "B")
    assert highest[2] == pytest.approx(1.0)
    assert lowest[2] == pytest.approx(-1.0)


def test_return_matrix_aligns_prices_before_calculating_returns():
    ticker_data = {
        "A": price_frame(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            [100.0, 200.0, 300.0],
        ),
        "B": price_frame(
            ["2024-01-01", "2024-01-03"],
            [50.0, 100.0],
        ),
    }

    returns = prepare_return_matrix(ticker_data, "price", "daily")

    assert returns.index.tolist() == [pd.Timestamp("2024-01-03")]
    assert returns.loc[pd.Timestamp("2024-01-03"), "A"] == pytest.approx(2.0)
    assert returns.loc[pd.Timestamp("2024-01-03"), "B"] == pytest.approx(1.0)


def test_return_matrix_honors_total_return_selection():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    ticker_data = {
        "A": price_frame(dates, [100.0, 100.0, 100.0], [100.0, 110.0, 99.0]),
        "B": price_frame(dates, [50.0, 50.0, 50.0], [50.0, 45.0, 54.0]),
    }

    returns = prepare_return_matrix(ticker_data, "total", "daily")

    assert returns["A"].tolist() == pytest.approx([.10, -.10])
    assert returns["B"].tolist() == pytest.approx([-.10, .20])


def test_monthly_returns_exclude_the_incomplete_current_month():
    dates = pd.to_datetime(["2023-12-29", "2024-01-31", "2024-02-12"])
    ticker_data = {
        "A": price_frame(dates, [100.0, 110.0, 121.0]),
        "B": price_frame(dates, [100.0, 90.0, 99.0]),
    }

    returns = prepare_return_matrix(
        ticker_data,
        "price",
        "monthly",
        as_of=pd.Timestamp("2024-02-15"),
    )

    assert returns.index.tolist() == [pd.Timestamp("2024-01-31")]
    assert returns.loc[pd.Timestamp("2024-01-31"), "A"] == pytest.approx(.10)


def test_correlation_rejects_small_or_constant_samples():
    small = pd.DataFrame({"A": [.1, .2], "B": [.2, .1]})
    with pytest.raises(ValueError, match="at least 3"):
        calculate_correlation_matrix(small, min_observations=3)

    constant = pd.DataFrame({"A": [.1, .1, .1], "B": [.1, .2, .3]})
    with pytest.raises(ValueError, match="constant return series: A"):
        calculate_correlation_matrix(constant, min_observations=3)


@pytest.mark.parametrize("multiindex", [False, True])
def test_daily_loader_preserves_close_and_adjusted_close(multiindex):
    daily = pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "Adj Close": [99.0, 101.0],
            "Volume": [10, 20],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    daily.index.name = "Date"
    if multiindex:
        daily.columns = pd.MultiIndex.from_product([daily.columns, ["TEST"]])

    load_daily_ticker_data.clear()
    try:
        with patch("utils.data_fetcher.yf.download", return_value=daily) as download:
            result = load_daily_ticker_data("TEST", "2024-01-01")

        assert download.call_args.kwargs["auto_adjust"] is False
        assert result.columns.tolist() == ["Date", "Close", "Adj Close"]
        assert result["Adj Close"].tolist() == [99.0, 101.0]
    finally:
        load_daily_ticker_data.clear()


def test_correlation_page_renders_total_return_heatmap():
    dates = pd.date_range("2026-01-02", periods=80, freq="B")
    base_returns = np.resize(np.array([.01, -.006, .004, .012, -.003]), len(dates) - 1)
    factors = {"SPY": 1.0, "QQQ": 1.4, "TLT": -0.5, "GLD": 0.3}

    def mocked_download(ticker, _start_date):
        adjusted = prices_from_returns(base_returns * factors[ticker])
        return price_frame(dates, adjusted, adjusted)

    page = Path(__file__).resolve().parents[1] / "pages" / "3_🔗_Correlation_Analysis.py"
    with patch(
        "utils.data_fetcher.load_daily_ticker_data",
        side_effect=mocked_download,
    ):
        app = AppTest.from_file(str(page)).run()
        app.button[0].click().run()

    assert not app.exception
    assert app.success
    assert [metric.label for metric in app.metric] == [
        "Highest correlation",
        "Lowest correlation",
        "Common observations",
    ]
    assert "total return correlation" in app.get("plotly_chart")[0].proto.spec
