"""Offline regressions for selectable returns and aligned comparison periods."""
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from utils.data_fetcher import load_ticker_data
from utils.ytd_analysis import calculate_ytd_returns
from utils.multi_ticker_analysis import (
    calculate_comparison_statistics,
    format_comparison_table,
)


@pytest.mark.parametrize('multiindex', [False, True])
def test_download_uses_price_close_and_resamples_months(multiindex):
    daily = pd.DataFrame({
        'Open': [100., 102., 104.], 'High': [103., 106., 109.],
        'Low': [99., 101., 103.], 'Close': [102., 104., 108.],
        'Adj Close': [90., 94., 108.], 'Volume': [10, 20, 30],
    }, index=pd.to_datetime(['2023-12-28', '2023-12-29', '2024-01-31']))
    daily.index.name = 'Date'
    if multiindex:
        daily.columns = pd.MultiIndex.from_product([daily.columns, ['TEST']])
    load_ticker_data.clear()
    try:
        with patch('utils.data_fetcher.yf.download', return_value=daily) as download:
            monthly = load_ticker_data('TEST', '2023-12-01')
        assert download.call_args.kwargs['auto_adjust'] is False
        assert monthly['Date'].tolist() == list(pd.to_datetime(['2023-12-31', '2024-01-31']))
        assert monthly['Close'].tolist() == [104., 108.]
        assert monthly['Adj Close'].tolist() == [94., 108.]
        assert monthly.loc[0, ['Open', 'High', 'Low', 'Volume']].tolist() == [100., 106., 99., 30]
        result = calculate_ytd_returns(monthly, 2024, 2024)
        assert result['YTD'].tolist() == pytest.approx([108 / 104 - 1])
    finally:
        load_ticker_data.clear()


def test_first_requested_year_keeps_prior_december_baseline():
    prices = pd.DataFrame({
        'Date': pd.to_datetime(['2022-12-31', '2023-01-31', '2023-12-31', '2024-01-31']),
        'Close': [100., 110., 120., 132.],
    })
    result = calculate_ytd_returns(prices, 2023, 2024)
    assert result['Year'].tolist() == [2023, 2023, 2024]
    assert result['YTD'].tolist() == pytest.approx([.1, .2, .1])
    pd.testing.assert_index_equal(prices.columns, pd.Index(['Date', 'Close']))


def test_total_return_uses_adjusted_close_and_price_default_is_unchanged():
    prices = pd.DataFrame({
        'Date': pd.to_datetime(['2023-12-31', '2024-01-31']),
        'Close': [100., 100.],
        'Adj Close': [90., 100.],
    })

    price_result = calculate_ytd_returns(prices, 2024, 2024)
    total_result = calculate_ytd_returns(prices, 2024, 2024, return_type='total')

    assert price_result['YTD'].tolist() == pytest.approx([0.])
    assert total_result['YTD'].tolist() == pytest.approx([100 / 90 - 1])


def test_total_return_does_not_fall_back_when_adjusted_close_is_missing():
    prices = pd.DataFrame({
        'Date': pd.to_datetime(['2023-12-31', '2024-01-31']),
        'Close': [100., 110.],
    })

    with pytest.raises(ValueError, match="Total return requires the 'Adj Close' column"):
        calculate_ytd_returns(prices, 2024, 2024, return_type='total')


def test_unknown_return_type_is_rejected():
    prices = pd.DataFrame({
        'Date': pd.to_datetime(['2023-12-31', '2024-01-31']),
        'Close': [100., 110.],
    })

    with pytest.raises(ValueError, match='Unsupported return type'):
        calculate_ytd_returns(prices, 2024, 2024, return_type='income')


def test_missing_december_does_not_invent_a_baseline():
    prices = pd.DataFrame({'Date': pd.to_datetime(['2023-11-30', '2024-01-31']), 'Close': [100., 110.]})
    assert calculate_ytd_returns(prices, 2024, 2024).empty


@pytest.mark.parametrize('month, expected_months', [(1, []), (3, [1, 2])])
def test_current_year_only_contains_completed_months(month, expected_months):
    prices = pd.DataFrame({
        'Date': pd.to_datetime(['2023-12-31', '2024-01-31', '2024-02-29', '2024-03-31']),
        'Close': [100., 110., 120., 130.],
    })
    with patch('utils.ytd_analysis.datetime') as clock:
        clock.now.return_value = datetime(2024, month, 15)
        result = calculate_ytd_returns(prices, 2024, 2024)
    assert result['Month'].tolist() == expected_months


@pytest.mark.parametrize('last_month, january', [(8, False), (12, True)])
@pytest.mark.parametrize('missing_side', ['current', 'prior'])
@pytest.mark.parametrize('missing', [None, pd.Series(dtype=float), pd.Series([.9], index=[6]), pd.Series([np.nan], index=[8])])
def test_missing_endpoint_is_na_not_an_earlier_return(last_month, january, missing_side, missing):
    data = {'current': pd.Series([.2], index=[last_month]), 'prior': pd.Series([.1], index=[last_month])}
    data[missing_side] = missing
    stats = calculate_comparison_statistics({'TEST': data}, 2024, 2023, last_month, january)
    assert pd.isna(stats.loc[0, f'{missing_side.title()} Return'])
    assert pd.isna(stats.loc[0, 'Difference'])
    table = format_comparison_table(stats, 2024, 2023, last_month, january)
    assert table.loc[0, 'Difference'] == 'N/A'
    assert table.iloc[0, 1 if missing_side == 'current' else 2] == 'N/A'


def test_comparison_uses_requested_month_even_with_unsorted_series():
    data = {'TEST': {'current': pd.Series([.2, .9], index=[8, 6]), 'prior': pd.Series([.1, .5], index=[8, 9])}}
    stats = calculate_comparison_statistics(data, 2024, 2023, 8, False)
    assert stats.loc[0, 'Current Return'] == pytest.approx(.2)
    assert stats.loc[0, 'Difference'] == pytest.approx(.1)


def test_historical_page_fetches_baseline_and_displays_selected_year():
    prices = pd.DataFrame({
        'Date': pd.to_datetime(['2003-12-31', '2004-01-31', '2004-12-31']),
        'Close': [100., 110., 120.],
    })
    page = Path(__file__).resolve().parents[1] / 'pages' / '2_📈_Historical_YTD.py'
    with patch('utils.data_fetcher.validate_ticker', return_value=True), patch(
        'utils.data_fetcher.load_ticker_data', return_value=prices
    ) as download:
        app = AppTest.from_file(str(page)).run()
        app.sidebar.button[0].click().run()
    assert not app.exception
    assert download.call_args.args[1] == '2003-12-01'
    assert app.metric[0].value == '20.0%'
    assert any('2004 - 2004' in element.value for element in app.markdown)


def test_historical_page_can_select_total_return():
    prices = pd.DataFrame({
        'Date': pd.to_datetime(['2003-12-31', '2004-01-31', '2004-12-31']),
        'Close': [100., 100., 100.],
        'Adj Close': [80., 90., 100.],
    })
    page = Path(__file__).resolve().parents[1] / 'pages' / '2_📈_Historical_YTD.py'
    with patch('utils.data_fetcher.validate_ticker', return_value=True), patch(
        'utils.data_fetcher.load_ticker_data', return_value=prices
    ):
        app = AppTest.from_file(str(page)).run()
        app.segmented_control[0].set_value('total').run()
        app.sidebar.button[0].click().run()

    assert not app.exception
    assert app.metric[0].value == '25.0%'
    assert 'Total return' in app.get('plotly_chart')[0].proto.spec
