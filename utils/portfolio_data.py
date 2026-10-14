"""Validated total-return inputs for portfolio analysis, optimization and simulation."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from utils.data_fetcher import load_daily_ticker_data, load_ticker_currency
from utils.portfolio import PortfolioDefinition


@dataclass
class PortfolioData:
    definition: PortfolioDefinition
    prices: pd.DataFrame
    returns: pd.DataFrame
    benchmark_prices: pd.Series | None
    coverage: pd.DataFrame
    currency: str
    notices: tuple[str, ...]


def prepare_portfolio_data(
    definition: PortfolioDefinition,
    ticker_data: Mapping[str, pd.DataFrame | None],
    currencies: Mapping[str, str | None],
) -> PortfolioData:
    """Align prices before returns, rejecting gaps rather than inventing daily returns.

The requested end date is inclusive. The benchmark is validated separately and
does not shorten the portfolio's history. Zero-weight holdings remain in the
universe so later optimization uses the same dates and securities.
"""
    series = {}
    errors = []
    coverage = []
    notices = []
    for ticker in definition.required_tickers:
        frame = ticker_data.get(ticker)
        if frame is None or frame.empty:
            errors.append(f"{ticker}: no price data available.")
            continue
        if not {"Date", "Adj Close"}.issubset(frame.columns):
            errors.append(f"{ticker}: adjusted-close data is unavailable; price returns cannot substitute.")
            continue
        try:
            dates = pd.DatetimeIndex(pd.to_datetime(frame["Date"]))
            if dates.tz is not None:
                dates = dates.tz_localize(None)
            dates = dates.normalize()
            prices = pd.Series(pd.to_numeric(frame["Adj Close"], errors="coerce").to_numpy(), index=dates)
        except (TypeError, ValueError):
            errors.append(f"{ticker}: invalid date or price data.")
            continue
        if dates.isna().any() or dates.duplicated().any():
            errors.append(f"{ticker}: missing or duplicate price dates.")
            continue
        prices = prices.sort_index().loc[str(definition.start_date):str(definition.end_date)]
        if prices.empty:
            errors.append(f"{ticker}: no observations in the requested period.")
            continue
        # Yahoo can publish the latest trading row before its closing prices.
        # End the history at the last reported price, without filling a gap or
        # dropping an interior observation (which would change daily returns).
        last_reported = prices.last_valid_index()
        if last_reported is not None and last_reported < prices.index[-1]:
            pending = prices.loc[prices.index > last_reported]
            notices.append(
                f"{ticker}: adjusted closing prices are unavailable for "
                f"{pending.index[0]:%m/%d/%Y}–{pending.index[-1]:%m/%d/%Y}. "
                f"History ends at the last reported close on {last_reported:%m/%d/%Y}; "
                "no prices were filled. Try again later for updated data."
            )
            prices = prices.loc[:last_reported]
        if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any():
            invalid = prices.index[~np.isfinite(prices.to_numpy()) | (prices.to_numpy() <= 0)]
            dates_text = ", ".join(day.strftime("%m/%d/%Y") for day in invalid[:3])
            errors.append(f"{ticker}: missing, non-finite, or non-positive adjusted prices on {dates_text}. "
                          "Choose a period excluding these dates or retry when source data is available.")
            continue
        currency = currencies.get(ticker)
        if not isinstance(currency, str) or not currency.strip():
            errors.append(f"{ticker}: quote currency could not be verified.")
            continue
        series[ticker] = prices
        coverage.append({
            "Ticker": ticker,
            "Role": "Holding" if ticker in definition.tickers else "Benchmark",
            "Currency": currency,
            "First available": prices.index[0].date(),
            "Last available": prices.index[-1].date(),
            "Price observations": len(prices),
        })
    if errors:
        raise ValueError("\n".join(errors))
    currency_set = {currencies[ticker] for ticker in definition.required_tickers}
    if len(currency_set) != 1:
        raise ValueError("All holdings and the benchmark must share a quote currency; currency conversion is not supported.")

    common_start = max(series[ticker].index[0] for ticker in definition.tickers)
    common_end = min(series[ticker].index[-1] for ticker in definition.tickers)
    if common_start >= common_end:
        raise ValueError("Holdings need at least two common price dates to calculate returns.")
    prices = pd.concat({ticker: series[ticker].loc[common_start:common_end] for ticker in definition.tickers}, axis=1)
    if prices.isna().any().any():
        affected = prices.columns[prices.isna().any()].tolist()
        raise ValueError(
            "Missing dates within the common period for " + ", ".join(affected)
            + ". Choose a different period or holdings with matching trading calendars. Prices are not forward-filled."
        )
    prices.index.name = "Date"
    returns = prices.pct_change(fill_method=None).iloc[1:]
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Price data produced non-finite returns.")
    if common_start.date() > definition.start_date or common_end.date() < definition.end_date:
        notices.append(
            f"Requested {definition.start_date:%m/%d/%Y} to {definition.end_date:%m/%d/%Y}; usable common history is "
            f"{common_start:%m/%d/%Y} to {common_end:%m/%d/%Y}. Availability and non-trading days can shorten the period."
        )
    if len(returns) < 252:
        notices.append("Less than one year of daily return observations is available. Later analytics may require more history.")
    benchmark_prices = None
    if definition.benchmark:
        benchmark_prices = series[definition.benchmark].reindex(prices.index).rename(definition.benchmark)
        if benchmark_prices.isna().any():
            raise ValueError(
                f"{definition.benchmark}: benchmark does not cover every portfolio date. "
                "Choose another benchmark, remove it, or shorten the requested period."
            )
    return PortfolioData(
        definition, prices, returns, benchmark_prices,
        pd.DataFrame(coverage), next(iter(currency_set)), tuple(notices),
    )


def load_portfolio_data(definition: PortfolioDefinition) -> PortfolioData:
    """Use cached source downloads; keep each user's allocation out of global caches."""
    frames = {ticker: load_daily_ticker_data(ticker, str(definition.start_date)) for ticker in definition.required_tickers}
    currencies = {ticker: load_ticker_currency(ticker) for ticker in definition.required_tickers}
    return prepare_portfolio_data(definition, frames, currencies)
