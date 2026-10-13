"""Historical portfolio accounting and statistics, independent of Streamlit.

All returns, weights, and risk contributions use fractional units. Accounting
uses adjusted-price ratios as reinvested total returns and allows fractional
holdings. There are no external flows, trading costs, taxes, or FX conversions.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.portfolio import REBALANCING_OPTIONS
from utils.portfolio_data import PortfolioData


TRADING_DAYS = 252
ZERO_TOLERANCE = 1e-12
METRIC_COLUMNS = ["Total return", "CAGR", "Volatility", "Sharpe ratio", "Max drawdown"]


@dataclass
class Backtest:
    values: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    rebalance_dates: pd.DatetimeIndex


@dataclass
class HistoricalAnalysis:
    backtest: Backtest
    growth: pd.DataFrame
    drawdowns: pd.DataFrame
    metrics: pd.DataFrame
    holdings: pd.DataFrame
    target_volatility: float


def _validate_prices(prices: pd.DataFrame) -> None:
    if prices.empty or len(prices) < 2:
        raise ValueError("At least two price observations are required.")
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError("Prices require valid dates.")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("Price dates must be unique and increasing.")
    if prices.columns.has_duplicates:
        raise ValueError("Price tickers must be unique.")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Prices must be finite, positive, and complete.")


def _aligned_weights(weights: pd.Series, columns: pd.Index) -> np.ndarray:
    if weights.index.has_duplicates or set(weights.index) != set(columns):
        raise ValueError("Weights must match the price tickers exactly.")
    aligned = weights.reindex(columns).to_numpy(dtype=float)
    if not np.isfinite(aligned).all() or (aligned < 0).any() or not np.isclose(aligned.sum(), 1, atol=1e-8, rtol=0):
        raise ValueError("Weights must be finite, non-negative, and total 100%.")
    # Remove only rounding noise accepted by the definition's sum tolerance.
    return aligned / aligned.sum()


def backtest_portfolio(
    prices: pd.DataFrame,
    weights: pd.Series,
    rebalancing: str = "Monthly",
    initial_value: float = 10_000.0,
) -> Backtest:
    """Apply returns to prior-close holdings before any period-end rebalance.

Rebalance at a close only if the following observation enters a new calendar
period. This includes weekend/holiday period ends and never treats the sample's
last (possibly partial-period) observation as a scheduled rebalance. Weights
recorded on a rebalance date are post-trade weights; total wealth is unchanged.
"""
    _validate_prices(prices)
    target = _aligned_weights(weights, prices.columns)
    if rebalancing not in REBALANCING_OPTIONS:
        raise ValueError("Choose a supported rebalancing frequency.")
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("Initial value must be finite and positive.")
    frequency = {"Monthly": "M", "Quarterly": "Q", "Annually": "Y"}.get(rebalancing)
    rebalance = np.zeros(len(prices), dtype=bool)
    if frequency:
        periods = prices.index.to_period(frequency)
        rebalance[:-1] = periods[:-1] != periods[1:]
    # Initial allocation already has the target weights.
    rebalance[0] = False
    relatives = prices.to_numpy()[1:] / prices.to_numpy()[:-1]
    holdings = target * initial_value
    values = np.empty(len(prices))
    weight_history = np.empty(prices.shape)
    values[0], weight_history[0] = initial_value, target
    for position, relative in enumerate(relatives, start=1):
        holdings = holdings * relative
        total = holdings.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Portfolio values became non-finite or non-positive.")
        values[position] = total
        if rebalance[position]:
            holdings = target * total
        weight_history[position] = holdings / total
    wealth = pd.Series(values, index=prices.index, name="Portfolio")
    return Backtest(
        wealth, wealth.pct_change(fill_method=None).iloc[1:],
        pd.DataFrame(weight_history, index=prices.index, columns=prices.columns),
        prices.index[rebalance],
    )


def performance_metrics(values: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    """CAGR uses elapsed calendar years; volatility and Sharpe use 252 daily periods.

Sharpe uses arithmetic mean excess daily return, not CAGR. Annual risk-free
input is an effective rate, converted to a constant equivalent daily rate.
Undefined sample statistics are NaN, never infinity or a misleading zero.
"""
    _validate_prices(values.to_frame())
    if not np.isfinite(risk_free_rate) or not -1 < risk_free_rate <= 1:
        raise ValueError("Risk-free rate must be greater than -100% and at most 100%.")
    returns = values.pct_change(fill_method=None).iloc[1:]
    years = (values.index[-1] - values.index[0]).total_seconds() / (365.25 * 24 * 60 * 60)
    total_return = values.iloc[-1] / values.iloc[0] - 1
    with np.errstate(over="ignore"):
        cagr = np.expm1(np.log1p(total_return) / years)
    if not np.isfinite(cagr):
        cagr = np.nan
    daily_std = returns.std(ddof=1) if len(returns) >= 2 else np.nan
    volatility = daily_std * np.sqrt(TRADING_DAYS)
    daily_rf = np.expm1(np.log1p(risk_free_rate) / TRADING_DAYS)
    sharpe = (
        (returns.mean() - daily_rf) / daily_std * np.sqrt(TRADING_DAYS)
        if np.isfinite(daily_std) and daily_std > ZERO_TOLERANCE else np.nan
    )
    return dict(zip(METRIC_COLUMNS, [
        float(total_return), float(cagr), float(volatility), float(sharpe),
        float((values / values.cummax() - 1).min()),
    ]))


def risk_decomposition(returns: pd.DataFrame, weights: pd.Series) -> tuple[pd.DataFrame, float]:
    """Euler volatility contributions using sample covariance at target weights.

Component i = w_i (Sigma w)_i / sigma; risk share = component / sigma.
Components sum to target-weight volatility, not drifted backtest volatility.
Negative shares and shares greater than 100% are preserved for hedging assets.
"""
    target = _aligned_weights(weights, returns.columns)
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns must be finite and complete.")
    result = pd.DataFrame(index=returns.columns)
    result["Target weight"] = target
    result["Volatility contribution"] = np.nan
    result["Risk contribution"] = np.nan
    if len(returns) < 2:
        return result, np.nan
    covariance = returns.cov(ddof=1).to_numpy() * TRADING_DAYS
    covariance_with_portfolio = covariance @ target
    variance = float(target @ covariance_with_portfolio)
    # A near-perfect hedge can leave a positive cancellation residual in w'Sw.
    if abs(variance) <= 16 * np.finfo(float).eps * np.max(np.abs(covariance)):
        variance = 0.0
    volatility = float(np.sqrt(max(variance, 0.0)))
    if volatility > ZERO_TOLERANCE:
        components = target * covariance_with_portfolio / volatility
        result["Volatility contribution"] = components
        result["Risk contribution"] = components / volatility
    return result, volatility


def analyze_portfolio(data: PortfolioData) -> HistoricalAnalysis:
    """Build comparable portfolio, benchmark, and standalone holding statistics."""
    definition = data.definition
    weights = pd.Series({h.ticker: h.weight for h in definition.holdings})
    backtest = backtest_portfolio(data.prices, weights, definition.rebalancing)
    growth = backtest.values.to_frame()
    if data.benchmark_prices is not None:
        benchmark = data.benchmark_prices
        if not benchmark.index.equals(data.prices.index):
            raise ValueError("Benchmark dates must match the portfolio exactly.")
        _validate_prices(benchmark.to_frame())
        growth[f"Benchmark ({definition.benchmark})"] = benchmark / benchmark.iloc[0] * backtest.values.iloc[0]
    metrics = pd.DataFrame({
        name: performance_metrics(growth[name], definition.risk_free_rate) for name in growth
    }).T
    # Derive returns from these prices rather than trusting a separately mutable matrix.
    returns = data.prices.pct_change(fill_method=None).iloc[1:]
    risk, target_volatility = risk_decomposition(returns, weights)
    holdings = pd.DataFrame({
        ticker: performance_metrics(data.prices[ticker], definition.risk_free_rate)
        for ticker in data.prices
    }).T
    holdings = risk.join(holdings)
    holdings.insert(1, "Ending weight", backtest.weights.iloc[-1])
    holdings.index.name = "Ticker"
    return HistoricalAnalysis(
        backtest, growth, growth / growth.cummax() - 1,
        metrics, holdings, target_volatility,
    )
