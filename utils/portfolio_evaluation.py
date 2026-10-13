"""Chronological holdout: estimate once, then evaluate without fitting on test returns."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.portfolio_analysis import backtest_portfolio, performance_metrics
from utils.portfolio_data import PortfolioData
from utils.portfolio_optimization import Estimates, OptimizationResult, estimate_parameters, optimize_portfolio


@dataclass
class HeldOutEvaluation:
    estimates: Estimates
    solution: OptimizationResult
    growth: pd.DataFrame
    metrics: pd.DataFrame
    test_observations: int


def evaluate_holdout(data: PortfolioData, caps: pd.Series, training_share: float = .7) -> HeldOutEvaluation:
    if not np.isfinite(training_share) or not .5 <= training_share <= .8:
        raise ValueError("Training share must be between 50% and 80%.")
    returns = data.prices.pct_change(fill_method=None).iloc[1:]
    count = int(len(returns) * training_share)
    if count < 60 or len(returns) - count < 30:
        raise ValueError("The split needs at least 60 training returns and 30 held-out returns. Extend the history or adjust the split.")
    estimates = estimate_parameters(returns.iloc[:count])
    current = pd.Series({h.ticker: h.weight for h in data.definition.holdings})
    solution = optimize_portfolio(estimates, current, caps, data.definition.risk_free_rate, frontier_points=0)
    # Last training close is the starting investment; the next return is the first test return.
    prices = data.prices.iloc[count:]
    growth = pd.DataFrame({name: backtest_portfolio(prices, weights, data.definition.rebalancing).values
                           for name, weights in solution.weights.items()})
    if data.benchmark_prices is not None:
        benchmark = data.benchmark_prices.reindex(prices.index)
        if benchmark.isna().any() or (benchmark <= 0).any():
            raise ValueError("Benchmark must cover the held-out price dates.")
        growth[f"Benchmark ({data.definition.benchmark})"] = 10_000 * benchmark / benchmark.iloc[0]
    metrics = pd.DataFrame({name: performance_metrics(values, data.definition.risk_free_rate, data.definition.minimum_acceptable_return)
                            for name, values in growth.items()}).T
    return HeldOutEvaluation(estimates, solution, growth, metrics, len(returns) - count)
