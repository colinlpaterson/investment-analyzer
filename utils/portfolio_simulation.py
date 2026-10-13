"""Correlated circular-block bootstrap with explicit end-of-month cash flows.

One modeled month has 21 trading days and one year has 12 modeled months.
Sampled dates are shared by every asset within a path. Only monthly portfolio
values are retained, keeping memory independent of daily horizon × asset count.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.portfolio import REBALANCING_OPTIONS


MIN_HISTORY = 60
DAYS_PER_MONTH = 21
MAX_WORK = 300_000_000


@dataclass(frozen=True)
class SimulationConfig:
    initial_value: float = 10_000.0
    months: int = 120
    monthly_contribution: float = 0.0
    rebalancing: str = "Monthly"
    paths: int = 5_000
    block_length: int = 21
    seed: int = 42
    target_value: float | None = None
    inflation_rate: float = 0.0

    def __post_init__(self):
        for name, value, lower, upper in [
            ("Horizon in months", self.months, 1, 480),
            ("Simulation count", self.paths, 1, 10_000),
            ("Block length", self.block_length, 1, 252),
            ("Random seed", self.seed, 0, 2**32 - 1),
        ]:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper}.")
        if not np.isfinite(self.initial_value) or self.initial_value <= 0:
            raise ValueError("Starting investment must be finite and positive.")
        if not np.isfinite(self.monthly_contribution) or self.monthly_contribution < 0:
            raise ValueError("Monthly contributions must be finite and non-negative.")
        if self.target_value is not None and (not np.isfinite(self.target_value) or self.target_value <= 0):
            raise ValueError("Target ending value must be finite and positive, or omitted.")
        if not np.isfinite(self.inflation_rate) or not -1 < self.inflation_rate <= 1:
            raise ValueError("Annual inflation must be greater than -100% and at most 100%.")
        if self.rebalancing not in REBALANCING_OPTIONS:
            raise ValueError("Choose a supported rebalancing schedule.")


@dataclass
class SimulationResult:
    config: SimulationConfig
    weights: pd.Series
    monthly_values: np.ndarray
    final_weights: np.ndarray
    quantiles: pd.DataFrame
    contributed: pd.Series
    probability_loss: float
    probability_target: float | None

    @property
    def ending_values(self) -> np.ndarray:
        return self.monthly_values[:, -1]


def sample_block_indices(observations: int, block_length: int, paths: int, rng) -> np.ndarray:
    """Uniform starting dates; contiguous, circular blocks, one row per path."""
    starts = rng.integers(0, observations, size=paths)
    return (starts[:, None] + np.arange(block_length)) % observations


def simulate_portfolio(returns: pd.DataFrame, weights: pd.Series, config: SimulationConfig) -> SimulationResult:
    if len(returns) < MIN_HISTORY:
        raise ValueError(f"Simulation requires at least {MIN_HISTORY} common daily returns.")
    if (not isinstance(returns.index, pd.DatetimeIndex) or returns.index.hasnans
            or returns.index.has_duplicates or not returns.index.is_monotonic_increasing):
        raise ValueError("Historical returns must have unique, increasing dates.")
    if returns.columns.has_duplicates or weights.index.has_duplicates or set(weights.index) != set(returns.columns):
        raise ValueError("Simulation weights must match the historical tickers exactly.")
    history = returns.to_numpy(dtype=float)
    if not np.isfinite(history).all() or (history <= -1).any():
        raise ValueError("Historical returns must be finite, complete, and greater than -100%.")
    target = weights.reindex(returns.columns).to_numpy(dtype=float, copy=True)
    if not np.isfinite(target).all() or (target < 0).any() or not np.isclose(target.sum(), 1, atol=1e-7, rtol=0):
        raise ValueError("Simulation weights must be non-negative and sum to 100%.")
    target /= target.sum()
    if config.block_length > len(history):
        raise ValueError("Block length cannot exceed the available history.")
    days = config.months * DAYS_PER_MONTH
    if days * config.paths * len(target) > MAX_WORK:
        raise ValueError("This run is too large. Reduce the number of simulations or the horizon and try again.")
    rng = np.random.default_rng(config.seed)
    holdings = np.tile(target * config.initial_value, (config.paths, 1))
    wealth = np.empty((config.paths, config.months + 1))
    wealth[:, 0] = config.initial_value
    every_months = {"Monthly": 1, "Quarterly": 3, "Annually": 12}.get(config.rebalancing)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for block_start in range(0, days, config.block_length):
                indices = sample_block_indices(len(history), config.block_length, config.paths, rng)
                for offset in range(min(config.block_length, days - block_start)):
                    holdings *= 1 + history[indices[:, offset]]
                    day = block_start + offset + 1
                    if day % DAYS_PER_MONTH == 0:
                        month = day // DAYS_PER_MONTH
                        # Fresh cash buys target proportions, without selling existing assets.
                        holdings += config.monthly_contribution * target
                        total = holdings.sum(axis=1)
                        if every_months is not None and month % every_months == 0:
                            holdings = total[:, None] * target
                        wealth[:, month] = total
            contributed = config.initial_value + np.arange(config.months + 1) * config.monthly_contribution
    except FloatingPointError as exc:
        raise ValueError("The selected history and horizon produced numerically extreme values. Shorten the horizon or change the history.") from exc
    if not np.isfinite(wealth).all() or not np.isfinite(contributed).all():
        raise ValueError("Simulation values are not finite. Reduce the horizon or investment amounts.")
    quantiles = pd.DataFrame(
        np.percentile(wealth, [10, 25, 50, 75, 90], axis=0).T,
        columns=["10th percentile", "25th percentile", "Median", "75th percentile", "90th percentile"],
        index=pd.Index(range(config.months + 1), name="Month"),
    )
    terminal = wealth[:, -1]
    # Treat floating-point equality as equality, not a loss or a missed target.
    equal_cost = np.isclose(terminal, contributed[-1], rtol=1e-12, atol=1e-8)
    loss = float(np.mean((terminal < contributed[-1]) & ~equal_cost))
    reached = None if config.target_value is None else float(np.mean(
        (terminal >= config.target_value) | np.isclose(terminal, config.target_value, rtol=1e-12, atol=1e-8)
    ))
    final_weights = np.divide(holdings, terminal[:, None], out=np.zeros_like(holdings), where=terminal[:, None] > 0)
    return SimulationResult(
        config, pd.Series(target, index=returns.columns), wealth, final_weights, quantiles,
        pd.Series(contributed, index=quantiles.index, name="Total contributed"), loss, reached,
    )
