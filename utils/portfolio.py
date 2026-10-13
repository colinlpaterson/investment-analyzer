"""Portfolio definitions and validation, independent of Streamlit and market data.

Weights and annual rates are stored as fractions; the editor accepts percentages.
"""

from dataclasses import dataclass
from datetime import date
import math
from typing import Iterable


REBALANCING_OPTIONS = ("Monthly", "Quarterly", "Annually", "Buy and hold")


@dataclass(frozen=True)
class Holding:
    ticker: str
    weight: float


@dataclass(frozen=True)
class PortfolioDefinition:
    holdings: tuple[Holding, ...]
    start_date: date
    end_date: date
    rebalancing: str = "Monthly"
    risk_free_rate: float = 0.0
    benchmark: str | None = None

    def __post_init__(self):
        if not self.holdings:
            raise ValueError("Add at least one holding.")
        tickers = [holding.ticker for holding in self.holdings]
        if any(not ticker or ticker != ticker.strip().upper() for ticker in tickers):
            raise ValueError("Tickers must be non-empty, uppercase symbols.")
        if len(set(tickers)) != len(tickers):
            raise ValueError("Duplicate tickers are not allowed; combine their weights.")
        if any(not math.isfinite(h.weight) or not 0 <= h.weight <= 1 for h in self.holdings):
            raise ValueError("Each weight must be finite and between 0% and 100%.")
        if not math.isclose(sum(h.weight for h in self.holdings), 1.0, abs_tol=1e-8, rel_tol=0):
            raise ValueError("Weights must total 100%. Use Normalize and save to rescale them.")
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date.")
        if self.end_date >= date.today():
            raise ValueError("End date must be before today to exclude an incomplete trading day.")
        if self.rebalancing not in REBALANCING_OPTIONS:
            raise ValueError("Choose a supported rebalancing frequency.")
        if not math.isfinite(self.risk_free_rate) or not -1 < self.risk_free_rate <= 1:
            raise ValueError("Annual risk-free rate must be greater than -100% and at most 100%.")
        if self.benchmark is not None and (
            not self.benchmark or self.benchmark != self.benchmark.strip().upper()
        ):
            raise ValueError("Benchmark must be an uppercase ticker or empty.")

    @property
    def tickers(self) -> tuple[str, ...]:
        """Include zero-weight holdings: they may be used by the future optimizer."""
        return tuple(holding.ticker for holding in self.holdings)

    @property
    def required_tickers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.tickers, *([self.benchmark] if self.benchmark else []))))


def portfolio_from_rows(
    rows: Iterable[dict],
    start_date: date,
    end_date: date,
    rebalancing: str = "Monthly",
    risk_free_rate_percent: float = 0.0,
    benchmark: str = "",
    normalize: bool = False,
) -> PortfolioDefinition:
    """Validate editor rows; never silently drop or merge a holding."""
    holdings = []
    for index, row in enumerate(rows, start=1):
        ticker = row.get("Ticker")
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError(f"Row {index}: enter a ticker or delete the row.")
        value = row.get("Weight (%)")
        try:
            weight = float(value) / 100
        except (TypeError, ValueError):
            raise ValueError(f"{ticker.strip().upper()}: enter a numeric weight.") from None
        if isinstance(value, bool) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{ticker.strip().upper()}: weight must be finite and non-negative.")
        holdings.append(Holding(ticker.strip().upper(), weight))
    if normalize:
        total = sum(h.weight for h in holdings)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("At least one positive weight is required to normalize.")
        holdings = [Holding(h.ticker, h.weight / total) for h in holdings]
    return PortfolioDefinition(
        tuple(holdings), start_date, end_date, rebalancing,
        float(risk_free_rate_percent) / 100, benchmark.strip().upper() or None,
    )
