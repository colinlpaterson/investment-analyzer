"""Reusable return-matrix and correlation analysis utilities."""

from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd

from utils.ytd_analysis import RETURN_TYPE_COLUMNS, ReturnType, get_return_label


ReturnFrequency = Literal["daily", "weekly", "monthly"]

FREQUENCY_LABELS = {
    "daily": "Daily",
    "weekly": "Weekly",
    "monthly": "Monthly",
}

MIN_OBSERVATIONS = {
    "daily": 30,
    "weekly": 12,
    "monthly": 6,
}


def normalize_tickers(tickers: list[str]) -> list[str]:
    """Normalize, de-duplicate, and preserve the order of ticker symbols."""
    normalized = []
    seen = set()

    for ticker in tickers:
        symbol = ticker.strip().upper()
        if symbol and symbol not in seen:
            normalized.append(symbol)
            seen.add(symbol)

    return normalized


def get_frequency_label(frequency: ReturnFrequency) -> str:
    """Return the display label for a supported observation frequency."""
    try:
        return FREQUENCY_LABELS[frequency]
    except KeyError as exc:
        raise ValueError(f"Unsupported return frequency: {frequency}") from exc


def prepare_return_matrix(
    ticker_data: Dict[str, pd.DataFrame],
    return_type: ReturnType,
    frequency: ReturnFrequency,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Create aligned periodic returns from per-ticker daily price data."""
    if return_type not in RETURN_TYPE_COLUMNS:
        raise ValueError(f"Unsupported return type: {return_type}")
    if frequency not in FREQUENCY_LABELS:
        raise ValueError(f"Unsupported return frequency: {frequency}")
    if len(ticker_data) < 2:
        raise ValueError("At least two tickers with valid data are required")

    value_column = RETURN_TYPE_COLUMNS[return_type]
    price_series = {}
    as_of = pd.Timestamp.now().normalize() if as_of is None else pd.Timestamp(as_of)

    for ticker, data in ticker_data.items():
        required_columns = {"Date", value_column}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise ValueError(
                f"{ticker}: {get_return_label(return_type)} requires "
                f"{', '.join(sorted(missing_columns))}"
            )

        prices = (
            data[["Date", value_column]]
            .dropna()
            .drop_duplicates(subset="Date", keep="last")
            .sort_values("Date")
            .set_index("Date")[value_column]
        )

        if frequency == "weekly":
            prices = prices.resample("W-FRI").last().dropna()
            current_week = as_of.to_period("W-FRI")
            prices = prices[prices.index.to_period("W-FRI") < current_week]
        elif frequency == "monthly":
            prices = prices.resample(pd.offsets.MonthEnd()).last().dropna()
            current_month = as_of.to_period("M")
            prices = prices[prices.index.to_period("M") < current_month]

        price_series[ticker] = prices

    # Align prices first so every return spans the same pair of observation dates.
    aligned_prices = pd.concat(price_series, axis=1, join="inner").dropna(how="any")
    returns = aligned_prices.pct_change(fill_method=None).dropna(how="any")
    returns.index.name = "Date"

    return returns


def calculate_correlation_matrix(
    returns: pd.DataFrame,
    min_observations: int,
) -> pd.DataFrame:
    """Calculate a Pearson correlation matrix after validating the sample."""
    if returns.shape[1] < 2:
        raise ValueError("At least two return series are required")
    if len(returns) < min_observations:
        raise ValueError(
            f"Only {len(returns)} common return observations are available; "
            f"at least {min_observations} are required"
        )

    constant_tickers = returns.columns[returns.nunique(dropna=True) < 2].tolist()
    if constant_tickers:
        raise ValueError(
            "Correlation is undefined for constant return series: "
            + ", ".join(constant_tickers)
        )

    return returns.corr(method="pearson")


def find_correlation_extremes(
    correlation_matrix: pd.DataFrame,
) -> Tuple[Tuple[str, str, float], Tuple[str, str, float]]:
    """Return the highest and lowest unique off-diagonal correlations."""
    upper_triangle = np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1)
    pairs = correlation_matrix.where(upper_triangle).stack()

    if pairs.empty:
        raise ValueError("At least one valid ticker pair is required")

    highest_pair = pairs.idxmax()
    lowest_pair = pairs.idxmin()

    highest = (*highest_pair, float(pairs.loc[highest_pair]))
    lowest = (*lowest_pair, float(pairs.loc[lowest_pair]))
    return highest, lowest
