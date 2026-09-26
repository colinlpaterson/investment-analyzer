"""
YTD analysis utilities for calculating year-to-date returns.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple


def calculate_ytd_returns(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Calculate year-to-date returns based on prior December baseline.
    
    YTD Return = (Current Month Close / Prior December Close) - 1
    
    This measures the cumulative price return from the prior year's December close
    to each month-end within the calendar year. This does NOT include dividends
    (total return), only price appreciation.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Date and Close columns
    start_year : int
        Starting year for analysis
    end_year : int
        Ending year for analysis
        
    Returns
    -------
    pd.DataFrame
        DataFrame with Year, Month, and YTD columns
    """
    # Extract year and month
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    
    # Preserve the preceding December before filtering the requested years.
    dec_baseline = df[df["Month"] == 12].copy()
    dec_baseline["Year"] = dec_baseline["Year"] + 1
    dec_baseline = dec_baseline.set_index("Year")["Close"]

    # Filter to specified date range AND only completed months
    df = df[df["Year"].between(start_year, end_year)].copy()
    
    # For current year, only include months that have been completed
    today = datetime.now()
    if end_year == today.year:
        # Only include months before the current month
        df = df[~((df["Year"] == end_year) & (df["Month"] >= today.month))].copy()
    
    # Calculate YTD return
    df["YTD"] = df["Close"] / df["Year"].map(dec_baseline) - 1
    
    # Remove rows where we couldn't calculate YTD
    df = df.dropna(subset=["YTD"])
    
    return df


def prepare_ytd_series(df: pd.DataFrame, current_year: int) -> Tuple[Dict[int, pd.Series], int, int]:
    """
    Prepare YTD series by year and determine which year to highlight.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Year, Month, and YTD columns
    current_year : int
        Current calendar year
        
    Returns
    -------
    tuple
        Dictionary of YTD series by year, year to highlight, and last month for highlighted year
    """
    years = sorted(df["Year"].unique())
    
    # Create dictionary of YTD series indexed by month for each year
    ytd_by_year = {
        year: df.loc[df["Year"] == year, ["Month", "YTD"]]
                .set_index("Month")["YTD"]
        for year in years
    }
    
    # Determine which year to highlight
    current_year_data = df[df["Year"] == current_year]
    
    if len(current_year_data) > 0:
        highlight_year = current_year
        last_month_highlight = current_year_data["Month"].max()
    else:
        highlight_year = max(years)
        last_month_highlight = df.loc[df["Year"] == highlight_year, "Month"].max()
    
    return ytd_by_year, highlight_year, last_month_highlight


def calculate_historical_average(ytd_by_year: Dict[int, pd.Series], 
                                  exclude_year: int = None) -> Tuple[float, float, int]:
    """
    Calculate average and standard deviation of full-year (December) YTD return.
    
    Parameters
    ----------
    ytd_by_year : dict
        Dictionary of YTD series by year
    exclude_year : int, optional
        Year to exclude from average (typically current year if incomplete)
        
    Returns
    -------
    tuple
        Average December YTD return, standard deviation, and number of years included
    """
    completed_years = [y for y in ytd_by_year.keys() if y != exclude_year]
    dec_returns = [ytd_by_year[y].get(12, np.nan) for y in completed_years]
    
    # Filter out NaN values
    dec_returns_clean = [r for r in dec_returns if not np.isnan(r)]
    
    avg = np.mean(dec_returns_clean) if dec_returns_clean else np.nan
    std = np.std(dec_returns_clean, ddof=1) if len(dec_returns_clean) > 1 else np.nan
    n_years = len(dec_returns_clean)
    
    return avg, std, n_years


def get_summary_statistics(ytd_by_year: Dict[int, pd.Series], 
                           highlight_year: int,
                           last_month_highlight: int) -> dict:
    """
    Calculate comprehensive summary statistics for display.
    
    Parameters
    ----------
    ytd_by_year : dict
        Dictionary of YTD series by year
    highlight_year : int
        Year being highlighted
    last_month_highlight : int
        Last available month for highlighted year
        
    Returns
    -------
    dict
        Dictionary of summary statistics
    """
    # Determine actual start year
    actual_start_year = min(ytd_by_year.keys())
    
    # Calculate historical average (exclude current year if incomplete)
    exclude_year = highlight_year if last_month_highlight < 12 else None
    avg_full_year_ytd, std_full_year_ytd, n_years = calculate_historical_average(
        ytd_by_year, exclude_year
    )
    
    # Find best and worst years
    year_end_returns = {}
    for year in ytd_by_year.keys():
        series = ytd_by_year[year]
        if 12 in series.index and not np.isnan(series.loc[12]):
            year_end_returns[year] = series.loc[12]
    
    best_year = None
    worst_year = None
    best_return = None
    worst_return = None
    
    if len(year_end_returns) >= 1:
        best_year = max(year_end_returns.keys(), key=lambda y: year_end_returns[y])
        worst_year = min(year_end_returns.keys(), key=lambda y: year_end_returns[y])
        best_return = year_end_returns[best_year]
        worst_return = year_end_returns[worst_year]
    
    # Current/highlighted year YTD
    current_ytd = None
    if highlight_year in ytd_by_year and len(ytd_by_year[highlight_year]) > 0:
        current_ytd = ytd_by_year[highlight_year].iloc[-1]
    
    return {
        'actual_start_year': actual_start_year,
        'highlight_year': highlight_year,
        'last_month_highlight': last_month_highlight,
        'avg_full_year_ytd': avg_full_year_ytd,
        'std_full_year_ytd': std_full_year_ytd,
        'n_years': n_years,
        'best_year': best_year,
        'best_return': best_return,
        'worst_year': worst_year,
        'worst_return': worst_return,
        'current_ytd': current_ytd
    }
