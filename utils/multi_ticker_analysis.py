"""
Multi-ticker comparison utilities for YTD analysis.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional


def determine_comparison_years() -> Tuple[int, int, int, bool]:
    """
    Returns (current_year, prior_year, last_completed_month, is_january)

    - January: Compare full prior year vs year before that
    - Other months: Compare YTD current year vs YTD prior year through last completed month
    """
    today = datetime.now()
    current_year = today.year
    current_month = today.month

    # January: last completed year is prior calendar year
    if current_month == 1:
        prior_year = current_year - 1
        return current_year, prior_year, 12, True

    # Other months: compare YTD through last completed month
    last_completed_month = current_month - 1
    prior_year = current_year - 1
    return current_year, prior_year, last_completed_month, False


def get_ytd_for_comparison(
    df: pd.DataFrame,
    year: int,
    last_month: int
) -> Optional[pd.Series]:
    """
    Extract YTD series for a specific year, truncated to last_month.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Year, Month, YTD columns
    year : int
        Year to extract
    last_month : int
        Last month to include (1-12)
        
    Returns
    -------
    pd.Series or None
        YTD series indexed by month, or None if no data
    """
    year_data = df[df["Year"] == year]
    
    if len(year_data) == 0:
        return None
    
    # Create series and truncate to last_month
    series = year_data.set_index("Month")["YTD"]
    
    # Only keep data up to last_month
    series = series[series.index <= last_month]
    
    return series if len(series) > 0 else None


def prepare_comparison_data(
    all_ytd_data: Dict[str, pd.DataFrame],
    current_year: int,
    prior_year: int,
    last_month: int
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Prepare data structure for multi-ticker comparison chart.
    
    Parameters
    ----------
    all_ytd_data : dict
        Dictionary mapping ticker to YTD DataFrame
    current_year : int
        Current/most recent year
    prior_year : int
        Prior year for comparison
    last_month : int
        Last month to include in comparison
        
    Returns
    -------
    dict
        Nested dict: {ticker: {'current': series, 'prior': series}}
    """
    comparison_data = {}
    
    for ticker, df in all_ytd_data.items():
        current_series = get_ytd_for_comparison(df, current_year, last_month)
        prior_series = get_ytd_for_comparison(df, prior_year, last_month)
        
        # Only include ticker if it has data for at least one period
        if current_series is not None or prior_series is not None:
            comparison_data[ticker] = {
                'current': current_series,
                'prior': prior_series
            }
    
    return comparison_data


def calculate_comparison_statistics(
    comparison_data: Dict[str, Dict[str, pd.Series]],
    compare_year: int,
    baseline_year: int,
    last_month: int,
    is_january: bool
) -> pd.DataFrame:
    """
    Calculate summary statistics for comparison table.
    
    Parameters
    ----------
    comparison_data : dict
        Nested dict with current and prior series for each ticker
    compare_year : int
        Current/display year
    baseline_year : int
        Prior year
    last_month : int
        Last month included
    is_january : bool
        Whether we're in January (full year comparison)
        
    Returns
    -------
    pd.DataFrame
        Summary statistics table
    """
    rows = []
    
    for ticker, data in comparison_data.items():
        current_series = data.get('current')
        prior_series = data.get('prior')
        
        # Compare the same endpoint; an earlier observation is not a substitute.
        current_return = current_series.get(last_month, np.nan) if current_series is not None else np.nan
        prior_return = prior_series.get(last_month, np.nan) if prior_series is not None else np.nan
        
        # Calculate difference
        if not np.isnan(current_return) and not np.isnan(prior_return):
            difference = current_return - prior_return
        else:
            difference = np.nan
        
        rows.append({
            'Ticker': ticker,
            'Current Return': current_return,
            'Prior Return': prior_return,
            'Difference': difference
        })
    
    df = pd.DataFrame(rows)
    
    # Sort by current return (descending)
    df = df.sort_values('Current Return', ascending=False, na_position='last')
    
    return df


def format_comparison_table(
    stats_df: pd.DataFrame,
    compare_year: int,
    baseline_year: int,
    last_month: int,
    is_january: bool
) -> pd.DataFrame:
    """
    Format the comparison statistics table for display.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        Raw statistics DataFrame
    compare_year : int
        Current year
    baseline_year : int
        Prior year
    last_month : int
        Last month
    is_january : bool
        Whether displaying January data
        
    Returns
    -------
    pd.DataFrame
        Formatted table with proper column names
    """
    from datetime import datetime
    
    # Determine period description
    if is_january:
        current_period = f"Full {compare_year}"
        prior_period = f"Full {baseline_year}"
    else:
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_str = month_names[last_month - 1]
        current_period = f"YTD {compare_year}\n(through {month_str})"
        prior_period = f"YTD {baseline_year}\n(through {month_str})"
    
    # Format the dataframe
    display_df = pd.DataFrame({
        'Ticker': stats_df['Ticker'],
        current_period: stats_df['Current Return'].apply(
            lambda x: f"{x*100:.1f}%" if not np.isnan(x) else "N/A"
        ),
        prior_period: stats_df['Prior Return'].apply(
            lambda x: f"{x*100:.1f}%" if not np.isnan(x) else "N/A"
        ),
        'Difference': stats_df['Difference'].apply(
            lambda x: f"{x*100:+.1f}%" if not np.isnan(x) else "N/A"
        )
    })
    
    return display_df
