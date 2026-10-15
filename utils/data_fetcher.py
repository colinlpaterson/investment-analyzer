"""
Data fetching utilities for Yahoo Finance price data and ticker validation.
"""
import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Optional


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_ticker_data(ticker: str, start_date: str) -> Optional[pd.DataFrame]:
    """
    Download daily ticker data from Yahoo Finance and resample to monthly.
    Uses daily data to get accurate month-end prices (last trading day).
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in 'YYYY-MM-DD' format
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with Date, OHLCV data (monthly), or None if error
    """
    try:
        # Download daily data instead of monthly for accuracy
        data = yf.download(
            ticker,
            start=start_date,
            # Yahoo's Close is split-adjusted; Adj Close also adjusts dividends.
            auto_adjust=False,
            progress=False
        )
        
        # Handle potential MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Check if data is empty
        if len(data) == 0:
            return None
        
        # Resample to monthly - take last trading day of each month
        monthly_aggregations = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        if 'Adj Close' in data.columns:
            monthly_aggregations['Adj Close'] = 'last'

        monthly_data = data.resample(pd.offsets.MonthEnd()).agg(monthly_aggregations)
        
        # Reset index to get Date as column
        monthly_data = monthly_data.reset_index()
        
        return monthly_data
    
    except Exception as e:
        st.error(f"Failed to download data for {ticker}: {e}")
        return None


@st.cache_data(ttl=3600, max_entries=100)
def load_daily_ticker_data(ticker: str, start_date: str) -> Optional[pd.DataFrame]:
    """Download daily close and adjusted-close data for correlation analysis."""
    try:
        data = yf.download(
            ticker,
            start=start_date,
            auto_adjust=False,
            progress=False,
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        if data.empty or "Close" not in data.columns:
            return None

        columns = ["Close"]
        if "Adj Close" in data.columns:
            columns.append("Adj Close")

        return data[columns].reset_index()
    except Exception:
        return None


@st.cache_data(ttl=3600, max_entries=100)
def load_ticker_currency(ticker: str) -> Optional[str]:
    """Read the quote currency without guessing from a ticker's exchange suffix."""
    try:
        currency = yf.Ticker(ticker).get_history_metadata().get("currency")
        return currency if isinstance(currency, str) and currency.strip() else None
    except Exception:
        return None


def validate_ticker(ticker: str) -> bool:
    """
    Validate if a ticker symbol exists and has data.
    Uses a simple data download test instead of .info API call.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
        
    Returns
    -------
    bool
        True if ticker is valid, False otherwise
    """
    if not ticker or len(ticker.strip()) == 0:
        return False
    
    try:
        # Simple validation: try to download 5 days of recent data
        # This is more reliable than using .info which often times out
        test_data = yf.download(ticker, period="5d", progress=False)
        
        # Handle MultiIndex if present
        if isinstance(test_data.columns, pd.MultiIndex):
            test_data.columns = test_data.columns.droplevel(1)
        
        # Check if we got any data back
        return len(test_data) > 0
    except:
        return False


def get_ticker_info(ticker: str) -> dict:
    """
    Get basic information about a ticker.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
        
    Returns
    -------
    dict
        Dictionary with ticker information
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        return {
            'name': info.get('shortName', ticker),
            'type': info.get('quoteType', 'Unknown'),
            'exchange': info.get('exchange', 'Unknown')
        }
    except:
        return {
            'name': ticker,
            'type': 'Unknown',
            'exchange': 'Unknown'
        }


# Common ticker definitions for dropdowns
POPULAR_TICKERS = {
    "Equity ETFs": {
        "SPY": "SPDR S&P 500 ETF",
        "QQQ": "Invesco QQQ (Nasdaq-100)",
        "DIA": "SPDR Dow Jones Industrial Average ETF",
        "IWM": "iShares Russell 2000 ETF",
        "VTI": "Vanguard Total Stock Market ETF",
        "VOO": "Vanguard S&P 500 ETF"
    },
    "Bond ETFs": {
        "TLT": "iShares 20+ Year Treasury Bond ETF",
        "AGG": "iShares Core U.S. Aggregate Bond ETF",
        "BND": "Vanguard Total Bond Market ETF",
        "LQD": "iShares iBoxx Investment Grade Corporate Bond ETF"
    },
    "Commodity ETFs": {
        "GLD": "SPDR Gold Shares",
        "SLV": "iShares Silver Trust",
        "USO": "United States Oil Fund",
        "DBA": "Invesco DB Agriculture Fund"
    },
    "Sector ETFs": {
        "XLF": "Financial Select Sector SPDR Fund",
        "XLE": "Energy Select Sector SPDR Fund",
        "XLK": "Technology Select Sector SPDR Fund",
        "XLV": "Health Care Select Sector SPDR Fund"
    }
}


def get_all_popular_tickers() -> list:
    """Get a flat list of all popular ticker symbols."""
    tickers = []
    for category in POPULAR_TICKERS.values():
        tickers.extend(category.keys())
    return sorted(tickers)


def get_ticker_description(ticker: str) -> str:
    """
    Get description for a ticker symbol.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
        
    Returns
    -------
    str
        Description of the ticker
    """
    # Search through all categories
    for category in POPULAR_TICKERS.values():
        if ticker in category:
            return category[ticker]
    
    # Fallback to fetching from yfinance
    info = get_ticker_info(ticker)
    return info.get('name', ticker)
