"""
Historical YTD Performance Analysis Page
Analyzes year-to-date price or total returns across multiple years for a single ticker.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_fetcher import (
    load_ticker_data, 
    POPULAR_TICKERS, 
    get_ticker_description,
    validate_ticker
)
from utils.ytd_analysis import (
    calculate_ytd_returns,
    get_return_label,
    prepare_ytd_series,
    get_summary_statistics
)
from utils.chart_builder import create_ytd_plotly_chart

# Page configuration
st.set_page_config(
    page_title="Historical YTD Analysis",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Historical Year-to-Date Performance")
st.markdown("""
Analyze year-to-date (YTD) price or total returns for any stock or ETF across multiple years.
Compare current year performance against historical trends to identify patterns and outliers.
""")

# Sidebar for inputs
st.sidebar.header("Analysis Parameters")

return_type = st.sidebar.segmented_control(
    "Return type",
    options=["price", "total"],
    default="price",
    required=True,
    format_func=get_return_label,
    key="historical_return_type",
    help=(
        "Price return excludes distributions. Total return uses Yahoo Finance's "
        "adjusted close as a proxy for reinvested distributions."
    ),
)
return_label = get_return_label(return_type)

# Ticker selection
ticker_input_method = st.sidebar.radio(
    "Ticker Selection Method",
    ["Popular Tickers", "Custom Ticker"],
    help="Choose from popular tickers or enter your own"
)

if ticker_input_method == "Popular Tickers":
    # Flatten popular tickers for dropdown
    all_tickers = []
    ticker_labels = {}
    for category, tickers in POPULAR_TICKERS.items():
        for symbol, name in tickers.items():
            all_tickers.append(symbol)
            ticker_labels[symbol] = f"{symbol} - {name}"
    
    selected_label = st.sidebar.selectbox(
        "Select Ticker",
        [ticker_labels[t] for t in sorted(all_tickers)],
        index=0
    )
    ticker = selected_label.split(" - ")[0]
else:
    ticker = st.sidebar.text_input(
        "Enter Ticker Symbol",
        value="AAPL",
        help="Enter any valid stock or ETF ticker symbol"
    ).upper().strip()

# Start year selection
current_year = datetime.now().year
start_year = st.sidebar.number_input(
    "Start Year",
    min_value=1990,
    max_value=current_year - 1,
    value=2004,
    help="First year to include in analysis"
)

# Analysis button
analyze_button = st.sidebar.button("📊 Analyze", type="primary", width="stretch")

# Info section
with st.sidebar.expander("ℹ️ About YTD Calculation"):
    if return_type == "price":
        methodology = (
            "Uses split-adjusted closing prices and excludes dividends and other distributions."
        )
    else:
        methodology = (
            "Uses Yahoo Finance adjusted close as a total-return proxy, assuming distributions "
            "are reinvested. It is not tax-aware and may differ from an investor's realized return."
        )
    st.markdown(f"""
    **YTD Return Formula:**
    ```
    YTD = (Current Month Value / Prior December Value) - 1
    ```
    
    - **Selected methodology:** {return_label}
    - {methodology}
    - Each year starts from prior December close
    - Current year shows completed months only
    - Historical years shown in grey for comparison
    """)

# Main content area
if analyze_button:
    if not ticker:
        st.error("Please enter a ticker symbol")
    else:
        with st.spinner(f"Loading data for {ticker}..."):
            # Validate ticker
            if not validate_ticker(ticker):
                st.error(f"❌ Invalid ticker symbol: {ticker}")
                st.info("Please check the ticker symbol and try again. Make sure it's a valid stock or ETF symbol.")
                st.stop()
            
            # Load data
            data = load_ticker_data(ticker, f"{start_year - 1}-12-01")
            
            if data is None or len(data) == 0:
                st.error(f"❌ No data available for {ticker} starting from {start_year}")
                st.info("Try a different ticker or start year.")
                st.stop()
            
            # Calculate YTD returns
            try:
                ytd_data = calculate_ytd_returns(
                    data, start_year, current_year, return_type=return_type
                )
            except ValueError as exc:
                st.error(f"❌ {exc}")
                st.stop()
            
            if len(ytd_data) == 0:
                st.error(f"❌ Insufficient data to calculate YTD returns for {ticker}")
                st.stop()
            
            # Prepare series and statistics
            ytd_by_year, highlight_year, last_month_highlight = prepare_ytd_series(
                ytd_data, current_year
            )
            
            summary_stats = get_summary_statistics(
                ytd_by_year, highlight_year, last_month_highlight
            )
            
            # Display ticker info
            ticker_desc = get_ticker_description(ticker)
            st.success(f"✅ Loaded {ticker}: {ticker_desc}")
            
            # Create and display chart
            fig = create_ytd_plotly_chart(
                ytd_by_year,
                highlight_year,
                last_month_highlight,
                summary_stats,
                ticker,
                return_label
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Display summary statistics
            st.subheader("📊 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current/Latest YTD",
                    f"{summary_stats['current_ytd']*100:.1f}%",
                    help=f"Year-to-date {return_label.lower()} for {summary_stats['highlight_year']}"
                )
            
            with col2:
                if summary_stats['best_year']:
                    st.metric(
                        "Best Year",
                        f"{summary_stats['best_return']*100:.1f}%",
                        delta=f"{summary_stats['best_year']}",
                        delta_color="off",
                        help="Highest full-year return in the dataset"
                    )
                else:
                    st.metric("Best Year", "N/A")
            
            with col3:
                if summary_stats['worst_year']:
                    st.metric(
                        "Worst Year",
                        f"{summary_stats['worst_return']*100:.1f}%",
                        delta=f"{summary_stats['worst_year']}",
                        delta_color="off",
                        help="Lowest full-year return in the dataset"
                    )
                else:
                    st.metric("Worst Year", "N/A")
            
            with col4:
                if not pd.isna(summary_stats['avg_full_year_ytd']):
                    st.metric(
                        "Historical Avg",
                        f"{summary_stats['avg_full_year_ytd']*100:.1f}%",
                        delta=f"σ = {summary_stats['std_full_year_ytd']*100:.1f}%",
                        delta_color="off",
                        help=f"Average full-year return across {summary_stats['n_years']} completed years"
                    )
                else:
                    st.metric("Historical Avg", "N/A")
            
            # Detailed information
            with st.expander("📋 Detailed Information"):
                st.markdown(f"""
                **Analysis Period:** {summary_stats['actual_start_year']} - {summary_stats['highlight_year']}
                
                **Data Source:** Yahoo Finance (yfinance)
                
                **Methodology:**
                - YTD return measured from prior December month-end to each subsequent month-end
                - Selected return type: **{return_label}**
                - {methodology}
                - {summary_stats['highlight_year']} line ends at last completed month
                - Average calculated across {summary_stats['n_years']} completed years
                - Dates label calendar month-end; prices come from the last trading day
                """)
            
            # Download data option
            st.subheader("💾 Export Data")
            
            # Prepare data for download
            value_column = "Close" if return_type == "price" else "Adj Close"
            value_label = (
                "Split-adjusted close"
                if return_type == "price"
                else "Dividend-adjusted close"
            )
            return_column = f"YTD {return_label} (%)"
            export_data = ytd_data[['Date', 'Year', 'Month', value_column, 'YTD']].copy()
            export_data = export_data.rename(
                columns={value_column: value_label, 'YTD': return_column}
            )
            export_data[return_column] = export_data[return_column] * 100
            
            # Resampling labels observations with calendar month-end dates
            # Format as date only (no time component)
            export_data['Date'] = pd.to_datetime(export_data['Date']).dt.date
            
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="Download YTD Data (CSV)",
                data=csv,
                file_name=f"{ticker}_{return_type}_return_ytd_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help=f"Download {return_label.lower()} data with calendar month-end labels"
            )

else:
    # Show placeholder content
    st.info("👈 Configure analysis parameters in the sidebar and click 'Analyze' to begin")
    
    st.markdown("""
    ### How to Use
    
    1. **Select a ticker** from popular options or enter a custom symbol
    2. **Choose a start year** for historical comparison (default: 2004)
    3. **Click 'Analyze'** to generate the visualization
    
    ### What You'll See
    
    - **Interactive chart** showing YTD performance across all years
    - **Current year** highlighted
    - **Best and worst years** marked with green and red dots
    - **Summary statistics** including averages and volatility
    - **Export options** to download the underlying data
    
    ### Example Tickers to Try
    
    - **SPY** - S&P 500 ETF
    - **QQQ** - Nasdaq-100 ETF  
    - **GLD** - Gold ETF
    - **TLT** - 20+ Year Treasury Bond ETF
    
    ### Return Types
    
    - **Price return** uses split-adjusted closing prices and excludes distributions.
    - **Total return** uses Yahoo Finance adjusted close as a proxy for reinvested distributions.
    """)

# Footer
st.markdown("---")
st.caption("Data provided by Yahoo Finance. Select price return or adjusted-close total return. For informational purposes only. Not financial advice.")
