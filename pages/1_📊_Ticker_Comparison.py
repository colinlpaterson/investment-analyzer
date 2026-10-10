"""
Multi-Ticker Comparison Page
Compare YTD performance across multiple tickers for current vs prior year.
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
    validate_ticker,
    get_all_popular_tickers
)
from utils.ytd_analysis import calculate_ytd_returns, get_return_label
from utils.multi_ticker_analysis import (
    determine_comparison_years,
    prepare_comparison_data,
    calculate_comparison_statistics,
    format_comparison_table
)
from utils.chart_builder import create_multi_ticker_comparison_chart

# Page configuration
st.set_page_config(
    page_title="Multi-Ticker Comparison",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for ticker list
if 'ticker_list' not in st.session_state:
    st.session_state.ticker_list = []

# Title and description
st.title("📊 Multi-Ticker YTD Comparison")
st.markdown("""
Compare year-to-date performance across multiple stocks and ETFs. 
The chart shows current period performance against the prior year's performance for the same time period.
""")

# Sidebar for inputs
st.sidebar.header("Ticker Selection")

return_type = st.sidebar.segmented_control(
    "Return type",
    options=["price", "total"],
    default="price",
    required=True,
    format_func=get_return_label,
    key="comparison_return_type",
    help=(
        "Price return excludes distributions. Total return uses Yahoo Finance's "
        "adjusted close as a proxy for reinvested distributions."
    ),
)
return_label = get_return_label(return_type)

# Determine comparison years for display
current_year, prior_year, last_month, is_january = determine_comparison_years()

# Determine the actual years to compare
if is_january:
    compare_year = prior_year          # 2025
    baseline_year = prior_year - 1     # 2024
else:
    compare_year = current_year        # 2026 (during year), compare YTD
    baseline_year = prior_year         # 2025


# Show what will be compared
if is_january:
    st.sidebar.info(
        f"📅 **Comparing:**\n\nFull Year {compare_year}\n\nvs\n\nFull Year {baseline_year}"
    )
else:
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    st.sidebar.info(
        f"📅 **Comparing:**\n\nYTD {compare_year} (through {month_names[last_month-1]})"
        f"\n\nvs\n\nYTD {baseline_year} (through {month_names[last_month-1]})"
    )

# Display current ticker list
st.sidebar.subheader("Selected Tickers")
if len(st.session_state.ticker_list) == 0:
    st.sidebar.write("*No tickers selected*")
else:
    # Display tickers with remove buttons
    for i, ticker in enumerate(st.session_state.ticker_list):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.write(f"{i+1}. **{ticker}**")
        with col2:
            if st.button("✖", key=f"remove_{i}", help=f"Remove {ticker}"):
                st.session_state.ticker_list.pop(i)
                st.rerun()

st.sidebar.markdown("---")

# Add ticker section
if len(st.session_state.ticker_list) < 5:
    st.sidebar.subheader("Add Ticker")
    
    ticker_input_method = st.sidebar.radio(
        "Selection Method",
        ["Popular Tickers", "Custom Ticker"],
        key="input_method"
    )
    
    new_ticker = None
    
    if ticker_input_method == "Popular Tickers":
        # Get all popular tickers that aren't already selected
        available_tickers = [t for t in get_all_popular_tickers() 
                           if t not in st.session_state.ticker_list]
        
        if len(available_tickers) > 0:
            # Create labels with descriptions
            ticker_labels = {}
            for t in available_tickers:
                desc = get_ticker_description(t)
                ticker_labels[t] = f"{t} - {desc}"
            
            selected_label = st.sidebar.selectbox(
                "Select Ticker",
                [ticker_labels[t] for t in sorted(available_tickers)],
                key="popular_select"
            )
            new_ticker = selected_label.split(" - ")[0]
        else:
            st.sidebar.info("All popular tickers already selected")
    else:
        new_ticker = st.sidebar.text_input(
            "Enter Ticker Symbol",
            key="custom_input",
            help="Enter any valid stock or ETF ticker symbol"
        ).upper().strip()
    
    if st.sidebar.button("➕ Add Ticker", type="secondary", width="stretch"):
        if new_ticker and new_ticker not in st.session_state.ticker_list:
            if validate_ticker(new_ticker):
                st.session_state.ticker_list.append(new_ticker)
                st.rerun()
            else:
                st.sidebar.error(f"Invalid ticker: {new_ticker}")
        elif new_ticker in st.session_state.ticker_list:
            st.sidebar.warning(f"{new_ticker} already in list")
        else:
            st.sidebar.warning("Please enter a ticker symbol")
else:
    st.sidebar.warning("⚠️ Maximum 5 tickers reached")

# Clear all button
if len(st.session_state.ticker_list) > 0:
    if st.sidebar.button("🗑️ Clear All", width="stretch"):
        st.session_state.ticker_list = []
        st.rerun()

st.sidebar.markdown("---")

# Analyze button
analyze_button = st.sidebar.button(
    "📊 Compare Tickers",
    type="primary",
    width="stretch",
    disabled=len(st.session_state.ticker_list) < 2
)

# Info section
with st.sidebar.expander("ℹ️ About This Comparison"):
    st.markdown(f"""
    **How It Works:**
    
    - **January**: Compares full prior year vs year before
    - **Other Months**: Compares YTD current year vs YTD prior year (same months)
    - **Return type**: {return_label}
    
    **Example (March 2026):**
    - Shows Jan-Feb 2026 vs Jan-Feb 2025
    - Apples-to-apples comparison
    
    **Chart Lines:**
    - Solid: Current period
    - Dashed: Prior period (except January)
    """)

# Main content area
if analyze_button:
    if len(st.session_state.ticker_list) < 2:
        st.warning("⚠️ Please add at least 2 tickers to compare")
    else:
        with st.spinner(f"Loading data for {len(st.session_state.ticker_list)} tickers..."):
            # Load data for all tickers
            all_ytd_data = {}
            failed_tickers = []
            
            for ticker in st.session_state.ticker_list:
                # We need baseline_year-1 data so baseline_year can use prior Dec close
                start_year = baseline_year - 1
                data = load_ticker_data(ticker, f"{start_year}-01-01")

                if data is None or len(data) == 0:
                    failed_tickers.append(ticker)
                    continue

                # Calculate YTD returns up through compare_year
                try:
                    ytd_data = calculate_ytd_returns(
                        data, start_year, compare_year, return_type=return_type
                    )
                except ValueError:
                    failed_tickers.append(ticker)
                    continue

                if len(ytd_data) == 0:
                    failed_tickers.append(ticker)
                    continue

                all_ytd_data[ticker] = ytd_data

            
            # Show warnings for failed tickers
            if failed_tickers:
                st.warning(
                    f"⚠️ Could not calculate {return_label.lower()} for: "
                    f"{', '.join(failed_tickers)}"
                )
            
            if len(all_ytd_data) < 2:
                st.error("❌ Need at least 2 tickers with valid data to compare")
                st.stop()
            
            # Prepare comparison data
            comparison_data = prepare_comparison_data(
                 all_ytd_data,
                compare_year,
                baseline_year,
                last_month
            )
            
            if len(comparison_data) < 2:
                st.error("❌ Insufficient data for comparison")
                st.stop()
            
            # Display success message
            st.success(f"✅ Loaded data for {len(comparison_data)} tickers")
            
            # Create and display chart
            fig = create_multi_ticker_comparison_chart(
                comparison_data,
                compare_year,
                baseline_year,
                last_month,
                is_january,
                return_label
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Calculate and display summary statistics
            st.subheader("📊 Performance Summary")
            
            stats_df = calculate_comparison_statistics(
                comparison_data,
                compare_year,
                baseline_year,
                last_month,
                is_january
            )
            
            display_df = format_comparison_table(
                stats_df,
                compare_year,
                baseline_year,
                last_month,
                is_january
            )
            
            # Display table with styling
            st.dataframe(
                display_df,
                hide_index=True,
                width="stretch",
                height=min(400, (len(display_df) + 1) * 35 + 3)
            )
            #Helper
            def safe_idx(op_series, fn):
                s = op_series.dropna()
                if s.empty:
                    return None
                return fn(s)
            # Key insights
            with st.expander("🔍 Key Insights"):
                # Best performer
                best_idx = safe_idx(stats_df["Current Return"], lambda s: s.idxmax())
                if best_idx is not None:
                    best_ticker = stats_df.loc[best_idx, "Ticker"]
                    best_return = stats_df.loc[best_idx, "Current Return"] * 100
                    st.markdown(f"**🏆 Best Performer:** {best_ticker} ({best_return:+.1f}%)")

                # Worst performer
                worst_idx = safe_idx(stats_df["Current Return"], lambda s: s.idxmin())
                if worst_idx is not None:
                    worst_ticker = stats_df.loc[worst_idx, "Ticker"]
                    worst_return = stats_df.loc[worst_idx, "Current Return"] * 100
                    st.markdown(f"**📉 Worst Performer:** {worst_ticker} ({worst_return:+.1f}%)")

                # Biggest improvement
                biggest_gain_idx = safe_idx(stats_df["Difference"], lambda s: s.idxmax())
                if biggest_gain_idx is not None:
                    gain_ticker = stats_df.loc[biggest_gain_idx, "Ticker"]
                    gain_diff = stats_df.loc[biggest_gain_idx, "Difference"] * 100
                    st.markdown(f"**📈 Most Improved:** {gain_ticker} ({gain_diff:+.1f}% vs prior period)")

                # Biggest decline
                biggest_decline_idx = safe_idx(stats_df["Difference"], lambda s: s.idxmin())
                if biggest_decline_idx is not None:
                    decline_ticker = stats_df.loc[biggest_decline_idx, "Ticker"]
                    decline_diff = stats_df.loc[biggest_decline_idx, "Difference"] * 100
                    st.markdown(f"**📉 Biggest Decline:** {decline_ticker} ({decline_diff:+.1f}% vs prior period)")
            
            # Export data
            st.subheader("💾 Export Data")
            
            # Prepare export data
            export_rows = []
            for ticker, data in comparison_data.items():
                current_series = data.get('current')
                prior_series = data.get('prior')
                
                for month in range(1, last_month + 1):
                    row = {
                        'Ticker': ticker,
                        'Return Type': return_label,
                        'Month': month,
                    }
                    
                    if current_series is not None and month in current_series.index:
                        row[f'{compare_year}_YTD_{return_type}_return'] = current_series.loc[month]
                    else:
                        row[f'{compare_year}_YTD_{return_type}_return'] = None
                    
                    if prior_series is not None and month in prior_series.index:
                        row[f'{baseline_year}_YTD_{return_type}_return'] = prior_series.loc[month]
                    else:
                        row[f'{baseline_year}_YTD_{return_type}_return'] = None
                    
                    export_rows.append(row)
            
            export_df = pd.DataFrame(export_rows)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="Download Comparison Data (CSV)",
                data=csv,
                file_name=f"ticker_comparison_{return_type}_return_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

else:
    # Show placeholder content
    st.info("👈 Add 2-5 tickers in the sidebar and click 'Compare Tickers' to begin")
    
    st.markdown("""
    ### How to Use
    
    1. **Add tickers** one at a time (minimum 2, maximum 5)
    2. **Choose from popular tickers** or enter custom symbols
    3. **Remove tickers** using the ✖ button if needed
    4. **Click 'Compare Tickers'** to generate the comparison
    
    ### What You'll See
    
    - **Interactive chart** comparing YTD performance
    - **Apples-to-apples comparison** (same time periods)
    - **Summary table** with current vs prior period returns
    - **Key insights** highlighting best/worst performers
    - **Export options** to download the data
    
    ### Comparison Logic
    
    **In January:**
    - Compares full prior year vs year before that
    - Example: In Jan 2026, shows full 2025 vs full 2024
    
    **Other Months:**
    - Compares YTD current year vs YTD prior year (same months)
    - Example: In Mar 2026, shows Jan-Feb 2026 vs Jan-Feb 2025
    
    ### Example Combinations to Try
    
    - **Market Indices**: SPY, QQQ, IWM
    - **Asset Classes**: SPY, TLT, GLD
    - **Sectors**: XLF, XLE, XLK, XLV
    """)

# Footer
st.markdown("---")
st.caption("Data provided by Yahoo Finance. For informational purposes only. Not financial advice.")
