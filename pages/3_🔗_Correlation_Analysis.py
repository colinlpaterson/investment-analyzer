"""Static return-correlation analysis for multiple stocks and ETFs."""

from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.chart_builder import create_correlation_heatmap
from utils.correlation_analysis import (
    MIN_OBSERVATIONS,
    calculate_correlation_matrix,
    find_correlation_extremes,
    get_frequency_label,
    normalize_tickers,
    prepare_return_matrix,
)
from utils.data_fetcher import get_all_popular_tickers, load_daily_ticker_data
from utils.ytd_analysis import RETURN_TYPE_COLUMNS, get_return_label


st.set_page_config(
    page_title="Correlation analysis",
    page_icon="🔗",
    layout="wide",
)

st.title("🔗 Correlation analysis")
st.markdown(
    "Compare how closely the periodic returns of stocks and ETFs have moved together. "
    "Correlation ranges from -1 (opposite movement) to +1 (similar movement)."
)

with st.sidebar:
    st.header("Analysis parameters")

    with st.form("correlation_parameters"):
        selected_tickers = st.multiselect(
            "Tickers",
            options=get_all_popular_tickers(),
            default=["SPY", "QQQ", "TLT", "GLD"],
            max_selections=10,
            accept_new_options=True,
            placeholder="Choose or enter 2–10 ticker symbols",
            help="Select popular tickers or type a custom Yahoo Finance symbol.",
        )

        return_type = st.segmented_control(
            "Return type",
            options=["price", "total"],
            default="total",
            required=True,
            format_func=get_return_label,
            help=(
                "Price return excludes distributions. Total return uses Yahoo Finance's "
                "adjusted close as a proxy for reinvested distributions."
            ),
        )

        lookback_years = st.selectbox(
            "Lookback period",
            options=[1, 3, 5, 10],
            index=1,
            format_func=lambda years: f"{years} year" if years == 1 else f"{years} years",
        )

        frequency = st.segmented_control(
            "Return frequency",
            options=["daily", "weekly", "monthly"],
            default="daily",
            required=True,
            format_func=get_frequency_label,
            help="Prices are aligned first, then converted to returns at this frequency.",
        )

        analyze = st.form_submit_button(
            "Calculate correlation",
            type="primary",
            icon=":material/analytics:",
            width="stretch",
        )

results_slot = st.container()

with st.expander("About the calculation"):
    st.markdown(
        """
        - Uses Pearson correlation calculated from periodic percentage returns, not price levels.
        - All securities are aligned to a common set of observation dates before returns are calculated.
        - Weekly and monthly analyses include completed calendar periods only.
        - Total return uses Yahoo Finance adjusted close as a proxy for reinvested distributions.
        - Correlation is historical, can change over time, and does not guarantee diversification.
        """
    )

st.markdown("---")
st.caption(
    "Data provided by Yahoo Finance. For informational purposes only. Not financial advice."
)

if not analyze:
    results_slot.info(
        "Configure 2–10 tickers in the sidebar and select **Calculate correlation**."
    )
else:
    tickers = normalize_tickers(selected_tickers)

    with results_slot:
        if len(tickers) < 2:
            st.error("Select at least two unique ticker symbols.")
            st.stop()

        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.DateOffset(years=lookback_years)
        ticker_data = {}
        failed_tickers = []
        value_column = RETURN_TYPE_COLUMNS[return_type]

        with st.spinner(f"Loading daily data for {len(tickers)} tickers..."):
            for ticker in tickers:
                data = load_daily_ticker_data(ticker, start_date.strftime("%Y-%m-%d"))
                if data is None or data.empty or value_column not in data.columns:
                    failed_tickers.append(ticker)
                    continue
                ticker_data[ticker] = data

        if failed_tickers:
            st.warning(
                f"Could not load {get_return_label(return_type).lower()} data for: "
                + ", ".join(failed_tickers)
            )

        if len(ticker_data) < 2:
            st.error("At least two tickers with valid data are required.")
            st.stop()

        try:
            returns = prepare_return_matrix(ticker_data, return_type, frequency)
            correlation_matrix = calculate_correlation_matrix(
                returns,
                min_observations=MIN_OBSERVATIONS[frequency],
            )
            highest, lowest = find_correlation_extremes(correlation_matrix)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        actual_start = returns.index.min().date()
        actual_end = returns.index.max().date()
        frequency_label = get_frequency_label(frequency)
        return_label = get_return_label(return_type)
        lookback_label = (
            f"{lookback_years} year"
            if lookback_years == 1
            else f"{lookback_years} years"
        )

        st.success(
            f"Calculated {frequency_label.lower()} correlations for "
            f"{len(correlation_matrix)} tickers."
        )
        st.caption(
            f"Common return period: **{actual_start:%b %d, %Y}** to "
            f"**{actual_end:%b %d, %Y}**"
        )

        metric_columns = st.columns(3)
        with metric_columns[0]:
            st.metric("Highest correlation", f"{highest[2]:.2f}", border=True)
            st.caption(f"{highest[0]} and {highest[1]}")
        with metric_columns[1]:
            st.metric("Lowest correlation", f"{lowest[2]:.2f}", border=True)
            st.caption(f"{lowest[0]} and {lowest[1]}")
        with metric_columns[2]:
            st.metric("Common observations", f"{len(returns):,}", border=True)
            st.caption(f"{frequency_label} returns")

        figure = create_correlation_heatmap(
            correlation_matrix,
            return_label,
            frequency_label,
            lookback_label,
        )
        st.plotly_chart(figure, width="stretch")

        with st.expander("Correlation matrix values"):
            st.dataframe(
                correlation_matrix,
                column_config={
                    ticker: st.column_config.NumberColumn(format="%.3f")
                    for ticker in correlation_matrix.columns
                },
                width="stretch",
            )

        st.download_button(
            "Download correlation matrix (CSV)",
            data=correlation_matrix.to_csv(),
            file_name=(
                f"correlation_matrix_{return_type}_{frequency}_"
                f"{datetime.now():%Y%m%d}.csv"
            ),
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
        )
