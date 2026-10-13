"""Portfolio setup and historical analysis."""

import streamlit as st

from utils.portfolio_ui import render_portfolio_builder
from utils.portfolio_analysis_ui import render_historical_analysis
from utils.portfolio_optimization_ui import render_optimization


st.set_page_config(page_title="Portfolio analysis", page_icon=":material/calculate:", layout="wide")
st.title("Portfolio analysis & optimization", icon=":material/calculate:")
st.markdown("Analyze historical growth, drawdowns, and the risk behind your allocation.")
data = render_portfolio_builder("portfolio_analysis")
if data is not None:
    view = st.segmented_control("Portfolio view", ["Analyze", "Optimize"], default="Analyze",
                                required=True, key="portfolio_analysis_view")
    if view == "Analyze":
        render_historical_analysis(data)
    else:
        render_optimization(data)
