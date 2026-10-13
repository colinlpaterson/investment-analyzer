"""Shared portfolio setup and correlated historical-block simulation."""

import streamlit as st

from utils.portfolio_ui import render_portfolio_builder
from utils.portfolio_simulation_ui import render_simulation


st.set_page_config(page_title="Portfolio simulation", page_icon="🎲", layout="wide")
st.title("🎲 Portfolio simulation")
st.markdown("Explore a range of possible outcomes for your allocation, contributions, and investment horizon.")
data = render_portfolio_builder("portfolio_simulation")
if data is not None:
    render_simulation(data)
