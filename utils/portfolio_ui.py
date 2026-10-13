"""The shared portfolio editor and data-readiness view."""

from datetime import date, timedelta
from dataclasses import replace
from math import isclose

import pandas as pd
import streamlit as st

from utils.portfolio import REBALANCING_OPTIONS, portfolio_from_rows
from utils.portfolio_data import PortfolioData, load_portfolio_data
from utils.portfolio_state import invalidate_portfolio_inputs, initialize_portfolio_state, save_portfolio, store_portfolio_data


def _inputs_changed() -> None:
    invalidate_portfolio_inputs(st.session_state)


def render_portfolio_builder(page_key: str) -> PortfolioData | None:
    """Both pages edit the same saved definition; widget keys remain page-local."""
    initialize_portfolio_state(st.session_state)
    saved = st.session_state["portfolio_definition"]
    prefix = f"{page_key}_{st.session_state['portfolio_revision']}"
    end_default = date.today() - timedelta(days=1)
    start_default = (pd.Timestamp(end_default) - pd.DateOffset(years=5)).date()
    rows = (
        [{"Ticker": h.ticker, "Weight (%)": h.weight * 100} for h in saved.holdings]
        if saved else [{"Ticker": "SPY", "Weight (%)": 60.0}, {"Ticker": "BND", "Weight (%)": 40.0}]
    )
    st.subheader("Build your portfolio")
    st.caption(
        "Enter stocks or ETFs quoted in the same currency. Weights must total 100%. "
        "Saved holdings and settings are shared between portfolio pages for this browser session."
    )
    with st.container(border=True):
        edited = st.data_editor(
            pd.DataFrame(rows), hide_index=True, num_rows="dynamic",
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", required=True),
                "Weight (%)": st.column_config.NumberColumn(
                    "Weight (%)", required=True, min_value=0.0, max_value=100.0, format="%.4f",
                ),
            },
            on_change=_inputs_changed, key=f"{prefix}_holdings",
        )
        left, right = st.columns(2)
        start = left.date_input("Start date", value=saved.start_date if saved else start_default,
                                max_value=end_default, format="MM/DD/YYYY", on_change=_inputs_changed, key=f"{prefix}_start")
        end = right.date_input("End date (inclusive)", value=saved.end_date if saved else end_default,
                               max_value=end_default, format="MM/DD/YYYY", on_change=_inputs_changed, key=f"{prefix}_end")
        rebalance = st.selectbox(
            "Rebalancing", REBALANCING_OPTIONS,
            index=REBALANCING_OPTIONS.index(saved.rebalancing) if saved else 0,
            help="Restore target weights at period-end closes. Buy and hold allows weights to drift throughout.",
            on_change=_inputs_changed, key=f"{prefix}_rebalancing",
        )
        with st.expander("Benchmark and assumptions"):
            benchmark = st.text_input("Benchmark ticker (optional)", value=saved.benchmark or "" if saved else "",
                                      on_change=_inputs_changed, key=f"{prefix}_benchmark")
            risk_free = st.number_input(
                "Annual risk-free rate (%)", min_value=-99.0, max_value=100.0,
                value=saved.risk_free_rate * 100 if saved else 0.0, step=0.25,
                help="A constant effective annual assumption for Sharpe ratios. 0% is a placeholder, not a live rate.",
                on_change=_inputs_changed, key=f"{prefix}_risk_free",
            )
            mar = st.number_input(
                "Annual minimum acceptable return (MAR) (%)", min_value=-99.0, max_value=100.0,
                value=saved.minimum_acceptable_return * 100 if saved else 0.0, step=0.25,
                help="Your annual return target for the Sortino ratio, converted to a constant daily target. "
                     "0% measures shortfalls below zero return. Independent of the Sharpe risk-free assumption.",
                on_change=_inputs_changed, key=f"{prefix}_mar",
            )
            st.caption(
                "Total returns use adjusted close as a proxy for reinvested distributions. "
                "Taxes, trading costs, and currency conversion are not included. "
                "Today's prices are excluded to avoid incomplete trading days."
            )
        # Local checks share the save-time validator and never request market data.
        try:
            portfolio_from_rows(edited.to_dict("records"), start, end, rebalance, risk_free, benchmark, minimum_acceptable_return_percent=mar)
        except (ValueError, TypeError) as exc:
            st.warning(str(exc))
        normalize_help = (
            "Normalize divides each entered weight by the sum of all weights, then multiplies by 100. "
            "This makes the total 100% while preserving the proportions. For example, weights of 20% "
            "and 30% become 40% and 60%. The normalized portfolio is then saved and its data checked."
        )
        submit = st.button("Save and check data", type="primary")
        with st.container(horizontal=True, vertical_alignment="center"):
            normalize = st.button("Normalize and save", icon=":material/balance:")
            st.caption("Normalization", help=normalize_help, width="content")
        st.caption("Edits hide previous results immediately. Save before switching pages to keep your changes.")
    # Also compare rendered values: programmatic/editor changes may bypass callbacks.
    if saved is not None and st.session_state["portfolio_data"] is not None:
        try:
            draft = portfolio_from_rows(edited.to_dict("records"), start, end, rebalance, risk_free, benchmark, minimum_acceptable_return_percent=mar)
            matches = (replace(draft, holdings=saved.holdings) == saved
                       and len(draft.holdings) == len(saved.holdings)
                       and all(a.ticker == b.ticker and isclose(a.weight, b.weight, rel_tol=0, abs_tol=1e-12)
                               for a, b in zip(draft.holdings, saved.holdings)))
        except (ValueError, TypeError):
            matches = False
        if not matches:
            invalidate_portfolio_inputs(st.session_state)
    if submit or normalize:
        try:
            definition = portfolio_from_rows(
                edited.to_dict("records"), start, end, rebalance, risk_free, benchmark, normalize, minimum_acceptable_return_percent=mar,
            )
        except (ValueError, TypeError) as exc:
            invalidate_portfolio_inputs(st.session_state)
            st.error(str(exc))
            st.info("These edits have not been saved. Correct the inputs and save again.")
            return
        save_portfolio(st.session_state, definition)
        with st.spinner("Checking adjusted prices, currencies, and common history…"):
            try:
                store_portfolio_data(st.session_state, load_portfolio_data(definition))
            except ValueError as exc:
                st.session_state["portfolio_data_error"] = str(exc)
        st.rerun()

    if st.session_state["portfolio_data_error"]:
        st.error(st.session_state["portfolio_data_error"])
        st.info("Your portfolio settings are saved, but data is not ready. Update the inputs or retry Save and check data.")
    if st.session_state["portfolio_dirty"]:
        st.info("Inputs changed. Save and check data to update the analysis.")
    data = st.session_state["portfolio_data"]
    if data is None:
        return
    st.subheader("Saved portfolio")
    st.dataframe(pd.DataFrame([
        {"Ticker": h.ticker, "Weight (%)": h.weight * 100} for h in data.definition.holdings
    ]), hide_index=True, column_config={"Weight (%)": st.column_config.NumberColumn(format="%.4f")})
    st.success(
        f"Data ready: {len(data.prices.columns)} holdings · {data.currency} · "
        f"{len(data.returns):,} daily return observations"
    )
    st.caption(
        f"Common price period: {data.prices.index[0]:%m/%d/%Y} to {data.prices.index[-1]:%m/%d/%Y}. "
        f"Rebalancing: {data.definition.rebalancing}. "
        f"Risk-free assumption: {data.definition.risk_free_rate:.2%} annually. "
        f"MAR: {data.definition.minimum_acceptable_return:.2%} annually. "
        f"Benchmark: {data.definition.benchmark or 'None'}."
    )
    for notice in data.notices:
        st.warning(notice)
    with st.expander("Data coverage"):
        st.dataframe(data.coverage, hide_index=True, column_config={
            "First available": st.column_config.DateColumn(format="MM/DD/YYYY"),
            "Last available": st.column_config.DateColumn(format="MM/DD/YYYY"),
        })
        st.caption("All entered holdings, including zero-weight candidates, must have valid data. Missing prices are not filled.")
    return data
