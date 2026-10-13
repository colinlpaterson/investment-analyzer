"""Simulation controls, allocation handoff, and conditional outcome summaries."""

import numpy as np
import pandas as pd
import streamlit as st

from utils.portfolio import REBALANCING_OPTIONS
from utils.portfolio_charts import simulation_distribution_chart, simulation_fan_chart
from utils.portfolio_data import PortfolioData
from utils.portfolio_simulation import MIN_HISTORY, SimulationConfig, simulate_portfolio
from utils.portfolio_exports import csv_bytes, portfolio_metadata
from utils.portfolio_state import invalidate_simulation_inputs, clear_simulation, select_simulation_allocation, simulation_allocations


def _inputs_changed() -> None:
    invalidate_simulation_inputs(st.session_state)


def render_simulation_handoff(names: list[str], key_prefix: str) -> None:
    with st.container(horizontal=True):
        for name in names:
            if st.button(f"Simulate {name.lower()}", key=f"{key_prefix}_{name}"):
                select_simulation_allocation(st.session_state, name)
                st.switch_page("pages/5_🎲_Portfolio_Simulation.py")


def render_simulation(data: PortfolioData) -> None:
    st.header("Monte Carlo simulation")
    if len(data.returns) < MIN_HISTORY:
        st.info(f"Simulation requires at least {MIN_HISTORY} common daily returns. Extend the history and save again.")
        return
    results = st.session_state["portfolio_results"]
    allocations = simulation_allocations(st.session_state)
    names = list(allocations)
    selected = results.get("simulation_selection", "Current")
    if selected not in allocations:
        selected = "Current"
        clear_simulation(st.session_state)
    prefix = f"simulation_{st.session_state['portfolio_revision']}_{results.get('simulation_selection_revision', 0)}"
    source = st.selectbox("Allocation to simulate", names, index=names.index(selected), key=f"{prefix}_source",
                          help="Use the saved current portfolio or an available optimized allocation. This does not change your current portfolio.")
    if source != results.get("simulation_selection"):
        clear_simulation(st.session_state)
        results["simulation_selection"] = source
    weights = allocations[source]
    st.dataframe((weights * 100).rename("Simulated weight (%)").to_frame(),
                 column_config={"Simulated weight (%)": st.column_config.NumberColumn(format="%.2f")})
    st.caption(f"Simulated allocation: {source}. The portfolio builder above continues to show your saved current allocation.")
    settings = results.get("simulation_settings", {})
    with st.container(border=True):
        left, right = st.columns(2)
        initial = left.number_input(f"Starting investment ({data.currency})", min_value=1.0, max_value=1e12,
                                    value=settings.get("initial", 10000.0), step=1000.0, on_change=_inputs_changed, key=f"{prefix}_initial")
        years = right.number_input("Horizon (years)", min_value=1, max_value=40, value=settings.get("years", 10), on_change=_inputs_changed, key=f"{prefix}_years")
        contribution = left.number_input(f"Monthly contribution ({data.currency})", min_value=0.0, max_value=1e9,
                                         value=settings.get("contribution", 0.0), step=100.0, on_change=_inputs_changed, key=f"{prefix}_contribution",
                                         help="Invested at the end of each simulated month in target proportions. It starts earning returns the following month.")
        target = right.number_input(f"Target ending value ({data.currency}; 0 = no target)", min_value=0.0, max_value=1e15,
                                     value=settings.get("target", 0.0), step=1000.0, on_change=_inputs_changed, key=f"{prefix}_target",
                                     help="An optional nominal ending-value goal, including contributions.")
        rebalance = st.selectbox("Simulation rebalancing", REBALANCING_OPTIONS,
                                 index=REBALANCING_OPTIONS.index(settings.get("rebalancing", data.definition.rebalancing)), on_change=_inputs_changed, key=f"{prefix}_rebalancing")
        with st.expander("Simulation assumptions"):
            paths = st.number_input("Number of simulations", min_value=100, max_value=10000, value=settings.get("paths", 5000), step=100, on_change=_inputs_changed, key=f"{prefix}_paths")
            block = st.number_input("Historical block length (trading days)", min_value=1, max_value=min(252, len(data.returns)),
                                    value=settings.get("block", 21), on_change=_inputs_changed, key=f"{prefix}_block",
                                    help="Consecutive historical daily returns sampled together. Longer blocks preserve longer stretches of observed behavior; blocks wrap from the end of the history to the start.")
            seed = st.number_input("Random seed", min_value=0, max_value=2**32 - 1, value=settings.get("seed", 42), on_change=_inputs_changed, key=f"{prefix}_seed",
                                   help="The same data, weights, settings, and seed reproduce the same scenarios.")
            inflation = st.number_input("Annual inflation assumption (%)", min_value=-5.0, max_value=20.0,
                                        value=settings.get("inflation", 0.0), step=.25, on_change=_inputs_changed, key=f"{prefix}_inflation",
                                        help="Used only for the supplementary ending-value column in today's money. Charts, contributions, targets, and probabilities remain nominal.")
        submitted = st.button("Run simulation", type="primary", key=f"{prefix}_run")
        st.caption("Edits hide previous outcomes. Run simulation to calculate new outcomes. One modeled month is 21 trading days; contributions are fixed nominal amounts.")
    if submitted:
        results["simulation_dirty"] = False
        clear_simulation(st.session_state)
        results["simulation_settings"] = dict(initial=initial, years=years, contribution=contribution, target=target,
                                               rebalancing=rebalance, paths=paths, block=block, seed=seed, inflation=inflation)
        try:
            config = SimulationConfig(initial, years * 12, contribution, rebalance, paths, block, seed,
                                      target if target > 0 else None, inflation / 100)
            with st.spinner("Sampling shared return blocks and simulating portfolio cash flows…"):
                result = simulate_portfolio(data.returns, weights, config)
            results["simulation"] = {"result": result, "source": source}
        except ValueError as exc:
            results["simulation_error"] = str(exc)
    if results.get("simulation_error"):
        st.error(results["simulation_error"])
    run = results.get("simulation")
    if run is None:
        if results.get("simulation_dirty"):
            st.info("Inputs changed. Run simulation again to update the outcomes.")
        return
    result, source = run["result"], run["source"]
    config = result.config
    st.subheader("Simulated outcomes")
    st.caption(
        f"Completed run: {source} · {config.paths:,} scenarios · {config.months / 12:g} years · "
        f"{config.rebalancing} · {config.block_length}-day blocks · seed {config.seed}. "
        f"Initial investment: {config.initial_value:,.2f} {data.currency}; monthly contribution: "
        f"{config.monthly_contribution:,.2f} {data.currency}. "
        f"Historical returns: {data.returns.index[0]:%m/%d/%Y}–{data.returns.index[-1]:%m/%d/%Y}."
    )
    st.info(
        "These are scenarios based on resampled historical returns, not forecasts or guarantees. "
        "The percentile bands are conditional on this history and model; the 10th percentile is not a worst case."
    )
    with st.container(horizontal=True):
        st.metric(f"Median ending value ({data.currency})", f"{np.median(result.ending_values):,.0f}",
                  help="The middle simulated outcome: half finish below this value and half above. Includes contributions; not a forecast.")
        st.metric(f"Total contributed ({data.currency})", f"{result.contributed.iloc[-1]:,.0f}",
                  help="Starting investment plus every monthly deposit. Investment gains are excluded.")
        st.metric("Chance of ending below contributions", f"{result.probability_loss:.1%}",
                  help="Share of simulated ending values below the initial investment plus all monthly contributions, in nominal terms. This is not the chance of experiencing a loss at any time along the way.")
        if result.probability_target is not None:
            st.metric("Chance of reaching target", f"{result.probability_target:.1%}",
                      help=f"Share of scenarios ending at or above the nominal target of {config.target_value:,.2f} {data.currency}.")
    st.plotly_chart(simulation_fan_chart(result.quantiles, result.contributed, data.currency), key="simulation_fan")
    st.caption("Bands show the distribution at each month. The median line connects monthly medians; it is not one simulated path.")
    st.subheader("Ending-value distribution")
    st.plotly_chart(simulation_distribution_chart(result.ending_values, result.contributed.iloc[-1], config.target_value, data.currency), key="simulation_distribution")
    terminal = pd.DataFrame({f"Nominal ending value ({data.currency})": np.percentile(result.ending_values, [10, 50, 90])},
                             index=["10th percentile", "Median", "90th percentile"])
    if config.inflation_rate != 0:
        terminal[f"Ending value in today's money ({data.currency})"] = terminal.iloc[:, 0] / (1 + config.inflation_rate) ** (config.months / 12)
        st.caption(f"Today's-money values use {config.inflation_rate:.2%} annual inflation. Other results remain nominal.")
    st.dataframe(terminal, column_config={column: st.column_config.NumberColumn(format="%.2f") for column in terminal})
    export = result.quantiles.join(result.contributed).add_suffix(f" ({data.currency})")
    metadata = portfolio_metadata(data)
    metadata.update({"Allocation": source, "Simulated weights (%)": "; ".join(
        f"{ticker}: {weight * 100:.12g}" for ticker, weight in result.weights.items()),
        "Seed": config.seed, "Paths": config.paths, "Block length (days)": config.block_length,
        "Horizon (months)": config.months, "Starting investment": config.initial_value,
        "Monthly contribution": config.monthly_contribution, "Simulation rebalancing": config.rebalancing,
        "Nominal target": config.target_value, "Annual inflation (%)": config.inflation_rate * 100})
    st.download_button("Download monthly outcome summary", csv_bytes(export, metadata),
                       file_name="portfolio_simulation_monthly.csv", mime="text/csv")
    ending = pd.DataFrame({f"Nominal ending value ({data.currency})": result.ending_values},
                          index=pd.RangeIndex(1, config.paths + 1, name="Scenario"))
    st.download_button("Download scenario ending values", csv_bytes(ending, metadata),
                       file_name="portfolio_simulation_endings.csv", mime="text/csv")
    with st.expander("Simulation calculation details"):
        st.markdown(
            "- **Sampling:** circular blocks of consecutive historical daily returns. A block start is drawn "
            "uniformly for each scenario; dates wrap at the end of the history. Every holding uses the same "
            "sampled dates, retaining observed cross-asset relationships and dependence within blocks. "
            "New blocks are independent; wrapping joins the end of the sample to its beginning.\n"
            "- **Accounting:** invest at target weights, apply daily total returns, then deposit contributions "
            "at each month-end in target proportions. Deposits earn returns starting the following month. "
            "Buy and hold does not sell existing holdings to restore weights.\n"
            "- **Rebalancing:** after returns and contributions, restore target weights every 1, 3, or 12 "
            "modeled months. A month has 21 trading days; these are not specific future calendar dates. "
            "Rebalancing is self-financing. Allocation caps apply to optimized target weights, not to drifted weights.\n"
            "- **Cash flows:** total contributed equals the starting investment plus all monthly contributions. "
            "Contributions increase wealth but are not investment returns. All probabilities refer to terminal "
            "nominal wealth, including contributions. Equality counts as reaching a target and not as a loss.\n"
            "- **Interpretation:** more scenarios reduce sampling noise, but do not eliminate uncertainty about "
            "future returns. Block length and the selected history influence outcomes. Unseen market regimes "
            "and parameter uncertainty are not modeled.\n"
            "- **Optimized allocations:** the weights come from optimization, but simulation resamples raw "
            "historical returns rather than generating returns from the optimizer's mean and covariance estimates. "
            "It is not an independent validation of the optimizer.\n"
            "- **Assumptions:** reinvested distributions, fractional holdings, fixed nominal contributions, "
            "and no taxes, trading costs, withdrawals, or currency conversion. The risk-free assumption is not "
            "used to generate returns.\n"
            "- **Reproducibility:** identical data, weights, configuration, seed, and numerical environment "
            "produce identical results. Refreshed market data can change them."
        )
        st.caption("Methodology: [time-series block bootstraps](https://bashtage.github.io/arch/doc/bootstrap/timeseries-bootstraps.html).")
