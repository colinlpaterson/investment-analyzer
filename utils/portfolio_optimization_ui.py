"""Constraint entry and comparisons of model-estimated portfolio allocations."""

import numpy as np
import pandas as pd
import streamlit as st

from utils.portfolio_charts import frontier_chart
from utils.portfolio_data import PortfolioData
from utils.portfolio_optimization import MIN_OBSERVATIONS, estimate_parameters, optimize_portfolio, weight_caps
from utils.portfolio_simulation_ui import render_simulation_handoff
from utils.portfolio_evaluation import evaluate_holdout
from utils.portfolio_exports import METRIC_HELP, csv_bytes, metric_table, portfolio_metadata
from utils.portfolio_charts import history_chart
from utils.portfolio_state import invalidate_optimization_inputs


def _inputs_changed() -> None:
    invalidate_optimization_inputs(st.session_state)


def render_optimization(data: PortfolioData) -> None:
    st.header("Portfolio optimization")
    st.caption(
        "Compare fully invested, long-only allocations using historical return estimates and "
        "a shrinkage covariance model. Risk means volatility, not capital loss."
    )
    if len(data.returns) < MIN_OBSERVATIONS:
        st.info(f"Optimization needs at least {MIN_OBSERVATIONS} common daily returns. Extend the portfolio date range and save again.")
        return
    results = st.session_state["portfolio_results"]
    settings = results.get("optimization_settings", {})
    prefix = f"optimization_{st.session_state['portfolio_revision']}"
    with st.container(border=True):
        maximum = st.number_input(
            "Maximum weight per holding (%)", min_value=1.0, max_value=100.0,
            value=settings.get("maximum", 100.0 if len(data.prices.columns) == 1 else 60.0), step=5.0,
            help="Each optimized holding must stay at or below this cap. Caps must collectively allow 100% allocation.",
            on_change=_inputs_changed, key=f"{prefix}_maximum",
        )
        st.caption("Optional ticker caps can tighten the global cap. Leave a cell empty to use the global cap; 0% excludes a ticker from optimized allocations.")
        overrides = st.data_editor(
            pd.DataFrame({"Ticker": data.prices.columns,
                          "Individual cap (%)": [settings.get("overrides", {}).get(ticker, np.nan) for ticker in data.prices]}),
            hide_index=True, disabled=["Ticker"], on_change=_inputs_changed, key=f"{prefix}_overrides",
            column_config={"Individual cap (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, format="%.2f")},
        )
        training = st.slider("Earlier history used for held-out training (%)", 50, 80,
                             value=settings.get("training", 70), step=5, key=f"{prefix}_training",
                             on_change=_inputs_changed,
                             help="Fit separate validation allocations on this earlier share of observations, then assess them on the remaining later history. The main frontier still uses full history.")
        submitted = st.button("Optimize portfolio", type="primary", key=f"{prefix}_submit")
        st.caption("Edits hide previous optimization results and simulations using optimized allocations. Click Optimize portfolio to calculate the new settings.")
    if "optimization" in results:
        expected_overrides = pd.DataFrame({"Ticker": data.prices.columns,
            "Individual cap (%)": [settings.get("overrides", {}).get(ticker, np.nan) for ticker in data.prices]})
        if maximum != settings.get("maximum") or training != settings.get("training", 70) or not overrides.equals(expected_overrides):
            invalidate_optimization_inputs(st.session_state)
    if submitted:
        invalidate_optimization_inputs(st.session_state)
        results["optimization_dirty"] = False
        try:
            if overrides["Ticker"].duplicated().any() or set(overrides["Ticker"]) != set(data.prices.columns):
                raise ValueError("Ticker constraints must match the saved portfolio.")
            per_ticker = {row["Ticker"]: float(row["Individual cap (%)"]) for row in overrides.to_dict("records")
                          if pd.notna(row["Individual cap (%)"])}
            results["optimization_settings"] = {"maximum": maximum, "overrides": per_ticker, "training": training}
            caps = weight_caps(data.prices.columns, maximum / 100, {ticker: cap / 100 for ticker, cap in per_ticker.items()})
            with st.spinner("Estimating returns and solving constrained allocations…"):
                estimates = estimate_parameters(data.returns)
                current = pd.Series({h.ticker: h.weight for h in data.definition.holdings})
                solution = optimize_portfolio(estimates, current, caps, data.definition.risk_free_rate)
                run = {"estimates": estimates, "caps": caps, "solution": solution}
                # A simple estimation-window sensitivity check, not an out-of-sample backtest.
                if len(data.returns) >= 2 * MIN_OBSERVATIONS:
                    recent_returns = data.returns.iloc[len(data.returns) // 2:]
                    try:
                        recent = estimate_parameters(recent_returns)
                        run["recent"] = optimize_portfolio(recent, current, caps, data.definition.risk_free_rate, frontier_points=0)
                        run["recent_estimates"] = recent
                    except ValueError as exc:
                        run["recent_error"] = str(exc)
                try:
                    run["validation"] = evaluate_holdout(data, caps, training / 100)
                except ValueError as exc:
                    run["validation_error"] = str(exc)
                results["optimization"] = run
        except (ValueError, TypeError) as exc:
            results["optimization_error"] = str(exc)
    if results.get("optimization_error"):
        st.error(results["optimization_error"])
    run = results.get("optimization")
    if run is None:
        if results.get("optimization_dirty"):
            st.info("Inputs changed. Optimize again to update the allocations and validation.")
        return
    estimates, caps, solution = run["estimates"], run["caps"], run["solution"]
    st.caption(
        f"Estimation returns: {estimates.start:%m/%d/%Y}–{estimates.end:%m/%d/%Y} "
        f"({estimates.observations:,} observations). Saved risk-free assumption: {data.definition.risk_free_rate:.2%} effective annually."
    )
    st.info(
        "These are model estimates using the selected history, not realized backtest metrics or promises "
        "of future performance. Weight caps limit concentration but do not remove estimation error."
    )
    if not solution.comparison.loc["Current", "Within caps"]:
        st.warning("Your current allocation exceeds one or more proposed caps. It remains visible for comparison; the optimized allocations obey the caps.")
    if solution.max_sharpe_reason:
        st.warning("Maximum Sharpe unavailable: " + solution.max_sharpe_reason)
    if solution.notices:
        if any("unavailable" in notice for notice in solution.notices):
            st.warning("Some frontier targets could not be solved. Gaps are left visible rather than presenting failed solutions.")
        with st.expander("Optimization status"):
            for notice in solution.notices:
                st.markdown(notice)
    st.subheader("Efficient frontier")
    st.plotly_chart(frontier_chart(solution.frontier, solution.comparison), key="portfolio_frontier_chart")
    st.caption("Each frontier point minimizes estimated volatility for a target return under the same caps. The line connects solved points.")
    order = ["Current", "Minimum volatility", "Maximum Sharpe"]
    comparison = solution.comparison.reindex(order).copy()
    comparison[["Estimated return", "Estimated volatility"]] *= 100
    st.subheader("Allocation comparison")
    st.dataframe(comparison, column_config={
        "Estimated return": st.column_config.NumberColumn("Estimated return (%)", format="%.2f",
            help="Weighted mean daily arithmetic returns × 252. This is a historical model estimate, not CAGR or a forecast."),
        "Estimated volatility": st.column_config.NumberColumn("Estimated volatility (%)", format="%.2f",
            help="Estimated annual return variability using shrinkage covariance. It measures volatility, not capital loss."),
        "Estimated Sharpe ratio": st.column_config.NumberColumn(format="%.2f",
            help=f"Estimated annualized arithmetic excess return per unit of volatility, using your {data.definition.risk_free_rate:.2%} effective annual risk-free assumption."),
    })
    allocations = solution.weights.reindex(columns=order).copy() * 100
    allocations.insert(0, "Maximum weight", caps * 100)
    allocations.index.name = "Ticker"
    st.dataframe(allocations, column_config={
        column: st.column_config.NumberColumn(f"{column} (%)", format="%.2f") for column in allocations
    })
    st.caption("Weights shown to two decimals may not sum to exactly 100% due to display rounding. Empty cells indicate an unavailable allocation.")
    metadata = portfolio_metadata(data)
    metadata["Caps (%)"] = "; ".join(f"{ticker}: {cap * 100:.12g}" for ticker, cap in caps.items())
    st.download_button("Download allocation comparison", csv_bytes(allocations.add_suffix(" (%)"), metadata),
                       file_name="portfolio_allocations_percent.csv", mime="text/csv")
    with st.expander("Export optimization results"):
        st.download_button("Download estimated metrics", csv_bytes(metric_table(solution.comparison), metadata),
                           file_name="portfolio_estimated_metrics.csv", mime="text/csv")
        st.download_button("Download frontier", csv_bytes(metric_table(solution.frontier).join(
            solution.frontier_weights.mul(100).add_suffix(" weight (%)")), metadata),
            file_name="portfolio_frontier.csv", mime="text/csv")
    render_simulation_handoff(list(solution.weights.columns), "optimization_simulate")
    with st.expander("Sensitivity to the estimation period"):
        if "recent" in run:
            recent, recent_estimates = run["recent"], run["recent_estimates"]
            st.caption(
                f"Full history versus the most recent half ({recent_estimates.start:%m/%d/%Y}–"
                f"{recent_estimates.end:%m/%d/%Y}), with identical caps. This is a sensitivity check, not a future-performance test."
            )
            table = pd.DataFrame(index=solution.weights.index)
            column_config = {}
            for name in order[1:]:
                if name in solution.weights and name in recent.weights:
                    table[f"{name}: full (%)"] = solution.weights[name] * 100
                    table[f"{name}: recent (%)"] = recent.weights[name] * 100
                    objective = (
                        "minimizes estimated portfolio volatility" if name == "Minimum volatility"
                        else "maximizes the estimated Sharpe ratio (excess return per unit of volatility)"
                    )
                    for window, period in [
                        ("full", f"the full selected history ({estimates.start:%m/%d/%Y}–{estimates.end:%m/%d/%Y})"),
                        ("recent", f"only the most recent half of the observations ({recent_estimates.start:%m/%d/%Y}–{recent_estimates.end:%m/%d/%Y})"),
                    ]:
                        column_config[f"{name}: {window} (%)"] = st.column_config.NumberColumn(
                            format="%.2f",
                            help=(
                                f"Percentage of the portfolio allocated to this ticker by the allocation that {objective}, "
                                f"using estimates from {period}. Both runs use the same holdings, weight caps, "
                                "and risk-free assumption. This is an allocation weight, not an investment return."
                            ),
                        )
                    change = (solution.weights[name] - recent.weights[name]).abs().max() * 100
                    st.markdown(f"**{name}:** largest change in a single holding's weight between full and recent history: {change:.2f} percentage points.")
            st.caption(
                "For each strategy, we compare every holding's full-history weight with its recent-history weight "
                "and report the largest absolute difference. A move from 20% to 25% is 5 percentage points. "
                "A displayed 0.00 means the largest difference rounds to zero (less than 0.005 percentage points). "
                "The windows may favor the same allocation, or binding caps may force the same weights. "
                "This does not establish that the allocation will remain stable in the future."
            )
            st.dataframe(table, column_config=column_config)
            sensitivity_metadata = dict(metadata, **{"Recent estimation start": f"{recent_estimates.start:%m/%d/%Y}",
                                                      "Recent estimation end": f"{recent_estimates.end:%m/%d/%Y}"})
            st.download_button("Download estimation sensitivity", csv_bytes(table, sensitivity_metadata),
                               file_name="portfolio_sensitivity.csv", mime="text/csv")
            if recent.max_sharpe_reason:
                st.caption("Recent-half maximum Sharpe unavailable: " + recent.max_sharpe_reason)
        elif "recent_error" in run:
            st.info("Sensitivity check unavailable: " + run["recent_error"])
        else:
            st.caption(f"At least {2 * MIN_OBSERVATIONS} daily returns are needed to compare two estimation windows.")
    with st.expander("Held-out historical evaluation"):
        evaluation = run.get("validation")
        if evaluation is None:
            st.info("Held-out evaluation unavailable: " + run["validation_error"])
        else:
            train, growth = evaluation.estimates, evaluation.growth
            st.caption(f"Fit: {train.start:%m/%d/%Y}–{train.end:%m/%d/%Y} ({train.observations} returns). "
                       f"Test returns: {growth.index[1]:%m/%d/%Y}–{growth.index[-1]:%m/%d/%Y} "
                       f"({evaluation.test_observations} returns). Start with 10,000 {data.currency} at the {growth.index[0]:%m/%d/%Y} close.")
            st.markdown("These separate allocations use only the earlier training returns. Target weights stay fixed during "
                        "the later test, with your saved rebalancing schedule. Current uses your entered weights. "
                        "The table reports realized test performance, not the full-history estimates above.")
            if evaluation.solution.max_sharpe_reason:
                st.warning("Training-period maximum Sharpe unavailable: " + evaluation.solution.max_sharpe_reason)
            st.dataframe(metric_table(evaluation.metrics), column_config={
                name if name in ("Sharpe ratio", "Sortino ratio") else name + " (%)": st.column_config.NumberColumn(
                    format="%.2f", help=METRIC_HELP[name]) for name in evaluation.metrics})
            st.plotly_chart(history_chart(growth, f"Held-out value ({data.currency})"), key="portfolio_holdout_chart")
            st.caption("Training allocations (%) — these may differ from the full-history allocations available for simulation.")
            training_weights = evaluation.solution.weights.mul(100).add_suffix(" (%)")
            st.dataframe(training_weights)
            st.info("This is one historical test, not proof of future superiority. Your ticker selection and constraints "
                    "may already reflect knowledge of this period. Changing settings after viewing test results makes "
                    "the period less independent. Short test windows can produce unstable annualized metrics.")
            heldout_metadata = dict(metadata, **{"Training return start": f"{train.start:%m/%d/%Y}",
                "Training return end": f"{train.end:%m/%d/%Y}", "Test return start": f"{growth.index[1]:%m/%d/%Y}",
                "Test return end": f"{growth.index[-1]:%m/%d/%Y}", "Evaluation": "Chronological held-out; no refitting"})
            for label, table, filename in [("metrics", metric_table(evaluation.metrics), "metrics"),
                                            ("growth", growth.add_suffix(f" ({data.currency})"), "growth"),
                                            ("training allocations", training_weights, "weights")]:
                st.download_button(f"Download held-out {label}", csv_bytes(table, heldout_metadata),
                                   file_name=f"portfolio_holdout_{filename}.csv", mime="text/csv")
    with st.expander("Optimization estimates and calculation details"):
        table = pd.DataFrame({"Estimated return (%)": estimates.returns * 100,
                              "Estimated volatility (%)": np.sqrt(np.diag(estimates.covariance)) * 100})
        st.dataframe(table, column_config={column: st.column_config.NumberColumn(format="%.2f") for column in table})
        st.markdown(
            "- **Return estimate:** average daily arithmetic return × 252. For a single period, the portfolio's "
            "return is the weighted sum of its holdings' returns, so their arithmetic means combine consistently "
            "in mean–variance optimization. CAGR measures compounded growth over a historical path and includes "
            "the effect of volatility on compounding; a weighted average of holding CAGRs is generally not the "
            "portfolio's CAGR. We therefore use arithmetic means to estimate returns for optimization and CAGR "
            "to describe historical compound growth. Multiplying by 252 is an annualization convention, not a "
            "forecast of compounded annual growth.\n"
            "- **Risk estimate:** annualized Ledoit–Wolf covariance shrinkage. It differs from the sample covariance "
            "used in historical risk decomposition.\n"
            "- **Covariance shrinkage intensity:** the weight given to a simpler risk model when blending it "
            "with the covariance estimated from historical returns. The Ledoit–Wolf model used here blends "
            "the empirical covariance with a target that gives every holding the same variance (the average "
            "empirical variance) and zero covariance between holdings. At 0% intensity, only empirical covariance "
            "is used; at 100%, only that target is used. For example, 30% intensity means 70% empirical covariance "
            "plus 30% target. The intensity is estimated from the data to reduce estimation noise and improve "
            "stability. It is not a confidence level or a percentage reduction in portfolio risk.\n"
            "- **Estimated Sharpe ratio:** annualized arithmetic excess return divided by modeled volatility. "
            "The saved effective annual risk-free rate is converted to a daily rate and multiplied by 252 "
            "to match arithmetic return estimates.\n"
            "- **Constraints:** fully invested, non-negative weights, with the displayed upper bounds. "
            "No shorting, leverage, trading costs, or taxes.\n"
            "- **Maximum Sharpe:** solved through a convex quadratic transformation when positive excess "
            "return is feasible. Undefined ratios and failed solves are not reported as allocations.\n"
            "- **Validation:** solver convergence, finite outputs, full investment, bounds, and target returns "
            "are checked before a solution is displayed (weight tolerance: 0.00001 percentage points)."
        )
        st.caption(f"Covariance shrinkage intensity: {estimates.shrinkage:.1%}.")
        st.caption(
            "References: [Ledoit–Wolf estimator](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html) · "
            "[SciPy solver](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)"
        )
