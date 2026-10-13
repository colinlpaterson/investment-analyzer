"""Historical portfolio results and plain-language methodology."""

import numpy as np
import streamlit as st

from utils.portfolio_analysis import METRIC_COLUMNS, analyze_portfolio
from utils.portfolio_charts import history_chart, risk_chart
from utils.portfolio_data import PortfolioData
from utils.portfolio_simulation_ui import render_simulation_handoff
from utils.portfolio_exports import METRIC_HELP, csv_bytes, metric_table, portfolio_metadata


def _formatted(value: float, percentage: bool = True) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.2%}" if percentage else f"{value:.2f}"


def _display_table(frame):
    """Convert fractional values to percentages for display, keeping numeric sorting."""
    displayed = frame.copy()
    columns = {}
    for column in frame:
        if column in ("Sharpe ratio", "Sortino ratio"):
            columns[column] = st.column_config.NumberColumn(format="%.2f", help=METRIC_HELP.get(column))
        else:
            displayed[column] *= 100
            suffix = " (pp)" if column == "Volatility contribution" else " (%)"
            columns[column] = st.column_config.NumberColumn(column + suffix, format="%.2f", help=METRIC_HELP.get(column))
    st.dataframe(displayed, column_config=columns)


def render_historical_analysis(data: PortfolioData) -> None:
    st.header("Historical performance")
    st.caption(
        "Hypothetical backtest of your saved allocation using reinvested total returns. "
        "No deposits, withdrawals, taxes, or trading costs."
    )
    results = st.session_state["portfolio_results"]
    try:
        if "historical" not in results:
            results["historical"] = analyze_portfolio(data)
        analysis = results["historical"]
    except ValueError as exc:
        st.error(f"Historical analysis is unavailable: {exc}")
        return
    portfolio_metrics = analysis.metrics.loc["Portfolio"]
    with st.container(horizontal=True):
        for label in METRIC_COLUMNS:
            help_text = METRIC_HELP.get(label)
            if label == "Sharpe ratio":
                help_text = (
                    "Annualized Sharpe ratio = mean daily excess return divided by the sample standard "
                    "deviation of daily returns, multiplied by √252. Excess return is the portfolio's "
                    "daily return minus the daily risk-free rate. "
                    f"This backtest uses your saved risk-free assumption of {data.definition.risk_free_rate:.2%} "
                    "per year, converted to a constant daily rate as (1 + annual rate)^(1/252) − 1. "
                    "This is a user-set assumption, not a live interest rate. "
                    "The ratio is undefined when volatility is zero."
                )
            elif label == "Sortino ratio":
                help_text += (
                    f" Saved MAR: {data.definition.minimum_acceptable_return:.2%} annually, "
                    "converted to a daily target as (1 + annual MAR)^(1/252) − 1. "
                    "Mean daily return above the target / daily downside deviation × √252. "
                    "Downside deviation is the square root of mean squared shortfalls across all days; "
                    "days meeting the target contribute zero."
                )
            st.metric(label, _formatted(portfolio_metrics[label], label not in ("Sharpe ratio", "Sortino ratio")),
                      border=True, help=help_text)
    st.caption("CAGR is annualized compound growth. Volatility, Sharpe, and Sortino use daily returns and 252 trading days per year.")
    if len(data.returns) < 2:
        st.warning("At least two daily returns are needed for volatility, Sharpe, Sortino, and risk decomposition. Extend the date range.")
    elif (data.prices.index[-1] - data.prices.index[0]).days < 365:
        st.warning("This backtest covers less than a year. Annualized figures can be extreme and are sensitive to the short sample.")
    st.subheader(f"Growth of 10,000 ({data.currency})")
    st.plotly_chart(history_chart(analysis.growth, f"Portfolio value ({data.currency})"), key="portfolio_growth_chart")
    st.subheader("Drawdown")
    st.caption("Decline from the previous peak. A return to 0% means that peak has been recovered.")
    st.plotly_chart(history_chart(analysis.drawdowns, "Drawdown", percentage=True), key="portfolio_drawdown_chart")
    if data.benchmark_prices is not None:
        st.subheader("Portfolio and benchmark")
        _display_table(analysis.metrics)
        st.caption("The benchmark is held throughout the same dates with distributions reinvested.")

    st.subheader("Where the risk comes from")
    st.caption(
        "Risk at the entered target weights, estimated using historical sample covariance. "
        "It differs from realized backtest risk when weights drift between rebalances."
    )
    st.metric("Volatility at target weights", _formatted(analysis.target_volatility))
    if analysis.holdings["Risk contribution"].notna().any():
        st.plotly_chart(risk_chart(analysis.holdings), key="portfolio_risk_chart")
        st.caption("Risk shares sum to 100%. Negative contributions indicate diversification; other shares can exceed 100%.")
    else:
        st.info("Risk shares are undefined when portfolio volatility is zero or there are too few observations.")
    st.subheader("Holdings")
    _display_table(analysis.holdings[["Target weight", "Ending weight", *METRIC_COLUMNS,
                                      "Risk contribution", "Volatility contribution"]])
    st.caption(
        "Holding performance metrics are standalone buy-and-hold results over the common period. "
        "Risk contribution is a share of portfolio risk; volatility contribution is in annualized percentage points. "
        "N/A or an empty cell means the statistic is undefined."
    )
    with st.expander("Calculation details"):
        st.markdown(
            "- **Accounting:** invest at the first available close and let weights drift with adjusted-price returns. "
            "Fractional holdings are allowed.\n"
            "- **Rebalancing:** restore the entered weights after the last observed close of a completed calendar "
            "month, quarter, or year. New weights apply to the following return. The final observation is not "
            "treated as a rebalance. Buy and hold never rebalances.\n"
            "- **CAGR:** ending value divided by starting value, raised to 1 / elapsed calendar years, minus 1; "
            "a year is 365.25 days.\n"
            "- **Volatility:** sample standard deviation of daily returns × √252.\n"
            "- **Sharpe ratio:** mean daily excess return / sample daily standard deviation × √252. "
            "The constant daily risk-free rate is (1 + annual rate)^(1/252) − 1. "
            "Zero volatility gives an undefined Sharpe ratio.\n"
            "- **Sortino ratio:** mean daily return above the daily MAR / daily downside deviation × √252. "
            "Daily MAR = (1 + annual MAR)^(1/252) − 1. Downside deviation is the square root of "
            "the average squared shortfall below daily MAR across all return observations, with zero "
            "shortfall on days meeting the target. Zero downside deviation or fewer than two returns "
            "gives an undefined Sortino ratio. MAR is independent of the Sharpe risk-free assumption.\n"
            "- **Maximum drawdown:** the worst decline from a prior peak, including the initial investment.\n"
            "- **Risk contribution:** wᵢ(Σw)ᵢ / (wᵀΣw), using annualized sample covariance Σ. "
            "Risk in this analysis is defined by volatility (variation in returns), not capital loss. "
            "This describes target weights, not a decomposition of the drifted backtest.\n"
            "- **Volatility contribution (pp):** each holding's contribution to the portfolio's annualized "
            "volatility at target weights, measured in percentage points (pp). It equals the holding's "
            "risk share multiplied by portfolio volatility. For example, a 25% risk share in a portfolio "
            "with 12% volatility contributes 3 percentage points. Contributions sum to target-weight "
            "portfolio volatility; negative contributions indicate a reduction in volatility.\n"
            "- **Annualization:** 252-day scaling assumes a conventional trading calendar and does not adjust "
            "for serial correlation in returns."
        )
        st.caption(
            "Methodology references: [Sharpe ratio](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm) · "
            "[Risk decomposition](https://docs.mosek.com/portfolio-cookbook/risk_parity.html)"
        )
        dates = analysis.backtest.rebalance_dates
        st.markdown(f"**Rebalance events:** {len(dates)}")
        if len(dates):
            st.dataframe({"Rebalance date": dates.date}, hide_index=True,
                         column_config={"Rebalance date": st.column_config.DateColumn(format="MM/DD/YYYY")})
    with st.expander("Export historical results"):
        metadata = portfolio_metadata(data)
        daily = analysis.growth.add_suffix(f" value ({data.currency})").join(
            analysis.drawdowns.mul(100).add_suffix(" drawdown (%)"))
        daily["Portfolio daily return (%)"] = analysis.backtest.returns * 100
        daily = daily.join(analysis.backtest.weights.mul(100).add_suffix(" closing weight (%)"))
        for label, table, filename in [("performance metrics", metric_table(analysis.metrics), "metrics"),
                                       ("holdings and risk", metric_table(analysis.holdings), "holdings"),
                                       ("daily performance", daily, "daily"),
                                       ("historical asset returns", data.returns.mul(100).add_suffix(" return (%)"), "returns")]:
            st.download_button(f"Download {label}", csv_bytes(table, metadata),
                               file_name=f"portfolio_historical_{filename}.csv", mime="text/csv")
    render_simulation_handoff(["Current"], "analysis_simulate")
