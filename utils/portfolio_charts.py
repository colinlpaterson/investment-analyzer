"""Portfolio charts using the app's existing Plotly chart library."""

import pandas as pd
import plotly.graph_objects as go


def history_chart(series: pd.DataFrame, y_title: str, percentage: bool = False) -> go.Figure:
    figure = go.Figure()
    for index, name in enumerate(series):
        figure.add_trace(go.Scatter(
            x=series.index, y=series[name], name=name, mode="lines",
            line={"width": 2.5, "dash": "solid" if index == 0 else "dash"},
            hovertemplate="%{x|%m/%d/%Y}<br>" + ("%{y:.2%}" if percentage else "%{y:,.2f}") + "<extra>%{fullData.name}</extra>",
        ))
    figure.update_layout(
        height=360, margin={"l": 10, "r": 10, "t": 20, "b": 10},
        hovermode="x unified", legend={"orientation": "h", "y": 1.12},
        xaxis={"title": None, "tickformat": "%m/%d/%Y"},
        yaxis={"title": y_title, "tickformat": ".0%" if percentage else ",.0f"},
    )
    return figure


def risk_chart(holdings: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for column, label in [("Target weight", "Allocation weight"), ("Risk contribution", "Share of portfolio risk")]:
        figure.add_trace(go.Bar(
            x=holdings.index, y=holdings[column], name=label,
            hovertemplate="%{x}<br>%{y:.2%}<extra>%{fullData.name}</extra>",
        ))
    figure.update_layout(
        barmode="group", height=360, margin={"l": 10, "r": 10, "t": 20, "b": 10},
        legend={"orientation": "h", "y": 1.12}, yaxis={"tickformat": ".0%", "zeroline": True},
    )
    return figure


def frontier_chart(frontier: pd.DataFrame, comparison: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=frontier["Estimated volatility"], y=frontier["Estimated return"],
        name="Constrained efficient frontier", mode="lines+markers", connectgaps=False,
        hovertemplate="Volatility: %{x:.2%}<br>Estimated return: %{y:.2%}<extra>Efficient frontier</extra>",
    ))
    symbols = {"Current": "circle", "Minimum volatility": "diamond", "Maximum Sharpe": "star"}
    for name, row in comparison.iterrows():
        figure.add_trace(go.Scatter(
            x=[row["Estimated volatility"]], y=[row["Estimated return"]], mode="markers", name=name,
            marker={"size": 13, "symbol": symbols[name]},
            hovertemplate="Volatility: %{x:.2%}<br>Estimated return: %{y:.2%}<extra>%{fullData.name}</extra>",
        ))
    figure.update_layout(
        height=430, margin={"l": 10, "r": 10, "t": 25, "b": 10},
        legend={"orientation": "h", "y": 1.15},
        xaxis={"title": "Estimated annualized volatility", "tickformat": ".0%"},
        yaxis={"title": "Estimated annualized arithmetic return", "tickformat": ".0%"},
    )
    return figure


def simulation_fan_chart(quantiles: pd.DataFrame, contributed: pd.Series, currency: str) -> go.Figure:
    figure = go.Figure()
    years = quantiles.index.to_numpy() / 12
    for lower, upper, name, color in [
        ("10th percentile", "90th percentile", "10th–90th percentile", "rgba(55,126,184,0.18)"),
        ("25th percentile", "75th percentile", "25th–75th percentile", "rgba(55,126,184,0.30)"),
    ]:
        figure.add_trace(go.Scatter(x=years, y=quantiles[lower], mode="lines", line={"width": 0},
                                   showlegend=False, hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=years, y=quantiles[upper], mode="lines", line={"width": 0},
                                   fill="tonexty", fillcolor=color, name=name, hoverinfo="skip"))
    for values, name, dash in [(quantiles["Median"], "Median", "solid"), (contributed, "Total contributed", "dash")]:
        figure.add_trace(go.Scatter(
            x=years, y=values, mode="lines", name=name, line={"width": 2, "dash": dash},
            hovertemplate="Year %{x:.2f}<br>%{y:,.2f}<extra>%{fullData.name}</extra>",
        ))
    figure.update_layout(height=420, xaxis_title="Years from start", yaxis_title=f"Nominal value ({currency})",
                         yaxis_tickformat=",.0f", legend={"orientation": "h", "y": 1.12},
                         margin={"l": 10, "r": 10, "t": 25, "b": 10})
    return figure


def simulation_distribution_chart(ending_values, contributed: float, target: float | None, currency: str) -> go.Figure:
    figure = go.Figure(go.Histogram(x=ending_values, nbinsx=50, histnorm="probability",
                                   name="Simulated outcomes", hovertemplate="Ending value: %{x:,.0f}<br>Share: %{y:.1%}<extra></extra>"))
    figure.add_vline(x=contributed, line_dash="dash", annotation_text="Total contributed", annotation_position="top left")
    if target is not None:
        figure.add_vline(x=target, line_dash="dot", annotation_text="Target", annotation_position="bottom right")
    figure.update_layout(height=360, xaxis_title=f"Nominal ending value ({currency})", yaxis_title="Share of simulations",
                         xaxis_tickformat=",.0f", yaxis_tickformat=".0%", margin={"l": 10, "r": 10, "t": 25, "b": 10})
    return figure
