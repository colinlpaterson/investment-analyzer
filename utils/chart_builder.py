"""
Chart building utilities using Plotly for interactive visualizations.
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Dict
import pandas as pd

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

COLOR_CURRENT_YEAR = "#01528a"  # Blue
COLOR_HISTORICAL = "#808080"  # Grey
COLOR_BEST = "#228B22"  # Forest Green
COLOR_WORST = "#DC143C"  # Crimson

theme_base = st.get_option("theme.base") or "light"
is_dark = theme_base == "dark"

TEXT_COLOR = "#f0f0f0" if is_dark else "#111111"
LEGEND_BG  = "rgba(0,0,0,0.35)" if is_dark else "rgba(255,255,255,0.9)"
GRID_COLOR = "rgba(255,255,255,0.18)" if is_dark else "rgba(128,128,128,0.2)"
PLOT_BG  = "rgba(0,0,0,0)" if is_dark else "white"
PAPER_BG = "rgba(0,0,0,0)" if is_dark else "white"

def create_ytd_plotly_chart(ytd_by_year: Dict[int, pd.Series],
                            highlight_year: int,
                            last_month_highlight: int,
                            summary_stats: dict,
                            ticker: str,
                            return_label: str = "Price return") -> go.Figure:
    """
    Create interactive YTD performance chart using Plotly.
    
    Parameters
    ----------
    ytd_by_year : dict
        Dictionary of YTD series by year
    highlight_year : int
        Year to highlight
    last_month_highlight : int
        Last available month for highlighted year
    summary_stats : dict
        Dictionary of summary statistics
    ticker : str
        Ticker symbol
    return_label : str
        Display label for the selected return methodology
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    months = np.arange(1, 13)
    years = sorted(ytd_by_year.keys())
    
    fig = go.Figure()
    
    # Plot historical years (grey lines)
    for year in years:
        if year == highlight_year:
            continue
            
        series = ytd_by_year[year].reindex(months)
        
        # Determine line style for best/worst years
        line_color = COLOR_HISTORICAL
        line_width = 1.0
        
        fig.add_trace(go.Scatter(
            x=months,
            y=series * 100,  # Convert to percentage
            mode='lines',
            name=str(year),
            line=dict(color=line_color, width=line_width),
            opacity=0.35,
            hovertemplate=f'<b>{year}</b><br>Month: %{{x}}<br>YTD: %{{y:.1f}}%<extra></extra>',
            showlegend=False
        ))
    
    # Plot highlighted year (gold line)
    highlight_series = ytd_by_year[highlight_year].reindex(months)
    highlight_series.loc[last_month_highlight + 1:] = np.nan
    
    fig.add_trace(go.Scatter(
        x=months,
        y=highlight_series * 100,
        mode='lines+markers',
        name=str(highlight_year),
        line=dict(color=COLOR_CURRENT_YEAR, width=3),
        marker=dict(size=8, color=COLOR_CURRENT_YEAR, 
                   line=dict(width=2, color='darkgoldenrod')),
        hovertemplate=f'<b>{highlight_year}</b><br>Month: %{{x}}<br>YTD: %{{y:.1f}}%<extra></extra>',
        showlegend=True
    ))
    
    # Add markers for best and worst years if available
    if summary_stats['best_year'] and summary_stats['best_year'] != highlight_year:
        fig.add_trace(go.Scatter(
            x=[12],
            y=[summary_stats['best_return'] * 100],
            mode='markers+text',
            name=f"Best: {summary_stats['best_year']}",
            marker=dict(size=12, color=COLOR_BEST, 
                       line=dict(width=2, color='white')),
            text=[f"{summary_stats['best_year']}<br>{summary_stats['best_return']*100:.1f}%"],
            textposition="middle right",
            textfont=dict(size=10, color=COLOR_BEST),
            hovertemplate=f"<b>Best Year: {summary_stats['best_year']}</b><br>Return: {summary_stats['best_return']*100:.1f}%<extra></extra>",
            showlegend=False
        ))
    
    if summary_stats['worst_year'] and summary_stats['worst_year'] != highlight_year:
        fig.add_trace(go.Scatter(
            x=[12],
            y=[summary_stats['worst_return'] * 100],
            mode='markers+text',
            name=f"Worst: {summary_stats['worst_year']}",
            marker=dict(size=12, color=COLOR_WORST,
                       line=dict(width=2, color='white')),
            text=[f"{summary_stats['worst_year']}<br>{summary_stats['worst_return']*100:.1f}%"],
            textposition="middle right",
            textfont=dict(size=10, color=COLOR_WORST),
            hovertemplate=f"<b>Worst Year: {summary_stats['worst_year']}</b><br>Return: {summary_stats['worst_return']*100:.1f}%<extra></extra>",
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{ticker} Year-to-Date {return_label} by Calendar Year ({summary_stats['actual_start_year']}–{summary_stats['highlight_year']})",
            x=0.5,
            xanchor='center',
            font=dict(size=18, color=TEXT_COLOR)
        ),
        xaxis=dict(
            title=dict(
                text="Month",
                font=dict(color=TEXT_COLOR, size=12)
            ),
            tickmode='array',
            tickvals=months,
            ticktext=MONTH_LABELS,
            tickfont=dict(color=TEXT_COLOR),
            range=[0.5, 13.5],
            showgrid=True,
            gridcolor=GRID_COLOR,
            linecolor=TEXT_COLOR,
            zerolinecolor=TEXT_COLOR
        ),
        yaxis=dict(
            title=dict(
                text=f"Year-to-Date {return_label} (%)",
                font=dict(color=TEXT_COLOR, size=12)
            ),
            tickformat='.1f',
            ticksuffix='%',
            tickfont=dict(color=TEXT_COLOR),
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor=TEXT_COLOR,
            linecolor=TEXT_COLOR
        ),
        hovermode='closest',
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        height=600,
        margin=dict(l=80, r=80, t=100, b=100),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor=LEGEND_BG,
            bordercolor="rgba(255,255,255,0.25)" if is_dark else "rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(color=TEXT_COLOR)
        )
    )
    
    return fig


def create_multi_ticker_comparison_chart(
    ticker_data: Dict[str, Dict[str, pd.Series]],
    compare_year: int,
    baseline_year: int,
    last_month_current: int,
    is_january: bool,
    return_label: str = "Price return",
) -> go.Figure:
    """
    Create multi-ticker comparison chart.
    
    Parameters
    ----------
    ticker_data : dict
        Nested dict: {ticker: {'current': series, 'prior': series}}
    compare_year : int
        Current/most recent year being displayed
    baseline_year : int
        Prior year for comparison
    last_month_current : int
        Last completed month in current period
    is_january : bool
        Whether we're in January (shows full years only)
    return_label : str
        Display label for the selected return methodology
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    months = np.arange(1, 13)
    
    # High-contrast color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig = go.Figure()
    
    for idx, (ticker, data) in enumerate(ticker_data.items()):
        color = colors[idx % len(colors)]
        
        current_series = data.get('current')
        prior_series = data.get('prior')
        
        # Prior year is always dashed with reduced alpha
        prior_line_style = 'dash'
        prior_alpha = 0.5
        prior_width = 2.0
        
        # Plot prior year
        if prior_series is not None and len(prior_series.dropna()) > 0:
            fig.add_trace(go.Scatter(
                x=months,
                y=prior_series.reindex(months) * 100,
                mode='lines',
                name=f"{ticker} {baseline_year}",
                line=dict(color=color, width=prior_width, dash=prior_line_style),
                opacity=prior_alpha,
                hovertemplate=f'<b>{ticker} {baseline_year}</b><br>Month: %{{x}}<br>Return: %{{y:.1f}}%<extra></extra>'
            ))
        
        # Plot current year
        if current_series is not None and len(current_series.dropna()) > 0:
            # Truncate to last completed month if not January
            display_series = current_series.reindex(months)
            if not is_january:
                display_series.loc[last_month_current + 1:] = np.nan
            
            fig.add_trace(go.Scatter(
                x=months,
                y=display_series * 100,
                mode='lines+markers',
                name=f"{ticker} {compare_year}",
                line=dict(color=color, width=2.5),
                marker=dict(size=6, color=color),
                hovertemplate=f'<b>{ticker} {compare_year}</b><br>Month: %{{x}}<br>Return: %{{y:.1f}}%<extra></extra>'
            ))
            
            # Add endpoint annotation
            last_val = display_series.dropna().iloc[-1] if len(display_series.dropna()) > 0 else None
            last_month_idx = display_series.dropna().index[-1] if len(display_series.dropna()) > 0 else None
            
            if last_val is not None and last_month_idx is not None:
                fig.add_annotation(
                    x=last_month_idx,
                    y=last_val * 100,
                    text=f"{ticker}<br>{last_val*100:.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    ax=30,
                    ay=0,
                    bgcolor='white',
                    bordercolor=color,
                    borderwidth=2,
                    font=dict(size=10, color=color, family='Arial Black')
                )
    
    # Build title based on comparison type
    if is_january:
        title_text = f"{return_label} Comparison<br><sub>Full Year {compare_year} vs Full Year {baseline_year}</sub>"
    else:
        title_text = f"Year-to-Date {return_label} Comparison<br><sub>{compare_year} YTD vs {baseline_year} YTD (through {MONTH_LABELS[last_month_current-1]})</sub>"

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            font=dict(size=18, color=TEXT_COLOR)
        ),
       xaxis=dict(
            title=dict(
                text="Month",
                font=dict(color=TEXT_COLOR, size=12)
            ),
            tickmode='array',
            tickvals=months,
            ticktext=MONTH_LABELS,
            tickfont=dict(color=TEXT_COLOR),
            range=[0.5, 13.5],
            showgrid=True,
            gridcolor=GRID_COLOR,
            linecolor=TEXT_COLOR,
            zerolinecolor=TEXT_COLOR
        ),
       yaxis=dict(
            title=dict(
                text=f"Year-to-Date {return_label} (%)",
                font=dict(color=TEXT_COLOR, size=12)
            ),
            tickformat='.1f',
            ticksuffix='%',
            tickfont=dict(color=TEXT_COLOR),
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor=TEXT_COLOR,
            linecolor=TEXT_COLOR
        ),
        hovermode='x unified',
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        height=600,
        margin=dict(l=80, r=80, t=120, b=80),
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.98,
            xanchor='left',
            x=0.02,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor="rgba(255,255,255,0.25)" if is_dark else "rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(color=TEXT_COLOR)
        ),
        showlegend=True
    )
    
    return fig
