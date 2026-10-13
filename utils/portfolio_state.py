"""Shared, non-widget session state used by both portfolio pages."""

from collections.abc import MutableMapping

import pandas as pd

from utils.portfolio import PortfolioDefinition
from utils.portfolio_data import PortfolioData


def initialize_portfolio_state(state: MutableMapping) -> None:
    state.setdefault("portfolio_definition", None)
    state.setdefault("portfolio_data", None)
    state.setdefault("portfolio_results", {})
    state.setdefault("portfolio_revision", 0)
    state.setdefault("portfolio_data_error", None)
    state.setdefault("portfolio_dirty", False)


def save_portfolio(state: MutableMapping, definition: PortfolioDefinition) -> None:
    """Commit validated inputs and invalidate dependent outputs before a reload."""
    initialize_portfolio_state(state)
    state["portfolio_definition"] = definition
    state["portfolio_data"] = None
    state["portfolio_results"] = {}
    state["portfolio_data_error"] = None
    state["portfolio_revision"] += 1
    state["portfolio_dirty"] = False


def invalidate_portfolio_inputs(state: MutableMapping) -> None:
    state["portfolio_data"] = None
    state["portfolio_results"] = {}
    state["portfolio_data_error"] = None
    state["portfolio_dirty"] = True


def invalidate_optimization_inputs(state: MutableMapping) -> None:
    """Drop optimized outputs without disturbing a current-portfolio simulation."""
    results = state["portfolio_results"]
    if results.get("simulation_selection", "Current") != "Current":
        select_simulation_allocation(state, "Current")
    elif results.get("simulation", {}).get("source", "Current") != "Current":
        clear_simulation(state)
    results.pop("optimization", None)
    results.pop("optimization_error", None)
    results["optimization_dirty"] = True


def invalidate_simulation_inputs(state: MutableMapping) -> None:
    clear_simulation(state)
    state["portfolio_results"]["simulation_dirty"] = True


def store_portfolio_data(state: MutableMapping, data: PortfolioData) -> None:
    if state.get("portfolio_definition") != data.definition:
        raise ValueError("Data belongs to a different portfolio. Save and check the current portfolio again.")
    state["portfolio_data"] = data
    state["portfolio_results"] = {}
    state["portfolio_data_error"] = None


def simulation_allocations(state: MutableMapping) -> dict[str, pd.Series]:
    """Available saved allocations; optimized weights never overwrite the current portfolio."""
    definition = state.get("portfolio_definition")
    if definition is None:
        return {}
    allocations = {"Current": pd.Series({h.ticker: h.weight for h in definition.holdings})}
    optimization = state.get("portfolio_results", {}).get("optimization")
    if optimization:
        for name in ("Minimum volatility", "Maximum Sharpe"):
            if name in optimization["solution"].weights:
                allocations[name] = optimization["solution"].weights[name].copy()
    return allocations


def clear_simulation(state: MutableMapping) -> None:
    results = state["portfolio_results"]
    results.pop("simulation", None)
    results.pop("simulation_error", None)


def select_simulation_allocation(state: MutableMapping, name: str) -> None:
    if name not in simulation_allocations(state):
        raise ValueError("That allocation is no longer available. Optimize again or use the current portfolio.")
    clear_simulation(state)
    results = state["portfolio_results"]
    results["simulation_selection"] = name
    # New widget identity lets handoff replace a prior page-local selection safely.
    results["simulation_selection_revision"] = results.get("simulation_selection_revision", 0) + 1
