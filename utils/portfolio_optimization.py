"""Long-only, fully invested mean-variance optimization with checked solutions.

SLSQP solves convex quadratic programs for minimum variance, frontier targets,
and the positive-excess-return maximum-Sharpe transformation. A solver result
is never accepted without independent checks on its values and constraints.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


MIN_OBSERVATIONS = 60
FEASIBILITY_TOL = 1e-7
RETURN_TOL = 1e-10
VOLATILITY_TOL = 1e-8


class OptimizationError(ValueError):
    """A solver failed or returned an invalid candidate."""


@dataclass
class Estimates:
    returns: pd.Series
    covariance: pd.DataFrame
    observations: int
    start: pd.Timestamp
    end: pd.Timestamp
    shrinkage: float


@dataclass
class OptimizationResult:
    weights: pd.DataFrame
    comparison: pd.DataFrame
    frontier: pd.DataFrame
    frontier_weights: pd.DataFrame
    max_sharpe_reason: str | None
    notices: tuple[str, ...]


def estimate_parameters(returns: pd.DataFrame) -> Estimates:
    if len(returns) < MIN_OBSERVATIONS:
        raise ValueError(f"Optimization requires at least {MIN_OBSERVATIONS} common daily returns; only {len(returns)} are available.")
    if returns.shape[1] == 0 or returns.columns.has_duplicates:
        raise ValueError("Return columns must contain unique tickers.")
    if not isinstance(returns.index, pd.DatetimeIndex) or returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("Returns must have unique, increasing dates.")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Daily returns must be finite, complete, and greater than -100%.")
    fitted = LedoitWolf().fit(values)
    return Estimates(
        returns.mean() * 252,
        pd.DataFrame(fitted.covariance_ * 252, index=returns.columns, columns=returns.columns),
        len(returns), returns.index[0], returns.index[-1], float(fitted.shrinkage_),
    )


def weight_caps(tickers: pd.Index, maximum: float, overrides: dict[str, float] | None = None) -> pd.Series:
    """Global cap and optional tighter ticker caps, all expressed as fractions."""
    if not np.isfinite(maximum) or not 0 < maximum <= 1:
        raise ValueError("Maximum weight must be greater than 0% and at most 100%.")
    caps = pd.Series(maximum, index=tickers, dtype=float)
    for ticker, cap in (overrides or {}).items():
        if ticker not in caps.index:
            raise ValueError(f"Unknown ticker in constraints: {ticker}.")
        if not np.isfinite(cap) or not 0 <= cap <= maximum:
            raise ValueError(f"{ticker}: individual cap must be between 0% and the global maximum ({maximum:.2%}).")
        caps.loc[ticker] = cap
    if caps.sum() < 1 - 1e-10:
        raise ValueError(f"Infeasible caps: together they allow only {caps.sum():.2%}. They must allow at least 100% for a fully invested portfolio.")
    return caps


def _check_weights(weights: np.ndarray, caps: np.ndarray) -> np.ndarray:
    candidate = np.asarray(weights, dtype=float)
    if candidate.shape != caps.shape or not np.isfinite(candidate).all():
        raise OptimizationError("Solver returned non-finite or incorrectly shaped weights.")
    if abs(candidate.sum() - 1) > FEASIBILITY_TOL or (candidate < -FEASIBILITY_TOL).any() or (candidate > caps + FEASIBILITY_TOL).any():
        raise OptimizationError("Solver returned weights that violate the allocation constraints.")
    # Clean floating-point bound noise only, then recheck the cleaned allocation.
    candidate = np.clip(candidate, 0, caps)
    candidate /= candidate.sum()
    if (candidate > caps + FEASIBILITY_TOL).any():
        raise OptimizationError("Rounded solver weights violate the allocation constraints.")
    return candidate


def _solve(objective, gradient, initial, bounds, constraints, label):
    try:
        result = minimize(
            objective, initial, jac=gradient, method="SLSQP", bounds=bounds,
            constraints=constraints, options={"ftol": 1e-11, "maxiter": 1000},
        )
    except Exception as exc:
        raise OptimizationError(f"{label} solver failed: {exc}") from exc
    if not result.success:
        raise OptimizationError(f"{label} solver did not converge: {result.message}")
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise OptimizationError(f"{label} solver returned non-finite values.")
    return np.asarray(result.x, dtype=float)


def _highest_return(mu: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Exact linear optimum for non-negative weights with upper bounds."""
    weights = np.zeros_like(mu)
    remaining = 1.0
    for position in np.argsort(-mu, kind="stable"):
        weights[position] = min(caps[position], remaining)
        remaining -= weights[position]
        if remaining <= 0:
            break
    return _check_weights(weights, caps)


def _minimum_variance(cov, caps, initial, mu=None, target=None):
    scale = max(float(np.max(np.abs(cov))), 1e-12)
    scaled_cov = cov / scale
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(len(w))}]
    if target is not None:
        return_scale = max(float(np.max(np.abs(mu))), 1e-6)
        constraints.append({
            "type": "ineq", "fun": lambda w: (mu @ w - target) / return_scale,
            "jac": lambda w: mu / return_scale,
        })
    candidate = _solve(
        lambda w: w @ scaled_cov @ w, lambda w: 2 * scaled_cov @ w,
        initial, list(zip(np.zeros(len(caps)), caps)), constraints, "Minimum-variance",
    )
    candidate = _check_weights(candidate, caps)
    if target is not None and mu @ candidate < target - FEASIBILITY_TOL * max(1, abs(target)):
        raise OptimizationError("Frontier solution does not meet its target return.")
    return candidate


def _maximum_sharpe(mu, cov, caps, rf, highest):
    excess = mu - rf
    if excess @ highest <= RETURN_TOL:
        return None, "No feasible allocation has a positive estimated return above the risk-free assumption. A meaningful positive-excess-return maximum Sharpe ratio is unavailable."
    # Let y = w / (e'w), after scaling e for numerical conditioning.
    # Minimize y'Sigma y subject to e'y=1 and 0<=y_i<=cap_i*sum(y).
    # This is a convex QP, avoiding a local search on a ratio objective.
    excess = excess / max(float(np.max(np.abs(excess))), 1e-12)
    scaled_cov = cov / max(float(np.max(np.abs(cov))), 1e-12)
    initial = highest / (excess @ highest)
    cap_jacobian = caps[:, None] * np.ones((len(caps), len(caps))) - np.eye(len(caps))
    constraints = [
        {"type": "eq", "fun": lambda y: excess @ y - 1, "jac": lambda y: excess},
        {"type": "ineq", "fun": lambda y: caps * y.sum() - y, "jac": lambda y: cap_jacobian},
    ]
    y = _solve(
        lambda y: y @ scaled_cov @ y, lambda y: 2 * scaled_cov @ y,
        initial, [(0, None)] * len(caps), constraints, "Maximum-Sharpe",
    )
    if y.shape != caps.shape or y.sum() <= 0 or abs(excess @ y - 1) > FEASIBILITY_TOL:
        raise OptimizationError("Maximum-Sharpe solver returned an invalid transformed solution.")
    weights = _check_weights(y / y.sum(), caps)
    volatility = np.sqrt(max(float(weights @ cov @ weights), 0))
    if volatility <= VOLATILITY_TOL:
        return None, "A feasible positive-excess-return allocation has effectively zero estimated volatility. Its Sharpe ratio is undefined; no finite maximum-Sharpe allocation is reported."
    if (mu - rf) @ weights <= RETURN_TOL:
        raise OptimizationError("Maximum-Sharpe solution has no positive estimated excess return.")
    return weights, None


def optimize_portfolio(
    estimates: Estimates, current: pd.Series, caps: pd.Series,
    risk_free_rate: float = 0.0, frontier_points: int = 30,
) -> OptimizationResult:
    tickers = estimates.returns.index
    if tickers.has_duplicates or len(tickers) == 0:
        raise ValueError("Expected returns must have unique tickers.")
    for values in [current, caps]:
        if values.index.has_duplicates or set(values.index) != set(tickers):
            raise ValueError("Weights and constraints must match the estimated tickers.")
    if set(estimates.covariance.index) != set(tickers) or set(estimates.covariance.columns) != set(tickers):
        raise ValueError("Covariance tickers must match expected returns.")
    mu = estimates.returns.to_numpy(dtype=float)
    cov = estimates.covariance.loc[tickers, tickers].to_numpy(dtype=float)
    upper = caps.reindex(tickers).to_numpy(dtype=float)
    if not np.isfinite(mu).all() or not np.isfinite(cov).all() or not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError("Return and covariance estimates must be finite and covariance symmetric.")
    if np.linalg.eigvalsh(cov).min() < -1e-10:
        raise ValueError("Covariance must be positive semidefinite.")
    if not np.isfinite(upper).all() or (upper < 0).any() or (upper > 1).any() or upper.sum() < 1 - 1e-10:
        raise ValueError("Infeasible weight caps: values must be between 0% and 100% and sum to at least 100%.")
    if not np.isfinite(risk_free_rate) or not -1 < risk_free_rate <= 1:
        raise ValueError("Risk-free rate must be greater than -100% and at most 100%.")
    if not isinstance(frontier_points, int) or frontier_points < 0:
        raise ValueError("Frontier point count must be a non-negative integer.")
    current_values = _check_weights(current.reindex(tickers).to_numpy(dtype=float), np.ones(len(tickers)))
    # Same daily excess-return convention as historical Sharpe, annualized arithmetically.
    rf = 252 * np.expm1(np.log1p(risk_free_rate) / 252)
    highest = _highest_return(mu, upper)
    initial = upper / upper.sum()
    fixed = abs(upper.sum() - 1) <= 1e-10 or len(tickers) == 1
    minimum = _check_weights(initial, upper) if fixed else _minimum_variance(cov, upper, initial)
    candidates = {"Current": current_values, "Minimum volatility": minimum}
    notices = []
    reason = None
    try:
        if fixed:
            vol = np.sqrt(max(float(minimum @ cov @ minimum), 0))
            if mu @ minimum - rf <= RETURN_TOL:
                maximum, reason = None, "The only feasible allocation has no positive estimated excess return; a meaningful maximum Sharpe ratio is unavailable."
            elif vol <= VOLATILITY_TOL:
                maximum, reason = None, "The only feasible allocation has zero estimated volatility, so its Sharpe ratio is undefined."
            else:
                maximum = minimum.copy()
        else:
            maximum, reason = _maximum_sharpe(mu, cov, upper, rf, highest)
        if maximum is not None:
            candidates["Maximum Sharpe"] = maximum
    except OptimizationError as exc:
        reason = str(exc)

    frontier_rows, frontier_allocations = [], []
    low, high = float(mu @ minimum), float(mu @ highest)
    targets = np.array([low]) if fixed or high - low <= RETURN_TOL else np.linspace(low, high, max(frontier_points, 2))
    if frontier_points == 0:
        targets = np.array([])
    for position, target in enumerate(targets):
        try:
            if position == 0:
                allocation = minimum
            else:
                fraction = (target - low) / (high - low)
                seed = (1 - fraction) * minimum + fraction * highest
                allocation = _minimum_variance(cov, upper, seed, mu, target)
            frontier_rows.append({"Estimated return": float(mu @ allocation),
                                  "Estimated volatility": np.sqrt(max(float(allocation @ cov @ allocation), 0))})
            frontier_allocations.append(allocation)
        except OptimizationError as exc:
            # Retain an explicit gap: plotting must not join across a failed solve.
            frontier_rows.append({"Estimated return": np.nan, "Estimated volatility": np.nan})
            frontier_allocations.append(np.full(len(tickers), np.nan))
            notices.append(f"Frontier target {target:.2%} unavailable: {exc}")
    comparison = []
    for name, allocation in candidates.items():
        ret = float(mu @ allocation)
        volatility = np.sqrt(max(float(allocation @ cov @ allocation), 0))
        comparison.append({
            "Allocation": name, "Estimated return": ret, "Estimated volatility": volatility,
            "Estimated Sharpe ratio": (ret - rf) / volatility if volatility > VOLATILITY_TOL else np.nan,
            "Within caps": bool((allocation <= upper + FEASIBILITY_TOL).all()),
        })
    if fixed:
        notices.append("These caps permit only one allocation; the efficient frontier is a single point.")
    elif high - low <= RETURN_TOL:
        notices.append("The efficient frontier collapses to a single point for these return estimates and constraints.")
    return OptimizationResult(
        pd.DataFrame(candidates, index=tickers), pd.DataFrame(comparison).set_index("Allocation"),
        pd.DataFrame(frontier_rows, columns=["Estimated return", "Estimated volatility"]),
        pd.DataFrame(frontier_allocations, columns=tickers), reason, tuple(notices),
    )
