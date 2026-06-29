"""
Risk engine: portfolio risk metrics computed from a returns matrix.

Kept deliberately framework-free (only numpy/pandas/scipy) so it can be unit
tested, reused in a notebook, or wired behind an API without dragging the UI
along. The Streamlit app in app.py is just one consumer of these functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a price panel (columns = assets)."""
    return prices.pct_change().dropna(how="all")


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Daily return series of the weighted portfolio."""
    weights = np.asarray(weights, dtype=float)
    return returns.to_numpy() @ weights


def annualized_volatility(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Volatility scaled to a yearly figure (sqrt-of-time rule)."""
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def annualized_return(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Geometric annualized return."""
    growth = (1.0 + returns).prod()
    n = returns.shape[0]
    return growth ** (TRADING_DAYS / n) - 1.0


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def historical_var(
    pnl: pd.Series, confidence: float = 0.95, horizon_days: int = 1
) -> float:
    """
    Historical VaR: empirical quantile of the loss distribution.

    Returned as a positive number representing the loss (e.g. 0.031 = 3.1%).
    No distributional assumption — it simply reads the tail off realized data.
    """
    q = np.percentile(pnl, (1.0 - confidence) * 100.0)
    return float(-q * np.sqrt(horizon_days))


def parametric_var(
    pnl: pd.Series, confidence: float = 0.95, horizon_days: int = 1
) -> float:
    """
    Variance-covariance (Gaussian) VaR. Fast and smooth, but understates tail
    risk when returns are fat-tailed — which is exactly why we show it side by
    side with the historical and Monte Carlo numbers.
    """
    mu, sigma = pnl.mean(), pnl.std(ddof=1)
    z = stats.norm.ppf(1.0 - confidence)
    daily = -(mu + z * sigma)
    return float(daily * np.sqrt(horizon_days))


def monte_carlo_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.95,
    horizon_days: int = 1,
    n_sims: int = 50_000,
    seed: int | None = 42,
) -> float:
    """
    Monte Carlo VaR under a multivariate-normal assumption calibrated to the
    sample mean vector and covariance matrix. Captures cross-asset correlation
    explicitly, which the single-series parametric VaR cannot.
    """
    rng = np.random.default_rng(seed)
    mu = returns.mean().to_numpy()
    cov = returns.cov().to_numpy()
    sims = rng.multivariate_normal(mu, cov, size=n_sims)
    port = sims @ np.asarray(weights, dtype=float)
    horizon = port * np.sqrt(horizon_days)
    q = np.percentile(horizon, (1.0 - confidence) * 100.0)
    return float(-q)


def expected_shortfall(
    pnl: pd.Series, confidence: float = 0.95, horizon_days: int = 1
) -> float:
    """
    Conditional VaR / Expected Shortfall: average loss *beyond* the VaR level.
    Answers "when it breaches, how bad on average?" — the question VaR ignores.
    """
    threshold = np.percentile(pnl, (1.0 - confidence) * 100.0)
    tail = pnl[pnl <= threshold]
    if len(tail) == 0:
        return historical_var(pnl, confidence, horizon_days)
    return float(-tail.mean() * np.sqrt(horizon_days))


def max_drawdown(pnl: pd.Series) -> float:
    """Worst peak-to-trough decline of the cumulative return path."""
    curve = (1.0 + pnl).cumprod()
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    return float(drawdown.min())


# --- Stress testing -------------------------------------------------------


def historical_event_stress(
    weights: dict[str, float],
    asset_factors: dict[str, dict],
    events: dict[str, dict],
    portfolio_value: float,
) -> pd.DataFrame:
    """
    Factor-based historical stress test.

    For each event, every position is shocked by its *factor's* realized move
    during that event, scaled by the asset's sensitivity to the factor. The
    portfolio impact is the weighted sum — so a long-gold position can offset
    an equity drawdown, and assets are not forced to move together.

    `weights` and `asset_factors` are keyed by asset. Any position whose
    inception year postdates the event is flagged as a proxy (its real history
    didn't exist then, so the factor move is an approximation).
    """
    rows = []
    for ev_name, ev in events.items():
        shocks = ev["shocks"]
        impact = 0.0
        proxied = []
        for asset, w in weights.items():
            meta = asset_factors[asset]
            factor_shock = shocks.get(meta["factor"], 0.0)
            impact += w * meta["sens"] * factor_shock
            # Flag as proxy only when we KNOW the asset postdates the event.
            # since=None means inception unknown -> don't assert proxy.
            since = meta.get("since")
            if since is not None and since > ev["year"]:
                proxied.append(asset)
        rows.append(
            {
                "Scenario": ev_name,
                "Portfolio impact": impact,
                "P&L": portfolio_value * impact,
                "Resulting value": portfolio_value * (1.0 + impact),
                "Proxied assets": ", ".join(proxied) if proxied else "—",
            }
        )
    return pd.DataFrame(rows)


def worst_historical_window(
    pnl: pd.Series, windows: tuple[int, ...] = (1, 5, 21)
) -> pd.DataFrame:
    """
    Empirical stress: the worst realized cumulative return over rolling windows
    of the portfolio's *own* history, with the dates it happened. No assumption
    at all — it's the actual worst stretch the book lived through in-sample.
    """
    rows = []
    for w in windows:
        rolling = (1.0 + pnl).rolling(w).apply(np.prod, raw=True) - 1.0
        worst = rolling.min()
        end = rolling.idxmin()
        start = pnl.index[max(0, pnl.index.get_loc(end) - w + 1)]
        rows.append(
            {
                "Window": f"{w}-day",
                "Worst return": float(worst),
                "From": start.date(),
                "To": end.date(),
            }
        )
    return pd.DataFrame(rows)


def portfolio_beta(returns: pd.DataFrame, weights: np.ndarray, market: pd.Series) -> float:
    """Beta of the weighted portfolio against a market return series."""
    port = pd.Series(portfolio_returns(returns, weights), index=returns.index)
    aligned = pd.concat([port, market], axis=1).dropna()
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(cov[0, 1] / cov[1, 1])
