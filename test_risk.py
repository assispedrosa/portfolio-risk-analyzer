"""
Smoke + sanity tests for the risk engine. Run with: python -m pytest -q
(or just `python test_risk.py` for a quick manual check).

These aren't exhaustive — they pin down the properties that actually matter:
VaR is non-negative and grows with confidence, ES is at least as large as VaR,
and the metrics line up with closed-form expectations on clean inputs.
"""

import numpy as np
import pandas as pd

import data
import risk


def _sample_returns(seed=1):
    prices = data.synthetic_prices(seed=seed)
    return risk.daily_returns(prices)


def test_var_is_non_negative_and_monotonic_in_confidence():
    rets = _sample_returns()
    w = np.ones(rets.shape[1]) / rets.shape[1]
    pnl = pd.Series(risk.portfolio_returns(rets, w), index=rets.index)
    v90 = risk.historical_var(pnl, 0.90)
    v99 = risk.historical_var(pnl, 0.99)
    assert v90 >= 0 and v99 >= 0
    assert v99 >= v90, "higher confidence must imply a larger VaR"


def test_expected_shortfall_dominates_var():
    rets = _sample_returns()
    w = np.ones(rets.shape[1]) / rets.shape[1]
    pnl = pd.Series(risk.portfolio_returns(rets, w), index=rets.index)
    var = risk.historical_var(pnl, 0.95)
    es = risk.expected_shortfall(pnl, 0.95)
    assert es >= var, "ES is the mean beyond VaR, so it can't be smaller"


def test_parametric_var_matches_gaussian_quantile():
    # On synthetic normal data the parametric VaR should track 1.645*sigma at 95%.
    rng = np.random.default_rng(0)
    pnl = pd.Series(rng.normal(0, 0.01, 100_000))
    v = risk.parametric_var(pnl, 0.95)
    assert abs(v - 1.645 * 0.01) < 5e-4


def test_horizon_scales_with_sqrt_time():
    rets = _sample_returns()
    w = np.ones(rets.shape[1]) / rets.shape[1]
    pnl = pd.Series(risk.portfolio_returns(rets, w), index=rets.index)
    v1 = risk.parametric_var(pnl, 0.95, horizon_days=1)
    v4 = risk.parametric_var(pnl, 0.95, horizon_days=4)
    assert abs(v4 / v1 - 2.0) < 0.05, "4-day VaR should be ~2x the 1-day VaR"


if __name__ == "__main__":
    test_var_is_non_negative_and_monotonic_in_confidence()
    test_expected_shortfall_dominates_var()
    test_parametric_var_matches_gaussian_quantile()
    test_horizon_scales_with_sqrt_time()
    print("All sanity checks passed.")
