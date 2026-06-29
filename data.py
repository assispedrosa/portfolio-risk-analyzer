"""
Data + reference layer.

Provides a price panel (synthetic or live), plus the reference data the risk
engine needs for *factor-based* stress testing:

  * a curated asset CATALOG (so the UI offers a pick-list instead of a fragile
    free-text box, and so every asset maps to a known risk factor);
  * a factor map for the synthetic sample book;
  * realized HISTORICAL_EVENTS expressed as per-factor shocks.

Stress shocks are applied per risk factor, not as a single market-wide number,
because assets do not all move together: in 2008 equities fell ~50% while gold
*rose*. Mapping each asset to a factor is what lets the engine reflect that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Risk factors a position can load on. An asset maps to a factor plus a
# sensitivity (its beta to that factor's shock).
FACTORS = ["Equity", "Credit", "Gold", "Crypto"]


# --- Synthetic sample book ------------------------------------------------

SAMPLE_ASSETS = {
    "US Equity":   dict(mu=0.09, sigma=0.18, factor="Equity", sens=1.0,  since=1960),
    "Intl Equity": dict(mu=0.07, sigma=0.20, factor="Equity", sens=1.1,  since=1970),
    "Corp Bonds":  dict(mu=0.04, sigma=0.07, factor="Credit", sens=1.0,  since=1970),
    "Gold":        dict(mu=0.05, sigma=0.15, factor="Gold",   sens=1.0,  since=1970),
    "Crypto":      dict(mu=0.25, sigma=0.65, factor="Crypto", sens=1.0,  since=2013),
}

_SAMPLE_CORR = np.array(
    [
        # USEq  Intl  Bond  Gold  Crypto
        [1.00, 0.82, 0.20, 0.05, 0.45],
        [0.82, 1.00, 0.18, 0.10, 0.40],
        [0.20, 0.18, 1.00, 0.30, 0.05],
        [0.05, 0.10, 0.30, 1.00, 0.15],
        [0.45, 0.40, 0.05, 0.15, 1.00],
    ]
)


def synthetic_prices(
    n_days: int = 3 * TRADING_DAYS, start_price: float = 100.0, seed: int | None = 7
) -> pd.DataFrame:
    """Correlated geometric Brownian motion for SAMPLE_ASSETS (deterministic)."""
    rng = np.random.default_rng(seed)
    names = list(SAMPLE_ASSETS)
    mu = np.array([SAMPLE_ASSETS[a]["mu"] for a in names])
    sigma = np.array([SAMPLE_ASSETS[a]["sigma"] for a in names])

    daily_mu = mu / TRADING_DAYS
    daily_sigma = sigma / np.sqrt(TRADING_DAYS)
    chol = np.linalg.cholesky(_SAMPLE_CORR)

    shocks = rng.standard_normal((n_days, len(names))) @ chol.T
    daily_ret = daily_mu + daily_sigma * shocks
    price_path = start_price * np.cumprod(1.0 + daily_ret, axis=0)

    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame(price_path, index=idx, columns=names)


# --- Live data: curated catalog ------------------------------------------

# ticker -> (display label, risk factor, sensitivity to factor, inception year)
# Inception lets the stress engine flag positions that did not exist during an
# event and were therefore mapped to a proxy factor. For tickers NOT in this
# catalog, inception is discovered automatically from price history (see
# live_prices) and the factor is inferred, so the catalog need not be complete.
CATALOG: dict[str, dict] = {
    # --- US equity (broad / style / size) ---
    "SPY":  dict(label="SPY — S&P 500",            factor="Equity", sens=1.0,  since=1993),
    "VOO":  dict(label="VOO — S&P 500",            factor="Equity", sens=1.0,  since=2010),
    "QQQ":  dict(label="QQQ — Nasdaq 100",         factor="Equity", sens=1.2,  since=1999),
    "DIA":  dict(label="DIA — Dow Jones 30",       factor="Equity", sens=0.9,  since=1998),
    "IWM":  dict(label="IWM — US Small Caps",      factor="Equity", sens=1.3,  since=2000),
    "VTV":  dict(label="VTV — US Value",           factor="Equity", sens=0.9,  since=2004),
    "VUG":  dict(label="VUG — US Growth",          factor="Equity", sens=1.15, since=2004),
    # --- US sectors ---
    "XLK":  dict(label="XLK — Technology",         factor="Equity", sens=1.25, since=1998),
    "XLF":  dict(label="XLF — Financials",         factor="Equity", sens=1.3,  since=1998),
    "XLE":  dict(label="XLE — Energy",             factor="Equity", sens=1.2,  since=1998),
    "XLV":  dict(label="XLV — Health Care",        factor="Equity", sens=0.8,  since=1998),
    "XLU":  dict(label="XLU — Utilities",          factor="Equity", sens=0.6,  since=1998),
    "XLP":  dict(label="XLP — Consumer Staples",   factor="Equity", sens=0.6,  since=1998),
    # --- Single names ---
    "AAPL": dict(label="AAPL — Apple",             factor="Equity", sens=1.2,  since=1980),
    "MSFT": dict(label="MSFT — Microsoft",         factor="Equity", sens=1.1,  since=1986),
    "NVDA": dict(label="NVDA — Nvidia",            factor="Equity", sens=1.7,  since=1999),
    "AMZN": dict(label="AMZN — Amazon",            factor="Equity", sens=1.3,  since=1997),
    "JPM":  dict(label="JPM — JPMorgan",           factor="Equity", sens=1.3,  since=1980),
    # --- International / regional ---
    "EFA":  dict(label="EFA — Developed ex-US",    factor="Equity", sens=1.1,  since=2001),
    "EEM":  dict(label="EEM — Emerging Markets",   factor="Equity", sens=1.3,  since=2003),
    "EWZ":  dict(label="EWZ — Brazil",             factor="Equity", sens=1.5,  since=2000),
    "FXI":  dict(label="FXI — China Large Cap",    factor="Equity", sens=1.4,  since=2004),
    # --- Brazil (B3, .SA) ---
    "PETR4.SA": dict(label="PETR4.SA — Petrobras", factor="Equity", sens=1.4, since=2000),
    "VALE3.SA": dict(label="VALE3.SA — Vale",      factor="Equity", sens=1.4, since=2000),
    "ITUB4.SA": dict(label="ITUB4.SA — Itaú",      factor="Equity", sens=1.2, since=2000),
    "BBDC4.SA": dict(label="BBDC4.SA — Bradesco",  factor="Equity", sens=1.2, since=2000),
    "BOVA11.SA":dict(label="BOVA11.SA — Ibovespa", factor="Equity", sens=1.0, since=2008),
    # --- Credit / rates ---
    "LQD":  dict(label="LQD — IG Corporate Bonds", factor="Credit", sens=1.0,  since=2002),
    "HYG":  dict(label="HYG — High-Yield Bonds",   factor="Credit", sens=1.6,  since=2007),
    "AGG":  dict(label="AGG — US Aggregate Bonds", factor="Credit", sens=0.7,  since=2003),
    "TLT":  dict(label="TLT — 20Y+ Treasuries",    factor="Credit", sens=-0.6, since=2002),
    "IEF":  dict(label="IEF — 7-10Y Treasuries",   factor="Credit", sens=-0.4, since=2002),
    # --- Gold / commodities ---
    "GLD":  dict(label="GLD — Gold",               factor="Gold",   sens=1.0,  since=2004),
    "IAU":  dict(label="IAU — Gold",               factor="Gold",   sens=1.0,  since=2005),
    "SLV":  dict(label="SLV — Silver",             factor="Gold",   sens=1.3,  since=2006),
    "DBC":  dict(label="DBC — Broad Commodities",  factor="Gold",   sens=0.8,  since=2006),
    # --- Crypto ---
    "BTC-USD": dict(label="BTC-USD — Bitcoin",     factor="Crypto", sens=1.0,  since=2014),
    "ETH-USD": dict(label="ETH-USD — Ethereum",    factor="Crypto", sens=1.2,  since=2017),
    "SOL-USD": dict(label="SOL-USD — Solana",      factor="Crypto", sens=1.5,  since=2020),
}

DEFAULT_LIVE_TICKERS = ["SPY", "EFA", "LQD", "GLD", "BTC-USD"]


def _infer_factor(ticker: str) -> str:
    """Best-effort factor for a ticker not in the catalog (heuristic, honest)."""
    if ticker.upper().endswith("-USD"):
        return "Crypto"
    return "Equity"  # default: treat unknowns as an equity exposure


def asset_factor_map(
    columns: list[str], inception: dict[str, int] | None = None
) -> dict[str, dict]:
    """
    Build {asset: {factor, sens, since, known}} for the panel's assets.

    Known assets (synthetic sample or CATALOG) use their curated factor/sens/
    inception. Unknown live tickers get an inferred factor, unit sensitivity,
    and their REAL inception year discovered from price history (passed in via
    `inception`) — so they are only flagged as a stress proxy when they truly
    did not exist during the event, not by default.
    """
    inception = inception or {}
    out = {}
    for c in columns:
        if c in SAMPLE_ASSETS:
            a = SAMPLE_ASSETS[c]
            out[c] = dict(factor=a["factor"], sens=a["sens"], since=a["since"], known=True)
        elif c in CATALOG:
            a = CATALOG[c]
            out[c] = dict(factor=a["factor"], sens=a["sens"], since=a["since"], known=True)
        else:
            out[c] = dict(
                factor=_infer_factor(c),
                sens=1.0,
                since=inception.get(c),  # real inception, or None if unknown
                known=False,
            )
    return out


def live_prices(
    tickers: list[str], lookback_years: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, int]]:
    """
    Adjusted close prices from Yahoo Finance.

    Returns (window_prices, full_prices, failed_tickers, inception_year):
      * window_prices: last `lookback_years`, used for vol/VaR/correlation so
        those metrics reflect the *current* regime;
      * full_prices: the entire common history, used for the worst-realized-
        window stress so it can reach back through 2008, 2020, etc.

    Fetches FULL history once (also giving each ticker's real inception year for
    accurate stress proxying). Robust to bad tickers: a symbol with no data
    becomes an all-NaN column, dropped BEFORE rows are aligned — otherwise one
    typo wipes the whole panel.
    """
    import yfinance as yf

    raw = yf.download(tickers, period="max", auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = pd.DataFrame(close)
    if len(tickers) == 1:
        close.columns = tickers

    close = close.dropna(axis=1, how="all")  # drop invalid/delisted tickers first
    failed = [t for t in tickers if t not in close.columns]

    # Real inception year per surviving ticker, from its first valid observation.
    inception = {c: int(close[c].first_valid_index().year) for c in close.columns}

    # Full common history (portfolio only "exists" once every constituent does).
    full = close.ffill().dropna(how="any")
    if full.empty:
        raise ValueError("No usable price data for any requested ticker.")

    # Recent risk window.
    cutoff = full.index.max() - pd.DateOffset(years=lookback_years)
    window = full.loc[full.index >= cutoff]
    return window, full, failed, inception


# --- Historical stress events --------------------------------------------

# Realized, approximate peak-to-trough shocks by risk factor during each event.
# These are illustrative but grounded in what actually happened; the point is
# that the shocks differ across factors and gold can rally while equity sinks.
HISTORICAL_EVENTS: dict[str, dict] = {
    "GFC 2008 (Sep'08–Mar'09)": dict(
        year=2008,
        shocks={"Equity": -0.50, "Credit": -0.18, "Gold": 0.06, "Crypto": -0.75},
    ),
    "Euro debt crisis (2011)": dict(
        year=2011,
        shocks={"Equity": -0.19, "Credit": -0.08, "Gold": 0.11, "Crypto": -0.40},
    ),
    "COVID crash (Feb–Mar 2020)": dict(
        year=2020,
        shocks={"Equity": -0.34, "Credit": -0.12, "Gold": -0.03, "Crypto": -0.50},
    ),
    "Inflation/rates shock (2022)": dict(
        year=2022,
        shocks={"Equity": -0.25, "Credit": -0.17, "Gold": -0.03, "Crypto": -0.65},
    ),
}
