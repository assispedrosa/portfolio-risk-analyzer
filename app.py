"""
Portfolio Risk Analyzer — Streamlit front end.

A focused quant-risk dashboard: given a portfolio, it reports volatility, a
three-method VaR comparison, Expected Shortfall, the correlation structure, and
named stress scenarios. Runs on deterministic synthetic data out of the box;
flip a switch to pull live prices from Yahoo Finance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data
import risk

st.set_page_config(page_title="Portfolio Risk Analyzer", layout="wide", page_icon="📉")

# --- Sidebar: portfolio definition ---------------------------------------
st.sidebar.title("Portfolio")

source = st.sidebar.radio(
    "Data source",
    ["Synthetic (offline)", "Live (Yahoo Finance)"],
    help="Synthetic data is deterministic and always available. Live mode pulls "
    "real adjusted prices for the tickers you enter.",
)

if source.startswith("Live"):
    # Pick-list from the curated catalog (valid tickers only), plus an optional
    # advanced box for anything not in the list.
    labels = {data.CATALOG[t]["label"]: t for t in data.CATALOG}
    default_labels = [data.CATALOG[t]["label"] for t in data.DEFAULT_LIVE_TICKERS]
    chosen = st.sidebar.multiselect("Assets", list(labels), default=default_labels)
    tickers = [labels[c] for c in chosen]

    extra = st.sidebar.text_input(
        "Add other tickers (advanced, comma separated)", "",
        help="Yahoo Finance symbols, e.g. PETR4.SA, AAPL. Unknown ones are "
        "skipped and reported, not allowed to break the portfolio.",
    )
    tickers += [t.strip().upper() for t in extra.split(",") if t.strip()]

    if not tickers:
        st.sidebar.warning("Pick at least one asset. Using synthetic data meanwhile.")
        prices_full = data.synthetic_prices(n_days=10 * 252)
        prices = prices_full.tail(3 * 252)
        live_inception = {}
    else:
        try:
            prices, prices_full, failed, live_inception = data.live_prices(tickers)
            if failed:
                st.sidebar.warning("Skipped (no data): " + ", ".join(failed))
            st.sidebar.success(
                f"Loaded {prices.shape[0]} days (risk window) / "
                f"{prices_full.shape[0]} days (full history) for "
                f"{len(prices.columns)} assets."
            )
        except Exception as exc:  # noqa: BLE001 - demo-friendly fallback
            st.sidebar.error(f"Live fetch failed ({exc}). Falling back to synthetic data.")
            prices_full = data.synthetic_prices(n_days=10 * 252)
            prices = prices_full.tail(3 * 252)
            live_inception = {}
else:
    prices_full = data.synthetic_prices(n_days=10 * 252)
    prices = prices_full.tail(3 * 252)
    live_inception = {}

assets = list(prices.columns)

st.sidebar.subheader("Weights")
st.sidebar.caption("Weights are normalized to sum to 100%.")
raw_weights = {a: st.sidebar.slider(a, 0.0, 1.0, round(1.0 / len(assets), 2), 0.05) for a in assets}
weight_vec = np.array([raw_weights[a] for a in assets], dtype=float)
if weight_vec.sum() == 0:
    weight_vec = np.ones(len(assets))
weight_vec = weight_vec / weight_vec.sum()

confidence = st.sidebar.select_slider(
    "Confidence level", options=[0.90, 0.95, 0.975, 0.99], value=0.95
)
horizon = st.sidebar.slider("VaR horizon (days)", 1, 20, 1)
portfolio_value = st.sidebar.number_input(
    "Portfolio value ($)", min_value=1_000, value=1_000_000, step=10_000
)

# --- Compute -------------------------------------------------------------
returns = risk.daily_returns(prices)
port_ret = pd.Series(risk.portfolio_returns(returns, weight_vec), index=returns.index)

# Per-asset weights + factor map drive the factor-based stress test.
weights_by_asset = {a: float(w) for a, w in zip(assets, weight_vec)}
factor_map = data.asset_factor_map(assets, live_inception)

# Portfolio return over the FULL history, for the worst-realized-window stress.
full_returns = risk.daily_returns(prices_full)
port_ret_full = pd.Series(
    risk.portfolio_returns(full_returns, weight_vec), index=full_returns.index
)
full_start, full_end = prices_full.index[0].date(), prices_full.index[-1].date()

span_start, span_end = prices.index[0].date(), prices.index[-1].date()
n_years = (prices.index[-1] - prices.index[0]).days / 365.25

ann_vol = risk.annualized_volatility(port_ret)
ann_ret = risk.annualized_return(port_ret)
hist_var = risk.historical_var(port_ret, confidence, horizon)
param_var = risk.parametric_var(port_ret, confidence, horizon)
mc_var = risk.monte_carlo_var(returns, weight_vec, confidence, horizon)
es = risk.expected_shortfall(port_ret, confidence, horizon)
mdd = risk.max_drawdown(port_ret)
sharpe = ann_ret / ann_vol if ann_vol else 0.0

# --- Header --------------------------------------------------------------
st.title("📉 Portfolio Risk Analyzer")
st.caption(
    "Volatility, multi-method Value-at-Risk, Expected Shortfall, correlation and "
    "stress testing for a multi-asset portfolio."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Annualized return", f"{ann_ret:6.1%}")
c2.metric("Annualized volatility", f"{ann_vol:6.1%}")
c3.metric("Return/Vol (Sharpe-like)", f"{sharpe:4.2f}")
c4.metric("Max drawdown", f"{mdd:6.1%}")

st.divider()

# --- VaR comparison ------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader(f"Value-at-Risk @ {confidence:.0%}, {horizon}-day horizon")
    var_df = pd.DataFrame(
        {
            "Method": ["Historical", "Parametric (Gaussian)", "Monte Carlo", "Expected Shortfall"],
            "Loss (%)": [hist_var, param_var, mc_var, es],
        }
    )
    var_df["Loss ($)"] = var_df["Loss (%)"] * portfolio_value
    fig = px.bar(
        var_df, x="Method", y="Loss (%)", text=var_df["Loss (%)"].map(lambda v: f"{v:.2%}")
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_tickformat=".1%", margin=dict(t=10, b=10), height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        var_df.style.format({"Loss (%)": "{:.2%}", "Loss ($)": "${:,.0f}"}),
        hide_index=True,
        use_container_width=True,
    )

with right:
    st.subheader("Reading the numbers")
    st.markdown(
        f"""
- **Historical VaR** reads the loss straight off realized data — no
  distribution assumed. At {confidence:.0%} over {horizon} day(s), losses are
  expected to exceed **{hist_var:.2%}** (${hist_var*portfolio_value:,.0f}) on
  roughly {(1-confidence):.0%} of days.
- **Parametric VaR** assumes Gaussian returns. When it sits *below* the
  historical number, the book has **fatter tails** than a normal curve.
- **Monte Carlo VaR** simulates 50k correlated draws, so it respects the
  cross-asset correlation the single-series methods miss.
- **Expected Shortfall** is the *average* loss once VaR is breached — the
  question VaR can't answer: **{es:.2%}**.
"""
    )

st.divider()

# --- Correlation + return path -------------------------------------------
g1, g2 = st.columns(2)

with g1:
    st.subheader("Correlation matrix")
    corr = risk.correlation_matrix(returns)
    heat = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
            zmin=-1,
            zmax=1,
        )
    )
    heat.update_layout(margin=dict(t=10, b=10), height=360)
    st.plotly_chart(heat, use_container_width=True)

with g2:
    st.subheader(f"Cumulative portfolio return — {span_start} to {span_end} (~{n_years:.1f}y)")
    curve = (1.0 + port_ret).cumprod() - 1.0
    line = px.area(curve, labels={"value": "Cumulative return", "index": "Date"})
    line.update_layout(yaxis_tickformat=".0%", showlegend=False, margin=dict(t=10, b=10), height=360)
    st.plotly_chart(line, use_container_width=True)

st.divider()

# --- Stress testing ------------------------------------------------------
st.subheader("Stress testing")
st.markdown(
    "Two complementary views. **Historical event scenarios** apply each crisis's "
    "*realized per-factor* shocks (equities, credit, gold, crypto move "
    "differently — gold can rally while equities fall), scaled by each position's "
    "sensitivity. **Worst realized window** reads the portfolio's own worst "
    "stretch straight off the data, no assumptions."
)

event_stress = risk.historical_event_stress(
    weights_by_asset, factor_map, data.HISTORICAL_EVENTS, portfolio_value
)
sc1, sc2 = st.columns([3, 2])
with sc1:
    st.markdown("**Historical event scenarios**")
    st.dataframe(
        event_stress.style.format(
            {
                "Portfolio impact": "{:.1%}",
                "P&L": "${:,.0f}",
                "Resulting value": "${:,.0f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    if (event_stress["Proxied assets"] != "—").any():
        st.caption(
            "⚠️ Proxied assets did not exist during that event; the factor shock "
            "is an approximation, not realized history."
        )
    inferred = [a for a, m in factor_map.items() if not m["known"]]
    if inferred:
        detail = ", ".join(f"{a} → {factor_map[a]['factor']}" for a in inferred)
        st.caption(
            f"ℹ️ Factor inferred for non-catalog tickers (inception read from "
            f"price history): {detail}."
        )
with sc2:
    sfig = px.bar(
        event_stress, x="Scenario", y="P&L", color="P&L", color_continuous_scale="RdYlGn"
    )
    sfig.update_layout(
        margin=dict(t=10, b=10), height=320, coloraxis_showscale=False, xaxis_tickangle=-20
    )
    st.plotly_chart(sfig, use_container_width=True)

st.markdown(
    f"**Worst realized window** — full available history {full_start} to "
    f"{full_end} (~{(prices_full.index[-1]-prices_full.index[0]).days/365.25:.1f}y)"
)
st.caption(
    "Computed over the entire common history of the current holdings so it can "
    "capture real crisis stretches (2008, 2020, …). The window starts when the "
    "newest constituent began trading — adding a young asset (e.g. a crypto) "
    "shortens how far back the portfolio can be evaluated."
)
worst = risk.worst_historical_window(port_ret_full, windows=(1, 5, 21, 63))
worst["P&L"] = worst["Worst return"] * portfolio_value
st.dataframe(
    worst.style.format({"Worst return": "{:.2%}", "P&L": "${:,.0f}"}),
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "Built with Python · numpy · scipy · pandas · plotly · streamlit. "
    "Synthetic data is deterministic; live mode uses Yahoo Finance."
)
