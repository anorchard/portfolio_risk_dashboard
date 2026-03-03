#!/usr/bin/env python3
"""
Portfolio Risk Dashboard — Streamlit App
=========================================
A comprehensive risk monitoring tool for a Single Family Office PM.

Features:
  • Portfolio input via sidebar (tickers + weights + AUM)
  • Key risk metrics: VaR, CVaR, Sharpe, Sortino, Max Drawdown, CAGR, Volatility
  • Correlation heatmap across holdings
  • Rolling volatility & rolling VaR charts
  • Drawdown chart
  • Monte Carlo VaR simulation
  • Cumulative returns vs benchmark
  • Monthly returns heatmap
  • Optional: Import positions from IBKR CSV flex report
  • One-click QuantStats HTML tearsheet export

Install dependencies:
  pip install streamlit yfinance pandas numpy scipy plotly quantstats seaborn matplotlib

Run:
  streamlit run portfolio_risk_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats as sp_stats
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import warnings
import io

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Portfolio Risk Dashboard")
st.markdown("*Real-time risk analytics for your portfolio — no Bloomberg required.*")

# ──────────────────────────────────────────────
# HELPER: PARSE WEIGHTS (ALLOWS '%' INPUT)
# ──────────────────────────────────────────────
def parse_weight_token(token: str) -> float:
    """
    Parse a weight token like '15.3', '15.3%', ' 15.3 % ' into a float 15.3.
    """
    token = token.strip()
    if token.endswith("%"):
        token = token[:-1]
    # Remove any stray spaces and thousand separators
    token = token.replace(" ", "").replace(",", "")
    return float(token)

# ──────────────────────────────────────────────
# SIDEBAR — Portfolio Input
# ──────────────────────────────────────────────
st.sidebar.header("⚙️ Portfolio Configuration")

input_method = st.sidebar.radio(
    "Input method",
    ["Manual Entry", "Upload IBKR CSV"],
    help="Enter tickers manually or upload an IBKR Flex Report CSV."
)

if input_method == "Manual Entry":
    default_tickers = "AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, V, UNH, XOM, META"
    default_weights = "12, 12, 10, 10, 10, 10, 10, 10, 8, 8"
    tickers_input = st.sidebar.text_area(
        "Tickers (comma-separated)", value=default_tickers
    )
    weights_input = st.sidebar.text_area(
        "Weights % (comma-separated, can use '15.3' or '15.3%')",
        value=default_weights,
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    weights_raw = [
        parse_weight_token(w) for w in weights_input.split(",") if w.strip()
    ]
    weights = np.array(weights_raw) / 100.0

else:
    uploaded_file = st.sidebar.file_uploader("Upload IBKR Flex CSV", type=["csv"])
    if uploaded_file is not None:
        ibkr_df = pd.read_csv(uploaded_file)
        # Attempt to parse common IBKR flex report columns
        if "Symbol" in ibkr_df.columns and "MarketValue" in ibkr_df.columns:
            ibkr_df = ibkr_df[ibkr_df["MarketValue"] > 0]
            tickers = ibkr_df["Symbol"].tolist()
            total_mv = ibkr_df["MarketValue"].sum()
            weights = (ibkr_df["MarketValue"] / total_mv).values
        elif "symbol" in ibkr_df.columns and "marketValue" in ibkr_df.columns:
            ibkr_df = ibkr_df[ibkr_df["marketValue"] > 0]
            tickers = ibkr_df["symbol"].tolist()
            total_mv = ibkr_df["marketValue"].sum()
            weights = (ibkr_df["marketValue"] / total_mv).values
        else:
            st.sidebar.error(
                "CSV must contain 'Symbol' and 'MarketValue' columns. "
                "Please use an IBKR Activity Flex Report."
            )
            st.stop()
    else:
        st.sidebar.info("Upload a file or switch to Manual Entry.")
        st.stop()

# Validate weights
if len(tickers) != len(weights):
    st.sidebar.error("Number of tickers must match number of weights.")
    st.stop()

if not np.isclose(weights.sum(), 1.0, atol=0.02):
    st.sidebar.warning(
        f"Weights sum to {weights.sum()*100:.1f}%. Normalising to 100%."
    )
    weights = weights / weights.sum()

portfolio_value = st.sidebar.number_input(
    "Portfolio Value (USD)", min_value=10_000, value=10_000_000, step=100_000,
    format="%d"
)

benchmark_ticker = st.sidebar.text_input("Benchmark", value="SPY")

lookback_years = st.sidebar.slider("Lookback (years)", 1, 10, 3)

confidence_level = st.sidebar.slider(
    "VaR Confidence Level", 0.90, 0.99, 0.95, 0.01
)

mc_simulations = st.sidebar.number_input(
    "Monte Carlo Simulations", min_value=1000, max_value=100_000,
    value=10_000, step=1000
)

# ──────────────────────────────────────────────
# DATA DOWNLOAD
# ──────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Downloading market data...")
def download_data(tickers, benchmark, years):
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    all_tickers = list(tickers) + [benchmark]
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(all_tickers[0])
    return data.dropna()

with st.spinner("Fetching data…"):
    price_data = download_data(tickers, benchmark_ticker, lookback_years)

# Check for missing tickers
missing = [t for t in tickers if t not in price_data.columns]
if missing:
    st.error(f"Could not find data for: {', '.join(missing)}. Check ticker symbols.")
    st.stop()

returns = price_data.pct_change().dropna()
portfolio_returns = (returns[tickers] * weights).sum(axis=1)
benchmark_returns = returns[benchmark_ticker]

# Align dates
common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
portfolio_returns = portfolio_returns.loc[common_idx]
benchmark_returns = benchmark_returns.loc[common_idx]

# NEW: guard against empty series
if portfolio_returns.empty or benchmark_returns.empty:
    st.error(
        "Not enough return history for the selected tickers/benchmark and lookback.\n\n"
        "Try:\n"
        "- Shorter lookback (e.g. 1–3 years)\n"
        "- Checking tickers are valid and have history\n"
        "- Using a broad ETF benchmark like SPY"
    )
    st.stop()


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────
def calc_var(returns_series, confidence):
    if returns_series.empty:
        return 0.0
    return np.percentile(returns_series, (1 - confidence) * 100)

def calc_cvar(returns_series, confidence):
    if returns_series.empty:
        return 0.0
    var = calc_var(returns_series, confidence)
    tail = returns_series[returns_series <= var]
    return tail.mean() if not tail.empty else 0.0

def calc_sharpe(returns_series, rf=0.04):
    excess = returns_series.mean() - rf / 252
    if returns_series.std() == 0:
        return 0.0
    return (excess / returns_series.std()) * np.sqrt(252)

def calc_sortino(returns_series, rf=0.04):
    excess = returns_series.mean() - rf / 252
    downside = returns_series[returns_series < 0].std()
    if downside == 0:
        return 0.0
    return (excess / downside) * np.sqrt(252)

def calc_max_drawdown(returns_series):
    cum = (1 + returns_series).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

def calc_cagr(returns_series):
    cum = (1 + returns_series).cumprod()
    n_years = len(returns_series) / 252
    if n_years == 0 or cum.iloc[-1] <= 0:
        return 0.0
    return (cum.iloc[-1]) ** (1 / n_years) - 1

def calc_beta(portfolio_ret, benchmark_ret):
    cov = np.cov(portfolio_ret, benchmark_ret)[0][1]
    var = np.var(benchmark_ret)
    return cov / var if var != 0 else 0.0

def monte_carlo_var(returns_series, n_sims, confidence, horizon=1):
    mu = returns_series.mean()
    sigma = returns_series.std()
    simulated = np.random.normal(mu * horizon, sigma * np.sqrt(horizon), n_sims)
    return np.percentile(simulated, (1 - confidence) * 100)

# ──────────────────────────────────────────────
# KEY METRICS
# ──────────────────────────────────────────────
daily_var = calc_var(portfolio_returns, confidence_level)
daily_cvar = calc_cvar(portfolio_returns, confidence_level)
mc_var = monte_carlo_var(portfolio_returns, mc_simulations, confidence_level)
sharpe = calc_sharpe(portfolio_returns)
sortino = calc_sortino(portfolio_returns)
max_dd = calc_max_drawdown(portfolio_returns)
ann_vol = portfolio_returns.std() * np.sqrt(252)
cagr = calc_cagr(portfolio_returns)
beta = calc_beta(portfolio_returns, benchmark_returns)

st.markdown("---")
st.subheader("📈 Key Risk Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Daily VaR (Historical)", f"{daily_var*100:.2f}%",
            delta=f"${daily_var * portfolio_value:,.0f}", delta_color="inverse")
col2.metric("Daily CVaR (ES)", f"{daily_cvar*100:.2f}%",
            delta=f"${daily_cvar * portfolio_value:,.0f}", delta_color="inverse")
col3.metric("Monte Carlo VaR", f"{mc_var*100:.2f}%",
            delta=f"${mc_var * portfolio_value:,.0f}", delta_color="inverse")
col4.metric("Max Drawdown", f"{max_dd*100:.2f}%")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Sharpe Ratio", f"{sharpe:.2f}")
col6.metric("Sortino Ratio", f"{sortino:.2f}")
col7.metric("CAGR", f"{cagr*100:.2f}%")
col8.metric("Beta vs " + benchmark_ticker, f"{beta:.2f}")

col9, col10, _, _ = st.columns(4)
col9.metric("Annualised Volatility", f"{ann_vol*100:.2f}%")
col10.metric("Portfolio Value", f"${portfolio_value:,.0f}")

# ──────────────────────────────────────────────
# CUMULATIVE RETURNS
# ──────────────────────────────────────────────
st.markdown("---")
st.subheader("📉 Cumulative Returns vs Benchmark")

cum_port = (1 + portfolio_returns).cumprod() - 1
cum_bench = (1 + benchmark_returns).cumprod() - 1

fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=cum_port.index, y=cum_port * 100,
    name="Portfolio", line=dict(color="#636EFA", width=2)
))
fig_cum.add_trace(go.Scatter(
    x=cum_bench.index, y=cum_bench * 100,
    name=benchmark_ticker, line=dict(color="#EF553B", width=2, dash="dash")
))
fig_cum.update_layout(
    yaxis_title="Cumulative Return (%)",
    template="plotly_white", height=400, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_cum, use_container_width=True)

# ──────────────────────────────────────────────
# DRAWDOWN
# ──────────────────────────────────────────────
st.subheader("📉 Drawdown")
cum_pnl = (1 + portfolio_returns).cumprod()
peak = cum_pnl.cummax()
drawdown = (cum_pnl - peak) / peak

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=drawdown.index, y=drawdown * 100,
    fill="tozeroy", fillcolor="rgba(239,85,59,0.2)",
    line=dict(color="#EF553B", width=1), name="Drawdown"
))
fig_dd.update_layout(
    yaxis_title="Drawdown (%)",
    template="plotly_white", height=300, hovermode="x unified"
)
st.plotly_chart(fig_dd, use_container_width=True)

# ──────────────────────────────────────────────
# CORRELATION HEATMAP
# ──────────────────────────────────────────────
st.markdown("---")
st.subheader("🔥 Correlation Heatmap")

corr_matrix = returns[tickers].corr()

fig_corr = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    aspect="auto",
)
fig_corr.update_layout(
    template="plotly_white", height=500,
    coloraxis_colorbar=dict(title="Corr")
)
st.plotly_chart(fig_corr, use_container_width=True)

# ──────────────────────────────────────────────
# ROLLING VOLATILITY & ROLLING VaR
# ──────────────────────────────────────────────
st.markdown("---")
col_rv, col_rvar = st.columns(2)

with col_rv:
    st.subheader("📊 Rolling 30d Volatility")
    rolling_vol = portfolio_returns.rolling(30).std() * np.sqrt(252) * 100
    fig_rvol = go.Figure()
    fig_rvol.add_trace(go.Scatter(
        x=rolling_vol.index, y=rolling_vol,
        line=dict(color="#636EFA", width=1.5), name="30d Vol"
    ))
    fig_rvol.update_layout(
        yaxis_title="Annualised Vol (%)",
        template="plotly_white", height=350, hovermode="x unified"
    )
    st.plotly_chart(fig_rvol, use_container_width=True)

with col_rvar:
    st.subheader(f"📊 Rolling 60d VaR ({confidence_level*100:.0f}%)")
    rolling_var = portfolio_returns.rolling(60).apply(
        lambda x: np.percentile(x, (1 - confidence_level) * 100), raw=True
    ) * 100
    fig_rvar = go.Figure()
    fig_rvar.add_trace(go.Scatter(
        x=rolling_var.index, y=rolling_var,
        line=dict(color="#EF553B", width=1.5), name="Rolling VaR"
    ))
    fig_rvar.update_layout(
        yaxis_title="Daily VaR (%)",
        template="plotly_white", height=350, hovermode="x unified"
    )
    st.plotly_chart(fig_rvar, use_container_width=True)

# ──────────────────────────────────────────────
# MONTE CARLO SIMULATION DISTRIBUTION
# ──────────────────────────────────────────────
st.markdown("---")
st.subheader("🎲 Monte Carlo Simulation — Return Distribution")

mc_mu = portfolio_returns.mean()
mc_sigma = portfolio_returns.std()
mc_sims = np.random.normal(mc_mu, mc_sigma, mc_simulations)
mc_var_val = np.percentile(mc_sims, (1 - confidence_level) * 100)

fig_mc = go.Figure()
fig_mc.add_trace(go.Histogram(
    x=mc_sims * 100, nbinsx=100,
    marker_color="#636EFA", opacity=0.7, name="Simulated Returns"
))
fig_mc.add_vline(
    x=mc_var_val * 100, line_dash="dash", line_color="red",
    annotation_text=f"VaR {confidence_level*100:.0f}%: {mc_var_val*100:.2f}%",
    annotation_position="top left"
)
fig_mc.update_layout(
    xaxis_title="Daily Return (%)", yaxis_title="Frequency",
    template="plotly_white", height=400
)
st.plotly_chart(fig_mc, use_container_width=True)

# ──────────────────────────────────────────────
# MONTHLY RETURNS HEATMAP
# ──────────────────────────────────────────────
st.markdown("---")
st.subheader("📅 Monthly Returns Heatmap")

monthly = portfolio_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
monthly_df = pd.DataFrame({
    "Year": monthly.index.year,
    "Month": monthly.index.month,
    "Return": monthly.values * 100
})
monthly_pivot = monthly_df.pivot_table(index="Year", columns="Month", values="Return")
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
monthly_pivot.columns = [month_labels[m - 1] for m in monthly_pivot.columns]

fig_monthly = px.imshow(
    monthly_pivot,
    text_auto=".1f",
    color_continuous_scale="RdYlGn",
    aspect="auto",
)
fig_monthly.update_layout(
    template="plotly_white", height=300,
    coloraxis_colorbar=dict(title="Return %")
)
st.plotly_chart(fig_monthly, use_container_width=True)

# ──────────────────────────────────────────────
# PORTFOLIO ALLOCATION PIE
# ──────────────────────────────────────────────
st.markdown("---")
st.subheader("🥧 Portfolio Allocation")

fig_pie = px.pie(
    names=tickers, values=weights * 100,
    hole=0.4,
)
fig_pie.update_traces(textposition="inside", textinfo="label+percent")
fig_pie.update_layout(template="plotly_white", height=400)
st.plotly_chart(fig_pie, use_container_width=True)

# ──────────────────────────────────────────────
# INDIVIDUAL POSITION RISK CONTRIBUTION
# ──────────────────────────────────────────────
st.markdown("---")
st.subheader("⚖️ Risk Contribution by Position")

cov_matrix = returns[tickers].cov() * 252
port_vol = np.sqrt(weights @ cov_matrix.values @ weights)
marginal_contrib = (cov_matrix.values @ weights) / port_vol
component_risk = weights * marginal_contrib
pct_risk_contrib = component_risk / port_vol * 100

risk_df = pd.DataFrame({
    "Ticker": tickers,
    "Weight %": weights * 100,
    "Risk Contribution %": pct_risk_contrib,
}).sort_values("Risk Contribution %", ascending=False)

fig_risk = px.bar(
    risk_df, x="Ticker", y="Risk Contribution %",
    color="Risk Contribution %",
    color_continuous_scale="Reds",
)
fig_risk.update_layout(template="plotly_white", height=400)
st.plotly_chart(fig_risk, use_container_width=True)

st.dataframe(
    risk_df.style.format({"Weight %": "{:.1f}", "Risk Contribution %": "{:.2f}"}),
    use_container_width=True, hide_index=True
)

# ──────────────────────────────────────────────
# STRESS TESTING — Simple Scenarios
# ──────────────────────────────────────────────
st.markdown("---")
st.subheader("🧪 Stress Testing — Scenario Analysis")

st.markdown(
    "Applies uniform shocks to all positions and estimates portfolio loss."
)

scenarios = {
    "Mild Correction (-5%)": -0.05,
    "Moderate Sell-off (-10%)": -0.10,
    "Severe Drawdown (-20%)": -0.20,
    "March 2020 Style (-35%)": -0.35,
    "GFC Style (-50%)": -0.50,
}

stress_results = []
for name, shock in scenarios.items():
    # Weight the shock by beta for a more realistic estimate
    est_loss = shock * beta * portfolio_value
    stress_results.append({
        "Scenario": name,
        "Market Shock": f"{shock*100:.0f}%",
        "Est. Portfolio Loss (beta-adj)": f"${est_loss:,.0f}",
        "Est. Portfolio Loss %": f"{shock * beta * 100:.1f}%",
    })

st.table(pd.DataFrame(stress_results))

# ──────────────────────────────────────────────
# QUANTSTATS TEARSHEET EXPORT
# ──────────────────────────────────────────────
st.markdown("---")
st.subheader("📄 Export Full QuantStats Tearsheet")

st.markdown(
    "Click below to generate a comprehensive HTML tearsheet using **QuantStats**. "
    "This replicates a Bloomberg PORT-style analytics report."
)

if st.button("🚀 Generate Tearsheet"):
    try:
        import quantstats as qs
        qs.extend_pandas()

        buf = io.StringIO()
        qs.reports.html(
            portfolio_returns,
            benchmark=benchmark_returns,
            output=buf,
            title="Portfolio Risk Tearsheet",
        )
        html_content = buf.getvalue()

        st.download_button(
            label="⬇️ Download Tearsheet (HTML)",
            data=html_content,
            file_name="portfolio_tearsheet.html",
            mime="text/html",
        )
        st.success("Tearsheet generated! Click above to download.")
    except ImportError:
        st.error("Install quantstats: `pip install quantstats`")

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data sourced from Yahoo Finance via yfinance. "
    "VaR and CVaR are estimates based on historical returns — not guarantees of future losses. "
    "Built for internal SFO use. Not financial advice."
)
