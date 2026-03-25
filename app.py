import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Quant Portfolio Analytics Suite", layout="wide")

# -----------------------------
# HEADER
# -----------------------------
st.title("📊 Quant Portfolio Analytics Suite v1.0")
st.markdown("Multi-Asset Portfolio Optimization & Risk Dashboard")
st.markdown("---")

# -----------------------------
# ASSET SELECTION
# -----------------------------
assets_dict = {
    "NIFTY 50": "^NSEI",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Gold": "GC=F",
    "Bonds (TLT)": "TLT",
    "Bitcoin": "BTC-USD"
}

selected_assets = st.multiselect(
    "Select Assets",
    list(assets_dict.keys()),
    default=["NIFTY 50", "S&P 500", "Gold", "Bitcoin"]
)

if len(selected_assets) < 2:
    st.warning("Please select at least 2 assets.")
    st.stop()

# -----------------------------
# PRESET PORTFOLIO BUTTONS
# -----------------------------
st.subheader("Portfolio Presets")

col1, col2, col3 = st.columns(3)

preset = None
if col1.button("Conservative"):
    preset = "conservative"
if col2.button("Balanced"):
    preset = "balanced"
if col3.button("Aggressive"):
    preset = "aggressive"

weights = []

st.subheader("Adjust Asset Weights")

for asset in selected_assets:
    if preset == "conservative":
        value = 70 / len(selected_assets)
    elif preset == "balanced":
        value = 100 / len(selected_assets)
    elif preset == "aggressive":
        value = 120 / len(selected_assets)
    else:
        value = 100 / len(selected_assets)

    weight = st.slider(f"{asset} (%)", 0, 100, int(value))
    weights.append(weight)

weights = np.array(weights)

if weights.sum() == 0:
    st.warning("Total weight cannot be zero.")
    st.stop()

weights = weights / weights.sum()

# -----------------------------
# INVESTMENT AMOUNT
# -----------------------------
investment = st.number_input("Enter Investment Amount ($)", value=10000)

# -----------------------------
# DOWNLOAD DATA
# -----------------------------
data = yf.download(
    [assets_dict[a] for a in selected_assets],
    start="2020-01-01"
)["Close"]

data = data.dropna()
returns = data.pct_change().dropna()

# -----------------------------
# PORTFOLIO CALCULATIONS
# -----------------------------
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

portfolio_return = np.dot(weights, mean_returns)
portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
sharpe_ratio = portfolio_return / portfolio_vol

portfolio_returns = returns.dot(weights)
cumulative_returns = (1 + portfolio_returns).cumprod()

# Max Drawdown
rolling_max = cumulative_returns.cummax()
drawdown = cumulative_returns / rolling_max - 1
max_drawdown = drawdown.min()

# Value at Risk (95%)
VaR = np.percentile(portfolio_returns, 5)

# -----------------------------
# KPI CARDS
# -----------------------------
st.markdown("## Portfolio Metrics")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Expected Return", f"{portfolio_return*100:.2f}%")
c2.metric("Volatility", f"{portfolio_vol*100:.2f}%")
c3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
c4.metric("Max Drawdown", f"{max_drawdown*100:.2f}%")
c5.metric("VaR (95%)", f"{VaR*100:.2f}%")

st.markdown("---")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Portfolio Overview", "Risk Analysis", "Performance", "Scenario Testing"]
)

# -----------------------------
# TAB 1: OVERVIEW
# -----------------------------
with tab1:
    st.subheader("Asset Allocation")

    fig1, ax1 = plt.subplots()
    ax1.pie(weights, labels=selected_assets, autopct="%1.1f%%")
    st.pyplot(fig1)

    allocation_values = weights * investment
    allocation_df = pd.DataFrame({
        "Asset": selected_assets,
        "Weight": weights,
        "Investment Value ($)": allocation_values
    })

    st.dataframe(allocation_df)

# -----------------------------
# TAB 2: RISK ANALYSIS
# -----------------------------
with tab2:
    st.subheader("Correlation Matrix")

    fig2, ax2 = plt.subplots()
    cax = ax2.matshow(returns.corr())
    fig2.colorbar(cax)

    ax2.set_xticks(range(len(selected_assets)))
    ax2.set_yticks(range(len(selected_assets)))
    ax2.set_xticklabels(selected_assets, rotation=90)
    ax2.set_yticklabels(selected_assets)

    st.pyplot(fig2)

    st.subheader("Drawdown Chart")
    st.line_chart(drawdown)

# -----------------------------
# TAB 3: PERFORMANCE
# -----------------------------
with tab3:
    st.subheader("Cumulative Returns")
    st.line_chart(cumulative_returns)

    st.subheader("Rolling Volatility (30 Days)")
    rolling_vol = portfolio_returns.rolling(30).std() * np.sqrt(252)
    st.line_chart(rolling_vol)

# -----------------------------
# TAB 4: SCENARIO TESTING
# -----------------------------
with tab4:
    st.subheader("Market Shock Simulator")

    shock = st.slider("Market Shock (%)", -30, 30, -10)

    shocked_return = portfolio_return + shock/100
    shocked_value = investment * (1 + shocked_return)

    st.write(f"Projected Portfolio Value After Shock: ${shocked_value:,.2f}")

# -----------------------------
# EFFICIENT FRONTIER
# -----------------------------
st.markdown("---")
st.subheader("Efficient Frontier Simulation")

num_portfolios = 5000
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    w = np.random.random(len(selected_assets))
    w /= np.sum(w)

    r = np.dot(w, mean_returns)
    v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

    results[0, i] = r
    results[1, i] = v
    results[2, i] = r / v

fig3, ax3 = plt.subplots()
ax3.scatter(results[1], results[0], c=results[2])
ax3.set_xlabel("Volatility")
ax3.set_ylabel("Return")
st.pyplot(fig3)

st.markdown("---")
st.caption("⚠ Past performance does not guarantee future results.")
