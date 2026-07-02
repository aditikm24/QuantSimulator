import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from models import AdvancedSimulationModels
from fetch_data import fetch_historical_data
import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Monte Carlo Simulator", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR EXACT LOOK ---
st.markdown("""
<style>
    /* Dark theme adjustments */
    .stApp {
        background-color: #0E1117;
    }
    .css-1d391kg {
        background-color: #1A1C24;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")

st.sidebar.markdown("### Parameter Selection Mode")
param_mode = st.sidebar.radio("", ["Auto-calculate from Historical Data", "Manual Parameter Override"])

# Period Mapping
period_mapping = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
lookback = st.sidebar.selectbox("Historical Lookback Period", list(period_mapping.keys()), index=3)

# Data Fetching & Parameter Estimation
st.sidebar.markdown("### Estimated Parameters")
try:
    if ticker:
        hist_data_dict = fetch_historical_data([ticker], period_mapping[lookback])
        if ticker in hist_data_dict and not hist_data_dict[ticker].empty:
            df = hist_data_dict[ticker]
            last_price = df['close'].iloc[-1]
            
            # Calculate actual historical drift and vol
            log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
            hist_mu = float(log_returns.mean() * 252)
            hist_sigma = float(log_returns.std() * np.sqrt(252))
        else:
            st.sidebar.error("Failed to fetch data.")
            st.stop()
except Exception as e:
    st.sidebar.error(f"Error fetching data: {e}")
    st.stop()

if param_mode == "Auto-calculate from Historical Data":
    mu = hist_mu
    sigma = hist_sigma
    st.sidebar.metric("Last Close Price", f"${last_price:.2f}")
    st.sidebar.metric("Annualized Drift (μ)", f"{mu*100:.2f}%")
    st.sidebar.metric("Annualized Volatility (σ)", f"{sigma*100:.2f}%")
else:
    st.sidebar.metric("Last Close Price", f"${last_price:.2f}")
    mu = st.sidebar.number_input("Annualized Drift (μ) %", value=hist_mu*100) / 100
    sigma = st.sidebar.number_input("Annualized Volatility (σ) %", value=hist_sigma*100) / 100

st.sidebar.markdown("### 🚀 Run Parameters")
num_paths = st.sidebar.slider("Number of Simulation Paths", min_value=10, max_value=1000, value=400, step=10)
time_horizon = st.sidebar.slider("Time Horizon (Trading Days)", min_value=10, max_value=252, value=190, step=10)

run_button = st.sidebar.button("Run Simulation", use_container_width=True, type="primary")

# --- MAIN BODY ---
if run_button:
    # Run Simulation
    sim_model = AdvancedSimulationModels(
        historical_data={ticker: df},
        tickers=[ticker],
        simulation_days=time_horizon,
        num_simulations=num_paths
    )
    
    # We will implement custom GBM simulation here to use the overridden mu/sigma if Manual is selected.
    # AdvancedSimulationModels automatically calculates it, so we replicate the core GBM here for the override feature.
    dt = 1 / 252
    paths = np.zeros((time_horizon, num_paths))
    paths[0] = last_price
    
    for t in range(1, time_horizon):
        z = np.random.standard_normal(num_paths)
        paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
        
    final_prices = paths[-1]
    
    # Analytics
    median_path = np.median(paths, axis=1)
    p05 = np.percentile(final_prices, 5)
    p50 = np.percentile(final_prices, 50)
    p95 = np.percentile(final_prices, 95)
    prob_profit = np.mean(final_prices > last_price) * 100
    var_95 = last_price - p05
    var_95_pct = (var_95 / last_price) * 100
    skewness = pd.Series(final_prices).skew()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Simulation Paths", "📊 Distribution & Risk Analytics", "📅 Historical Data"])
    
    with tab1:
        st.markdown("### Simulated Price Trajectories")
        st.markdown("Showing the projected price walks over time. The golden bold line represents the median path which represents the 50th percentile outcome.")
        
        fig = go.Figure()
        
        # Plot all paths
        days = list(range(time_horizon))
        for i in range(min(num_paths, 200)): # Limit to 200 for rendering performance
            fig.add_trace(go.Scatter(x=days, y=paths[:, i], mode='lines', line=dict(color='rgba(65, 105, 225, 0.1)', width=1), showlegend=False, hoverinfo='skip'))
            
        # Plot median path
        fig.add_trace(go.Scatter(x=days, y=median_path, mode='lines', line=dict(color='orange', width=3), name='Median Path (50th Percentile)'))
        
        # Plot starting baseline
        fig.add_trace(go.Scatter(x=[0, time_horizon-1], y=[last_price, last_price], mode='lines', line=dict(color='red', width=1, dash='dash'), name='Starting Price Baseline'))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Trading Days", yaxis_title="Stock Price ($)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.markdown("### Terminal Price Distribution")
        st.markdown(f"This histogram shows the probability density distribution of the stock price at the end of the {time_horizon}-day simulation window.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig2 = go.Figure(data=[go.Histogram(x=final_prices, nbinsx=50, marker_color='royalblue')])
            
            # Add vertical lines for percentiles
            fig2.add_vline(x=p05, line_dash="dash", line_color="red", annotation_text=f"Worst-case (5th Percentile): ${p05:.2f}", annotation_position="top left", annotation_font_color="red")
            fig2.add_vline(x=p50, line_dash="dash", line_color="orange", annotation_text=f"Median Final Price: ${p50:.2f}", annotation_position="top right", annotation_font_color="orange")
            fig2.add_vline(x=p95, line_dash="dash", line_color="green", annotation_text=f"Best-case (95th Percentile): ${p95:.2f}", annotation_position="top right", annotation_font_color="green")
            
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Terminal Stock Price ($)", yaxis_title="Path Count",
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig2, use_container_width=True)
            
        with col2:
            st.markdown("### 🛡️ Risk & Probability Analysis")
            st.markdown(f"**Probability of Profit** (Ending Price > Starting Price): `{prob_profit:.1f}%`")
            st.markdown(f"**Value at Risk** (95% Confidence Level):")
            st.markdown(f"There is a 95% statistical probability that the maximum loss will not exceed **{var_95_pct:.1f}%** (or **${var_95:.2f}** per share).")
            st.markdown(f"**Distribution Skewness**:")
            st.markdown("The distribution is log-normal (skewed to the right), showing that stock prices have a bounded downside ($0) but theoretically unlimited upside.")
            
            stats_df = pd.DataFrame({
                "Metric": ["Minimum Final Price", "10th Percentile", "Median (50th)", "90th Percentile", "Maximum Final Price"],
                "Value": [f"${np.min(final_prices):.2f}", f"${np.percentile(final_prices, 10):.2f}", f"${p50:.2f}", f"${np.percentile(final_prices, 90):.2f}", f"${np.max(final_prices):.2f}"]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
    with tab3:
        st.markdown("### Historical Data")
        st.dataframe(df, use_container_width=True)
else:
    st.info("Configure your parameters in the sidebar and click 'Run Simulation' to begin.")
