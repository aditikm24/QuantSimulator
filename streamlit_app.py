import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from fetch_data import get_historical_data
from models import AdvancedSimulationModels

st.set_page_config(page_title="Advanced Monte Carlo Simulator", layout="wide", page_icon="📈")

# --- Custom CSS to give a premium feel ---
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    h1, h2, h3 {
        color: #f8fafc;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #a0aec0;
        font-size: 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b;
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("Configuration")

tickers_input = st.sidebar.text_input("Ticker(s) (comma-separated)", "AAPL")
lookback = st.sidebar.selectbox("Historical Lookback Period", ["1 Month", "6 Months", "1 Year", "2 Years", "5 Years"], index=2)
model_choice = st.sidebar.selectbox("Financial Model (Single Asset Only)", ["Geometric Brownian Motion (GBM)", "Merton Jump Diffusion"])
strike_price_input = st.sidebar.number_input("Strike Price (Optional, Single Asset)", value=0.0, step=1.0)

st.sidebar.markdown("### Run Parameters")
num_sims = st.sidebar.slider("Number of Simulation Paths", min_value=10, max_value=1000, value=100, step=10)
days_to_sim = st.sidebar.slider("Time Horizon (Trading Days)", min_value=5, max_value=252, value=30, step=5)

if st.sidebar.button("Run Simulation", type="primary", use_container_width=True):
    
    # Process Inputs
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    lookback_map = {"1 Month": 30, "6 Months": 180, "1 Year": 365, "2 Years": 730, "5 Years": 1825}
    hist_days = lookback_map[lookback]
    
    try:
        with st.spinner("Fetching data and running simulations..."):
            # 1. Fetch Data
            hist_data = get_historical_data(tickers, hist_days)
            
            # 2. Init Model
            model = AdvancedSimulationModels(hist_data)
            
            # 3. Run Simulation
            is_portfolio = len(tickers) > 1
            if is_portfolio:
                sim_res = model.run_portfolio_simulation(days_to_sim, num_sims)
                paths = sim_res["paths"]
            else:
                if model_choice == "Geometric Brownian Motion (GBM)":
                    paths = model.run_gbm_simulation(days_to_sim, num_sims)
                else:
                    paths = model.run_jump_diffusion_simulation(days_to_sim, num_sims)
            
            # Calculate metrics
            terminal_prices = paths[:, -1]
            median_path = np.median(paths, axis=0)
            p5 = np.percentile(terminal_prices, 5)
            p50 = np.percentile(terminal_prices, 50)
            p95 = np.percentile(terminal_prices, 95)
            start_price = paths[0, 0]
            prob_profit = np.mean(terminal_prices > start_price) * 100
            
            # Generate Future Dates
            last_date_str = hist_data['dates'][-1]
            last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
            future_dates = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days_to_sim + 1)]
            
            # Calculate Historical Prices for charting
            if is_portfolio:
                hist_prices_array = np.array([hist_data['prices'][t] for t in tickers])
                hist_prices = np.sum(hist_prices_array, axis=0).tolist()
            else:
                hist_prices = hist_data['prices'][tickers[0]]
            
            # --- Main Content Area ---
            st.title("Simulated Price Trajectories")
            
            # Tabs
            tab1, tab2, tab3 = st.tabs(["📈 Simulation Paths", "📊 Distribution & Risk Analytics", "🧠 Advanced Quant Metrics"])
            
            with tab1:
                st.markdown("Showing the projected price walks over time. The **golden bold line** represents the median path.")
                fig1 = go.Figure()
                
                # Plot Historical Data
                fig1.add_trace(go.Scatter(
                    x=hist_data['dates'], y=hist_prices, mode='lines',
                    line=dict(color='#3b82f6', width=2),
                    name='Historical Data'
                ))
                
                # Plot subset of paths for performance
                max_plot_paths = min(num_sims, 50)
                for i in range(max_plot_paths):
                    fig1.add_trace(go.Scatter(
                        x=future_dates, y=paths[i], mode='lines', 
                        line=dict(color='rgba(56, 189, 248, 0.1)', width=1),
                        showlegend=False, hoverinfo='none'
                    ))
                
                # Plot Median Path
                fig1.add_trace(go.Scatter(
                    x=future_dates, y=median_path, mode='lines',
                    line=dict(color='#fbbf24', width=3),
                    name='Median Path (50th Percentile)'
                ))
                
                fig1.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Date",
                    yaxis_title="Stock Price ($)" if not is_portfolio else "Portfolio Value ($)",
                    height=500,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig1, use_container_width=True)
                
            with tab2:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### Terminal Price Distribution")
                    fig2 = go.Figure(data=[go.Histogram(x=terminal_prices, nbinsx=50, marker_color='#3b82f6')])
                    
                    # Add vlines for percentiles
                    fig2.add_vline(x=p5, line_dash="dash", line_color="#ef4444", annotation_text="5th Percentile")
                    fig2.add_vline(x=p50, line_dash="dash", line_color="#fbbf24", annotation_text="Median")
                    fig2.add_vline(x=p95, line_dash="dash", line_color="#10b981", annotation_text="95th Percentile")
                    
                    fig2.update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis_title="Terminal Price ($)",
                        yaxis_title="Path Count",
                        height=400,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col2:
                    st.markdown("### Risk & Probability Analysis")
                    st.markdown(f"**Probability of Profit:** `{prob_profit:.1f}%`")
                    st.markdown(f"**Value at Risk (95% Confidence):**")
                    loss_95 = start_price - p5
                    st.markdown(f"> There is a 95% statistical probability that the maximum loss will not exceed **${loss_95:.2f}** over the period.")
                    
                    st.divider()
                    st.metric("Minimum Final Value", f"${np.min(terminal_prices):.2f}")
                    st.metric("Median (50th)", f"${p50:.2f}")
                    st.metric("Maximum Final Value", f"${np.max(terminal_prices):.2f}")
            
            with tab3:
                if is_portfolio:
                    st.markdown("### Portfolio Risk (VaR)")
                    st.info("The portfolio simulator uses Cholesky Decomposition on the historical covariance matrix to correlate random shocks across all assets.")
                    
                    st.metric("Initial Portfolio Value", f"${sim_res['initial_value']:.2f}")
                    st.metric("95% Value at Risk", f"-${abs(sim_res['var_95']):.2f}")
                    st.metric("99% Value at Risk", f"-${abs(sim_res['var_99']):.2f}")
                else:
                    if strike_price_input > 0:
                        st.markdown("### Options Pricing & Black-Scholes Greeks")
                        options_data = model.price_european_options(paths, strike_price_input, days_to_sim)
                        
                        colA, colB = st.columns(2)
                        with colA:
                            st.metric("Call Price", f"${options_data['call_price']:.2f}")
                            st.metric("Put Price", f"${options_data['put_price']:.2f}")
                        
                        greeks = options_data['greeks']
                        if greeks:
                            with colB:
                                st.metric("Delta (Call)", f"{greeks['delta_call']:.4f}")
                                st.metric("Gamma", f"{greeks['gamma']:.4f}")
                                st.metric("Theta (Daily Call)", f"{greeks['theta_call']:.4f}")
                                st.metric("Vega", f"{greeks['vega']:.4f}")
                    else:
                        st.warning("Enter a Strike Price in the sidebar to calculate Options Pricing and Greeks.")
            
            # --- Sidebar Readout ---
            st.sidebar.divider()
            st.sidebar.markdown("### Estimated Parameters")
            if not is_portfolio:
                st.sidebar.metric("Last Close Price", f"${model.S0:.2f}")
                st.sidebar.metric("Annualized Drift (μ)", f"{model.mu * 100:.2f}%")
                st.sidebar.metric("Annualized Volatility (σ)", f"{model.sigma * 100:.2f}%")
            else:
                st.sidebar.success("Multi-Asset Portfolio Mode Active")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Enter parameters in the sidebar and click **Run Simulation** to begin.")
