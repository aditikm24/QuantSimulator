# Advanced Monte Carlo Quant Simulator

A high-performance, full-stack web application built in Python for quantitative financial modeling. This engine simulates future asset trajectories, calculates Value at Risk (VaR) for correlated portfolios, and prices European options using advanced stochastic calculus. 

Designed for **Global Markets** and **Wholesale Strategy** analysis, it moves beyond basic academic models to incorporate real-world market dynamics like volatility smiles and tail-risk events.

## 🚀 Key Features

* **Advanced Stochastic Models**: 
  * **Geometric Brownian Motion (GBM)**: For standard equity trajectory forecasting.
  * **Merton Jump Diffusion**: Incorporates Poisson processes to model sudden market crashes and spikes, accurately reflecting "fat-tail" risk.
* **Portfolio Risk Engine (VaR)**: Input multiple tickers to instantly calculate the historical covariance matrix. The simulation uses **Cholesky Decomposition** to correlate random price shocks, generating an accurate 95% and 99% Value at Risk (VaR) for the aggregated portfolio.
* **Options Pricing & Greeks**: Prices European Call and Put options using Monte Carlo simulated terminal distributions. Automatically calculates analytical Black-Scholes Greeks ($\Delta$, $\Gamma$, $\Theta$, $\nu$) for risk management.
* **Lightning Fast Vectorization**: All Monte Carlo simulations are powered by highly optimized, vectorized `NumPy` operations, running thousands of paths in milliseconds.
* **Live Market Data**: Integrates with `yfinance` to automatically pull, clean, and calculate daily logarithmic returns and volatility from live historical market data.
* **Interactive UI**: A premium, custom-built FastAPI web frontend utilizing `Plotly.js` for dynamic, zoomable, and interactive financial charting.

## 🧮 Mathematical Foundations

The core engine is driven by the discretized solution to the Geometric Brownian Motion stochastic differential equation:

$$ S_t = S_{t-1} \cdot \exp \left( \left( \mu - \frac{\sigma^2}{2} \right) dt + \sigma \sqrt{dt} Z \right) $$

For the **Merton Jump Diffusion** model, a compound Poisson process is added to the drift and diffusion components to simulate discrete market shocks:

$$ S_t = S_{t-1} \cdot \exp \left( \left( \mu - \frac{\sigma^2}{2} - \lambda k \right) dt + \sigma \sqrt{dt} Z \right) \cdot \prod_{i=1}^{N_t} Y_i $$

## 🛠️ Tech Stack

* **Backend Engine**: Python, FastAPI, NumPy, Pandas, SciPy
* **Data Ingestion**: yfinance
* **Frontend Dashboard**: Vanilla HTML/CSS/JS, Plotly.js, Glassmorphism UI
* **Deployment**: Render (Docker/Gunicorn)

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aditikm24/QuantSimulator.git
   cd QuantSimulator
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI server:**
   ```bash
   uvicorn main:app --reload
   ```

4. Open your browser and navigate to `http://127.0.0.1:8000`.

## 📈 Usage Guide

* **Single Stock & Options**: Enter a single ticker (e.g., `AAPL`) and a Strike Price. The engine will simulate the stock's future path and calculate the Option Premium and Greeks.
* **Portfolio Risk (VaR)**: Enter multiple comma-separated tickers (e.g., `AAPL, MSFT, GOOG`). The engine will automatically switch to Portfolio Mode, correlate the assets using historical covariance, and output the 95% and 99% VaR.
