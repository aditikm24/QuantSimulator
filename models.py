import numpy as np
import pandas as pd
from scipy.stats import norm

class AdvancedSimulationModels:
    def __init__(self, historical_data_dict, risk_free_rate=0.05):
        """
        historical_data_dict: dictionary returned by fetch_data.get_historical_data
        risk_free_rate: Annualized risk-free rate (e.g., 0.05 for 5%)
        """
        self.r = risk_free_rate
        self.log_returns = historical_data_dict['log_returns']
        self.tickers = list(historical_data_dict['prices'].keys())
        
        # Last prices for all tickers
        self.last_prices = np.array([historical_data_dict['prices'][t][-1] for t in self.tickers])
        
        # Single asset parameters (used if only 1 ticker is provided)
        if len(self.tickers) == 1:
            self.daily_mean = self.log_returns[self.tickers[0]].mean()
            self.daily_vol = self.log_returns[self.tickers[0]].std()
            self.mu = self.daily_mean * 252
            self.sigma = self.daily_vol * np.sqrt(252)
            self.S0 = self.last_prices[0]
        else:
            # Multi-asset parameters
            self.cov_matrix = self.log_returns.cov() * 252
            self.mu_vec = self.log_returns.mean().values * 252
            self.sigma_vec = self.log_returns.std().values * np.sqrt(252)

    def run_gbm_simulation(self, days_to_simulate, num_simulations=10):
        """ Runs GBM for a single asset. """
        dt = 1 / 252
        simulations = np.zeros((num_simulations, days_to_simulate + 1))
        simulations[:, 0] = self.S0
        
        for t in range(1, days_to_simulate + 1):
            Z = np.random.standard_normal(num_simulations)
            simulations[:, t] = simulations[:, t-1] * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z)
            
        return simulations

    def run_jump_diffusion_simulation(self, days_to_simulate, num_simulations=10, lam=0.5, jump_mean=-0.05, jump_std=0.1):
        """ Runs Jump Diffusion for a single asset. """
        dt = 1 / 252
        simulations = np.zeros((num_simulations, days_to_simulate + 1))
        simulations[:, 0] = self.S0
        
        k = np.exp(jump_mean + 0.5 * jump_std**2) - 1
        
        for t in range(1, days_to_simulate + 1):
            Z = np.random.standard_normal(num_simulations)
            N = np.random.poisson(lam * dt, num_simulations)
            J = np.where(N > 0, np.random.normal(N * jump_mean, np.sqrt(N) * jump_std), 0)
            
            drift = (self.mu - 0.5 * self.sigma**2 - lam * k) * dt
            diffusion = self.sigma * np.sqrt(dt) * Z
            
            simulations[:, t] = simulations[:, t-1] * np.exp(drift + diffusion + J)
            
        return simulations

    def run_portfolio_simulation(self, days_to_simulate, num_simulations=1000):
        """
        Runs Correlated GBM for multiple assets and returns the aggregate portfolio value over time.
        Assumes equal weighting (1 share of each stock) for simplicity.
        """
        num_assets = len(self.tickers)
        dt = 1 / 252
        
        # Cholesky decomposition of the covariance matrix to correlate random shocks
        L = np.linalg.cholesky(self.cov_matrix.values)
        
        # Array to hold the portfolio value over time
        portfolio_sims = np.zeros((num_simulations, days_to_simulate + 1))
        
        # Initial portfolio value (sum of 1 share of each asset)
        portfolio_sims[:, 0] = np.sum(self.last_prices)
        
        # We will simulate all assets simultaneously for all simulations
        # current_prices shape: (num_simulations, num_assets)
        current_prices = np.tile(self.last_prices, (num_simulations, 1))
        
        drift = (self.mu_vec - 0.5 * self.sigma_vec**2) * dt
        
        for t in range(1, days_to_simulate + 1):
            # Independent standard normals: shape (num_assets, num_simulations)
            Z = np.random.standard_normal((num_assets, num_simulations))
            
            # Correlated normals: L * Z -> shape (num_assets, num_simulations)
            correlated_Z = np.dot(L, Z)
            
            # Transpose back to (num_simulations, num_assets) for easy broadcasting
            correlated_Z = correlated_Z.T
            
            diffusion = correlated_Z * np.sqrt(dt)
            
            # Update prices for all assets in all simulations
            current_prices = current_prices * np.exp(drift + diffusion)
            
            # Sum across assets to get portfolio value
            portfolio_sims[:, t] = np.sum(current_prices, axis=1)
            
        # Calculate VaR
        terminal_portfolio_values = portfolio_sims[:, -1]
        initial_portfolio_value = portfolio_sims[0, 0]
        returns = (terminal_portfolio_values - initial_portfolio_value) / initial_portfolio_value
        
        var_95 = np.percentile(returns, 5) * initial_portfolio_value
        var_99 = np.percentile(returns, 1) * initial_portfolio_value
        
        return {
            "paths": portfolio_sims,
            "var_95": var_95,
            "var_99": var_99,
            "initial_value": initial_portfolio_value
        }

    def calculate_bs_greeks(self, K, T):
        """ Calculates analytical Black-Scholes Greeks for European Options. """
        if len(self.tickers) > 1:
            return None # Greeks are for single assets in this context
            
        S = self.S0
        r = self.r
        sigma = self.sigma
        
        if T <= 0 or sigma <= 0:
            return None
            
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Call Greeks
        delta_call = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta_call = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                      - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365 # Daily decay
        vega = S * norm.pdf(d1) * np.sqrt(T) * 0.01 # Per 1% change in vol
        
        # Put Greeks
        delta_put = delta_call - 1
        theta_put = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365 # Daily decay
                     
        return {
            "delta_call": float(delta_call),
            "delta_put": float(delta_put),
            "gamma": float(gamma),
            "theta_call": float(theta_call),
            "theta_put": float(theta_put),
            "vega": float(vega)
        }

    def price_european_options(self, simulated_paths, strike_price, days_to_maturity):
        """ Prices options using Monte Carlo and returns pricing + Greeks. """
        terminal_prices = simulated_paths[:, -1]
        T = days_to_maturity / 252 
        
        call_payoffs = np.maximum(terminal_prices - strike_price, 0)
        put_payoffs = np.maximum(strike_price - terminal_prices, 0)
        
        call_price = np.exp(-self.r * T) * np.mean(call_payoffs)
        put_price = np.exp(-self.r * T) * np.mean(put_payoffs)
        
        greeks = self.calculate_bs_greeks(strike_price, T)
        
        return {
            "call_price": call_price,
            "put_price": put_price,
            "greeks": greeks
        }
