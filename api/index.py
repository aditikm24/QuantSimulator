from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from .fetch_data import fetch_data
from .models import AdvancedSimulationModels

app = FastAPI()

class SimulationRequest(BaseModel):
    ticker: str
    historical_days: int = 365
    simulation_days: int = 30
    model: str = "gbm"
    num_simulations: int = 10
    strike_price: float = None

@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    try:
        tickers = [t.strip().upper() for t in req.ticker.split(',')]
        is_portfolio = len(tickers) > 1

        # Fetch Data
        hist_data = fetch_data(tickers, req.historical_days)
        
        sim_model = AdvancedSimulationModels(hist_data)
        
        if is_portfolio:
            sim_results, var_metrics = sim_model.simulate_portfolio(req.simulation_days, req.num_simulations)
            
            # Format historical for UI (using the portfolio value)
            hist_dates = list(hist_data[tickers[0]].index.strftime('%Y-%m-%d'))
            # Calculate historical portfolio value (assuming equal weighting starting at 1.0)
            returns = pd.DataFrame({tkr: hist_data[tkr]['Log_Return'] for tkr in tickers}).dropna()
            port_returns = returns.mean(axis=1)
            port_value = (1 + port_returns).cumprod() * 10000 # Start at 10k
            hist_prices = list(port_value)
            # Match lengths
            hist_dates = hist_dates[-len(hist_prices):]
            
            return {
                "is_portfolio": True,
                "historical": {
                    "dates": hist_dates,
                    "prices": hist_prices
                },
                "simulation": sim_results,
                "var_metrics": var_metrics
            }
        else:
            ticker = tickers[0]
            if req.model == "gbm":
                sim_results = sim_model.simulate_gbm(ticker, req.simulation_days, req.num_simulations)
            else:
                sim_results = sim_model.simulate_merton_jump(ticker, req.simulation_days, req.num_simulations)
                
            response = {
                "is_portfolio": False,
                "historical": {
                    "dates": list(hist_data[ticker].index.strftime('%Y-%m-%d')),
                    "prices": list(hist_data[ticker]['Close'])
                },
                "simulation": sim_results
            }
            
            if req.strike_price is not None:
                current_price = hist_data[ticker]['Close'].iloc[-1]
                options_data = sim_model.calculate_options_premium(
                    sim_results['paths'], 
                    req.strike_price,
                    current_price,
                    req.simulation_days
                )
                response["options"] = options_data
                
            return response
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
