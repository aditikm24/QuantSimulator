from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import fetch_data
from models import AdvancedSimulationModels
import numpy as np

app = FastAPI()

static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

class SimulationRequest(BaseModel):
    ticker: str
    historical_days: int = 365
    simulation_days: int = 30
    model: str = "gbm" # "gbm" or "jump"
    num_simulations: int = 10
    strike_price: float | None = None

@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    try:
        # Parse tickers (comma separated)
        tickers = [t.strip().upper() for t in req.ticker.split(',') if t.strip()]
        if not tickers:
            raise ValueError("Please provide at least one ticker.")
            
        is_portfolio = len(tickers) > 1

        # Fetch historical data
        hist_data_dict = fetch_data.get_historical_data(tickers, req.historical_days)
        
        # Initialize models
        models = AdvancedSimulationModels(hist_data_dict)
        
        options_pricing = None
        var_metrics = None
        
        if is_portfolio:
            # Multi-asset: Portfolio Simulation
            sim_result = models.run_portfolio_simulation(req.simulation_days, num_simulations=req.num_simulations)
            simulations = sim_result["paths"]
            var_metrics = {
                "var_95": float(sim_result["var_95"]),
                "var_99": float(sim_result["var_99"]),
                "initial_value": float(sim_result["initial_value"])
            }
            
            # Aggregate historical prices for the portfolio (equal weight, 1 share each)
            hist_prices_array = np.array([hist_data_dict['prices'][t] for t in tickers])
            hist_prices = np.sum(hist_prices_array, axis=0).tolist()
            
        else:
            # Single asset simulation
            if req.model.lower() == "jump":
                simulations = models.run_jump_diffusion_simulation(req.simulation_days, req.num_simulations)
            else:
                simulations = models.run_gbm_simulation(req.simulation_days, req.num_simulations)
                
            hist_prices = hist_data_dict['prices'][tickers[0]]
            
            # Options Pricing only available for single stock
            if req.strike_price is not None:
                options_pricing = models.price_european_options(simulations, req.strike_price, req.simulation_days)

        # Format data for JSON response
        hist_dates = hist_data_dict['dates']
        
        # Generate simulation dates
        last_date_str = hist_dates[-1]
        last_date = fetch_data.datetime.strptime(last_date_str, '%Y-%m-%d')
        sim_dates = [(last_date + fetch_data.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(req.simulation_days + 1)]
        
        response = {
            "is_portfolio": is_portfolio,
            "historical": {
                "dates": hist_dates,
                "prices": hist_prices
            },
            "simulation": {
                "dates": sim_dates,
                "paths": simulations.tolist()
            },
            "options": options_pricing,
            "var_metrics": var_metrics
        }
        
        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
