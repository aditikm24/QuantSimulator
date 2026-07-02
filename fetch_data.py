import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_historical_data(tickers, days: int):
    """
    Fetches historical stock data for the given tickers for a specified number of days.
    tickers: list of strings (e.g. ['AAPL', 'MSFT'])
    Returns: A dictionary with:
      - 'dates': list of string dates
      - 'prices': dict mapping ticker -> list of closing prices
      - 'log_returns': pandas DataFrame of daily log returns for all tickers
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Download data
    data = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')
    if data.empty:
        raise ValueError(f"No data found for tickers {tickers}")
        
    # If a single ticker is passed, yfinance returns a single level column index.
    # If multiple, it returns a MultiIndex (PriceType, Ticker).
    # We only want 'Close' prices.
    
    if len(tickers) == 1:
        ticker = tickers[0]
        close_df = pd.DataFrame(data['Close'])
        close_df.columns = [ticker]
    else:
        # data['Close'] is already a DataFrame where columns are the tickers
        close_df = data['Close']
        
    # Drop rows with NaN values (e.g. market holidays or missing data)
    close_df = close_df.dropna()
    
    # Calculate daily log returns
    log_returns = np.log(close_df / close_df.shift(1)).dropna()
    
    # Prepare the output format
    dates = close_df.index.strftime('%Y-%m-%d').tolist()
    
    prices = {}
    for ticker in tickers:
        prices[ticker] = close_df[ticker].tolist()
        
    return {
        'dates': dates,
        'prices': prices,
        'log_returns': log_returns
    }
