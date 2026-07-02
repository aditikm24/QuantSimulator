import pandas as pd
from datetime import datetime
import urllib.request
import json
import math
import numpy as np

def fetch_data(tickers, days=365):
    """
    Fetches historical adjusted close prices for given tickers using Yahoo Finance's raw JSON API.
    This avoids the 'yfinance' library which hangs on Vercel Serverless.
    Returns a dictionary of Pandas DataFrames with 'Close' and 'Log_Return' columns.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
        
    data_dict = {}
    
    for ticker in tickers:
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range={days}d"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                raw_data = json.loads(response.read().decode())
                
                if raw_data.get('chart', {}).get('error'):
                    raise ValueError(f"Error fetching {ticker}")
                    
                result = raw_data['chart']['result'][0]
                timestamps = result['timestamp']
                close_prices = result['indicators']['quote'][0]['close']
                
                # Filter out nulls
                valid_data = [(ts, price) for ts, price in zip(timestamps, close_prices) if price is not None]
                if not valid_data:
                    raise ValueError(f"No valid price data for {ticker}")
                    
                dates = [datetime.fromtimestamp(ts) for ts, price in valid_data]
                prices = [price for ts, price in valid_data]
                
                df = pd.DataFrame({'Close': prices}, index=dates)
                df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
                df = df.dropna()
                
                data_dict[ticker] = df
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {ticker}: {str(e)}")
            
    return data_dict
