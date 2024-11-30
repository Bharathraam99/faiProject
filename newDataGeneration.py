import requests
import pandas as pd
import os
from datetime import datetime, timedelta

# Alpha Vantage API key
API_KEY = "QEFVQOMF9QYHO5D8"  # Replace with your API key

stocks = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "ADBE", "INTC", 
    "BRK.B", "JNJ", "PG", "XOM", "V", "KO", "DIS", "MA", "AMD", "CVX"
]
# Output directory for saving CSV files
output_dir = "new_dataset/historical_data"
os.makedirs(output_dir, exist_ok=True)

# Function to fetch stock data and save it as a CSV
def fetch_and_save_stock_data(stock_symbol):
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": stock_symbol,
        "apikey": API_KEY,
        "outputsize": "full",  # Fetch as much data as possible
        "datatype": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if "Time Series (Daily)" in data:
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient="index")
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. adjusted close": "Adjusted Close",
            "6. volume": "Volume",
            "7. dividend amount": "Dividend Amount",
            "8. split coefficient": "Split Coefficient"
        })
        df.index = pd.to_datetime(df.index)  # Convert index to datetime
        df = df.sort_index()  # Sort by date
        
        # Filter for the last 3 years
        three_years_ago = datetime.now() - timedelta(days=3*365)
        df = df[df.index >= three_years_ago]
        
        # Reset index to include date as a column
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)
        
        output_file = os.path.join(output_dir, f"{stock_symbol}.csv")
        df.to_csv(output_file, index=False)  # Save without row numbers
        print(f"3 years of data for {stock_symbol} saved to {output_file}")
    else:
        print(f"Failed to fetch data for {stock_symbol}: {data.get('Note', 'Unknown error')}")

# Fetch and save data for all stocks
for stock in stocks:
    fetch_and_save_stock_data(stock)
