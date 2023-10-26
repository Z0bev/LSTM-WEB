import os
import yfinance as yf
import pandas as pd

def get_top_10_companies():
    # Replace the symbols with the tickers of the 10 most popular companies you want to fetch data for
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "PG", "JNJ"]
    return symbols

def download_stock_data(tickers):
    all_data = {}
    script_dir = os.path.dirname(__file__)
    
    for ticker in tickers:
        try:
            data = yf.download(ticker,start="2020-7-21", end="2023-7-21", period="1y", interval="1d", prepost=True, threads=True, proxy=None)
            all_data[ticker] = data
            file_path = os.path.join(script_dir, f"{ticker}.csv")
            data.to_csv(file_path)
            
        except:
            print(f"Failed to download data for {ticker}.")
    
    return all_data

if __name__ == "__main__":
    top_10_tickers = get_top_10_companies()
    stock_data = download_stock_data(top_10_tickers)