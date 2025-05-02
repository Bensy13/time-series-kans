import yfinance as yf
import pandas as pd

# Tickers: Stocks + Indices
tickers = ['NVDA', 'AAPL', 'KO', '^GSPC', '^DJI']
start_date = '2018-01-01'
end_date = '2024-12-31'
file_path = r"../data/stock_data.parquet"
# Download full data (default includes OHLCV)
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

# Flatten to a MultiIndex DataFrame
flattened = pd.concat({
    ticker: data[ticker][['Open', 'High', 'Low', 'Close', 'Volume']]
    for ticker in ['NVDA', 'AAPL', 'KO', '^GSPC', '^DJI']
}, axis=1)

# Rename index for easier access
flattened.columns.names = ['Ticker', 'Feature']
flattened = flattened.ffill().dropna()

# Save to Parquet
flattened.to_parquet(file_path)

print(f"Full OHLCV data saved at {file_path}")
print(flattened.tail())
