import yfinance as yf
import pandas as pd

# Set the symbol and the original date range
symbol = "NQ=F"
original_start_date = "2023-01-01"
original_end_date = "2023-12-31"

# Fetch the original daily stock data
original_data = yf.download(symbol, start=original_start_date, end=original_end_date)

# Set the sub-section date range
sub_start_date = "2023-06-01"
sub_end_date = "2023-09-01"

# Extract the sub-section of the data
sub_data = original_data[(original_data.index >= sub_start_date) & (original_data.index <= sub_end_date)]

# Count the number of candlesticks in the sub-section
candlestick_count = len(sub_data)

print(f"Number of Candlesticks in {symbol} from {sub_start_date} to {sub_end_date}: {candlestick_count}")
