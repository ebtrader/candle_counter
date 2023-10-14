import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Set the symbol and the original date range
symbol = "NQ=F"
original_start_date = "2023-07-15"
original_end_date = "2023-10-15"

# Fetch the original daily stock data
original_data = yf.download(symbol, start=original_start_date, end=original_end_date)

# Define the aggregation interval (e.g., n=5)
aggregation_interval = 5

# Resample the original data to the specified aggregation interval
aggregated_data = original_data.resample(f'{aggregation_interval}D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum',
})

# Create a candlestick chart using Plotly with aggregated data
fig = go.Figure(data=[go.Candlestick(x=aggregated_data.index,
                open=aggregated_data['Open'],
                high=aggregated_data['High'],
                low=aggregated_data['Low'],
                close=aggregated_data['Close'],
                name=f"{symbol} Candlestick ({aggregation_interval}-day Aggregation)")])


# Set the chart title and labels
fig.update_layout(
    title=f"{symbol} Candlestick Chart ({original_start_date} to {original_end_date})",
    xaxis_title="Date",
    yaxis_title="Price",
)

# Set the x-axis range to display only the desired date range
fig.update_xaxes(range=[original_start_date, original_end_date])

# Show the plot
fig.write_html('NQ_tick_by_tick.html', auto_open=True)
