import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Set the symbol and the original date range
symbol = "NQ=F"
original_start_date = "2023-01-01"
original_end_date = "2023-12-31"

# Fetch the original daily stock data
original_data = yf.download(symbol, start=original_start_date, end=original_end_date)

# Set the sub-section date range
sub_start_date = "2023-06-01"
sub_end_date = "2023-09-01"

# Create a candlestick chart using Plotly
fig = go.Figure(data=[go.Candlestick(x=original_data.index,
                open=original_data['Open'],
                high=original_data['High'],
                low=original_data['Low'],
                close=original_data['Close'],
                name="Original Data")])

# Add a rectangle shape to indicate the sub-dates
fig.add_vrect(
    x0=sub_start_date,
    x1=sub_end_date,
    fillcolor="rgba(0, 255, 0, 0.2)",
    layer="below",
    line_width=0,
)

# Set the chart title and labels
fig.update_layout(
    title=f"{symbol} Candlestick Chart ({original_start_date} to {original_end_date})",
    xaxis_title="Date",
    yaxis_title="Price",
)

# Show the plot
fig.show()
