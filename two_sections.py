import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Set the symbol and the original date range
symbol = "NQ=F"
original_start_date = "2023-01-01"
original_end_date = "2023-12-31"

# Fetch the original daily stock data
original_data = yf.download(symbol, start=original_start_date, end=original_end_date)

# Define sub-sections with their respective start and end dates
sub_sections = [
    {"start_date": "2023-06-01", "end_date": "2023-09-01"},
    {"start_date": "2023-02-01", "end_date": "2023-04-03"},
]

# Create a candlestick chart using Plotly
fig = go.Figure(data=[go.Candlestick(x=original_data.index,
                open=original_data['Open'],
                high=original_data['High'],
                low=original_data['Low'],
                close=original_data['Close'],
                name="Original Data")])

# Iterate through sub-sections and add shapes and counter numbers
for sub_section in sub_sections:
    sub_start_date = sub_section["start_date"]
    sub_end_date = sub_section["end_date"]

    # Calculate the numerical indices of the sub-section
    sub_start_idx = original_data.index.get_loc(sub_start_date)
    sub_end_idx = original_data.index.get_loc(sub_end_date)

    # Add a rectangle shape to indicate the sub-dates
    fig.add_vrect(
        x0=sub_start_idx,
        x1=sub_end_idx,
        fillcolor="rgba(0, 255, 0, 0.2)",
        layer="below",
        line_width=0,
    )

    # Count the number of candlesticks in the sub-section
    sub_data = original_data[sub_start_date:sub_end_date]
    candlestick_count = len(sub_data)

    # Calculate the y-position for the counter numbers with a larger buffer
    y_position = sub_data['High'] + (sub_data['High'].max() - sub_data['High'].min()) * 0.2

    # Add counter numbers above the candles for the sub-dates
    candle_counter = go.Scatter(x=sub_data.index, y=y_position, mode='text', text=[f'{c}' for c in range(1, len(sub_data)+1)])

    # Add the counter numbers to the figure
    fig.add_trace(candle_counter)

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
