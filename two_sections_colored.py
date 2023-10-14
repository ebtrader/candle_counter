import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Set the symbol and the original date range
symbol = "NQ=F"
original_start_date = "2023-07-01"
original_end_date = "2023-10-15"

# Fetch the original daily stock data
original_data = yf.download(symbol, start=original_start_date, end=original_end_date)

# Define sub-sections with their respective start and end dates
sub_sections = [
    {"start_date": "2023-07-21", "end_date": "2023-08-18"},
    {"start_date": "2023-08-18", "end_date": "2023-09-01"},
]

# Create a candlestick chart using Plotly
fig = go.Figure(data=[go.Candlestick(x=original_data.index,
                open=original_data['Open'],
                high=original_data['High'],
                low=original_data['Low'],
                close=original_data['Close'],
                name="Original Data")])

# Iterate through sub-sections and add colored backgrounds
for i, sub_section in enumerate(sub_sections):
    sub_start_date = sub_section["start_date"]
    sub_end_date = sub_section["end_date"]

    # Calculate the numerical indices of the sub-section
    sub_start_idx = original_data.index.get_loc(sub_start_date)
    sub_end_idx = original_data.index.get_loc(sub_end_date)

    # Add a rectangle shape to indicate the sub-dates with alternating colors
    color = "rgba(0, 255, 0, 0.2)" if i % 2 == 0 else "rgba(255, 0, 0, 0.2)"
    fig.add_vrect(
        x0=sub_start_idx,
        x1=sub_end_idx,
        fillcolor=color,
        layer="below",
        line_width=0,
    )

# Set the chart title and labels
fig.update_layout(
    title=f"{symbol} Candlestick Chart ({original_start_date} to {original_end_date})",
    xaxis_title="Date",
    yaxis_title="Price",
)

# Set the x-axis range to display only the desired date range
fig.update_xaxes(range=[original_start_date, original_end_date])

# Show the plot
fig.show()
