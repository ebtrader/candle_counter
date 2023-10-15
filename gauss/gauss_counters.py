import yfinance as yf
import numpy as np
import math
import pandas as pd
import plotly.graph_objects as go

ticker="NQ=F"
# df = yf.download(tickers=ticker, period=period, interval=interval)

original_start_date = "2013-01-01"
original_end_date = "2014-12-31"

df = yf.download(tickers=ticker, start=original_start_date, end=original_end_date)

df = df.reset_index()

df7 = df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                         'Volume': 'volume'}, inplace=False)

# print(df7)
# df7.to_csv('daily.csv')

# interval = int(details['interval'])
n = 5

df3 = df7.groupby(np.arange(len(df7)) // n).max()  # high

df4 = df7.groupby(np.arange(len(df7)) // n).min()  # low

df5 = df7.groupby(np.arange(len(df7)) // n).first()  # open

df6 = df7.groupby(np.arange(len(df7)) // n).last()  # close

agg_df = pd.DataFrame()

agg_df['date'] = df6['date']
agg_df['low'] = df4['low']
agg_df['high'] = df3['high']
agg_df['open'] = df5['open']
agg_df['close'] = df6['close']

df2 = agg_df
print(df2)
num_periods = 21

# recursive
df2['test'] = 5
df2.loc[0, 'diff'] = df2.loc[0, 'test'] * 0.4
df2.loc[1, 'diff'] = df2.loc[1, 'test'] * 0.4
df2.loc[2, 'diff'] = df2.loc[2, 'test'] * 0.4
df2.loc[3, 'diff'] = df2.loc[3, 'test'] * 0.4

for i in range(4, len(df2)):
    df2.loc[i, 'diff'] = df2.loc[i, 'test'] + df2.loc[i - 1, 'diff'] + df2.loc[i - 2, 'diff']

# Gauss
num_periods_gauss = 15.5
df2['symbol'] = 2 * math.pi / num_periods_gauss
df2['beta'] = (1 - np.cos(df2['symbol'])) / ((1.414) ** (0.5) - 1)
df2['alpha'] = - df2['beta'] + (df2['beta'] ** 2 + df2['beta'] * 2) ** 2

# Gauss equation
# initialize
df2.loc[0, 'gauss'] = df2.loc[0, 'close']
df2.loc[1, 'gauss'] = df2.loc[1, 'close']
df2.loc[2, 'gauss'] = df2.loc[2, 'close']
df2.loc[3, 'gauss'] = df2.loc[3, 'close']
df2.loc[4, 'gauss'] = df2.loc[4, 'close']

for i in range(4, len(df2)):
    df2.loc[i, 'gauss'] = df2.loc[i, 'close'] * df2.loc[i, 'alpha'] ** 4 + (4 * (1 - df2.loc[i, 'alpha'])) * \
                          df2.loc[i - 1, 'gauss'] \
                          - (6 * ((1 - df2.loc[i, 'alpha']) ** 2) * df2.loc[i - 2, 'gauss']) \
                          + (4 * (1 - df2.loc[i, 'alpha']) ** 3) * df2.loc[i - 3, 'gauss'] \
                          - ((1 - df2.loc[i, 'alpha']) ** 4) * df2.loc[i - 4, 'gauss']

# ATR
num_periods_ATR = 21
multiplier = 1

df2['ATR_diff'] = df2['high'] - df2['low']
df2['ATR'] = df2['ATR_diff'].ewm(span=num_periods_ATR, adjust=False).mean()

df2['Line'] = df2['gauss']

# upper bands and ATR

df2['upper_band'] = df2['Line'] + multiplier * df2['ATR']
df2['lower_band'] = df2['Line'] - multiplier * df2['ATR']

multiplier_1 = 1.6
multiplier_2 = 2.3

# new multipliers
multiplier_3 = 0.5

df2['upper_band_1'] = df2['Line'] + multiplier_1 * df2['ATR']
df2['lower_band_1'] = df2['Line'] - multiplier_1 * df2['ATR']

df2['upper_band_2'] = df2['Line'] + multiplier_2 * df2['ATR']
df2['lower_band_2'] = df2['Line'] - multiplier_2 * df2['ATR']

df2['upper_band_3'] = df2['Line'] + multiplier_3 * df2['ATR']
df2['lower_band_3'] = df2['Line'] - multiplier_3 * df2['ATR']

# forecasting begins

# calculate atr
df4 = pd.DataFrame()
df4['date'] = df2['date']
df4['close'] = df2['ATR']
df4['open'] = df2['ATR']
df4['high'] = df2['ATR']
df4['low'] = df2['ATR']

fig1 = go.Figure(data=[go.Candlestick(x=df2['date'],
                                      open=df2['open'],
                                      high=df2['high'],
                                      low=df2['low'],
                                      close=df2['close'], showlegend=True)]

                 )

fig1.add_trace(
    go.Scatter(
        x=df2['date'],
        y=df2['upper_band'].round(2),
        name='upper band',
        mode="lines",
        line=go.scatter.Line(color="gray"),
        showlegend=True)
)

fig1.add_trace(
    go.Scatter(
        x=df2['date'],
        y=df2['lower_band'].round(2),
        name='lower band',
        mode="lines",
        line=go.scatter.Line(color="gray"),
        showlegend=True)
)

fig1.add_trace(
    go.Scatter(
        x=df2['date'],
        y=df2['upper_band_1'].round(2),
        name='upper band_1',
        mode="lines",
        line=go.scatter.Line(color="gray"),
        showlegend=True)
)

fig1.add_trace(
    go.Scatter(
        x=df2['date'],
        y=df2['lower_band_1'].round(2),
        name='lower band_1',
        mode="lines",
        line=go.scatter.Line(color="gray"),
        showlegend=True)
)

fig1.add_trace(
    go.Scatter(
        x=df2['date'],
        y=df2['upper_band_3'].round(2),
        name='upper band_3',
        mode="lines",
        line=go.scatter.Line(color="gray"),
        showlegend=True)
)

fig1.add_trace(
    go.Scatter(
        x=df2['date'],
        y=df2['lower_band_3'].round(2),
        name='lower band_3',
        mode="lines",
        line=go.scatter.Line(color="gray"),
        showlegend=True)
)

fig1.add_trace(
    go.Scatter(
        x=df2['date'],
        y=df2['Line'].round(2),
        name="gauss",
        mode="lines",
        line=go.scatter.Line(color="blue"),
        showlegend=True)
)

# get index to be date
print(df2)
df3 = df2.set_index('date')
print(df3)

# Define sub-sections with their respective start and end dates
sub_sections = [
    {"start_date": "2013-01-08", "end_date": "2013-01-30"}
]

# Iterate through sub-sections and add shapes, counter numbers, and lines
for sub_section in sub_sections:
    sub_start_date = sub_section["start_date"]
    sub_end_date = sub_section["end_date"]

    sub_start_idx = df3.index.get_loc(sub_start_date)
    sub_end_idx = df3.index.get_loc(sub_end_date)

    # Add a rectangle shape to indicate the sub-dates
    fig1.add_vrect(
        x0=sub_start_date,
        x1=sub_end_date,
        fillcolor="rgba(0, 255, 0, 0.2)",
        layer="below",
        line_width=0,
    )

    # Count the number of candlesticks in the sub-section
    sub_data = df3[sub_start_date:sub_end_date]
    candlestick_count = len(sub_data)

    # Calculate the y-position for the counter numbers with a larger buffer
    y_position = sub_data['high'] + (sub_data['high'].max() - sub_data['high'].min()) * 0.5

    # Calculate the corresponding x-values for the counter numbers based on the aggregation interval
    x_values = df3.index[sub_start_idx:sub_end_idx + 1]

    # Add counter numbers above the candles for the sub-dates
    candle_counter = go.Scatter(x=x_values, y=y_position, mode='text',
                                text=[f'{c}' for c in range(1, len(x_values) + 1)])

    # Add the counter numbers to the figure
    fig1.add_trace(candle_counter)

    # Draw a horizontal line at the highest and lowest levels for the sub-section
    highest_level = sub_data['high'].max()
    lowest_level = sub_data['low'].min()
    highest_line = go.Scatter(x=[sub_start_date, sub_end_date], y=[highest_level, highest_level],
                              mode='lines', name=f'Highest Level', line=dict(color='red', width=1))
    lowest_line = go.Scatter(x=[sub_start_date, sub_end_date], y=[lowest_level, lowest_level],
                             mode='lines', name=f'Lowest Level', line=dict(color='blue', width=1))

    # Calculate the midpoint between the high and low
    middle_level = (highest_level + lowest_level) / 2
    middle_line = go.Scatter(x=[sub_start_date, sub_end_date], y=[middle_level, middle_level],
                             mode='lines', name=f'Middle Level', line=dict(color='purple', width=1, dash='dash'))

    fig1.add_trace(highest_line)
    fig1.add_trace(lowest_line)
    fig1.add_trace(middle_line)

# Set the chart title and labels
fig1.update_layout(
    title=f"{ticker} Candlestick Chart ({original_start_date} to {original_end_date})",
    xaxis_title="Date",
    yaxis_title="Price",
)

# Set the x-axis range to display only the desired date range
fig1.update_xaxes(range=[original_start_date, original_end_date])

# Show the plot
fig1.write_html('NQ_tick_by_tick.html', auto_open=True)
