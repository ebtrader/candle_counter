import yfinance as yf
import numpy as np
import math
import pandas as pd
import plotly.graph_objects as go

ticker="NQ=F"
# df = yf.download(tickers=ticker, period=period, interval=interval)
df = yf.download(tickers = ticker, start='2013-01-01', end='2014-12-31')

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

fig1.update_layout(
    title=ticker, width=1800, height=1200, hovermode='x unified'
)
# Show the plot
fig1.write_html('NQ_tick_by_tick.html', auto_open=True)
