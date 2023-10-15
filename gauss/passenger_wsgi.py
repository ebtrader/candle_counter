import sys, os

INTERP = os.path.join(os.environ['HOME'], '', 'venv', 'bin', 'python3')
if sys.executable != INTERP:
    os.execl(INTERP, INTERP, *sys.argv)
sys.path.append(os.getcwd())

from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import plotly
import plotly.graph_objects as go
import numpy as np
import math
import json
from scipy.signal import argrelextrema

application = Flask(__name__)

@application.route('/simple')
def output_simple():
    out_text = 'hello world now'
    return render_template('output.html', arithmetic=out_text)

@application.route('/numbers')
def hello():
    ticker = 'NQ=F'
    data = yf.download(tickers=ticker, period='6mo')
    return data.to_html(header='true', table_id='table')

@application.route('/chart', methods=['GET', 'POST'])
def graph():
    if request.method == "POST":
        details = request.form

        ticker = details['ticker']

        period = details['period']
        interval = details['interval']

        df = yf.download(tickers=ticker, period=period, interval=interval)
        #df = yf.download(tickers = ticker, start='2013-01-01', end='2014-12-31')

        df = df.reset_index()

        Order = 5

        max_idx = argrelextrema(df['Close'].values, np.greater, order=Order)[0]
        min_idx = argrelextrema(df['Close'].values, np.less, order=Order)[0]


        fig1 = go.Figure(data=[go.Candlestick(x=df['Date'],
                                              open=df['Open'],
                                              high=df['High'],
                                              low=df['Low'],
                                              close=df['Close'], showlegend=False)])
        Size = 15
        Width = 1

        fig1.add_trace(
            go.Scatter(
                name='Sell Here!',
                mode='markers',
                x=df.iloc[max_idx]['Date'],
                y=df.iloc[max_idx]['High'],
                marker=dict(
                    symbol=46,
                    color='darkred',
                    size=Size,
                    line=dict(
                        color='MediumPurple',
                        width=Width
                    )
                ),
                showlegend=True
            )
        )

        fig1.add_trace(
            go.Scatter(
                name='Buy Here!',
                mode='markers',
                x=df.iloc[min_idx]['Date'],
                y=df.iloc[min_idx]['Low'],
                marker=dict(
                    symbol=45,
                    color='forestgreen',
                    size=Size,
                    line=dict(
                        color='MediumPurple',
                        width=Width
                    )
                ),
                showlegend=True
            )
        )

        # fig1.show()

        fig1.update_layout(
            title=ticker, xaxis_rangeslider_visible=False
        )

        graphJSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('neckline.html', graphJSON=graphJSON)
    return render_template('dropdown.html')

@application.route('/weekly', methods=['GET', 'POST'])
def weekly_graph():
    if request.method == "POST":
        details = request.form

        ticker = details['ticker']

        period = details['period']
        interval = details['interval']

        df = yf.download(tickers=ticker, period=period, interval=interval)
        #df = yf.download(tickers = ticker, start='2013-01-01', end='2014-12-31')

        df = df.reset_index()

        df7 = df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                                 'Volume': 'volume'}, inplace=False)

        # print(df7)
        # df7.to_csv('daily.csv')

        # interval = int(details['interval'])
        n = 1

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

        # print(df2)
        num_periods = 21
        # df2['SMA'] = TA.SMA(df2, 21)
        # df2['FRAMA'] = TA.FRAMA(df2, 10)
        # df2['TEMA'] = TA.TEMA(df2, num_periods)
        # df2['VWAP'] = TA.VWAP(df2)

        # how to get previous row's value
        # df2['previous'] = df2['lower_band'].shift(1)

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

        graphJSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('neckline.html', graphJSON=graphJSON)
    return render_template('dropdown1.html')

@application.route('/buffer', methods=['GET', 'POST'])
def buffer_graph():
    if request.method == "POST":
        details = request.form

        ticker = details['ticker']

        period = details['period']
        interval = details['interval']

        df = yf.download(tickers=ticker, period=period, interval=interval)
        #df = yf.download(tickers = ticker, start='2013-01-01', end='2014-12-31')

        df = df.reset_index()

        Order = 5

        max_idx = argrelextrema(df['Close'].values, np.greater, order=Order)[0]
        min_idx = argrelextrema(df['Close'].values, np.less, order=Order)[0]

        high_dates = df.iloc[max_idx]['Date']
        highs = df.iloc[max_idx]['High']

        # print(high_dates)
        # print(highs)

        df1 = pd.DataFrame(high_dates)
        df2 = pd.DataFrame(highs)

        # print(df1)
        # print(df2)

        df5 = df1.join(df2)
        df5.rename(columns={'High': 'Price'}, inplace=True)
        df5['Position'] = 'High'
        # print(df5)

        low_dates = df.iloc[min_idx]['Date']
        lows = df.iloc[min_idx]['Low']

        # print(low_dates)
        # print(lows)

        df3 = pd.DataFrame(low_dates)
        df4 = pd.DataFrame(lows)

        # print(df3)
        # print(df4)

        df6 = df3.join(df4)
        df6.rename(columns={'Low': 'Price'}, inplace=True)
        df6['Position'] = 'Low'
        # print(df6)

        frames = [df5, df6]
        df7 = pd.concat(frames)
        # print(df7)

        df8 = df7.sort_index(axis=0)
        # print(df8)

        df8['diff'] = abs(df8['Price'].diff())
        # print(df8)

        pct = .20

        df8['tenpct'] = pct * df8['diff']
        # print(df8)

        df8['diff'] = df8['diff'].fillna(0)
        df8['tenpct'] = df8['tenpct'].fillna(0)
        df8['buff_high'] = df8['Price'] - df8['tenpct']
        df8['buff_low'] = df8['Price'] + df8['tenpct']

        df8['buffer'] = np.where(df8.Position.str.contains('High'), df8['buff_high'],
                                 np.where(df8.Position.str.contains('Low'), df8['buff_low'], 0))
        print(df8)

        high_df = df8[df8['Position'] == 'High']
        print(high_df)

        low_df = df8[df8['Position'] == 'Low']
        print(low_df)

        fig1 = go.Figure(data=[go.Candlestick(x=df['Date'],
                                              open=df['Open'],
                                              high=df['High'],
                                              low=df['Low'],
                                              close=df['Close'], showlegend=False)])
        Size = 15
        Width = 1

        fig1.add_trace(
            go.Scatter(
                x=df.iloc[max_idx]['Date'],
                y=df.iloc[max_idx]['High'],
                name='upper band',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.add_trace(
            go.Scatter(
                name='Sell Calls!',
                mode='markers',
                x=df.iloc[max_idx]['Date'],
                y=df.iloc[max_idx]['High'],
                marker=dict(
                    symbol=46,
                    color='darkred',
                    size=Size,
                    line=dict(
                        color='MediumPurple',
                        width=Width
                    )
                ),
                showlegend=True
            )
        )

        fig1.add_trace(
            go.Scatter(
                x=df.iloc[min_idx]['Date'],
                y=df.iloc[min_idx]['Low'],
                name='lower band',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.add_trace(
            go.Scatter(
                name='Sell Puts!',
                mode='markers',
                x=df.iloc[min_idx]['Date'],
                y=df.iloc[min_idx]['Low'],
                marker=dict(
                    symbol=45,
                    color='forestgreen',
                    size=Size,
                    line=dict(
                        color='MediumPurple',
                        width=Width
                    )
                ),
                showlegend=True
            )
        )

        fig1.add_trace(
            go.Scatter(
                x=low_df['Date'],
                y=low_df['buffer'],
                name='Low Buffer',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.add_trace(
            go.Scatter(
                x=high_df['Date'],
                y=high_df['buffer'],
                name='High Buffer',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.update_layout(
            title=ticker, width=1800, height=1200, hovermode='x unified'
        )

        graphJSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('neckline.html', graphJSON=graphJSON)
    return render_template('dropdown1.html')

@application.route('/buffer1', methods=['GET', 'POST'])
def buffer_start_end_graph():
    if request.method == "POST":
        details = request.form

        ticker = details['ticker']

        start_date = str(details['start'])
        end_date = str(details['end'])

        # df = yf.download(tickers=ticker, period=period, interval=interval)
        df = yf.download(tickers=ticker, start=start_date, end=end_date, interval='1wk')

        df = df.reset_index()

        Order = 5

        max_idx = argrelextrema(df['Close'].values, np.greater, order=Order)[0]
        min_idx = argrelextrema(df['Close'].values, np.less, order=Order)[0]

        high_dates = df.iloc[max_idx]['Date']
        highs = df.iloc[max_idx]['High']

        # print(high_dates)
        # print(highs)

        df1 = pd.DataFrame(high_dates)
        df2 = pd.DataFrame(highs)

        # print(df1)
        # print(df2)

        df5 = df1.join(df2)
        df5.rename(columns={'High': 'Price'}, inplace=True)
        df5['Position'] = 'High'
        # print(df5)

        low_dates = df.iloc[min_idx]['Date']
        lows = df.iloc[min_idx]['Low']

        # print(low_dates)
        # print(lows)

        df3 = pd.DataFrame(low_dates)
        df4 = pd.DataFrame(lows)

        # print(df3)
        # print(df4)

        df6 = df3.join(df4)
        df6.rename(columns={'Low': 'Price'}, inplace=True)
        df6['Position'] = 'Low'
        # print(df6)

        frames = [df5, df6]
        df7 = pd.concat(frames)
        # print(df7)

        df8 = df7.sort_index(axis=0)
        # print(df8)

        df8['diff'] = abs(df8['Price'].diff())
        # print(df8)

        pct = .20

        df8['tenpct'] = pct * df8['diff']
        # print(df8)

        df8['diff'] = df8['diff'].fillna(0)
        df8['tenpct'] = df8['tenpct'].fillna(0)
        df8['buff_high'] = df8['Price'] - df8['tenpct']
        df8['buff_low'] = df8['Price'] + df8['tenpct']

        df8['buffer'] = np.where(df8.Position.str.contains('High'), df8['buff_high'],
                                 np.where(df8.Position.str.contains('Low'), df8['buff_low'], 0))
        print(df8)

        high_df = df8[df8['Position'] == 'High']
        print(high_df)

        low_df = df8[df8['Position'] == 'Low']
        print(low_df)

        fig1 = go.Figure(data=[go.Candlestick(x=df['Date'],
                                              open=df['Open'],
                                              high=df['High'],
                                              low=df['Low'],
                                              close=df['Close'], showlegend=False)])
        Size = 15
        Width = 1

        fig1.add_trace(
            go.Scatter(
                x=df.iloc[max_idx]['Date'],
                y=df.iloc[max_idx]['High'],
                name='upper band',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.add_trace(
            go.Scatter(
                name='Sell Calls!',
                mode='markers',
                x=df.iloc[max_idx]['Date'],
                y=df.iloc[max_idx]['High'],
                marker=dict(
                    symbol=46,
                    color='darkred',
                    size=Size,
                    line=dict(
                        color='MediumPurple',
                        width=Width
                    )
                ),
                showlegend=True
            )
        )

        fig1.add_trace(
            go.Scatter(
                x=df.iloc[min_idx]['Date'],
                y=df.iloc[min_idx]['Low'],
                name='lower band',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.add_trace(
            go.Scatter(
                name='Sell Puts!',
                mode='markers',
                x=df.iloc[min_idx]['Date'],
                y=df.iloc[min_idx]['Low'],
                marker=dict(
                    symbol=45,
                    color='forestgreen',
                    size=Size,
                    line=dict(
                        color='MediumPurple',
                        width=Width
                    )
                ),
                showlegend=True
            )
        )

        fig1.add_trace(
            go.Scatter(
                x=low_df['Date'],
                y=low_df['buffer'],
                name='Low Buffer',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.add_trace(
            go.Scatter(
                x=high_df['Date'],
                y=high_df['buffer'],
                name='High Buffer',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.update_layout(
            title=ticker, width=1800, height=1200, hovermode='x unified'
        )

        graphJSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('neckline.html', graphJSON=graphJSON)
    return render_template('start_end.html')

@application.route('/forecast', methods=['GET', 'POST'])
def forecast_graph():
    if request.method == "POST":
        details = request.form

        ticker = details['ticker']

        period = details['period']
        interval = details['interval']

        df = yf.download(tickers=ticker, period=period, interval=interval)
        #df = yf.download(tickers = ticker, start='2013-01-01', end='2014-12-31')

        df = df.reset_index()

        Order = 5

        max_idx = argrelextrema(df['Close'].values, np.greater, order=Order)[0]
        min_idx = argrelextrema(df['Close'].values, np.less, order=Order)[0]

        # get the high necklines

        # high_dates = df.iloc[max_idx]['Date']
        highs = df.iloc[max_idx]['High']

        # print(high_dates)
        # print(highs)

        # df1 = pd.DataFrame(high_dates)
        df2 = pd.DataFrame(highs)

        # print(df1)
        # print(df2)

        # df5 = df1.join(df2)
        df2.rename(columns={'High': 'Price'}, inplace=True)
        df2['Position'] = 'High'
        print(df2)
        df2.to_csv('highs.csv')

        # forecast high

        last_high_close = df2['Price'].iloc[-1]
        print(last_high_close)
        second_last_high_close = df2['Price'].iloc[-2]
        print(second_last_high_close)

        diff_high = last_high_close - second_last_high_close
        print(diff_high)

        new_high = last_high_close + diff_high
        print(new_high)

        diff_high_index = df2.index[-1] - df2.index[-2]
        print(diff_high_index)

        daily_incremental_high = diff_high / diff_high_index
        print(daily_incremental_high)

        new_high_index = df2.index[-1] + diff_high_index

        df2.loc[new_high_index] = [new_high, 'High']
        # print(df2)

        new_high1 = new_high + diff_high
        new_high_index1 = new_high_index + diff_high_index

        df2.loc[new_high_index1] = [new_high1, 'High']
        print(df2)

        # get each date point for the high forecast

        high_range1 = np.arange(last_high_close, new_high1, daily_incremental_high, dtype=list)
        print(high_range1)

        high_range1_df = pd.DataFrame(high_range1, columns=['Price'])
        high_range1_df['Position'] = 'High'
        print(high_range1_df)

        starting_high_index = int(df2.index[-3])
        print(starting_high_index)
        new_high_index1 = int(new_high_index1)
        range_high = np.arange(starting_high_index, new_high_index1, 1, dtype=list)
        print(range_high)
        high_range1_df.index = range_high
        print(high_range1_df)

        # drop last 3 rows
        df2a = df2.copy()
        df2a.drop(df2a.tail(3).index, inplace=True)
        print(df2a)

        frames = [df2a, high_range1_df]
        forecast_high_line = pd.concat(frames)
        print(forecast_high_line)

        # df10 = df3.interpolate(method='linear', limit_direction='forward', axis=0)
        # print(df10)

        # forecast low

        # low_dates = df.iloc[min_idx]['Date']
        lows = df.iloc[min_idx]['Low']

        # print(low_dates)
        # print(lows)

        # df3 = pd.DataFrame(low_dates)
        df4 = pd.DataFrame(lows)

        # print(df3)
        # print(df4)

        # df6 = df3.join(df4)
        df4.rename(columns={'Low': 'Price'}, inplace=True)
        df4['Position'] = 'Low'
        print(df4)

        # forecast low

        last_low_close = df4['Price'].iloc[-1]
        print(last_low_close)
        second_last_low_close = df4['Price'].iloc[-2]
        print(second_last_low_close)

        diff_low = last_low_close - second_last_low_close
        print(diff_low)

        new_low = last_low_close + diff_low
        print(new_low)

        diff_low_index = df4.index[-1] - df4.index[-2]
        print(diff_low_index)

        new_low_index = df4.index[-1] + diff_low_index

        df4.loc[new_low_index] = [new_low, 'Low']
        # print(df2)

        new_low1 = new_low + diff_low
        new_low_index1 = new_low_index + diff_low_index

        df4.loc[new_low_index1] = [new_low1, 'Low']
        print(df4)

        # get the points for the high forecast

        daily_incremental_low = diff_low / diff_low_index
        print(daily_incremental_low)

        low_range1 = np.arange(last_low_close, new_low1, daily_incremental_low, dtype=list)
        print(low_range1)

        low_range1_df = pd.DataFrame(low_range1, columns=['Price'])
        low_range1_df['Position'] = 'Low'
        print(low_range1_df)

        starting_low_index = int(df4.index[-3])
        print(starting_low_index)
        new_low_index1 = int(new_low_index1)
        range_low = np.arange(starting_low_index, new_low_index1, 1, dtype=list)
        print(range_low)
        low_range1_df.index = range_low
        print(low_range1_df)

        # drop last 3 rows
        df4a = df4.copy()
        df4a.drop(df4a.tail(3).index, inplace=True)
        print(df4a)

        frames = [df4a, low_range1_df]
        forecast_low_line = pd.concat(frames)
        print(forecast_low_line)

        frames = [df2, df4]
        df7 = pd.concat(frames)
        # print(df7)

        df8 = df7.sort_index(axis=0)
        # print(df8)

        df8['diff'] = abs(df8['Price'].diff())
        # print(df8)

        pct = .20

        df8['tenpct'] = pct*df8['diff']
        # print(df8)

        df8['diff'] = df8['diff'].fillna(0)
        df8['tenpct'] = df8['tenpct'].fillna(0)
        df8['buff_high'] = df8['Price'] - df8['tenpct']
        df8['buff_low'] = df8['Price'] + df8['tenpct']

        df8['buffer'] = np.where(df8.Position.str.contains('High'), df8['buff_high'],
                                 np.where(df8.Position.str.contains('Low'), df8['buff_low'], 0))
        # print(df8)

        high_df = df8[df8['Position'] == 'High']
        # print(high_df)

        low_df = df8[df8['Position'] == 'Low']
        # print(low_df)

        fig1 = go.Figure(data=[go.Candlestick(x=df.index,
                                              open=df['Open'],
                                              high=df['High'],
                                              low=df['Low'],
                                              close=df['Close'], showlegend=False)])
        Size = 15
        Width = 1

        fig1.add_trace(
            go.Scatter(
                x=df2.index,
                y=df2['Price'],
                name='upper band',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.add_trace(
            go.Scatter(
                x=forecast_high_line.index,
                y=forecast_high_line['Price'],
                name='upper band',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.add_trace(
            go.Scatter(
                x=forecast_low_line.index,
                y=forecast_low_line['Price'],
                name='lower band',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )


        fig1.add_trace(
            go.Scatter(
                name='Sell Calls!',
                mode='markers',
                x=df.iloc[max_idx].index,
                y=df.iloc[max_idx]['High'],
                marker=dict(
                    symbol=46,
                    color='darkred',
                    size=Size,
                    line=dict(
                        color='MediumPurple',
                        width=Width
                    )
                ),
                showlegend=True
            )
        )

        fig1.add_trace(
            go.Scatter(
                x=df4.index,
                y=df4['Price'],
                name='lower band',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.add_trace(
            go.Scatter(
                name='Sell Puts!',
                mode='markers',
                x=df.iloc[min_idx].index,
                y=df.iloc[min_idx]['Low'],
                marker=dict(
                    symbol=45,
                    color='forestgreen',
                    size=Size,
                    line=dict(
                        color='MediumPurple',
                        width=Width
                    )
                ),
                showlegend=True
            )
        )

        fig1.add_trace(
            go.Scatter(
                x=low_df.index,
                y=low_df['buffer'],
                name='Low Buffer',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.add_trace(
            go.Scatter(
                x=high_df.index,
                y=high_df['buffer'],
                name='High Buffer',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.add_vline(x=df2.index[-3], line_width=3, line_dash="dash", line_color="darkred")

        fig1.add_vline(x=df4.index[-3], line_width=3, line_dash="dash", line_color="green")

        fig1.update_layout(
                    title=ticker
                )

        graphJSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('neckline.html', graphJSON=graphJSON)
    return render_template('dropdown2.html')

@application.route('/forecast1', methods=['GET', 'POST'])
def forecast_graph1():
    if request.method == "POST":
        details = request.form

        ticker = details['ticker']

        period = details['period']
        interval = details['interval']

        df = yf.download(tickers=ticker, period=period, interval=interval)
        #df = yf.download(tickers = ticker, start='2013-01-01', end='2014-12-31')

        df = df.reset_index()

        Order = 5

        max_idx = argrelextrema(df['Close'].values, np.greater, order=Order)[0]
        min_idx = argrelextrema(df['Close'].values, np.less, order=Order)[0]

        # get the high necklines

        # high_dates = df.iloc[max_idx]['Date']
        highs = df.iloc[max_idx]['High']

        # print(high_dates)
        # print(highs)

        # df1 = pd.DataFrame(high_dates)
        df2 = pd.DataFrame(highs)

        # print(df1)
        # print(df2)

        # df5 = df1.join(df2)
        df2.rename(columns={'High': 'Price'}, inplace=True)
        df2['Position'] = 'High'
        print(df2)
        df2.to_csv('highs.csv')

        # forecast high

        last_high_close = df2['Price'].iloc[-1]
        print(last_high_close)
        second_last_high_close = df2['Price'].iloc[-2]
        print(second_last_high_close)

        diff_high = last_high_close - second_last_high_close
        print(diff_high)

        new_high = last_high_close + diff_high
        print(new_high)

        diff_high_index = df2.index[-1] - df2.index[-2]
        print(diff_high_index)

        daily_incremental_high = diff_high / diff_high_index
        print(daily_incremental_high)

        new_high_index = df2.index[-1] + diff_high_index

        df2.loc[new_high_index] = [new_high, 'High']
        # print(df2)

        new_high1 = new_high + diff_high
        new_high_index1 = new_high_index + diff_high_index

        df2.loc[new_high_index1] = [new_high1, 'High']
        print(df2)

        # get each date point for the high forecast

        high_range1 = np.arange(last_high_close, new_high1, daily_incremental_high, dtype=list)
        print(high_range1)

        high_range1_df = pd.DataFrame(high_range1, columns=['Price'])
        high_range1_df['Position'] = 'High'
        print(high_range1_df)

        starting_high_index = int(df2.index[-3])
        print(starting_high_index)
        new_high_index1 = int(new_high_index1)
        range_high = np.arange(starting_high_index, new_high_index1, 1, dtype=list)
        print(range_high)
        high_range1_df.index = range_high
        print(high_range1_df)

        # drop last 3 rows
        df2a = df2.copy()
        df2a.drop(df2a.tail(3).index, inplace=True)
        print(df2a)

        frames = [df2a, high_range1_df]
        forecast_high_line = pd.concat(frames)
        print(forecast_high_line)

        # df10 = df3.interpolate(method='linear', limit_direction='forward', axis=0)
        # print(df10)

        # forecast low

        # low_dates = df.iloc[min_idx]['Date']
        lows = df.iloc[min_idx]['Low']

        # print(low_dates)
        # print(lows)

        # df3 = pd.DataFrame(low_dates)
        df4 = pd.DataFrame(lows)

        # print(df3)
        # print(df4)

        # df6 = df3.join(df4)
        df4.rename(columns={'Low': 'Price'}, inplace=True)
        df4['Position'] = 'Low'
        print(df4)

        # forecast low

        last_low_close = df4['Price'].iloc[-1]
        print(last_low_close)
        second_last_low_close = df4['Price'].iloc[-2]
        print(second_last_low_close)

        diff_low = last_low_close - second_last_low_close
        print(diff_low)

        new_low = last_low_close + diff_low
        print(new_low)

        diff_low_index = df4.index[-1] - df4.index[-2]
        print(diff_low_index)

        new_low_index = df4.index[-1] + diff_low_index

        df4.loc[new_low_index] = [new_low, 'Low']
        # print(df2)

        new_low1 = new_low + diff_low
        new_low_index1 = new_low_index + diff_low_index

        df4.loc[new_low_index1] = [new_low1, 'Low']
        print(df4)

        # get the points for the high forecast

        daily_incremental_low = diff_low / diff_low_index
        print(daily_incremental_low)

        low_range1 = np.arange(last_low_close, new_low1, daily_incremental_low, dtype=list)
        print(low_range1)

        low_range1_df = pd.DataFrame(low_range1, columns=['Price'])
        low_range1_df['Position'] = 'Low'
        print(low_range1_df)

        starting_low_index = int(df4.index[-3])
        print(starting_low_index)
        new_low_index1 = int(new_low_index1)
        range_low = np.arange(starting_low_index, new_low_index1, 1, dtype=list)
        print(range_low)
        low_range1_df.index = range_low
        print(low_range1_df)

        # drop last 3 rows
        df4a = df4.copy()
        df4a.drop(df4a.tail(3).index, inplace=True)
        print(df4a)

        frames = [df4a, low_range1_df]
        forecast_low_line = pd.concat(frames)
        print(forecast_low_line)

        frames = [df2, df4]
        df7 = pd.concat(frames)
        # print(df7)

        df8 = df7.sort_index(axis=0)
        # print(df8)

        df8['diff'] = abs(df8['Price'].diff())
        # print(df8)

        pct = .20

        df8['tenpct'] = pct*df8['diff']
        # print(df8)

        df8['diff'] = df8['diff'].fillna(0)
        df8['tenpct'] = df8['tenpct'].fillna(0)
        df8['buff_high'] = df8['Price'] - df8['tenpct']
        df8['buff_low'] = df8['Price'] + df8['tenpct']

        df8['buffer'] = np.where(df8.Position.str.contains('High'), df8['buff_high'],
                                 np.where(df8.Position.str.contains('Low'), df8['buff_low'], 0))
        # print(df8)

        high_df = df8[df8['Position'] == 'High']
        # print(high_df)

        low_df = df8[df8['Position'] == 'Low']
        # print(low_df)

        fig1 = go.Figure(data=[go.Candlestick(x=df.index,
                                              open=df['Open'],
                                              high=df['High'],
                                              low=df['Low'],
                                              close=df['Close'], showlegend=False)])
        Size = 15
        Width = 1

        fig1.add_trace(
            go.Scatter(
                x=df2.index,
                y=df2['Price'],
                name='upper band',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.add_trace(
            go.Scatter(
                x=forecast_high_line.index,
                y=forecast_high_line['Price'],
                name='upper band',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.add_trace(
            go.Scatter(
                x=forecast_low_line.index,
                y=forecast_low_line['Price'],
                name='lower band',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )


        fig1.add_trace(
            go.Scatter(
                name='Sell Calls!',
                mode='markers',
                x=df.iloc[max_idx].index,
                y=df.iloc[max_idx]['High'],
                marker=dict(
                    symbol=46,
                    color='darkred',
                    size=Size,
                    line=dict(
                        color='MediumPurple',
                        width=Width
                    )
                ),
                showlegend=True
            )
        )

        fig1.add_trace(
            go.Scatter(
                x=df4.index,
                y=df4['Price'],
                name='lower band',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.add_trace(
            go.Scatter(
                name='Sell Puts!',
                mode='markers',
                x=df.iloc[min_idx].index,
                y=df.iloc[min_idx]['Low'],
                marker=dict(
                    symbol=45,
                    color='forestgreen',
                    size=Size,
                    line=dict(
                        color='MediumPurple',
                        width=Width
                    )
                ),
                showlegend=True
            )
        )

        fig1.add_trace(
            go.Scatter(
                x=low_df.index,
                y=low_df['buffer'],
                name='Low Buffer',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.add_trace(
            go.Scatter(
                x=high_df.index,
                y=high_df['buffer'],
                name='High Buffer',
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig1.add_vline(x=df2.index[-3], line_width=3, line_dash="dash", line_color="darkred")

        fig1.add_vline(x=df4.index[-3], line_width=3, line_dash="dash", line_color="green")

        fig1.update_layout(
                    title=ticker, width=1800, height=1200, hovermode='x unified'
                )

        graphJSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('neckline.html', graphJSON=graphJSON)
    return render_template('dropdown2.html')
