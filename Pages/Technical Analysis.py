import streamlit as st
import sqlalchemy
import pymysql
import ta
import pandas_ta
import pandas as pd
import numpy as np
from tabulate import tabulate
from datetime import date
import plotly.graph_objects as go

st.set_page_config(
    page_title='Day Stock Predictor App',
    page_icon="❄",
    layout='wide'
)

# Data Retrieval from SQL Workbench

pymysql.install_as_MySQLdb()
engine = sqlalchemy.create_engine('mysql://root:12345@localhost:3306/')

#Function to get tables from SQL Workbench

def getTables(Schema):
    query = f"""SELECT table_name FROM information_schema.tables
    Where table_schema ='{Schema}'"""
    df = pd.read_sql(query, engine)
    df['Schema'] = Schema
    return df

#Function to fecth stock prices from SQL Workbench

def getPrices(tables_list):
    price = []
    for table, schema in zip(tables_list.TABLE_NAME, tables_list.Schema):
        sql = schema + '.' + f'`{table}`'
        price.append(pd.read_sql(f"SELECT Date, Close FROM {sql}", engine))
    return price

#Function to run Simple Moving Average Strategy

def SMA_Strategy(data):
    data['SMA 30'] = pandas_ta.sma(data['Close'], 30)
    data['SMA 100'] = pandas_ta.sma(data['Close'], 100)
    buySignalCost = []
    sellSignalCost = []
    buySignal = []
    sellSignal = []
    position = False
    for i in range(len(data)):
        if data['SMA 30'][i] > data['SMA 100'][i]:
            if position == False:
                buySignal.append("Buy")
                sellSignal.append(np.nan)
                buySignalCost.append(data.Close[i])
                sellSignalCost.append(np.nan)
                position = True
            else:
                buySignal.append("Hold/Buy")
                sellSignal.append(np.nan)
                buySignalCost.append(np.nan)
                sellSignalCost.append(np.nan)
        elif data['SMA 30'][i] < data['SMA 100'][i]:
            if position == True:
                buySignal.append(np.nan)
                sellSignal.append("Sell")
                sellSignalCost.append(data.Close[i])
                buySignalCost.append(np.nan)
                position = False
            else:
                buySignal.append(np.nan)
                sellSignal.append(np.nan)
                buySignalCost.append(np.nan)
                sellSignalCost.append(np.nan)
        else:
            buySignal.append(np.nan)
            sellSignal.append(np.nan)
            buySignalCost.append(np.nan)
            sellSignalCost.append(np.nan)

    data['SMA_Buy_Signal_price'] = buySignalCost
    data['SMA_Sell_Signal_price'] = sellSignalCost
    data['SMA_Buy_Signal'] = buySignal
    data['SMA_Sell_Signal'] = sellSignal
    return data

#Function to run Moving Average Convergance/Divergence Strategy

def MACD_Strategy(data, risk):
    macd = pandas_ta.macd(data['Close'])
    data = pd.concat([data, macd], axis=1).reindex(data.index)
    buySignalCost = []
    sellSignalCost = []
    buySignal = []
    sellSignal = []
    position = False

    for i in range(0, len(data)):
        if data['MACD_12_26_9'][i] > data['MACDs_12_26_9'][i]:
            sellSignal.append(np.nan)
            sellSignalCost.append(np.nan)
            if position == False:
                buySignal.append("Buy")
                buySignalCost.append(data.Close[i])
                position = True
            else:
                buySignal.append("Hold/Buy")
                buySignalCost.append(np.nan)
        elif data['MACD_12_26_9'][i] < data['MACDs_12_26_9'][i]:
            buySignal.append(np.nan)
            buySignalCost.append(np.nan)
            if position == True:
                sellSignal.append("Sell")
                sellSignalCost.append(data.Close[i])
                position = False
            else:
                sellSignal.append(np.nan)
                sellSignalCost.append(np.nan)
        elif position == True and data['Close'][i] < buySignalCost[-1] * (1 - risk):
            sellSignal.append("Sell")
            buySignal.append(np.nan)
            buySignalCost.append(np.nan)
            sellSignalCost.append(data.Close[i])
            position = False
        elif position == True and data['Close'][i] < data['Close'][i - 1] * (1 - risk):
            sellSignal.append("Sell")
            buySignal.append(np.nan)
            buySignalCost.append(np.nan)
            sellSignalCost.append(data.Close[i])
            position = False
        else:
            sellSignal.append(np.nan)
            buySignal.append(np.nan)
            buySignalCost.append(np.nan)
            sellSignalCost.append(np.nan)

    data['MACD_Buy_Signal_price'] = buySignalCost
    data['MACD_Sell_Signal_price'] = sellSignalCost
    data['MACD_Buy_Signal'] = buySignal
    data['MACD_Sell_Signal'] = sellSignal
    return data

#Function to run Bollinger Bands Strategy

def BB_strategy(data):
    buySignalCost = []
    sellSignalCost = []
    buySignal = []
    sellSignal = []
    position = False
    bb = pandas_ta.bbands(data['Close'], length=20, std=2)
    data = pd.concat([data, bb], axis=1).reindex(data.index)

    for i in range(len(data)):
        if data['Close'][i] < data['BBL_20_2.0'][i]:
            if position == False:
                buySignal.append("Buy")
                sellSignal.append(np.nan)
                buySignalCost.append(data.Close[i])
                sellSignalCost.append(np.nan)
                position = True
            else:
                buySignal.append("Hold/Buy")
                sellSignal.append(np.nan)
                buySignalCost.append(np.nan)
                sellSignalCost.append(np.nan)
        elif data['Close'][i] > data['BBU_20_2.0'][i]:
            if position == True:
                buySignal.append(np.nan)
                sellSignal.append("Sell")
                buySignalCost.append(np.nan)
                sellSignalCost.append(data.Close[i])
                position = False  # To indicate that I actually went there
            else:
                sellSignal.append(np.nan)
                buySignal.append(np.nan)
                buySignalCost.append(np.nan)
                sellSignalCost.append(np.nan)
        else:
            sellSignal.append(np.nan)
            buySignal.append(np.nan)
            buySignalCost.append(np.nan)
            sellSignalCost.append(np.nan)

    data['bb_Buy_Signal_price'] = buySignalCost
    data['bb_Sell_Signal_price'] = sellSignalCost
    data['bb_Buy_Signal'] = buySignal
    data['bb_Sell_Signal'] = sellSignal

    return data

#Load data from SQL to a python dictionary for further processing

dic = {
    'cost': BB_strategy(MACD_Strategy(SMA_Strategy(getPrices(getTables('Nasdaq'))[0]), .02)),
    'kdp': BB_strategy(MACD_Strategy(SMA_Strategy(getPrices(getTables('Nasdaq'))[1]), .02)),
    'khc': BB_strategy(MACD_Strategy(SMA_Strategy(getPrices(getTables('Nasdaq'))[2]), .02)),
    'mdlz': BB_strategy(MACD_Strategy(SMA_Strategy(getPrices(getTables('Nasdaq'))[3]), .02)),
    'mnst': BB_strategy(MACD_Strategy(SMA_Strategy(getPrices(getTables('Nasdaq'))[4]), .02)),
    'pep': BB_strategy(MACD_Strategy(SMA_Strategy(getPrices(getTables('Nasdaq'))[5]), .02)),
    'wba': BB_strategy(MACD_Strategy(SMA_Strategy(getPrices(getTables('Nasdaq'))[6]), .02))
}

# Code to get the ticker discripton to feed in to Front-End

tickers = [i.upper() for i in dic.keys()]
Ticker_df = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]
StockDesc = [(Ticker_df.loc[Ticker_df['Ticker'] == i, 'Company'].item()) for i in tickers]
tickerDesc = st.selectbox("Select a Company for Evaluation: ", StockDesc, 0)
ticker = Ticker_df.loc[Ticker_df['Company'] == tickerDesc, 'Ticker'].item().lower()
data = dic[ticker]

# ********** SMA ***************
fig = go.Figure()
fig.add_trace(go.Scatter(y=data["Close"], name="Close", mode="lines"))
fig.add_trace(go.Scatter(y=data["SMA 30"], name="SMA 30", mode="lines"))
fig.add_trace(go.Scatter(y=data["SMA 100"], name="SMA 100", mode="lines"))
fig.add_trace(
    go.Scatter(y=data['SMA_Buy_Signal_price'], mode='markers', name="Buy/Hold Signal", marker_symbol='triangle-up',
               marker=dict(color='green', size=10, opacity=0.9)))
fig.add_trace(
    go.Scatter(y=data['SMA_Sell_Signal_price'], mode='markers', name="Sell Signal", marker_symbol='triangle-down',
               marker=dict(color='red', size=10, opacity=0.9)))

fig.update_layout(
    title=f"{tickerDesc}'s Stock Buying/Selling Signals - SMA Strategy", xaxis_title="Date",
    yaxis_title="Closing Price", width=1100, height=500
)
st.write(fig)

# ********** MACD Histogram ***************

def MACD_color(data):
    MACD_color = []
    for i in range(1, len(data)):
        if data['MACDh_12_26_9'][i] > data['MACDh_12_26_9'][i - 1]:
            MACD_color.append(True)
        else:
            MACD_color.append(False)
    MACD_color.append(False)
    return MACD_color


data['positive'] = MACD_color(data)

# ********** MACD ***************
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Closing Price Chart", "MACD Histogram"))

fig.add_trace(go.Scatter(y=data["Close"], name="Closing Price", mode="lines"), row=1, col=1)
fig.add_trace(
    go.Scatter(y=data['MACD_Buy_Signal_price'], mode='markers', name="Buy/Hold Signal", marker_symbol='triangle-up',
               marker=dict(color='green', size=6, opacity=0.9)), row=1, col=1)
fig.add_trace(
    go.Scatter(y=data['MACD_Sell_Signal_price'], mode='markers', name="Sell Signal", marker_symbol='triangle-down',
               marker=dict(color='red', size=6, opacity=0.9)), row=1, col=1)

fig.add_trace(go.Scatter(y=data["MACD_12_26_9"], name="MACD Line", mode="lines"), row=2, col=1)
fig.add_trace(go.Scatter(y=data["MACDs_12_26_9"], name="Signal Line", mode="lines"), row=2, col=1)
fig.add_trace(go.Bar(y=data['MACDh_12_26_9'], name='MACD Histogram', opacity=0.9,
                     marker_color=data.positive.map({True: 'green', False: 'red'})), row=2, col=1)
fig.update_layout(title=f"{tickerDesc}'s Stock Buying/Selling Signals - MACD Strategy", xaxis_title="Date",
                  yaxis_title="Closing Price", width=1100, height=800)
st.write(fig)

# ********** BB ***************

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Closing Price Chart", "Bollinger Bands"))

fig.add_trace(go.Scatter(y=data["Close"], name="Closing Price", mode="lines"), row=1, col=1)
fig.add_trace(
    go.Scatter(y=data['bb_Buy_Signal_price'], mode='markers', name="Buy/Hold Signal", marker_symbol='triangle-up',
               marker=dict(color='green', size=6, opacity=0.9)), row=1, col=1)
fig.add_trace(
    go.Scatter(y=data['bb_Sell_Signal_price'], mode='markers', name="Sell Signal", marker_symbol='triangle-down',
               marker=dict(color='red', size=6, opacity=0.9)), row=1, col=1)

fig.add_trace(go.Scatter(y=data["BBM_20_2.0"], name="Middle", mode="lines"), row=2, col=1)
fig.add_trace(go.Scatter(y=data["BBU_20_2.0"], name="Upper", mode="lines"), row=2, col=1)
fig.add_trace(go.Scatter(y=data["BBL_20_2.0"], name="Upper", mode="lines"), row=2, col=1)
fig.update_layout(title=f"{tickerDesc}'s Stock Buying/Selling Signals - BB Strategy", xaxis_title="Date",
                  yaxis_title="Closing Price", width=1100, height=800)
st.write(fig)
