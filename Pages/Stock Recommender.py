import streamlit as st
import math
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import sqlalchemy
import pymysql
import pandas_ta
from tabulate import tabulate
import plotly.graph_objects as go


# *********************** Data Loading to SQL Workbench ********************************
pymysql.install_as_MySQLdb()

engine = sqlalchemy.create_engine('mysql://root:12345@localhost:3306/')

#Function to Create Schema In SQL Workbench
def schemacreator(index):
    engine = sqlalchemy.create_engine('mysql://root:12345@localhost:3306/')
    engine.execute(sqlalchemy.schema.CreateSchema(index))

#Function to get tables from SQL Workbench

def getTables(Schema):
    query = f"""SELECT table_name FROM information_schema.tables
    Where table_schema ='{Schema}'"""
    df = pd.read_sql(query,engine)
    df['Schema'] = Schema
    return df

#Function to fecth stock prices from SQL Workbench

def getPrices(tables_list):
    price = []
    for table,schema in zip(tables_list.TABLE_NAME,tables_list.Schema):
        sql = schema + '.' + f'`{table}`'
        price.append(pd.read_sql(f"SELECT Date, Close FROM {sql}", engine))
    return price

#Function to run LSTM Model

def lstm(data):
    close_prices = data['Close']

    values = close_prices.values
    training_data_len = math.ceil(len(values) * 0.99)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values.reshape(-1, 1))
    train_data = scaled_data[0: training_data_len, :]

    x_train = []
    y_train = []

    for i in range(5, len(train_data)):
        x_train.append(train_data[i - 5:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len - 5:, :]
    x_test = []
    y_test = values[training_data_len:]

    for i in range(5, len(test_data)):
        x_test.append(test_data[i - 5:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    model = keras.Sequential()
    model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(layers.LSTM(100, return_sequences=False))
    model.add(layers.Dense(25))
    model.add(layers.Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=3)

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)

    return predictions, rmse

#Load data from SQL to a python dictionary for further processing

dic = {
    'cost' : getPrices(getTables('Nasdaq'))[0],
    'kdp' : getPrices(getTables('Nasdaq'))[1],
    'khc' : getPrices(getTables('Nasdaq'))[2],
    'mdlz' : getPrices(getTables('Nasdaq'))[3],
    'mnst' : getPrices(getTables('Nasdaq'))[4],
    'pep' : getPrices(getTables('Nasdaq'))[5],
    'wba' : getPrices(getTables('Nasdaq'))[6]
}

# Code to get the ticker discripton to feed in to Front-End

tickers = [i.upper() for i in dic.keys()]
Ticker_df = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]
StockDesc = [(Ticker_df.loc[Ticker_df['Ticker'] == i, 'Company'].item()) for i in tickers]
tickerDesc = st.selectbox("Select a Company for Evaluation: ", StockDesc, 0)
ticker = Ticker_df.loc[Ticker_df['Company'] == tickerDesc, 'Ticker'].item().lower()
data = dic[ticker]

#Run LSTM Model

pred, rmse = lstm(data)

close_prices = data['Close']
values = close_prices.values
training_data_len = math.ceil(len(values)* 0.99)


data = data.filter(['Close'])
train = data[:training_data_len]
validation = data[training_data_len:]
validation['Predictions'] = pred

# Display Actual/Predicted Stock Price Graph - LSTM Model

fig = go.Figure()
fig.add_trace(go.Scatter(y=train["Close"], name="Closing Price", mode="lines"))
fig.add_trace(go.Scatter(x = validation.index, y=validation["Close"], name="Actual Closing Price", mode="lines"))
fig.add_trace(go.Scatter(x = validation.index, y=validation["Predictions"], name="Simulated Closing Price", mode="lines"))
fig.update_layout(
   title=f"{tickerDesc} Stock Price Evaluation - LSTM Model", xaxis_title="Date", yaxis_title="Closing Price", width=800, height=500)
st.write(fig)
temp = ""

st.markdown(f"{tickerDesc} Stock Prices for the Next Few Days Would be: ")
for i in validation["Predictions"]:
    # temp = temp + '\n' + str(round(i, 2))
    st.button(str(round(i, 2)))

# st.Markdown("Stock Prices for the Next Few Days Would be")
