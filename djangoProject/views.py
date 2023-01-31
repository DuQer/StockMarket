from urllib import request
from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Scatter
import json

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from datetime import datetime, timedelta, date

import datetime as dt
import sqlite3

from django.shortcuts import render

from django.http import HttpResponseRedirect

from .forms import NameForm, PredictionForm

from djangoProject.DBModule import DBInterface
import hashlib

import io
import urllib, base64
import matplotlib.pyplot as plt
from chartjs.views.lines import BaseLineChartView

dbInterface = DBInterface()


def index(request):
    data = yf.download(
        tickers=['EURUSD=X', 'JPY=X', 'GBPUSD=X'],
        group_by='ticker',
        threads=True,
        period='1mo',
        interval='1d'
    )

    data.reset_index(level=0, inplace=True)

    fig_left = go.Figure()

    fig_left.add_trace(
        go.Scatter(x=data['Date'], y=data['EURUSD=X']['Adj Close'], name="EURUSD=X")
    )
    fig_left.add_trace(
        go.Scatter(x=data['Date'], y=data['JPY=X']['Adj Close'], name="JPY=X")
    )
    fig_left.add_trace(
        go.Scatter(x=data['Date'], y=data['GBPUSD=X']['Adj Close'], name="GBPUSD=X")
    )

    fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    plot_div_left = plot(fig_left, auto_open=False, output_type='div')

    # ================================================ To show recent stocks ==============================================

    df1 = yf.download(tickers='EURUSD=X', period='1d', interval='1d')
    df2 = yf.download(tickers='JPY=X', period='1d', interval='1d')
    df3 = yf.download(tickers='GBPUSD=X', period='1d', interval='1d')

    df1.insert(0, "Ticker", "EURUSD=X")
    df2.insert(0, "Ticker", "JPY=X")
    df3.insert(0, "Ticker", "GBPUSD=X")

    df = pd.concat([df1, df2, df3], axis=0)
    df.reset_index(level=0, inplace=True)
    df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    convert_dict = {'Date': object}
    df = df.astype(convert_dict)
    df.drop('Date', axis=1, inplace=True)

    json_records = df.reset_index().to_json(orient='records')
    recent_stocks = []
    recent_stocks = json.loads(json_records)

    return render(request, 'index.html', {
        'plot_div_left': plot_div_left,
        'recent_stocks': recent_stocks
    })


def ticker(request):
    # ================================================= Load Ticker Table ================================================
    ticker_df = pd.read_csv('converter/MyTickers.csv')
    json_ticker = ticker_df.reset_index().to_json(orient='records')
    ticker_list = []
    ticker_list = json.loads(json_ticker)

    return render(request, '../templates/ticker.html', {
        'ticker_list': ticker_list[0:20]
    })


def get_data(request):
    if request.method == 'POST':
        form = NameForm(request.POST).data.dict()
        hash_password = hashlib.sha256(form['password'].encode()).hexdigest()
        exists = dbInterface.login(form['username'], hash_password)
        print(exists)
    else:
        form = NameForm()

    return render(request, 'name.html', form)

def prediction(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST).data.dict()

        ticker_symbol = form['ticker_symbol'].upper()
        num_of_days = form['number_of_days']
    else:
        return render(request, 'YahooError.html')

    if not num_of_days.isnumeric():
        return render(request, 'FormatDaysError.html')
    if num_of_days.isnumeric() < 0:
        return render(request, 'FormatDaysError.html')
    if num_of_days.isnumeric() > 365:
        return render(request, 'FormatDaysError.html')

    converted_num = int(num_of_days)
    start_date = datetime.now()
    end_date = start_date + timedelta(days=converted_num)

    arr = []
    for i in range(0,converted_num):
        x = datetime.now() + timedelta(days=i)
        y = datetime.combine(x, datetime.min.time())
        arr.append(int(y.timestamp())*10**9)

    data = yf.download(
        tickers=ticker_symbol,
        group_by='ticker',
        threads=True,
        period='1mo',
        interval='1d',
        start='2021-01-01',
        end='2023-01-01'
    )

    data['Date'] = data.index
    data.reset_index(drop=True, inplace=True)
    data["Date"] = pd.to_numeric(data["Date"])
    prediction_table = pd.DataFrame(data[['Date', 'Adj Close']], index=None)

    X = prediction_table.drop(columns='Adj Close')
    y = prediction_table.drop(columns='Date')
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=0.2)

    linear = GradientBoostingRegressor()
    linear.fit(xtrain, ytrain)

    price_pred = linear.predict(xtest)

    date = pd.DataFrame(xtest, columns=['Date'], index=None)
    date = date.reset_index(drop=True)

    predicted = pd.DataFrame(price_pred.flatten(), columns=['Predicted'])
    actual = ytest.reset_index(drop=True)
    actual.columns = ['Actual']

    testdf = pd.concat([date, predicted, actual], axis=1)

    testdf['Absolute error'] = abs(testdf['Actual'] - testdf['Predicted'])
    print(f"Mean squared error: {mean_squared_error(testdf['Actual'], testdf['Predicted'])}")
    print(f"Model score: {linear.score(xtest, ytest)}")

    # print(testdf)

    arr_predict = np.array(arr)
    arr_predict = arr_predict.reshape(-1, 1)

    price_pred = linear.predict(arr_predict)
    # print(price_pred)

    print(linear.predict(np.array(1615033200000000000).reshape(1,-1)))
    print(linear.predict(np.array(1675724400000000000).reshape(1,-1)))
    print(linear.predict(np.array(1676674800000000000).reshape(1,-1)))
    print(linear.predict(np.array(1676847600000000000).reshape(1,-1)))
    print(linear.predict(np.array(1675292400000000000).reshape(1,-1)))

    return render(request, 'name.html', form)
