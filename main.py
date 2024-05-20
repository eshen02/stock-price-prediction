# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.src.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping

cur_data = 0
stock_ticker = ['AAPL', 'INTC', 'NKE']


def create_dataset(dataset, time_step):
    data_x, data_y = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step), 0]
        data_x.append(a)
        data_y.append(dataset[i + time_step, 0])

    return np.array(data_x), np.array(data_y)


def create_dataset_augmented(dataset, time_step):
    data_x, data_y = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step)]
        data_x.append(a)
        data_y.append(dataset[i + time_step, 0])

    return np.array(data_x), np.array(data_y)


def lstm_augmented():
    df1 = {}
    df2 = {}

    df1[0] = pd.read_csv('AAPL.csv', usecols=['Date', 'Close/Last'])
    df2[0] = pd.read_csv('AAPL_analyst_rating.csv', usecols=['Date', 'rating_cnt_strong_buys', 'rating_cnt_mod_buys', 'rating_cnt_holds', 'rating_cnt_mod_sells', 'rating_cnt_strong_sells'])

    df1[1] = pd.read_csv('INTC.csv', usecols=['Date', 'Close/Last'])
    df2[1] = pd.read_csv('INTC_analyst_rating.csv', usecols=['Date', 'rating_cnt_strong_buys', 'rating_cnt_mod_buys', 'rating_cnt_holds', 'rating_cnt_mod_sells', 'rating_cnt_strong_sells'])

    df1[2] = pd.read_csv('NKE.csv', usecols=['Date', 'Close/Last'])
    df2[2] = pd.read_csv('NKE_analyst_rating.csv', usecols=['Date', 'rating_cnt_strong_buys', 'rating_cnt_mod_buys', 'rating_cnt_holds', 'rating_cnt_mod_sells', 'rating_cnt_strong_sells'])

    df1[cur_data]['Close/Last'] = df1[cur_data]['Close/Last'].str.replace('$', '').astype(float)

    df2[cur_data]['Date'] = pd.to_datetime(df2[cur_data]['Date'], format='%Y-%m-%d')
    df1[cur_data]['Date'] = pd.to_datetime(df1[cur_data]['Date'], format='%m/%d/%Y')

    start_date = {}
    start_date[0] = '01/17/2018'
    start_date[1] = '01/30/2018'
    start_date[2] = '01/05/2018'

    end_date = '12/31/2018'
    df1[cur_data] = df1[cur_data][(df1[cur_data]['Date'] >= start_date[cur_data]) & (df1[cur_data]['Date'] <= end_date)]

    data_file = pd.merge(df1[cur_data], df2[cur_data], on="Date", how="outer")
    data_file.ffill(inplace=True)

    dates = data_file['Date']

    data_file.drop('Date', axis=1, inplace=True)  # Drop the date column if it's not needed as a feature

    close_price = data_file['Close/Last']
    orig_close_price = data_file['Close/Last'].values.reshape(-1, 1)

    scaler_price = MinMaxScaler()
    close_price = scaler_price.fit_transform(np.array(close_price).reshape(-1, 1))

    data_file.drop('Close/Last', axis=1, inplace=True)

    ratings = []
    r_data = np.array(data_file)

    scale_value = np.max(r_data) * 1.5

    if scale_value == 0:
        scale_value = 1

    for i in range(len(r_data)):
        arr = []
        for j in range(len(r_data[i])):
            arr.append(r_data[i, j] / scale_value)

        ratings.append(arr)

    data = np.concatenate((close_price, ratings), axis=1)

    train_size = int(len(data) * 0.75)
    train_data, test_data = data[0:train_size, :], data[train_size:len(data), :]

    time_step = 30

    input_train, output_train = create_dataset_augmented(train_data, time_step)
    input_test, output_test = create_dataset_augmented(test_data, time_step)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(input_train.shape[1], 6)))
    model.add(Dropout(.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

    model.fit(input_train, output_train, validation_data=(input_test, output_test), epochs=75, batch_size=16,
              verbose=1, callbacks=early_stopping)

    train_predict = model.predict(input_train)
    test_predict = model.predict(input_test)

    # reshape predicted train data to original shape
    train_reshaped = np.zeros((train_predict.shape[0], 6), dtype=train_predict.dtype)
    train_reshaped[:train_predict.shape[0], :train_predict.shape[1]] = train_predict

    # reshape predicted test data to original shape
    test_reshaped = np.zeros((test_predict.shape[0], 6), dtype=test_predict.dtype)
    test_reshaped[:test_predict.shape[0], :test_predict.shape[1]] = test_predict

    train_predict = scaler_price.inverse_transform(train_predict)
    test_predict = scaler_price.inverse_transform(test_predict)

    look_back = 30
    train_predict_plot = np.empty_like(orig_close_price)
    train_predict_plot[:] = np.nan
    train_predict_plot[look_back: len(train_predict) + look_back] = train_predict

    test_predict_plot = np.empty_like(orig_close_price)
    test_predict_plot[:] = np.nan
    test_predict_plot[len(train_predict) + 2 * look_back: len(data)] = test_predict

    mse = np.mean((orig_close_price[len(train_predict) + 2 * look_back: len(data)] - test_predict) ** 2)
    print("Mean Squared Error: " + str(mse))

    mape = np.mean(np.abs((orig_close_price[len(train_predict) + 2 * look_back: len(data)] - test_predict) / orig_close_price[len(train_predict) + 2 * look_back: len(data)])) * 100
    print("Mean Absolute Percentage Error: " + str(mape))

    plt.xlabel('Date')
    plt.ylabel('Close Price')

    plt.title(stock_ticker[cur_data] + ' Close Prices Over Time With Analyst Rating')

    plt.plot(dates, orig_close_price, label='Actual')
    plt.plot(dates, train_predict_plot, label='Training')
    plt.plot(dates, test_predict_plot, label='Prediction')
    plt.legend()

    print(plt.show())


def lstm_model():
    data_file = {}

    data_file[0] = pd.read_csv('AAPL.csv', usecols=['Date', 'Close/Last'])
    data_file[1] = pd.read_csv('INTC.csv', usecols=['Date', 'Close/Last'])
    data_file[2] = pd.read_csv('NKE.csv', usecols=['Date', 'Close/Last'])

    data_file[cur_data]['Close/Last'] = data_file[cur_data]['Close/Last'].str.replace('$', '').astype(float)

    # print(data_file.head())
    data_file[cur_data] = data_file[cur_data].iloc[::-1]

    data_file[cur_data]['Date'] = pd.to_datetime(data_file[cur_data]['Date'], format='%m/%d/%Y')

    start_date = {}
    start_date[0] = '01/17/2018'
    start_date[1] = '01/30/2018'
    start_date[2] = '01/05/2018'

    end_date = '12/31/2018'
    data_file[cur_data] = data_file[cur_data][(data_file[cur_data]['Date'] >= start_date[cur_data]) & (data_file[cur_data]['Date'] <= end_date)]

    dates = data_file[cur_data]['Date']

    close_data = data_file[cur_data].reset_index()['Close/Last']

    scaler = MinMaxScaler()
    close_data = scaler.fit_transform(np.array(close_data).reshape(-1, 1))

    time_step = 30

    train_size = int(len(close_data) * 0.75)
    train_data, test_data = close_data[0:train_size, :], close_data[train_size:len(close_data), :1]

    input_train, output_train = create_dataset(train_data, time_step)
    input_test, output_test = create_dataset(test_data, time_step)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(input_train.shape[1], 1)))
    model.add(Dropout(.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

    model.fit(input_train, output_train, validation_data=(input_test, output_test), epochs=75, batch_size=16,
              verbose=1, callbacks=early_stopping)

    train_predict = model.predict(input_train)
    test_predict = model.predict(input_test)

    # transform to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    close_data = scaler.inverse_transform(close_data);

    look_back = 30
    train_predict_plot = np.empty_like(close_data)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back: len(train_predict) + look_back, :] = train_predict

    test_predict_plot = np.empty_like(close_data)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + 2 * look_back: len(close_data), :] = test_predict

    mse = np.mean((close_data[len(train_predict) + 2 * look_back: len(close_data)] - test_predict) ** 2)
    print("Mean Squared Error: " + str(mse))

    mape = np.mean(np.abs((close_data[len(train_predict) + 2 * look_back: len(close_data)] - test_predict) / close_data[len(train_predict) + 2 * look_back: len(close_data)])) * 100
    print("Mean Absolute Percentage Error: " + str(mape))

    plt.xlabel('Date')
    plt.ylabel('Close Price')

    plt.title(stock_ticker[cur_data] + ' Close Price Over Time Without Analyst Rating')

    plt.plot(dates, close_data, label='Actual')
    plt.plot(dates, train_predict_plot, label='Training')
    plt.plot(dates, test_predict_plot, label='Prediction')

    plt.legend()

    print(plt.show())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ticker = input("Enter stock ticker: ")
    if ticker == 'AAPL':
        cur_data = 0
    elif ticker == 'INTC':
        cur_data = 1
    elif ticker == 'NKE':
        cur_data = 2
    else:
        print("Invalid Input")
        exit()
    lstm_model()
    lstm_augmented()
    print("finished!")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
