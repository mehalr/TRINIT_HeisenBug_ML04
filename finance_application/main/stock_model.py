import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
from finance.settings import BASE_DIR

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def get_stock_data(symbol):
    api_key = '8Z7MAVSSG2H39LQN'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol={symbol}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()
    return data


def create_dataframe(data):
    df = pd.DataFrame(data["Time Series (Daily)"])
    df = df.T
    df.reset_index(inplace=True)
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df = df[:1000]
    df = df.iloc[::-1]
    print(df)
    return df


def LSTM_ALGO(df, sym):
    # Split data into training set and test set
    dataset_train = df.iloc[0:int(0.8 * len(df)), :]
    dataset_test = df.iloc[int(0.8 * len(df)):, :]
    ############# NOTE #################
    # TO PREDICT STOCK PRICES OF NEXT N DAYS, STORE PREVIOUS N DAYS IN MEMORY WHILE TRAINING
    # HERE N=7
    ###dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
    training_set = df.iloc[:, 4:5].values  # 1:2, to store as numpy array else Series obj will be stored
    # select cols using above manner to select as float64 type, view in var explorer

    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))  # Scaled values btween 0,1
    training_set_scaled = sc.fit_transform(training_set)
    # In scaling, fit_transform for training, transform for test

    # Creating data stucture with 7 timesteps and 1 output.
    # 7 timesteps meaning storing trends from 7 days before current day to predict 1 next output
    X_train = []  # memory with 7 days from day i
    y_train = []  # day i
    for i in range(7, len(training_set_scaled)):
        X_train.append(training_set_scaled[i - 7:i, 0])
        y_train.append(training_set_scaled[i, 0])
    # Convert list to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_forecast = np.array(X_train[-1, 1:])
    X_forecast = np.append(X_forecast, y_train[-1])
    # Reshaping: Adding 3rd dimension
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # .shape 0=row,1=col
    X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))
    # For X_train=np.reshape(no. of rows/samples, timesteps, no. of cols/features)

    # Initialise RNN
    regressor = Sequential()

    # Add first LSTM layer
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    # units=no. of neurons in layer
    # input_shape=(timesteps,no. of cols/features)
    # return_seq=True for sending recc memory. For last layer, retrun_seq=False since end of the line
    regressor.add(Dropout(0.1))

    # Add 2nd LSTM layer
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))

    # Add 3rd LSTM layer
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))

    # Add 4th LSTM layer
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.1))

    # Add o/p layer
    regressor.add(Dense(units=1))

    # Compile
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    # Training
    regressor.fit(X_train, y_train, epochs=25, batch_size=32)
    # For lstm, batch_size=power of 2

    # Testing
    ###dataset_test=pd.read_csv('IBM_Stock_Price_Test.csv')
    dataset_test['close'] = dataset_test['close'].apply(lambda x: float(x))
    real_stock_price = dataset_test.iloc[:, 4:5].values

    # To predict, we need stock prices of 7 days before the test set
    # So combine train and test set to get the entire data set
    dataset_total = pd.concat((dataset_train['close'], dataset_test['close']), axis=0)
    testing_set = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values
    testing_set = testing_set.reshape(-1, 1)
    # -1=till last row, (-1,1)=>(80,1). otherwise only (80,0)

    # Feature scaling
    testing_set = sc.transform(testing_set)

    # Create data structure
    X_test = []
    for i in range(7, len(testing_set)):
        X_test.append(testing_set[i - 7:i, 0])
        # Convert list to numpy arrays
    X_test = np.array(X_test)

    # Reshaping: Adding 3rd dimension
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Testing Prediction
    predicted_stock_price = regressor.predict(X_test)

    # with open('model_pkl', 'wb') as files:
    #     pickle.dump(regressor, files)


    # Getting original prices back from scaled values
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(real_stock_price, label='Actual Price')
    plt.plot(predicted_stock_price, label='Predicted Price')

    plt.legend(loc=4)
    plt.savefig('LSTM.png')
    plt.close(fig)

    error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

    # Forecasting Prediction
    forecasted_stock_price = regressor.predict(X_forecast)

    # Getting original prices back from scaled values
    forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)

    lstm_pred = forecasted_stock_price[0, 0]
    print()
    print("##############################################################################")
    print("Tomorrow's IBM", " Closing Price Prediction by LSTM: ", lstm_pred)
    print("LSTM RMSE:", error_lstm)
    print("##############################################################################")
    return lstm_pred, error_lstm

