import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

print('Imports Complete')
# ----------------------------- read data -----------------------------
df = pd.read_csv('LSTM-Multivariate_pollution.csv')
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe())
# ----------------------------- data pre-processing -----------------------------
col_names = ['pollution', 'dew', 'temperature', 'pressure', 'wind_dir', 'wind_speed', 'snow', 'rain']
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
# print(df.shape)

df.index = pd.to_datetime(df['date'], format='%Y.%m.%d %H:%M:%S')
# print(df.head())
df.drop('date', axis=1, inplace=True)
df.columns = col_names
features = df.values
# print(df.head())

columns = [0, 1, 2, 3, 5, 6, 7]
plt.figure(figsize=(20,14))
for i, c in enumerate(columns, 1):
    plt.subplot(len(columns), 1, i)
    plt.plot(features[:, c])
    plt.title(df.columns[c], y=0.75, loc="right")
# plt.show()

plt.matshow(df.corr())
plt.xticks(range(len(col_names)), col_names)
plt.yticks(range(len(col_names)), col_names)
plt.colorbar()
# plt.show()

# print(df["wind_dir"].unique())
wind_dir_encoder = LabelEncoder()
df["wind_dir"] = wind_dir_encoder.fit_transform(df["wind_dir"])
df["wind_dir"] = df["wind_dir"].astype(float)
# print(df.head())

values = df.values
target = df['pollution']
# plt.plot(target)
# plt.show()

# How to Convert a Time Series to a Supervised Learning Problem in Python
# https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
# The function is defined with default parameters so that if you call it with just your data, it will construct a DataFrame with t-1
# as X and t as y
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]

    df = pd.DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_dataset = scaler.fit_transform(values)
# print(scaled_dataset.shape[1])
reframed = series_to_supervised(scaled_dataset, 1, 1)
# print(reframed.head())
# print(reframed.shape)

reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
# print(reframed.head())
# print(reframed.shape)

values = reframed.values
# We train the model on the 1st 3 years and then test on the last year (for now)
n_train_hours = 365 * 24 * 3

train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D :- (no.of samples, no.of timesteps, no.of features)
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


#----------------------------- Design Model ----------------------------
model = Sequential()
model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

history = model.fit(train_X, train_y, epochs=50, batch_size=32, validation_split=0.2, verbose=1, shuffle=False)

plt.figure(figsize=(15,6))
plt.plot(history.history['loss'], label='train', linewidth = 2.5)
plt.plot(history.history['val_loss'], label='test',  linewidth = 2.5)
plt.legend()
plt.show()

# ---------------------------- Pridict --------------------------------
prediction = model.predict(test_X)
prediction = prediction.ravel()

ture_test = test[:, 8]

poll = np.array(df["pollution"])

meanop = poll.mean()
stdop = poll.std()

ture_test = ture_test * stdop + meanop
prediction = prediction * stdop + meanop

plt.figure(figsize=(15, 6))
plt.xlim([1000, 1250])
plt.ylabel("ppm")
plt.xlabel("hrs")
plt.plot(ture_test, c="g", alpha=0.90, linewidth=2.5)
plt.plot(prediction, c="b", alpha=0.75)
plt.title("Testing(Validation) data")
plt.show()

rmse = np.sqrt(mse(ture_test, prediction))
print("Test(Validation) RMSE =", rmse)