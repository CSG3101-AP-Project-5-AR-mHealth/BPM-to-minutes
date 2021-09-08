import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
from zipfile import ZipFile
import os
from os.path import exists
import json

sns.set()
# sns.set_palette(“hls”, 2)

json_path = "heart_rate.json"

# Retrieved from kaggle
# TODO: share link where data was retrieved from
if not exists(json_path):
    zip_path = "heart_rate.json.zip"
    zip_file = ZipFile(zip_path)
    zip_file.extractall()

with open(json_path) as f:
    d = json.load(f)

# Normalise Json
df = pd.json_normalize(d)

# Convert dateTime object
df['dateTime'] = pd.to_datetime(df['dateTime'])

# Remove confidence
df = df.drop(['value.confidence'], axis=1)

# Show the data we are working with
# df1.shape
# df1.info()
# print(df1.head(-10))

df.sort_values(by="dateTime")
df = df.set_index('dateTime')

# Drop any nans and round to zero
df = df.resample('1Min').mean().dropna().round(0)
print(df.head(5))

# Split data for testing
df_test_data = df.loc[(df.index > '2020-01-12 00:00:00') & (df.index <= '2020-01-12 23:59:59')]
df = df.loc[(df.index < '2020-01-12 00:00:00') | (df.index > '2020-01-12 23:59:59')]

# Visualising training and testing data
datasets = [df, df_test_data]
for idx, dfs in enumerate(datasets):
    fig, axs = plt.subplots(figsize=(10, 6))
    axs.plot(dfs['value.bpm'])
    axs.tick_params(axis='x', rotation=60)
    axs.set(ylabel='Heart Rate (BPM)', xlabel="Date")
    plt.tight_layout()
    # plt.show()
    fig.savefig(str(idx + 1) + "_data_fig.png")

# TODO: correlation
# Make a correlation coefficient
# print(df.corr())

# https://blog.keras.io/building-autoencoders-in-keras.html or
# https://scikit-learn.org/stable/modules/ensemble.html#forest
# Preprocessing - Scale the data from 0 to 1
scaler = MinMaxScaler()
scaler = scaler.fit(df)
scale_train = pd.DataFrame(scaler.transform(df))
scale_test = pd.DataFrame(scaler.transform(df_test_data))


# reshape to [samples, time_steps, n_features]
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


time_steps = 30
X_train, y_train = create_dataset(scale_train, scale_train, time_steps)
X_test, y_test = create_dataset(scale_test, scale_test, time_steps)
print("[samples, time_steps, n_features]")
print(X_train.shape, y_train.shape)

# model building
model = keras.Sequential()
model.add(
    keras.layers.LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, name='encoder_1'))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.LSTM(128, return_sequences=True, name='encoder_2'))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.LSTM(64, return_sequences=False, name='encoder_3'))
model.add(keras.layers.RepeatVector(n=X_train.shape[1], name='encoder_decoder'))
model.add(keras.layers.LSTM(64, return_sequences=True, name='decoder_1'))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.LSTM(128, return_sequences=True, name='decoder_2'))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.LSTM(256, return_sequences=True, name='decoder_3'))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
model.compile(loss='mae', optimizer='adam')
model.summary()

# fitting on training data
if not exists("saved_model"):
    history = model.fit(X_train, X_train, epochs=1, batch_size=256,
                        validation_split=0.1, verbose=1, shuffle=False)
    model.save("saved_model")
else:
    history = keras.models.load_model("saved_model")

# plotting loss
# fig = plt.figure()
# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.ylabel('Loss')
# plt.xlabel('No. Epochs')
# plt.legend()
# plt.show()

# Predicting on test data
X_pred = model.predict(X_train, verbose=1)
X_pred_2d = pd.DataFrame(X_pred[:, 0, :]).astype(float)
X_pred_2d.columns = ['HR Pred']
X_train_2d = pd.DataFrame(X_train[:, 0, :]).astype(float)
X_train_2d.columns = ['HR Test']

# Plot the test data together
print("Plotting Test Data")
fig, axs = plt.subplots(2, figsize=(12, 6))
axs[0].plot(X_pred_2d['HR Pred'])
axs[1].plot(X_train_2d['HR Test'])
plt.setp(axs[0].get_xticklabels(), visible=False)
axs[0].tick_params(axis='x', rotation=70)
axs[1].tick_params(axis='x', rotation=70)
axs[0].set(ylabel='HR Prediction')
axs[1].set(ylabel='HR Training', xlabel='Time Step (per minute)')
plt.tight_layout()
# plt.show()

# Calculate error
print("Calculating Error")
predictions = X_pred_2d['HR Pred']
train_inputs = X_train_2d['HR Test']
anomaly = pd.DataFrame(np.abs(predictions.values - train_inputs.values))
anomaly = anomaly.mean(axis=1)
ax = sns.displot(anomaly, kde=True)
# ax.set_xlabel('Loss')
# ax.set_ylabel('Frequency')

threshold = round(np.quantile(anomaly, 0.98), 3)
print('98th percentile loss value from training: ' + str(threshold))

# Predicting
X_pred = model.predict(X_test, verbose=1)
X_pred = pd.DataFrame(X_pred[:, 0, :]).astype(float)
X_pred.columns = ['HR Pred']
X_test_data = pd.DataFrame(X_test[:, 0, :]).astype(float)
X_test_data.columns = ['HR Test']

difference = pd.DataFrame(np.abs(X_pred.values - X_test_data.values))
difference['mae loss'] = difference.mean(axis=1)
difference['threshold'] = threshold
difference['anomaly'] = difference['mae loss'] > difference['threshold']
difference['index'] = difference.index
X_pred['index'] = X_pred.index
X_test_data['index'] = X_test_data.index
X_test_data = X_test_data.join(difference['anomaly'])

X_test_data_original = pd.DataFrame(scaler.inverse_transform(X_test_data[['HR Test']]))
X_test_data = pd.concat([X_test_data, X_test_data_original], axis=1)
X_test_data.columns = ['HR Test', 'Index', 'Anomaly', 'Heart Rate (BPM)']

sns.lmplot(x='index', y='mae loss', data=difference,
                 fit_reg=False, hue='anomaly', scatter_kws={"s": 10}, legend=True, height=5, aspect=2)
# plt.set(xlabel='Time Steps (per minute)', ylabel='MAE Loss')
plt.show()
