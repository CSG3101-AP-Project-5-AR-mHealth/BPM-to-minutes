import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from zipfile import ZipFile
import os
from os.path import exists
import json

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
    plt.show()
    fig.savefig(str(idx+1) + "_data_fig.png")

# TODO: correlation
# Make a correlation coefficient
# print(df.corr())
