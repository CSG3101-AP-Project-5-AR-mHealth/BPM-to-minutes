import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from zipfile import ZipFile
import os
from os.path import exists
import json

json_path = "heart_rate.json"

if not exists(json_path):
    zip_path = "heart_rate.json.zip"
    zip_file = ZipFile(zip_path)
    zip_file.extractall()

with open(json_path) as f:
    d = json.load(f)

# Normalise Json
df1 = pd.json_normalize(d)

# Convert dateTime object
df1['dateTime'] = pd.to_datetime(df1['dateTime'])

# Remove confidence
df1 = df1.drop(['value.confidence'], axis=1)

# Show the data we are working with
# df1.shape
# df1.info()
# print(df1.head(-10))

df1.sort_values(by="dateTime")
df1 = df1.set_index('dateTime')

# Drop any nans and round to zero
df1 = df1.resample('1Min').mean().dropna().round(0)
print(df1.head(5))

df1.plot()
plt.show()

