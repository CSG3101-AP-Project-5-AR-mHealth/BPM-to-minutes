import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from zipfile import ZipFile
import os
import json

zip_path = "heart_rate.json.zip"
zip_file = ZipFile(zip_path)
zip_file.extractall()
json_path = "heart_rate.json"
with open(json_path) as f:
    d = json.load(f)

df1 = pd.json_normalize(d)
df1.shape
df1.info()
print(df1.head(10))
print(df1.head(-10))
# df1.sort_values(by="dateTime")
# df1.set_index('dateTime')
# df1.resample('1Min').mean()

df1.plot(x="dateTime", y="value.bpm")
plt.show()
