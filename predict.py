import os
import joblib
import pandas as pd


file_name = input('Please enter your Y-STR file name:')
model = input('Please enter a model name:')
df = pd.read_csv(f'{file_name}.csv', index_col = 0, header = 0)
feature_data = df.columns.tolist()
x = df[feature_data].values
rfc = joblib.load('f{os.getcwd()}predict_y_haplogroup/model/{model}')
print(rfc.predict(x))
