import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


file_name = input('Please enter your Y-STR file name:')
df = pd.read_csv(f'{file_name}.csv', index_col = 0, header = 0)
feature_data = df.columns.tolist()
x = df[feature_data].values
rfc = RandomForestClassifier()
print(rfc.predict(x))
