import os
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


df = pd.read_csv('database.csv', index_col = 0, header = 0)
feature_data = df.columns.tolist()
x = df[feature_data].values
y = df.index
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=30)
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
print(rfc.score(x_test, y_test))
date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
joblib.dump(rfc,f'{os.getcwd()}/predict_y_haplogroup/model/ensemble_str{date}.joblib')
rfc_pre = rfc.predict(x_test)
report = classification_report(y_test, y_pred = rfc_pre, output_dict = True)
df = pd.DataFrame(report).transpose()
df.to_csv(f'{os.getcwd()}/predict_y_haplogroup/report/ensemble_str{date}.csv', index = True)
