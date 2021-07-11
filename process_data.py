# %%
import pandas as pd
import numpy as np

df = pd.read_csv('database.csv', index_col = 0, header = 0)

new_index = []
index = df.index
for i in range(len(index)):
    new_index.append(index[i][0])
df.index = new_index
df.to_csv('database.csv')
# %%
