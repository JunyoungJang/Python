import numpy as np
import pandas as pd

df = pd.read_csv('train_data_linear_regression.txt', sep='\s+')
df = np.array(df, np.float32)
print df
print type(df[0, 0])

x_data = df[:, 0:-1]
print x_data
y_data = np.expand_dims( df[:, -1], axis=1)
print y_data


