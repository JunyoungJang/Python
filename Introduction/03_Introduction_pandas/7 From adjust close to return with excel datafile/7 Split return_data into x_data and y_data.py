import pandas as pd
import numpy as np

df = pd.read_excel('/Users/sungchul/Dropbox/Data/DOW30.xlsx', sheetname=0)
df = df.iloc[:, range(0, df.shape[1], 2)]
df = df.dropna(axis=1)
df = np.array(df)

def Daily_Return_Computation(Daily_Adjust_Close):
    c = Daily_Adjust_Close
    r = (c[1:,:]-c[0:-1,:])/c[0:-1,:]
    return r

retrun_data = Daily_Return_Computation(Daily_Adjust_Close=df)

x_data = np.expand_dims(retrun_data[:,-1], axis=1)
y_data = retrun_data[:,0:-1]
print type(x_data), x_data.shape # <type 'numpy.ndarray'> (3019, 1)
print type(y_data), y_data.shape # <type 'numpy.ndarray'> (3019, 29)
