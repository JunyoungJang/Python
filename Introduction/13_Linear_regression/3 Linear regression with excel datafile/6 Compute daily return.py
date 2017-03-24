import pandas as pd
import numpy as np

df = pd.read_excel('/Users/sungchul/Dropbox/Data/DOW30.xlsx', sheetname=0)
df = df.iloc[:, range(0, df.shape[1], 2)]
df = df.dropna(axis=1)
df = np.array(df, np.float32)

def Daily_Return_Computation(Daily_Adjust_Close):
    c = Daily_Adjust_Close
    r = (c[1:,:]-c[0:-1,:])/c[0:-1,:]
    return r

retrun_data = Daily_Return_Computation(Daily_Adjust_Close=df)