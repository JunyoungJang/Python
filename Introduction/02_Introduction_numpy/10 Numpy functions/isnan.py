import pandas as pd
import numpy as np

df = pd.read_excel('sample.xlsx')
df = np.array(df)
print df

where_are_NaNs = np.isnan(df)
df[where_are_NaNs] = -700
print df





