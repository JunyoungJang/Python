import pandas as pd
import numpy as np

df = pd.read_excel('supermarkets.xlsx')
df = pd.read_excel('supermarkets.xlsx', sheetname=0)
print df
print type(df)
print df.shape


