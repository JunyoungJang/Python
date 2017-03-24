import pandas as pd
import numpy as np

df = pd.read_excel('sample.xlsx')
print df
# df.dropna(axis=0) # droping rows - NOT WORKING
df = df.dropna(axis=0) # droping rows
print df

df2 = pd.read_excel('sample.xlsx')
print df2
# df2.dropna(axis=0) # droping columns - NOT WORKING
df2 = df2.dropna(axis=1) # droping columns
print df






