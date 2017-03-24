import pandas as pd
import numpy as np

# by default we assume there is a header
# header=None declares that there is no header
df = pd.read_csv('supermarkets.csv', header=None)
print df
print type(df)
print df.shape


