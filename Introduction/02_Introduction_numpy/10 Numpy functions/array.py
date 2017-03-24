import pandas as pd
import numpy as np

df = pd.DataFrame([[40., 170., 70.], [20., 180., 60.]], columns=['Age','Height','Weight'], index=['Lee','Kim'])
df = np.array(df)
print df
print type(df)
print df.shape
