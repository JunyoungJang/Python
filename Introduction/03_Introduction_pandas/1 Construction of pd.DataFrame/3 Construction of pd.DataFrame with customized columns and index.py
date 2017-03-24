import pandas as pd

df = pd.DataFrame([[40., 170., 70.], [20., 180., 60.]], columns=['Age','Height','Weight'], index=['Lee','Kim'])
print df
print type(df)
print df.shape
