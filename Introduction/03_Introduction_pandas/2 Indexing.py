import pandas as pd

df = pd.DataFrame([[40., 170., 70.], [20., 180., 60.]])
print df
print type(df)
print df.shape

# print df[1, 1] # Error
print df.iloc[1, 1]
print type(df.iloc[1, 1])
print df.iloc[1, 1].shape

print df.iloc[1, :]
print type(df.iloc[1, :])
print df.iloc[1, :].shape

print df.iloc[:, 1]
print type(df.iloc[:, 1])
print df.iloc[:, 1].shape

