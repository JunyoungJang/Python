import pandas as pd

# import os
# print os.getcwd()

df = pd.read_excel('/Users/sungchul/Dropbox/Data/DOW30.xlsx', sheetname=0)
print df
print type(df), df.shape

