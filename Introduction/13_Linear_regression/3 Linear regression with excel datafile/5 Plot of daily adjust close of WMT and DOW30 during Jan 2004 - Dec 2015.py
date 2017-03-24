import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('/Users/sungchul/Dropbox/Data/DOW30.xlsx', sheetname=0)
df = df.iloc[:, range(0, df.shape[1], 2)]
df = df.dropna(axis=1)
df = np.array(df, np.float32)

plt.subplot(1, 2, 1)
plt.plot(df[:,-3])
plt.title('Daily adjust close of WMT during Jan 2004 - Dec 2015')
plt.subplot(1, 2, 2)
plt.plot(df[:,-1])
plt.title('Daily adjust close of DOW 30 during Jan 2004 - Dec 2015')
plt.show()






