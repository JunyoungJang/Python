import pandas as pd
import numpy as np

df = pd.read_csv('supermarkets.csv')                      # csv = (commonly known as) comma separated values
df = pd.read_csv('supermarkets-commas.txt')               # csv = (commonly known as) comma separated values
df = pd.read_csv('supermarkets-semi-colons.txt', sep=';') # csv = (in fact) character separated values
print df
print type(df)
print df.shape






