import pandas as pd
import numpy as np


df = pd.read_csv('Tickers.csv', sep=';', header=0, index_col=False, skip_blank_lines=True, keep_default_na=False)
df.index.name = 'Index'
df.replace('', np.nan, inplace=True)
df.dropna(axis=0, inplace=True)
df.to_csv('MyTickers.csv', sep=',')