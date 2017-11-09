from pandas.io.data import Options
spy = Options('spy', 'yahoo')
chain = spy.get_all_data()