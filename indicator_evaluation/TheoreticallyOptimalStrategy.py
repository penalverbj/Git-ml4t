import datetime as dt
import pandas as pd
import util


def author():
    return 'jpb6'


def testPolicy(symbol='AAPL', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    prices = pd.DataFrame(util.get_data([symbol], pd.date_range(sd, ed)))
    if (symbol != 'SPY'):
        prices = prices.drop(['SPY'], axis=1)

    prices['shift'] = prices.shift(periods=-1) - prices
    prices['shift'] /= abs(prices['shift'])

    prices.fillna(method='bfill', inplace=True)
    trades = pd.DataFrame(data=0, index=prices.index, columns={symbol})
    trades[symbol] = prices.iloc[0, -1] * 1000

    for row, col in prices[1:].iterrows():
        if col['shift'] == 1 and trades[row-1] != 2000:
            trades.loc[row] = 2000
        elif col['shift'] == -1 and trades[row-1] != -2000:
            trades.loc[row] = -2000
        else:
            trades.loc[row] = 0

    trades.loc[-1] = 0
    return trades


