import util
import datetime as dt
import pandas as pd
import marketsimcode
import matplotlib.pyplot as plt
import indicators

def author():
  return 'jpb6' # replace tb34 with your Georgia Tech username.

def manualStrategy(symbol, start_date, end_date, starting_value):
    sym = symbol[0]
    data = pd.DataFrame(util.get_data([sym], pd.date_range(start_date, end_date)))
    prices = data.drop(['SPY'], axis=1)
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    norm_price = prices[sym] / prices[sym][0]

    trades = prices.copy(deep=True)
    trades[:] = 0
    dates = trades.index

    macd = indicators.macd(prices, 12, 36, 10)
    cci = indicators.cci(prices, 14)
    sma = indicators.sma(prices, 24)

    pos_curr = 0
    a_last = 0

    for date in dates:
        a_last += 1

        price_today = prices.loc[date]
        sma_today = sma.loc[date]
        macd_today = macd.loc[date]
        cci_today = cci.loc[date]

        if price_today[sym] > sma_today.loc["mean"]:
            sma_vote = 1
        elif price_today[sym] < sma_today.loc["mean"]:
            sma_vote = -1
        else:
            sma_vote = 0

        if macd_today.loc['signal'] > macd_today.loc['macd']:
            macd_vote = 2
        elif macd_today.loc['signal'] < macd_today.loc['macd']:
            macd_vote = -10
        else:
            macd_vote = 1

        if cci_today.loc['cci'] > 100:
            cci_vote = 2
        elif cci_today.loc['cci'] < -100:
            cci_vote = -5
        else:
            cci_vote = 0

        final_vote = macd_vote + cci_vote + sma_vote
        if final_vote >= 3:
            action = 1000 - pos_curr

        elif final_vote <= -3:
            action = - 1000 - pos_curr
        else:
            action = -pos_curr

        if a_last >= 5:
            trades.loc[date] = action
            pos_curr += action
            a_last = 0

    return trades


if __name__ == "__main__":
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["JPM"]
    trades = manualStrategy(symbols, start_date, end_date, starting_value=100000)

    manualPortVals = marketsimcode.compute_portvals(orders_df=trades, impact=0, commission=0)
    crManual = (manualPortVals[-1] / manualPortVals[0]) - 1
    drManual = (manualPortVals[1:] / manualPortVals.shift(1)) - 1
    std_drManual = drManual.std()
    mean_drManual = drManual.mean()
    manualNorm = manualPortVals / manualPortVals[0]

    benchmarkPortVals = pd.DataFrame(util.get_data(["JPM"], pd.date_range(start_date, end_date)))
    benchmarkPortVals.fillna(method='ffill', inplace=True)
    benchmarkPortVals.fillna(method='bfill', inplace=True)
    benchmarkPortVals = benchmarkPortVals.drop(['SPY'], axis=1)
    crBenchmark = (benchmarkPortVals.iloc[-1] / benchmarkPortVals.iloc[0]) - 1
    drBenchmark = (benchmarkPortVals.iloc[1:] / benchmarkPortVals.shift(1)) - 1
    std_drBenchmark = drBenchmark.std()
    mean_drBenchmark = drBenchmark.mean()
    benchmarkNorm = benchmarkPortVals / benchmarkPortVals.iloc[0]

    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(7)
    plt.plot(manualNorm, label="Manual Strategy", color="red")
    plt.plot(benchmarkNorm, label="Benchmark", color="purple")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.title("Normalized Portfolio Values")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value ($)")
    plt.show()
    plt.clf()
    plt.close(f)



