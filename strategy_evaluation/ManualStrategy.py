import util
import datetime as dt
import pandas as pd
import marketsimcode
import matplotlib.pyplot as plt
import indicators


def author():
    return 'jpb6'  # replace tb34 with your Georgia Tech username.


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
    cci = indicators.cci(prices, 14, symbol)
    sma = indicators.sma(prices, 24)

    short = []
    long = []
    holdings = 0
    for date in dates:
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
            macd_vote = 1
        elif macd_today.loc['signal'] < macd_today.loc['macd']:
            macd_vote = -1
        else:
            macd_vote = 1

        if cci_today.loc['cci'] >= 100:
            cci_vote = 1
        elif cci_today.loc['cci'] <= -100:
            cci_vote = -1
        else:
            cci_vote = 0

        final_vote = macd_vote + cci_vote + sma_vote
        if holdings == 0:
            if final_vote >= 2:
                trades.loc[date] = 1000
                long.append(date)
                holdings = 1000
            elif final_vote <= -2:
                trades.loc[date] = -1000
                short.append(date)
                holdings = -1000
        elif holdings == 1000:
            if final_vote <= -2:
                trades.loc[date] = -2000
                short.append(date)
                holdings = -1000
        elif holdings == -1000:
            if final_vote >= 2:
                trades.loc[date] = 2000
                long.append(date)
                holdings = 1000

    return trades, long, short

def inSample():
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 30)
    symbols = ["JPM"]
    trades, long, short = manualStrategy(symbols, start_date, end_date, starting_value=100000)
    manualPortVals = marketsimcode.compute_portvals(orders_df=trades, impact=0.005, commission=9.95)
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

    f = open("p8_inSample_results.txt", "a")
    f.truncate(0)
    f.write(
        "MANUAL STRAT VALS: \n CR = " + str(crManual) + "\n STD_DR = " + str(std_drManual) + "\n MEAN_DR = " + str(
            mean_drManual) +
        "\n\nBENCHMARK STRAT VALS: \n CR = " + str(crBenchmark) + "\n STD_DR = " + str(
            std_drBenchmark) + "\n MEAN_DR = " + str(mean_drBenchmark) + "\n")
    f.close()

    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(7)
    plt.plot(manualNorm, label="Manual Strategy", color="red")
    plt.plot(benchmarkNorm, label="Benchmark", color="purple")
    for date in short:
        plt.axvline(date, color="black")
    for date in long:
        plt.axvline(date, color="blue")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.title("In-Sample Normalized Portfolio Values")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value ($)")
    plt.savefig("ManualInSample.png")
    plt.clf()
    plt.close(f)


def outSample():
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2011, 12, 30)
    symbols = ["JPM"]
    trades, long, short = manualStrategy(symbols, start_date, end_date, starting_value=100000)

    manualPortVals = marketsimcode.compute_portvals(orders_df=trades, impact=0.005, commission=9.95)
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

    f = open("p8_outSample_results.txt", "a")
    f.truncate(0)
    f.write(
        "MANUAL STRAT VALS: \n CR = " + str(crManual) + "\n STD_DR = " + str(std_drManual) + "\n MEAN_DR = " + str(
            mean_drManual) +
        "\n\nBENCHMARK STRAT VALS: \n CR = " + str(crBenchmark) + "\n STD_DR = " + str(
            std_drBenchmark) + "\n MEAN_DR = " + str(mean_drBenchmark) + "\n")
    f.close()

    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(7)
    plt.plot(manualNorm, label="Manual Strategy", color="red")
    plt.plot(benchmarkNorm, label="Benchmark", color="purple")
    for date in short:
        plt.axvline(date, color="black")
    for date in long:
        plt.axvline(date, color="blue")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.title("Out-of-Sample Normalized Portfolio Values")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value ($)")
    plt.savefig("ManualOutSample.png")
    plt.clf()
    plt.close(f)


if __name__ == "__main__":
    inSample()
    outSample()
