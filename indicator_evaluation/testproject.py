import pandas as pd

import TheoreticallyOptimalStrategy as tos
import util
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
import datetime as dt
import indicators


def author():
    return 'jpb6'


def part_1():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    optimalTrades = tos.testPolicy(symbol='JPM', sd=sd, ed=ed)
    optimalPortVals = compute_portvals(orders_df=optimalTrades, impact=0, commission=0)
    crOptimal = (optimalPortVals[-1] / optimalPortVals[0]) - 1
    drOptimal = (optimalPortVals[1:] / optimalPortVals.shift(1)) - 1
    std_drOptimal = drOptimal.std()
    mean_drOptimal = drOptimal.mean()
    optimalNorm = optimalPortVals / optimalPortVals[0]

    benchmarkPortVals = pd.DataFrame(util.get_data(["JPM"], pd.date_range(sd, ed)))
    benchmarkPortVals.fillna(method='ffill', inplace=True)
    benchmarkPortVals.fillna(method='bfill', inplace=True)
    benchmarkPortVals = benchmarkPortVals.drop(['SPY'], axis=1)
    crBenchmark = (benchmarkPortVals.iloc[-1] / benchmarkPortVals.iloc[0]) - 1
    drBenchmark = (benchmarkPortVals.iloc[1:] / benchmarkPortVals.shift(1)) - 1
    std_drBenchmark = drBenchmark.std()
    mean_drBenchmark = drBenchmark.mean()
    benchmarkNorm = benchmarkPortVals / benchmarkPortVals.iloc[0]

    f = open("p6_results.txt", "a")
    f.truncate(0)
    f.write(
        "OPTIMAL STRAT VALS: \n CR = " + str(crOptimal) + "\n STD_DR = " + str(std_drOptimal) + "\n MEAN_DR = " + str(
            mean_drOptimal) +
        "\n\nBENCHMARK STRAT VALS: \n CR = " + str(crBenchmark) + "\n STD_DR = " + str(
            std_drBenchmark) + "\n MEAN_DR = " + str(mean_drBenchmark) + "\n")
    f.close()

    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(7)
    plt.plot(optimalNorm, label="Theoretical Optimal", color="red")
    plt.plot(benchmarkNorm, label="Benchmark", color="purple")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.title("Normalized Portfolio Values")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value ($)")
    plt.savefig("Part1-BenchmarkVsTheoretical.png")
    plt.clf()
    plt.close(f)


def part_2():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    data = pd.DataFrame(util.get_data(["JPM"], pd.date_range(sd, ed)))
    data = data.drop(['SPY'], axis=1)

    #SMA
    sma12 = indicators.sma(data, 12)
    sma36 = indicators.sma(data, 36)
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(7)
    plt.plot(data, label="JPM Data")
    plt.plot(sma12, label="SMA-12")
    plt.plot(sma36, label="SMA-36")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.title("Simple Moving Averages of JPM")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
    plt.savefig("sma.png")
    plt.clf()
    plt.close(f)

    #Bolinger Bands
    bb = indicators.bolinger_bands(data, window=24, threshold=2)
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(7)
    plt.plot(data, label="JPM Data")
    plt.plot(bb["upper"], label="Upper BB")
    plt.plot(bb["lower"], label="Lower BB")
    plt.plot(bb["mean"], label="Mean")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.title("Bollinger Bands of JPM")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
    plt.savefig("bb.png")
    plt.clf()
    plt.close(f)

    #EMA
    ema12 = indicators.ema(data, 12)
    ema36 = indicators.ema(data, 36)
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(7)
    plt.plot(data, label="JPM Data")
    plt.plot(ema12, label="EMA-12")
    plt.plot(ema36, label="EMA-36")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.title("Exponential Moving Averages of JPM")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
    plt.savefig("ema.png")
    plt.clf()
    plt.close(f)

    #MACD
    macd = indicators.macd(data, 12, 36, 10)
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(7)
    # plt.plot(data, label="JPM Data")
    plt.plot(macd["macd"], label="MACD")
    plt.plot(macd['signal'], label="Signal")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.title("Moving Average Convergence/Divergence of JPM")
    plt.xlabel("Date")
    plt.ylabel("Arbitrary Price %")
    plt.savefig("macd.png")
    plt.clf()
    plt.close(f)

    #CCI
    cci = indicators.cci(data, 14)
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(7)
    # plt.plot(data, label="JPM Data")
    plt.plot(cci, label="CCI")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.title("Commodity Channel Index of JPM")
    plt.xlabel("Date")
    plt.ylabel("CCI")
    plt.savefig("cci.png")
    plt.clf()
    plt.close(f)

if __name__ == "__main__":
    part_1()
    # part_2()
