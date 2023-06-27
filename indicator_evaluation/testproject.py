import pandas as pd

import TheoreticallyOptimalStrategy as tos
import util
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
import datetime as dt


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
    benchmarkPortVals = benchmarkPortVals.drop(['SPY'], axis=1)
    crBenchmark = (benchmarkPortVals.iloc[-1] / benchmarkPortVals.iloc[0]) - 1
    drBenchmark = (benchmarkPortVals.iloc[1:] / benchmarkPortVals.shift(1)) - 1
    std_drBenchmark = drBenchmark.std()
    mean_drBenchmark = drBenchmark.mean()
    benchmarkNorm = benchmarkPortVals / benchmarkPortVals.iloc[0]

    f = open("p6_results.txt", "a")
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


def part_2():
    pass


if __name__ == "__main__":
    part_1()
