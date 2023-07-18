import util
import ManualStrategy
import StrategyLearner
import pandas as pd
import marketsimcode
import datetime as dt
import matplotlib.pyplot as plt


def author():
    return 'jpb6'  # replace tb34 with your Georgia Tech username.


def inSample():
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 30)
    symbols = ["JPM"]
    trades, long, short = ManualStrategy.manualStrategy(symbols, start_date, end_date, starting_value=100000)
    manualPortVals = marketsimcode.compute_portvals(orders_df=trades, impact=9.95, commission=0.005)
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

    learner = StrategyLearner.StrategyLearner(verbose=True, impact=0.000)
    learner.add_evidence(symbol='JPM', sd=start_date, ed=end_date, sv=100000)
    trades = learner.testPolicy(symbol='JPM', sd=start_date, ed=end_date, sv=100000)

    portvals = marketsimcode.compute_portvals(orders_df=trades, impact=9.95, commission=0.005)
    cr = (portvals[-1] / portvals[0]) - 1
    dr = (portvals[1:] / portvals.shift(1)) - 1
    std_dr = dr.std()
    mean_dr = dr.mean()
    norm = portvals / portvals[0]

    f = open("p8_exp1_results.txt", "a")
    f.truncate(0)
    f.write(
        "MANUAL STRAT VALS: \n CR = " + str(crManual) + "\n STD_DR = " + str(std_drManual) + "\n MEAN_DR = " + str(
            mean_drManual) +
        "\n\nBENCHMARK STRAT VALS: \n CR = " + str(crBenchmark) + "\n STD_DR = " + str(
            std_drBenchmark) + "\n MEAN_DR = " + str(mean_drBenchmark) + "\n" +
        "\n\nBAGLEARNER STRAT VALS: \n CR = " + str(cr) + "\n STD_DR = " + str(
            std_dr) + "\n MEAN_DR = " + str(mean_dr) + "\n")
    f.close()

    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(7)
    plt.plot(manualNorm, label="Manual Strategy", color="red")
    plt.plot(benchmarkNorm, label="Benchmark", color="purple")
    plt.plot(norm, label="BagLearner Strategy", color="pink")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.title("In-Sample Normalized Portfolio Values")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value ($)")
    plt.savefig("experiment1.png")
    plt.clf()
    plt.close(f)


def outSample():
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2011, 12, 30)
    symbols = ["JPM"]
    trades, long, short = ManualStrategy.manualStrategy(symbols, start_date, end_date, starting_value=100000)
    manualPortVals = marketsimcode.compute_portvals(orders_df=trades, impact=9.95, commission=0.005)
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

    learner = StrategyLearner.StrategyLearner(verbose=True, impact=0.000)
    learner.add_evidence(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 30), sv=100000)
    trades = learner.testPolicy(symbol='JPM', sd=start_date, ed=end_date, sv=100000)

    portvals = marketsimcode.compute_portvals(orders_df=trades, impact=9.95, commission=0.005)
    cr = (portvals[-1] / portvals[0]) - 1
    dr = (portvals[1:] / portvals.shift(1)) - 1
    std_dr = dr.std()
    mean_dr = dr.mean()
    norm = portvals / portvals[0]

    f = open("p8_exp1OUT_results.txt", "a")
    f.truncate(0)
    f.write(
        "MANUAL STRAT VALS: \n CR = " + str(crManual) + "\n STD_DR = " + str(std_drManual) + "\n MEAN_DR = " + str(
            mean_drManual) +
        "\n\nBENCHMARK STRAT VALS: \n CR = " + str(crBenchmark) + "\n STD_DR = " + str(
            std_drBenchmark) + "\n MEAN_DR = " + str(mean_drBenchmark) + "\n" +
        "\n\nBAGLEARNER STRAT VALS: \n CR = " + str(cr) + "\n STD_DR = " + str(
            std_dr) + "\n MEAN_DR = " + str(mean_dr) + "\n")
    f.close()

    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(7)
    plt.plot(manualNorm, label="Manual Strategy", color="red")
    plt.plot(benchmarkNorm, label="Benchmark", color="purple")
    plt.plot(norm, label="BagLearner Strategy", color="pink")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.title("Out-Sample Normalized Portfolio Values")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value ($)")
    plt.savefig("experiment1OUT.png")
    plt.clf()
    plt.close(f)


if __name__ == "__main__":
    # inSample()
    outSample()
