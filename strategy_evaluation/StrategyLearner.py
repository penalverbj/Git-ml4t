""""""
import numpy as np
from matplotlib import pyplot as plt

from strategy_evaluation import marketsimcode

"""  		  	   		  		 			  		 			 	 	 		 		 	
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		  		 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			 	 	 		 		 	
or edited.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Student Name: Jose Penalver Bartolome (replace with your name)  		  	   		  		 			  		 			 	 	 		 		 	
GT User ID: jpb6 (replace with your User ID)  		  	   		  		 			  		 			 	 	 		 		 	
GT ID: 903376324 (replace with your GT ID)  		  	   		  		 			  		 			 	 	 		 		 	
"""

import datetime as dt
import random
import QLearner as q
import pandas as pd
import util as ut
import indicators
import BagLearner as bl
import RTLearner as rt


class StrategyLearner(object):
    """  		  	   		  		 			  		 			 	 	 		 		 	
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			 	 	 		 		 	
        If verbose = False your code should not generate ANY output.  		  	   		  		 			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		  		 			  		 			 	 	 		 		 	
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  		 			  		 			 	 	 		 		 	
    :type impact: float  		  	   		  		 			  		 			 	 	 		 		 	
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  		 			  		 			 	 	 		 		 	
    :type commission: float  		  	   		  		 			  		 			 	 	 		 		 	
    """

    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Constructor method  		  	   		  		 			  		 			 	 	 		 		 	
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 1}, bags=20, boost=False, verbose=False)

    # this method should create a QLearner, and train it for trading  		  	   		  		 			  		 			 	 	 		 		 	
    def add_evidence(
            self,
            symbol="IBM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 1, 1),
            sv=10000,
    ):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Trains your strategy learner over a given time frame.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
        :param symbol: The stock symbol to train on  		  	   		  		 			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		  		 			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		  		 			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		  		 			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		  		 			  		 			 	 	 		 		 	
        :type sv: int  		  	   		  		 			  		 			 	 	 		 		 	
        """
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)
        prices = prices_all[syms]

        sma, macd, cci = get_indicators(prices, symbol)

        inds = pd.concat((sma, macd, cci), axis=1)
        inds.fillna(0, inplace=True)
        inds = inds[:-5]
        trainX = inds.values

        trainY = []
        for i in range(prices.shape[0] - 5):
            ratio = (prices.ix[i + 5, symbol] - prices.ix[i, symbol]) / prices.ix[i, symbol]
            if ratio > (0.02 + self.impact):
                trainY.append(1)
            elif ratio < (-0.02 - self.impact):
                trainY.append(-1)
            else:
                trainY.append(0)

        trainY = np.array(trainY)

        self.learner.add_evidence(trainX, trainY)

    def testPolicy(
            self,
            symbol="IBM",
            sd=dt.datetime(2009, 1, 1),
            ed=dt.datetime(2010, 1, 1),
            sv=10000,
    ):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Tests your learner using data outside of the training data  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
        :param symbol: The stock symbol that you trained on on  		  	   		  		 			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		  		 			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		  		 			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		  		 			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		  		 			  		 			 	 	 		 		 	
        :type sv: int  		  	   		  		 			  		 			 	 	 		 		 	
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  		 			  		 			 	 	 		 		 	
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  		 			  		 			 	 	 		 		 	
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  		 			  		 			 	 	 		 		 	
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  		 			  		 			 	 	 		 		 	
        :rtype: pandas.DataFrame  		  	   		  		 			  		 			 	 	 		 		 	
        """

        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)
        prices = prices_all[syms]
        trades = prices_all[syms].copy()
        trades.loc[:] = 0

        sma, macd, cci = get_indicators(prices, symbol)

        inds = pd.concat((sma, macd, cci), axis=1)
        inds.fillna(0, inplace=True)
        testX = inds.values

        testY = self.learner.query(testX)

        flag = 0
        for i in range(0, prices.shape[0] - 1):
            if flag == 0:
                if testY[i] > 0:
                    trades.values[i, :] = 1000
                    flag = 1
                elif testY[i] < 0:
                    trades.values[i, :] = -1000
                    flag = -1

            elif flag == 1:
                if testY[i] < 0:
                    trades.values[i, :] = -2000
                    flag = -1
                elif testY[i] == 0:
                    trades.values[i, :] = -1000
                    flag = 0

            else:
                if testY[i] > 0:
                    trades.values[i, :] = 2000
                    flag = 1
                elif testY[i] == 0:
                    trades.values[i, :] = 1000
                    flag = 0

        if flag == -1:
            trades.values[prices.shape[0] - 1, :] = 1000
        elif flag == 1:
            trades.values[prices.shape[0] - 1, :] = -1000

        return trades


def author():
    return 'jpb6'  # replace tb34 with your Georgia Tech username.


def get_indicators(prices, symbol):
    macd = indicators.macd(prices, 12, 36, 10)
    cci = indicators.cci(prices, 14)
    sma = indicators.sma(prices, 24)
    sma.rename(columns={sma.columns[0]: symbol}, inplace=True)

    cci = (cci > 100) * 1
    macd = (macd['macd'] < macd['signal']) * 1
    sma = (prices > sma) * 1
    sma.rename(columns={sma.columns[0]: 'mean'}, inplace=True)

    return sma, macd, cci

if __name__ == "__main__":
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 30)
    symbol = "JPM"
    learner = StrategyLearner(verbose=True, impact=0.000)
    learner.add_evidence(symbol=symbol, sd=start_date, ed=end_date, sv=100000)
    trades = learner.testPolicy(symbol=symbol, sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)

    portvals = marketsimcode.compute_portvals(orders_df=trades, impact=9.95, commission=0.005)
    cr = (portvals[-1] / portvals[0]) - 1
    dr = (portvals[1:] / portvals.shift(1)) - 1
    std_dr = dr.std()
    mean_dr= dr.mean()
    norm = portvals / portvals[0]

    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(7)
    plt.plot(norm, label="BagLearner Strategy", color="red")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.title("BagLearner Normalized Portfolio Values")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value ($)")
    plt.savefig("BagLearnerTest.png")
    plt.clf()
    plt.close(f)
