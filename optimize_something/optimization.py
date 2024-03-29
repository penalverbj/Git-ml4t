""""""  		  	   		  		 			  		 			 	 	 		 		 	
"""MC1-P2: Optimize a portfolio.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
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
  		  	   		  		 			  		 			 	 	 		 		 	
import numpy as np  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
import matplotlib.pyplot as plt  		  	   		  		 			  		 			 	 	 		 		 	
import pandas as pd  		  	   		  		 			  		 			 	 	 		 		 	
from util import get_data, plot_data
from scipy import optimize as spo
  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
# This is the function that will be tested by the autograder  		  	   		  		 			  		 			 	 	 		 		 	
# The student must update this code to properly implement the functionality  		  	   		  		 			  		 			 	 	 		 		 	
def optimize_portfolio(  		  	   		  		 			  		 			 	 	 		 		 	
    sd=dt.datetime(2008, 1, 1),  		  	   		  		 			  		 			 	 	 		 		 	
    ed=dt.datetime(2009, 1, 1),  		  	   		  		 			  		 			 	 	 		 		 	
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		  		 			  		 			 	 	 		 		 	
    gen_plot=False,  		  	   		  		 			  		 			 	 	 		 		 	
):  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		  		 			  		 			 	 	 		 		 	
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		  		 			  		 			 	 	 		 		 	
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		  		 			  		 			 	 	 		 		 	
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		  		 			  		 			 	 	 		 		 	
    statistics.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			 	 	 		 		 	
    :type sd: datetime  		  	   		  		 			  		 			 	 	 		 		 	
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			 	 	 		 		 	
    :type ed: datetime  		  	   		  		 			  		 			 	 	 		 		 	
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		  		 			  		 			 	 	 		 		 	
        symbol in the data directory)  		  	   		  		 			  		 			 	 	 		 		 	
    :type syms: list  		  	   		  		 			  		 			 	 	 		 		 	
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		  		 			  		 			 	 	 		 		 	
        code with gen_plot = False.  		  	   		  		 			  		 			 	 	 		 		 	
    :type gen_plot: bool  		  	   		  		 			  		 			 	 	 		 		 	
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		  		 			  		 			 	 	 		 		 	
        standard deviation of daily returns, and Sharpe ratio  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: tuple  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
    # Read in adjusted closing prices for given symbols, date range  		  	   		  		 			  		 			 	 	 		 		 	
    dates = pd.date_range(sd, ed)  		  	   		  		 			  		 			 	 	 		 		 	
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all.fillna(method='ffill', inplace=True)
    prices_all.fillna(method='bfill', inplace=True)
    prices = prices_all[syms]  # only portfolio symbols  		  	   		  		 			  		 			 	 	 		 		 	
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
    SPY_norm = prices_SPY / prices_SPY.iloc[0]

    # find the allocations for the optimal portfolio  		  	   		  		 			  		 			 	 	 		 		 	
    # note that the values here ARE NOT meant to be correct for a test case
    num_syms = len(syms)
    allocs_guess = np.zeros(num_syms)# add code here to find the allocations
    allocs_guess.fill(1.0/num_syms)

    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    bounds = tuple((0, 1) for ag in allocs_guess)

    allocs = spo.minimize(sharpe_func, allocs_guess, args=prices, bounds=bounds, constraints=cons,
                          options={"disp": "TRUE"})
    port_val, cr, adr, sddr, sr = get_portfolio_stats(allocs.x, prices)
  		  	   		  		 			  		 			 	 	 		 		 	
    if gen_plot:
        # add code to plot here  		  	   		  		 			  		 			 	 	 		 		 	
        df_temp = pd.concat(  		  	   		  		 			  		 			 	 	 		 		 	
            [port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1  		  	   		  		 			  		 			 	 	 		 		 	
        )  		  	   		  		 			  		 			 	 	 		 		 	
        plt.plot(port_val, label="Portfolio")
        plt.plot(SPY_norm, label="SPY")
        plt.legend(loc="best")
        plt.title("Daily Portfolio Value VS. SPY")
        plt.ylabel("Normalized Price")
        plt.xlabel("Date")
        plt.grid(linestyle="--")
        plt.savefig("Figure1.png")
  		  	   		  		 			  		 			 	 	 		 		 	
    return allocs.x, cr, adr, sddr, sr

def sharpe_func(X, prices):
    return get_portfolio_stats(X, prices)[4] * -1

def get_portfolio_stats(allocs, prices):
    normed_prices = prices / prices.iloc[0]
    alloced_prices = normed_prices * allocs
    port_val = alloced_prices.sum(axis=1)

    daily_returns = (port_val.iloc[1:].values / port_val.iloc[:-1]) - 1
    daily_returns = daily_returns[1:]

    cr = (port_val.iloc[-1] / port_val.iloc[0]) - 1
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = np.sqrt(252) * (adr / sddr)

    return port_val, cr, adr, sddr, sr
  		  	   		  		 			  		 			 	 	 		 		 	
def test_code():  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    This function WILL NOT be called by the auto grader.  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]
    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )

    # start_date = dt.datetime(2010, 6, 1)
    # end_date = dt.datetime(2010, 12, 31)
    # symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    # dates = pd.date_range(start_date, end_date)
    # prices_all = get_data(symbols, dates)  # automatically adds SPY
    # prices_all.fillna(method='ffill', inplace=True)
    # prices_all.fillna(method='bfill', inplace=True)
    # prices = prices_all[syms]  # only portfolio symbols
    # allocations = [0.2, 0.3, 0.4, 0.1]
    # _, cr, adr, sddr, sr = get_portfolio_stats(
    #     allocs=allocations, prices=prices
    # )
  		  	   		  		 			  		 			 	 	 		 		 	
    # Print statistics
    print('\n')
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")  		  	   		  		 			  		 			 	 	 		 		 	
    print(f"Sharpe Ratio: {sr}")  		  	   		  		 			  		 			 	 	 		 		 	
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		  		 			  		 			 	 	 		 		 	
    print(f"Average Daily Return: {adr}")  		  	   		  		 			  		 			 	 	 		 		 	
    print(f"Cumulative Return: {cr}")  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		  		 			  		 			 	 	 		 		 	
    # This code WILL NOT be called by the auto grader  		  	   		  		 			  		 			 	 	 		 		 	
    # Do not assume that it will be called  		  	   		  		 			  		 			 	 	 		 		 	
    test_code()  		  	   		  		 			  		 			 	 	 		 		 	
