import pandas as pd
import util


def author():
    return 'jpb6'


def compute_portvals(
        orders_df,
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    sym = orders_df.columns[0]
    dates = orders_df.index
    start_date = dates[0]
    end_date = dates[-2]
    prices_df = pd.DataFrame(util.get_data([sym], pd.date_range(start_date, end_date)))
    prices_df['cash'] = 1

    trades_df = prices_df.copy(deep=True)
    holdings_df = prices_df.copy(deep=True)

    for date in dates:
        if orders_df.ix[date, sym] != 0:
            cost = orders_df.ix[date, sym] * prices_df.ix[date, sym]
            trades_df.ix[date, 'cash'] = -1 * cost - abs(commission + cost*impact)
    trades_df.fillna(0)

    holdings_df.ix[start_date, 'cash'] = start_val + trades_df.ix[start_date, 'cash']
    holdings_df.iloc[0, :-1] = trades_df.iloc[0, :-1]

    for i in range(1, trades_df.shape[0]):
        t = trades_df.iloc[i, :]
        h = holdings_df.iloc[i - 1, :]
        holdings_df.iloc[i, :] = t + h

    portvals = holdings_df.multiply(prices_df).sum(axis=1)
    return portvals
