import TheoreticallyOptimalStrategy as tos
from marketsimcode import compute_portvals

def author():
    return 'jpb6'

def test_code():
    optimalTrades = tos.testPolicy(symbol='JPM')
    optimalPortVals = compute_portvals(orders_df=optimalTrades, impact=0, commission=0)
    print(optimalPortVals)

if __name__ == "__main__":
    test_code()