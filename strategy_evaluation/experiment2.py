import StrategyLearner
import datetime as dt
import pandas as pd
import util
import matplotlib.pyplot as plt
import marketsimcode

def author():
  return 'jpb6' # replace tb34 with your Georgia Tech username.

def impactChange():
  start_date = dt.datetime(2008, 1, 1)
  end_date = dt.datetime(2009, 12, 30)

  learner = StrategyLearner.StrategyLearner(verbose=True, impact=0.005)
  learner.add_evidence(symbol='JPM', sd=start_date, ed=end_date, sv=100000)
  trades = learner.testPolicy(symbol='JPM', sd=start_date, ed=end_date, sv=100000)

  portvalsImpact = marketsimcode.compute_portvals(orders_df=trades, impact=0.005, commission=9.95)
  cr = (portvalsImpact[-1] / portvalsImpact[0]) - 1
  dr = (portvalsImpact[1:] / portvalsImpact.shift(1)) - 1
  std_dr = dr.std()
  mean_dr = dr.mean()
  norm = portvalsImpact / portvalsImpact[0]

  learner = StrategyLearner.StrategyLearner(verbose=True, impact=0.005)
  learner.add_evidence(symbol='JPM', sd=start_date, ed=end_date, sv=100000)
  trades = learner.testPolicy(symbol='JPM', sd=start_date, ed=end_date, sv=100000)

  portvalsNoImpact = marketsimcode.compute_portvals(orders_df=trades, impact=0.005, commission=9.95)
  crNo = (portvalsNoImpact[-1] / portvalsNoImpact[0]) - 1
  drNo = (portvalsNoImpact[1:] / portvalsNoImpact.shift(1)) - 1
  std_drNo = drNo.std()
  mean_drNo = drNo.mean()
  normNo = portvalsNoImpact / portvalsNoImpact[0]

  learner = StrategyLearner.StrategyLearner(verbose=True, impact=0.01)
  learner.add_evidence(symbol='JPM', sd=start_date, ed=end_date, sv=100000)
  trades = learner.testPolicy(symbol='JPM', sd=start_date, ed=end_date, sv=100000)

  portvalsImpact1 = marketsimcode.compute_portvals(orders_df=trades, impact=0.01, commission=9.95)
  cr1 = (portvalsImpact1[-1] / portvalsImpact1[0]) - 1
  dr1 = (portvalsImpact1[1:] / portvalsImpact1.shift(1)) - 1
  std_dr1 = dr1.std()
  mean_dr1 = dr1.mean()
  norm1 = portvalsImpact1 / portvalsImpact1[0]

  learner = StrategyLearner.StrategyLearner(verbose=True, impact=0.1)
  learner.add_evidence(symbol='JPM', sd=start_date, ed=end_date, sv=100000)
  trades = learner.testPolicy(symbol='JPM', sd=start_date, ed=end_date, sv=100000)

  portvalsImpact2 = marketsimcode.compute_portvals(orders_df=trades, impact=0.1, commission=9.95)
  cr2 = (portvalsImpact2[-1] / portvalsImpact2[0]) - 1
  dr2 = (portvalsImpact2[1:] / portvalsImpact2.shift(1)) - 1
  std_dr2 = dr2.std()
  mean_dr2 = dr2.mean()
  norm2 = portvalsImpact2 / portvalsImpact2[0]

  f = open("p8_exp2_impact_test_results.txt", "a")
  f.truncate(0)
  f.write(
    "IMPACT .005 VALS: \n CR = " + str(cr) + "\n STD_DR = " + str(std_dr) + "\n MEAN_DR = " + str(
      mean_dr) +
    "\n\nNO IMPACT VALS: \n CR = " + str(crNo) + "\n STD_DR = " + str(
      std_drNo) + "\n MEAN_DR = " + str(mean_drNo) + "\n" +
    "\n\nIMPACT .01 VALS: \n CR = " + str(cr1) + "\n STD_DR = " + str(
      std_dr1) + "\n MEAN_DR = " + str(mean_dr1) + "\n" +
    "\n\nIMPACT .1 VALS: \n CR = " + str(cr2) + "\n STD_DR = " + str(
      std_dr2) + "\n MEAN_DR = " + str(mean_dr2) + "\n"
  )
  f.close()

  f = plt.figure()
  f.set_figwidth(10)
  f.set_figheight(7)
  plt.plot(norm, label="Impact .005", color="red")
  plt.plot(normNo, label="No Impact", color="purple")
  plt.plot(norm1, label="Impact .01", color="pink")
  plt.plot(norm2, label="Impact .1", color="blue")
  plt.grid(linestyle="--")
  plt.legend(loc="best")
  plt.title("Effect of Impact on BagLearner")
  plt.xlabel("Date")
  plt.ylabel("Normalized Value ($)")
  plt.savefig("experiment2Impact.png")
  plt.clf()
  plt.close(f)


def experiment2():
  impactChange()

if __name__ == "__main__":
    experiment2()
