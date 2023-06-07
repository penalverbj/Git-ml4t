""""""
"""  		  	   		  		 			  		 			 	 	 		 		 	
Test a learner.  (c) 2015 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
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
"""

import math
import sys

import numpy as np

import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bg
import InsaneLearner as il
import matplotlib.pyplot as plt


def LinRegTest(train_x, train_y, test_x, test_y):
    # create a learner and train it
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    learner.add_evidence(train_x, train_y)  # train it
    print(learner.author())

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")


def DTTest(train_x, train_y, test_x, test_y):
    learner = dt.DTLearner()
    learner.add_evidence(train_x, train_y)
    pred_y = learner.query(train_x)
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")


def RTTest(train_x, train_y, test_x, test_y):
    learner = rt.RTLearner()
    learner.add_evidence(train_x, train_y)
    pred_y = learner.query(train_x)
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")


def BagTest(train_x, train_y, test_x, test_y):
    learner = bg.BagLearner(learner=dt.DTLearner)
    learner.add_evidence(train_x, train_y)
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")


def experiment1(train_x, train_y, test_x, test_y):
    train_rmses = []
    test_rmses = []
    for leaf in range(1, 101):
        learner = dt.DTLearner(leaf_size=leaf)
        learner.add_evidence(train_x, train_y)

        train_pred_y = learner.query(train_x)
        train_rmse = math.sqrt(((train_y - train_pred_y) ** 2).sum() / train_y.shape[0])
        test_pred_y = learner.query(test_x)
        test_rmse = math.sqrt(((test_y - test_pred_y) ** 2).sum() / test_y.shape[0])

        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)

    ticks = range(1, 101)
    plt.plot(ticks, train_rmses, label="training sample")
    plt.plot(ticks, test_rmses, label="testing sample")
    plt.title("Leaf Size vs RMSE with DT")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.savefig("Figure1.png")
    plt.clf()


def experiment2(train_x, train_y, test_x, test_y):
    train_rmses = []
    test_rmses = []
    for leaf in range(1, 101):
        learner = bg.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf}, bags=20)
        learner.add_evidence(train_x, train_y)

        train_pred_y = learner.query(train_x)
        train_rmse = math.sqrt(((train_y - train_pred_y) ** 2).sum() / train_y.shape[0])
        test_train_y = learner.query(test_x)
        test_rmse = math.sqrt(((test_y - test_train_y) ** 2).sum() / test_y.shape[0])

        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)

    ticks = range(1, 101)
    plt.plot(ticks, train_rmses, label="in sample")
    plt.plot(ticks, test_rmses, label="out sample")
    plt.title("Leaf Size vs RMSE with BagLearner with 20 bags using DT")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.savefig("Figure2.png")
    plt.clf()

    train_rmses = []
    test_rmses = []
    for leaf in range(1, 101):
        learner = bg.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf}, bags=40)
        learner.add_evidence(train_x, train_y)

        train_pred_y = learner.query(train_x)
        train_rmse = math.sqrt(((train_y - train_pred_y) ** 2).sum() / train_y.shape[0])
        test_train_y = learner.query(test_x)
        test_rmse = math.sqrt(((test_y - test_train_y) ** 2).sum() / test_y.shape[0])

        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)

    ticks = range(1, 101)
    plt.plot(ticks, train_rmses, label="in sample")
    plt.plot(ticks, test_rmses, label="out sample")
    plt.title("Leaf Size vs RMSE with BagLearner with 40 bags using DT")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.savefig("Figure3.png")
    plt.clf()

def experiment3_1(train_x, train_y, test_x, test_y):
    # Mean Absolute Error = sum(predictions - trueVal) / numDataPoints
    train_DT= []
    test_DT = []
    train_RT = []
    test_RT = []
    for leaf in range(1, 101):
        DT = dt.DTLearner(leaf_size=leaf)
        RT = rt.RTLearner(leaf_size=leaf)
        DT.add_evidence(train_x, train_y)
        RT.add_evidence(train_x, train_y)

        train_pred_y_DT = DT.query(train_x)
        train_pred_y_RT = RT.query(train_x)
        train_dt_mae = np.sum(np.abs(train_pred_y_DT - train_y)) / train_y.shape[0]
        train_rt_mae = np.sum(np.abs(train_pred_y_RT - train_y)) / train_y.shape[0]
        train_DT.append(train_dt_mae)
        train_RT.append(train_rt_mae)

        test_pred_y_DT = DT.query(test_x)
        test_pred_y_RT = RT.query(test_x)
        test_dt_mae = np.sum(np.abs(test_pred_y_DT - test_y)) / test_y.shape[0]
        test_rt_mae = np.sum(np.abs(test_pred_y_RT - test_y)) / test_y.shape[0]
        test_DT.append(test_dt_mae)
        test_RT.append(test_rt_mae)

    ticks = range(1, 101)
    plt.plot(ticks, train_DT, label="training sample DT")
    plt.plot(ticks, train_RT, label="training sample RT")
    plt.title("Leaf Size vs MAE with DT and RT (Training)")
    plt.xlabel("Leaf Size")
    plt.ylabel("MAE")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.savefig("Figure4.png")
    plt.clf()

    ticks = range(1, 101)
    plt.plot(ticks, test_DT, label="testing sample DT")
    plt.plot(ticks, test_RT, label="testing sample RT")
    plt.title("Leaf Size vs MAE with DT and RT (Testing)")
    plt.xlabel("Leaf Size")
    plt.ylabel("MAE")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.savefig("Figure5.png")
    plt.clf()

def experiment3_2(train_x, train_y, test_x, test_y):
    #mean absolute percentage error = mean(abs(true - pred) / true) * 100
    train_DT= []
    test_DT = []
    train_RT = []
    test_RT = []
    for leaf in range(1, 101):
        DT = dt.DTLearner(leaf_size=leaf)
        RT = rt.RTLearner(leaf_size=leaf)
        DT.add_evidence(train_x, train_y)
        RT.add_evidence(train_x, train_y)

        train_pred_y_DT = DT.query(train_x)
        train_pred_y_RT = RT.query(train_x)
        train_dt_mape = np.mean(np.abs(train_y - train_pred_y_DT) / train_y) * 100
        train_rt_mape = np.mean(np.abs(train_y - train_pred_y_RT) / train_y) * 100
        train_DT.append(train_dt_mape)
        train_RT.append(train_rt_mape)

        test_pred_y_DT = DT.query(test_x)
        test_pred_y_RT = RT.query(test_x)
        test_dt_mape = np.mean(np.abs(test_y - test_pred_y_DT) / test_y) * 100
        test_rt_mape = np.mean(np.abs(test_y - test_pred_y_RT) / test_y) * 100
        test_DT.append(test_dt_mape)
        test_RT.append(test_rt_mape)

    ticks = range(1, 101)
    plt.plot(ticks, train_DT, label="training sample DT")
    plt.plot(ticks, test_DT, label="testing sample DT")
    plt.plot(ticks, train_RT, label="training sample RT")
    plt.plot(ticks, test_RT, label="testing sample RT")
    plt.title("Leaf Size vs MAPE comparing DT and RT")
    plt.xlabel("Leaf Size")
    plt.ylabel("MAPE (%)")
    plt.grid(linestyle="--")
    plt.legend(loc="best")
    plt.savefig("Figure6.png")
    plt.clf()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array(
        [list(map(str, s.strip().split(","))) for s in inf.readlines()]
    )
    # only Istanbul.csv has dates, so this checks for it and removes them
    if sys.argv[1] == "Data/Istanbul.csv":
        data = data[1:, 1:]
    data = data.astype('float')
    np.random.seed(903376324)  # this is my GTID
    np.random.shuffle(data)  # shuffles whole dataset

    # compute how much of the data is training and testing  		  	   		  		 			  		 			 	 	 		 		 	
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data  		  	   		  		 			  		 			 	 	 		 		 	
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]
    # most of the section above this comment within main() came with the project

    experiment1(train_x, train_y, test_x, test_y)
    experiment2(train_x, train_y, test_x, test_y)
    experiment3_1(train_x, train_y, test_x, test_y)
    experiment3_2(train_x, train_y, test_x, test_y)
