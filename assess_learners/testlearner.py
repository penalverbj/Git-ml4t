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
    in_sample_rsmes = []
    out_sample_rsmes = []
    for leaf in range(1, 101):
        learner = dt.DTLearner(leaf_size=leaf)
        learner.add_evidence(train_x, train_y)
        print("learner made")
        # in sample
        # in_predY = learner.query(train_x)
        # in_rmse = math.sqrt(((train_y - in_predY) ** 2).sum() / train_y.shape[0])
        # print(in_rmse)
        # out sample
        # out_predY = learner.query(test_x)
        # out_rmse = math.sqrt(((test_y - out_predY) ** 2).sum() / test_y.shape[0])
        #
        # in_sample_rsmes.append(in_rmse)
        # out_sample_rsmes.append(out_rmse)

    # xi = range(1, 101)
    # plt.plot(xi, in_sample_rsmes, label="in sample")
    # plt.plot(xi, out_sample_rsmes, label="out sample")
    #
    # plt.title("Figure 1 - Leaf Size and Overfitting in DT")
    # plt.xlabel("Leaf Size")
    # plt.ylabel("RMSE")
    # plt.xticks(np.insert(np.arange(5, 101, step=5), 0, 1))
    # plt.grid()
    # plt.legend()
    # plt.savefig("figure1.png")
    # plt.clf()
    #
    # xi = range(1, 21)
    # plt.plot(xi, in_sample_rsmes[:20], label="in sample")
    # plt.plot(xi, out_sample_rsmes[:20], label="out sample")
    #
    # plt.title("Figure 2 - Leaf Size and Overfitting in DT (Zoomed In)")
    # plt.xlabel("Leaf Size")
    # plt.ylabel("RMSE")
    # plt.xticks(np.insert(np.arange(5, 21, step=5), 0, 1))
    # plt.grid()
    # plt.legend()
    # plt.savefig("figure2.png")
    # plt.clf()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array(
        [list(map(str, s.strip().split(","))) for s in inf.readlines()]
    )
    if sys.argv[1] == "Data/Istanbul.csv":
        data = data[1:, 1:]
    data = data.astype('float')

    # compute how much of the data is training and testing  		  	   		  		 			  		 			 	 	 		 		 	
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data  		  	   		  		 			  		 			 	 	 		 		 	
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    experiment1(train_x, train_y, test_x, test_y)
