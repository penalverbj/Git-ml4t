""""""
"""  		  	   		  		 			  		 			 	 	 		 		 	
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	
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

import math

import numpy as np


# this function should return a dataset (X and Y) that will work  		  	   		  		 			  		 			 	 	 		 		 	
# better for linear regression than decision trees  		  	   		  		 			  		 			 	 	 		 		 	
def best_4_lin_reg(seed=1489683273):
    """  		  	   		  		 			  		 			 	 	 		 		 	
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		  		 			  		 			 	 	 		 		 	
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		  		 			  		 			 	 	 		 		 	
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
    :param seed: The random seed for your data generation.  		  	   		  		 			  		 			 	 	 		 		 	
    :type seed: int  		  	   		  		 			  		 			 	 	 		 		 	
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: numpy.ndarray  		  	   		  		 			  		 			 	 	 		 		 	
    """
    np.random.seed(seed)
    features = 6
    rows = 500
    x = np.random.random((rows, features)) * 200 - 100
    # basic linear combination of features by adding them
    y = np.sum(x, axis=1)
    return x, y


def best_4_dt(seed=1489683273):
    """  		  	   		  		 			  		 			 	 	 		 		 	
    Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		  		 			  		 			 	 	 		 		 	
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		  		 			  		 			 	 	 		 		 	
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
    :param seed: The random seed for your data generation.  		  	   		  		 			  		 			 	 	 		 		 	
    :type seed: int  		  	   		  		 			  		 			 	 	 		 		 	
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: numpy.ndarray  		  	   		  		 			  		 			 	 	 		 		 	
    """
    np.random.seed(seed)
    features = 6
    rows = 500
    x = np.random.random((rows, features)) * 200 - 100
    avg = np.mean(x)
    y = np.zeros(rows)
    for r in range(rows):
        if (x[r,3] < avg):
            y[r] = 5
        else:
            y[r] = -5
    return x, y


def author():
    """  		  	   		  		 			  		 			 	 	 		 		 	
    :return: The GT username of the student  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: str  		  	   		  		 			  		 			 	 	 		 		 	
    """
    return "jpb6"  # Change this to your user ID

# if __name__ == "__main__":
#     x,y = best_4_dt()
#     print(x.shape)
#     print(y.shape)
#
#     print(x[0,:])
#     print(y[0])
