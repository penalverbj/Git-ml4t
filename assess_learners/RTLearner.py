""""""
"""  		  	   		  		 			  		 			 	 	 		 		 	
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	

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

import numpy as np
from random import randint


class RTLearner(object):
    """
    This is a Linear Regression Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.tree = None

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jpb6"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        y = np.array([data_y]).T
        xy = np.append(data_x, y, axis=1)
        self.tree = self.build_tree(xy)

    def build_tree(self, data):
        if (np.all(data[:, -1] == data[0, -1], axis=0)):
            return np.array([[-404, data[0, -1], -1, -1]])

        if (data.shape[0] <= self.leaf_size):
            mean = np.mean(data[:, -1])
            return np.array([[-404, mean, -1, -1]])

        else:
            rand_feature_idx = randint(0, data.shape[1] - 2)
            split_val = np.median(data[:, rand_feature_idx])

            maximum = max(data[:, rand_feature_idx])
            if (maximum == split_val):
                return np.array([[-404, np.mean(data[:, -1]), -1, -1]])

            left = self.build_tree(data[data[:, rand_feature_idx] <= split_val])
            right = self.build_tree(data[data[:, rand_feature_idx] > split_val])

            root = np.array([[rand_feature_idx, split_val, 1, left.shape[0] + 1]])
            sub_left = np.append(root, left, axis=0)
            return np.append(sub_left, right, axis=0)


    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        num_rows = points.shape[0]
        out = np.empty(num_rows)
        for r in range(num_rows):
            out[r] = float(self.query_tuple(points[r, :]))
        return out


    def query_tuple(self, tree_tuple):
        row_idx = 0
        while (self.tree[row_idx, 0] != -404):
            f = float(self.tree[row_idx, 0])
            split_val = float(self.tree[row_idx, 1])

            if tree_tuple[int(f)] <= split_val:
                row_idx += int(float(self.tree[row_idx, 2]))
            else:
                row_idx += int(float(self.tree[row_idx, 3]))
        return self.tree[row_idx, 1]


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
