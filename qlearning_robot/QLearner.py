""""""
"""  		  	   		  		 			  		 			 	 	 		 		 	
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
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

import random as rand

import numpy as np


class QLearner(object):
    """  		  	   		  		 			  		 			 	 	 		 		 	
    This is a Q learner object.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
    :param num_states: The number of states to consider.  		  	   		  		 			  		 			 	 	 		 		 	
    :type num_states: int  		  	   		  		 			  		 			 	 	 		 		 	
    :param num_actions: The number of actions available..  		  	   		  		 			  		 			 	 	 		 		 	
    :type num_actions: int  		  	   		  		 			  		 			 	 	 		 		 	
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		  		 			  		 			 	 	 		 		 	
    :type alpha: float  		  	   		  		 			  		 			 	 	 		 		 	
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		  		 			  		 			 	 	 		 		 	
    :type gamma: float  		  	   		  		 			  		 			 	 	 		 		 	
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		  		 			  		 			 	 	 		 		 	
    :type rar: float  		  	   		  		 			  		 			 	 	 		 		 	
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		  		 			  		 			 	 	 		 		 	
    :type radr: float  		  	   		  		 			  		 			 	 	 		 		 	
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		  		 			  		 			 	 	 		 		 	
    :type dyna: int  		  	   		  		 			  		 			 	 	 		 		 	
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		  		 			  		 			 	 	 		 		 	
    """

    def __init__(
            self,
            num_states=100,
            num_actions=4,
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=0,
            verbose=False,
    ):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Constructor method  		  	   		  		 			  		 			 	 	 		 		 	
        """
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.num_states = num_states
        self.Q_table = np.zeros(shape=(num_states, num_actions))
        if dyna != 0:
            self.Tc = np.full(shape=(num_states, num_actions, num_states), fill_value=0.0001)
            self.T = self.Tc / np.sum(self.Tc, axis=2, keepdims=True)
            self.R = np.full(self.Q_table.shape, fill_value=-1)

    def querysetstate(self, s):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Update the state without updating the Q-table  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
        :param s: The new state  		  	   		  		 			  		 			 	 	 		 		 	
        :type s: int  		  	   		  		 			  		 			 	 	 		 		 	
        :return: The selected action  		  	   		  		 			  		 			 	 	 		 		 	
        :rtype: int  		  	   		  		 			  		 			 	 	 		 		 	
        """
        self.s = s
        action = rand.randint(0, self.num_actions)
        if self.verbose:
            print(f"s = {s}, a = {action}")
        return action

    def bellman(self, s, a, s_prime, r):
        return (1 - self.alpha) * self.Q_table[s, a] + self.alpha * (r + self.gamma * self.Q_table[s_prime, np.argmax(self.Q_table[s_prime])])
        # return np.add(np.multiply((1 - self.alpha), self.Q_table[s, a]), np.multiply(self.alpha, np.add(r, np.multiply(self.gamma, self.Q_table[s_prime, np.argmax(self.Q_table[s_prime])]))))

    def query(self, s_prime, r):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Update the Q table and return an action  		  	   		  		 			  		 			 	 	 		 		 	

        :param s_prime: The new state  		  	   		  		 			  		 			 	 	 		 		 	
        :type s_prime: int  		  	   		  		 			  		 			 	 	 		 		 	
        :param r: The immediate reward
        :type r: float
        :return: The selected action  		  	   		  		 			  		 			 	 	 		 		 	
        :rtype: int  		  	   		  		 			  		 			 	 	 		 		 	
        """
        sq = self.s
        aq = self.a
        num_actions = self.num_actions
        self.Q_table[sq][aq] = self.bellman(sq, aq, s_prime, r)

        if np.random.random() > self.rar:
            action = np.random.randint(0, num_actions)
        else:
            action = np.argmax(self.Q_table[s_prime])

        self.rar *= self.radr

        # if self.dyna != 0:
        #     self.Tc[sq][aq][s_prime] += 1
        #     self.T = self.Tc / np.sum(self.Tc, axis=2, keepdims=True)
        #     self.R[sq][aq] = (1 - self.alpha) * self.R[sq][aq] + (self.alpha * r)
        #
        #     for i in range(self.dyna):
        #         s = np.random.randint(0, self.num_states)
        #         a = np.random.randint(0, self.num_actions)
        #         sd_prime = np.random.multinomial(1, self.T[s][a]).argmax()
        #         self.Q_table[s][a] = self.bellman(s, a, sd_prime, self.R[s][a])

        self.s = s_prime
        self.a = action
        return action

    def author(self):
        return 'jpb6'  # replace tb34 with your Georgia Tech username.


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
