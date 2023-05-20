""""""
import matplotlib.pyplot as plt

"""Assess a betting strategy.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
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
  		  	   		  		 			  		 			 	 	 		 		 	
import numpy as np  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
def author():  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    :return: The GT username of the student  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: str  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    return "jpb6"  # replace tb34 with your Georgia Tech username.
  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
def gtid():  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    :return: The GT ID of the student  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: int  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    return 903376324  # replace with your GT ID number
  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
def get_spin_result(win_prob):  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
    :param win_prob: The probability of winning  		  	   		  		 			  		 			 	 	 		 		 	
    :type win_prob: float  		  	   		  		 			  		 			 	 	 		 		 	
    :return: The result of the spin.  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: bool  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    result = False  		  	   		  		 			  		 			 	 	 		 		 	
    if np.random.random() <= win_prob:  		  	   		  		 			  		 			 	 	 		 		 	
        result = True  		  	   		  		 			  		 			 	 	 		 		 	
    return result

def betting_episode():
    winnings_array = np.zeros(1000)
    winnings = 0
    i = 1
    # 18 total black squares
    # 38 total squares
    # prob = 18 / 38
    win_prob = 18 / 38.0
    while (winnings < 80 or i <= 1000):
        won = False
        bet_amount = 1
        while not won:
            won = get_spin_result(win_prob)
            if won == True:
                winnings = winnings + bet_amount
            else:
                winnings = winnings - bet_amount
                bet_amount = bet_amount * 2

            winnings_array[i] = winnings
            i = i + 1
            if (i > 1000):
                return (winnings, winnings_array)
            if (winnings >= 80):
                winnings_array[i:winnings_array.shape[0]] = winnings
                return (winnings, winnings_array)

    return (winnings, winnings_array)

def betting_episode_2():
    winnings_array = np.zeros(1000)
    winnings = 0
    i = 1
    # 18 total black squares
    # 38 total squares
    # prob = 18 / 38
    win_prob = 18 / 38.0
    while (winnings < 80 or i <= 1000):
        won = False
        bet_amount = 1
        while not won:
            won = get_spin_result(win_prob)
            if won == True:
                winnings = winnings + bet_amount
            else:
                winnings = winnings - bet_amount
                bet_amount = bet_amount * 2
                if (winnings - bet_amount < -256):
                    bet_amount = winnings + 256

            winnings_array[i] = winnings
            i = i + 1
            if (i > 1000):
                return (winnings, winnings_array)
            if (winnings >= 80):
                winnings_array[i:winnings_array.shape[0]] = winnings
                return (winnings, winnings_array)
            if (winnings <= -256):
                winnings_array[i:winnings_array.shape[0]] = winnings
                return (winnings, winnings_array)

    return (winnings, winnings_array)

def experiment_1_1():
    data = np.zeros((10, 1000))
    for i in range(10):
        data[i] = betting_episode()[1]
    for d in data:
        plt.plot(d, label="Test")
    plt.title("Figure 1: Results over 10 episodes with infinite bankroll")
    plt.xlabel("Episode")
    plt.ylabel("Winning")
    plt.xlim(0,300)
    plt.ylim(-256, 100)
    plt.legend(loc="best")
    plt.gca().legend(('Episode 0', 'Episode 1', 'Episode 2', 'Episode 3', 'Episode 4', 'Episode 5', 'Episode 6', 'Episode 7',
                      'Episode 8', 'Episode 9'))
    plt.savefig("figure1.png")
    plt.clf()

def experiment_1_2and3():
    data = np.zeros((1000, 1000))
    for i in range(1000):
        data[i] = betting_episode()[1]
    data = np.array(data)
    means = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    bolinger1 = means - std
    bolinger2 = means + std
    plt.plot(means, label="Mean")
    plt.plot(bolinger1, label="STD Add")
    plt.plot(bolinger2, label="STD Sub")
    plt.title("Figure 2: Mean winnings over 1000 episodes with infinite bankroll")
    plt.xlabel("Episode")
    plt.ylabel("Mean Winning")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend(loc="best")
    plt.savefig("figure2.png")

    plt.clf()

    median = np.median(data, axis=0)
    std = np.std(data, axis=0)
    bolinger1 = median - std
    bolinger2 = median + std
    plt.plot(median, label="Median")
    plt.plot(bolinger1, label="STD Add")
    plt.plot(bolinger2, label="STD Sub")
    plt.title("Figure 3: Median winnings over 1000 episodes with infinite bankroll")
    plt.xlabel("Episode")
    plt.ylabel("Median Winning")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend(loc="best")
    plt.savefig("figure3.png")

    plt.clf()

def experiment_1_4and5():
    data = np.zeros((1000, 1000))
    for i in range(1000):
        data[i] = betting_episode_2()[1]
    data = np.array(data)
    means = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    bolinger1 = means - std
    bolinger2 = means + std
    plt.plot(means, label="Mean")
    plt.plot(bolinger1, label="STD Add")
    plt.plot(bolinger2, label="STD Sub")
    plt.title("Figure 4: Mean winnings over 1000 episodes with finite bankroll")
    plt.xlabel("Episode")
    plt.ylabel("Mean Winning")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend(loc="best")
    plt.savefig("figure4.png")

    plt.clf()

    median = np.median(data, axis=0)
    std = np.std(data, axis=0)
    bolinger1 = median - std
    bolinger2 = median + std
    plt.plot(median, label="Median")
    plt.plot(bolinger1, label="STD Add")
    plt.plot(bolinger2, label="STD Sub")
    plt.title("Figure 5: Median winnings over 1000 episodes with finite bankroll")
    plt.xlabel("Episode")
    plt.ylabel("Median Winning")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend(loc="best")
    plt.savefig("figure5.png")

    plt.clf()

def test_code():
    """  		  	   		  		 			  		 			 	 	 		 		 	
    Method to test your code  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    win_prob = 0.60  # set appropriately to the probability of a win  		  	   		  		 			  		 			 	 	 		 		 	
    np.random.seed(gtid())  # do this only once
    print("\n")
    print(get_spin_result(.6))  # test the roulette spin
    # add your code here to implement the experiments
    #code was added above as individual methods
    experiment_1_1()
    experiment_1_2and3()
    experiment_1_4and5()

  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		  		 			  		 			 	 	 		 		 	
    test_code()  		  	   		  		 			  		 			 	 	 		 		 	
