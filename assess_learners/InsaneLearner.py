import numpy as np
import BagLearner as bg
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learner_list = []
        for i in range(20):
            self.learner_list.append(bg.BagLearner(lrl.LinRegLearner, kwargs={}, bags=20))
    def author(self):
        return 'jpb6'
    def add_evidence(self, data_x, data_y):
        for l in self.learner_list:
            l.add_evidence(data_x, data_y)
    def query(self, points):
        temp = []
        for l in self.learner_list:
            temp.append(l.query(points))
        return np.mean(temp, axis=0)
