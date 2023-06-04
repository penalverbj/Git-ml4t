import numpy as np
import BagLearner as bg
import LinRegLearner as lrl


class InsaneLearner(object):

    def __init__(self, verbose=False):
        self.learner_list = []
        for i in range(20):
            self.learner_list.append(bg.BagLearner(lrl.LinRegLearner, kwargs={}, bags=20))

    def author(self):
        return 'jpb6'  # replace tb34 with your Georgia Tech username

    def addEvidence(self, data_x, data_y):
        for learner in self.learner_list:
            learner.add_evidence(data_x, data_y)

    def query(self, points):
        temp = np.empty(len(self.learner_list))
        i = 0
        for l in self.learner_list:
            temp[i] = l.query(points)
            i += 1
        out = np.mean(temp, axis=0)
        return out