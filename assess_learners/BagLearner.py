import numpy as np


class BagLearner(object):

    def __init__(self, learner, kwargs=None, bags=20, boost=False, verbose=False):
        if kwargs is None:
            kwargs = {"leaf_size": 1}
        self.learner = learner
        self.learner_list = []
        self.bags = bags
        for i in range(0, bags):
            self.learner_list.append(learner(**kwargs))

    def author(self):
        return 'jpb6'

    def add_evidence(self, data_x, data_y):
        idxs = np.linspace(0, data_x.shape[0] - 1, data_x.shape[0]).astype(int)

        for l in self.learner_list:
            i = np.random.choice(idxs, idxs.size)
            x = np.take(data_x, i, axis=0)
            y = np.take(data_y, i, axis=0)
            l.add_evidence(x, y)

    def query(self, points):
        temp = np.empty(len(self.learner_list))
        i = 0
        for l in self.learner_list:
            temp[i] = l.query(points)
            i += 1
        out = np.mean(temp, axis=0)
        return out
