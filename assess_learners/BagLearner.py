import numpy as np


class BagLearner(object):

    def __init__(self, learner, kwargs=None, bags=20, boost=False, verbose=False):
        if kwargs is None:
            kwargs = {"leaf_size": 1}
        self.learner = learner
        self.learner_list = []
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        for i in range(bags):
            self.learner_list.append(learner(**kwargs))

    def author(self):
        return 'jpb6'

    def add_evidence(self, data_x, data_y):
        for l in self.learner_list:
            idxs = np.random.choice(range(data_x.shape[0]), data_x.shape[0], replace=True)
            x = data_x[idxs]
            y = data_y[idxs]
            l.add_evidence(x, y)

    def query(self, points):
        temp = []
        for l in self.learner_list:
            temp.append(l.query(points))
        return np.mean(temp, axis=0)

if __name__ == '__main__':
    print('BagLearner Main')
