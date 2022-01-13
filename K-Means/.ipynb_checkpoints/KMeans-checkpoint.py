import numpy as np
from math import sqrt
from collections import Counter
from sklearn.metrics import accuracy_score
import random

class KMeans:
    def __init__(self, n_clusters=3, random_state=0):
        assert n_clusters >=1, " must be valid"
        self._k = n_clusters
        self._random_state = random_state
        self._X = None
        self._center = None
        self.cluster_centers_ = None
        
    def distance(self, M, N):
        return (np.sum((M - N) ** 2, axis = 1))** 0.5
        
    def fit_predict(self, X):
        k = self._k
        
        if self._random_state:
            random.seed(self._random_state)
        init_k_index = [random.randint(0, X.shape[0]) for i in np.arange(k)]
        print(init_k_index)
        center = np.array(X[init_k_index])
        nearest = np.array([(np.min(self.distance(center, item)) == self.distance(center, item)).tolist() for item in X])
        print(np.array([np.argmin(self.distance(center, item))) for item in X]))
        last_center = center
        n_iters = 10
        while n_iters > 0:
            dots_index = [[] for i in np.arange(k)]
            dots = [[] for i in np.arange(k)]
#             print(n_iters)
#             print(center)
            last_center = center
            for m in np.arange(k):
                for i, j in enumerate(nearest[:, m]):
                    if j:
                        dots_index[m].append(i)
                        if dots[m] == []:
                            dots[m] = X[i]
                        else:
                            dots[m] = np.vstack((dots[m], X[i]))


            for m in np.arange(k):
                center[m] = np.average(np.array(dots[m]), axis=0)
            self.cluster_centers_ = center
            if n_iters < 0:
                print('not enough n_iters')
            n_iters = n_iters - 1
        return center, dots