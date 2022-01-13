import numpy as np
from math import sqrt
from collections import Counter
from sklearn.metrics import accuracy_score
import random

class KMeans:
    def __init__(self, n_clusters=3, random_state=0):
        assert n_clusters >=1, " must be valid"
        self._n_clusters = n_clusters
        self._random_state = random_state
        self._X = None
        self._center = None
        self.cluster_centers_ = None

    def distance(self, M, N):
        return (np.sum((M - N) ** 2, axis = 1))** 0.5

    def _generate_labels(self, center, X):
        return np.array([np.argmin(self.distance(center, item)) for item in X])

    def _generate_centers(self, labels, X):
        return np.array([np.average(X[labels == i], axis=0) for i in np.arange(self._n_clusters)])

    def fit_predict(self, X):
        k = self._n_clusters

        # 设置随机数
        if self._random_state:
            random.seed(self._random_state)

        # 生成随机中心点的索引
        center_index = [random.randint(0, X.shape[0]) for i in np.arange(k)]

        center = X[center_index]

        # print('init center: ', center)

        n_iters = 1e3
        while n_iters > 0:


            # 记录上一个迭代的中心点坐标
            last_center = center

            # 根据上一批中心点，计算各个点所属的类
            labels = self._generate_labels(last_center, X)
            self.labels_ = labels

            # 新的中心点坐标
            center = self._generate_centers(labels, X)

            # print('n center: ', center)

            # 暴露给外头的参数
            # 中心点
            self.cluster_centers_ = center

            # 返回节点对应的分类 {0, 1, ..., n}


            # 如果新计算得到的中心点，和上一次计算得到的点相同，说明迭代已经稳定了。
            if (last_center == center).all():

                self.labels_ = self._generate_labels(center, X)
                break

            n_iters = n_iters - 1
        return self
