#coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score

class Perceptron:

    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        assert X.ndim == 2, "X must be 2 dimensional"
        assert y.ndim == 1, "y must be 1 dimensional"
        assert X.shape[0] == y.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        
        for _ in range(self.n_iter):
            assert X[self.predict(X)!=y].shape[0] == y[self.predict(X)!=y].shape[0], \
                "the size of X_train must be equal to the size of y_train"
            
            # M 表示错误分类的点
            M = X[self.predict(X)!=y]
            M_target = y[self.predict(X)!=y]

            if (len(M) > 0):
                
                # 每次迭代抽取一个错误点进行梯度下降 直到错误分类点的集合大小为0
                M_predict = np.array(self.predict(M))
            
                x_i = M[0]
                M_target_i = np.array([M_target[0]])
                M_predict_i = np.array([M_predict[0]])
                
                update = self.eta * (M_target_i - M_predict_i)
                self.w_[1:] += update * x_i
                self.w_[0] += update
            
        return self


    def _compute_value(self, X):
        assert X.ndim == 2, "the single input must be one dimenssional"
        
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        assert X.ndim == 2, "function predict: X must be 2 dimensional"
        return np.where(self._compute_value(X) >= 0.0, 1, -1)
    
    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)