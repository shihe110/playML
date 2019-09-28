import numpy as np
from .metrics import r2_score


class LinearRegression:

    def __init__(self):
        """初始化 Linear Regression 模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None


    def fit(self, X_train, y_train):
        """根据训练数据集x_train, y_train 训练 linear regression 模型"""
        assert X_train.ndim == 1, \
            "Simple Linear Regression can only slove single featrue training data"
        assert len(X_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train,y_train,使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta))**2) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            res = np.empty(len(theta))
            res[0] = np.sum(X_b.dot(theta) - y)
            for i in range(1, len(theta)):
                res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            return res * 2 / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones(len(X_train.shape[0], 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self




    def predict(self, X_predict):
        """给定带预测数据集x_predict, 返回表示x_predict的结果向量"""
        assert X_predict.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)


    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)


    def __repr__(self):
        return "LiearRegression()"