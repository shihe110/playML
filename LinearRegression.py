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