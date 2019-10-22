import numpy as np

class PCA:

    def __init__(self, n_components):
        """初始化PCA"""
        assert n_components >= 1, "n_components must be valid"
        self.n_components = n_components
        self.components_ = None

    def fit(self, X, eta=0.01, n_iters=1e4):
        """获取数据集X的前n个主成分"""
        assert self.n_components <= X.shape[1], \
            "n_components must be greater than the feature number of X"

        def demean(X):
            return X - np.mean(X, axis=0)

        def f(w, X):
            return np.sum((X.dot(w) ** 2)) / len(X)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_component(X, initial_w, eta=0.01, epsilon=1e-8):

            w = direction(initial_w)
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break

                cur_iter += 1

            return w

        def transform(self, X):
            """将给定的X，映射到各主成分分量重"""
            assert X.shape[1] == self.components_.shape[1]

            return X.dot(self.components_.T)

        def inverse_transform(self, X):
            """"""
            assert X.shape[1] == self.components_.shape[0]

            return X.dot(self.components_)

        def __repr__(self):
            return "PCA(n_components=%d)" % self.n_components