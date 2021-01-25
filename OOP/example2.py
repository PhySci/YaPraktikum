from sklearn.datasets import load_iris
from sklearn.base import TransformerMixin
import numpy as np
from matplotlib import pyplot as plt


class RemoveOutliers(TransformerMixin):

    def __init__(self, lb=5, ub=95):
        self._lower_bound = lb
        self._upper_bound = ub

    def fit(self, X):
        self._min_value = np.percentile(X, self._lower_bound)
        self._max_value = np.percentile(X, self._upper_bound)
        return self

    def transform(self, X):
        X2 = X
        X2[X2 < self._min_value] = self._min_value
        X2[X2 > self._max_value] = self._max_value
        return X2


if __name__ == '__main__':
    iris = load_iris()
    X = iris.get('data')
    y = iris.get('target')
    feature1 = X[:, 0]
    feature2 = X[:, 1]
    plt.scatter(feature1, feature2)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    ro = RemoveOutliers(40, 60)
    feature1_transformed = ro.fit_transform(feature1)
    plt.scatter(feature1_transformed, feature2)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
