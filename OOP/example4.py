from sklearn.datasets import load_iris
from example2 import RemoveOutliers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from joblib import load, dump
from pprint import pprint
import pandas as pd


class AbstractIrisClassifier(ClassifierMixin):

    def __init__(self):
        lp = 5
        hp = 95
        self._preprocess_feature1 = RemoveOutliers(lp, hp)
        self._preprocess_feature2 = RemoveOutliers(lp, hp)
        self._preprocess_feature3 = RemoveOutliers(lp, hp)
        self._preprocess_feature4 = RemoveOutliers(lp, hp)
        self._clf = None


    def fit(self, X, y):
        X2 = X
        X2[:, 0] = self._preprocess_feature1.fit_transform(X2[:, 0])
        X2[:, 1] = self._preprocess_feature2.fit_transform(X2[:, 1])
        X2[:, 2] = self._preprocess_feature3.fit_transform(X2[:, 2])
        X2[:, 3] = self._preprocess_feature4.fit_transform(X2[:, 3])

        self._clf.fit(X2, y)
        return self

    def predict(self, X):
        X2 = X
        X2[:, 0] = self._preprocess_feature1.transform(X2[:, 0])
        X2[:, 1] = self._preprocess_feature2.transform(X2[:, 1])
        X2[:, 2] = self._preprocess_feature3.transform(X2[:, 2])
        X2[:, 3] = self._preprocess_feature4.transform(X2[:, 3])
        return self._clf.predict(X2)

    def save(self):
        dump(self, 'iris_clf.jbl')

    def load(self):
        instance = load('iris_clf.jbl')
        for k, v in instance.__dict__.items():
            setattr(self, k, v)


class RFIrisClassifier(AbstractIrisClassifier):

    def __init__(self):
        super(RFIrisClassifier, self).__init__()
        self._clf = RandomForestClassifier()

    def save(self):
       dump(self, 'iris_clf_rf.jbl')


class KNNIrisClassifier(AbstractIrisClassifier):

    def __init__(self):
        super(KNNIrisClassifier, self).__init__()
        self._clf = KNeighborsClassifier()

    def save(self):
        dump(self, 'iris_clf_knn.jbl')


if __name__ == '__main__':
    iris = load_iris()
    X = iris.get('data')
    y = iris.get('target')

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = RFIrisClassifier()
    clf.fit(X_train, y_train)
    y_pr = clf.predict(X_test)
    report = classification_report(y_test, y_pr, output_dict=True)
    report = pd.DataFrame.from_dict(report).transpose()
    pprint(report)

    print('-'*80)
    clf2 = KNNIrisClassifier()
    clf2.fit(X_train, y_train)
    y_pr2 = clf2.predict(X_test)
    report2 = classification_report(y_test, y_pr2, output_dict=True)
    report2 = pd.DataFrame.from_dict(report2).transpose()
    pprint(report2)





