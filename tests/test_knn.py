import unittest

import torch
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split

from src.knn import KNN, KMeans


class TestKNN(unittest.TestCase):
    def test_classification(self):
        X, y = make_moons(noise=0.1, random_state=0)
        X = torch.tensor(X)
        y = torch.tensor(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        m = KNN(10, 2)
        m.train(X_train, y_train)
        preds = m.predict(X_test)
        error = torch.abs(preds - y_test).sum() / len(preds)
        self.assertLess(error, 0.10)


class TestKMeans(unittest.TestCase):
    def test_clustering(self):
        X, y = make_blobs(n_features=2, centers=2, random_state=0)
        X = torch.tensor(X)
        y = torch.tensor(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        m = KMeans(2)
        m.train(X_train)
        preds = m.predict(X_test)
        error = torch.abs(preds - y_test).sum() / len(preds)
        self.assertTrue(error < 0.1 or error > 0.9)  # order of labels is arbitrary
