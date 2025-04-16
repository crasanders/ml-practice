import unittest

import torch
from sklearn.model_selection import train_test_split

from src.naive_bayes import CategoricalNB


class TestCategoricalNaiveBayes(unittest.TestCase):
    def test_classification(self):
        n_samples = 100
        n_features = 10

        X1 = torch.ones((n_samples, n_features)) * 0.25  # first class tends to have 0s
        X2 = torch.ones((n_samples, n_features)) * 0.75  # second class tends to have 1s

        y1 = torch.zeros((n_samples,))
        y2 = torch.ones((n_samples,))

        X = torch.bernoulli(torch.vstack((X1, X2)))
        y = torch.hstack((y1, y2))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        m = CategoricalNB()
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        error = torch.abs(preds - y_test).sum() / len(preds)
        self.assertTrue(error < 0.1)
