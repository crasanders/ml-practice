import unittest

import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from src.mlp import MLPClassifier


class TestMLPClassifier(unittest.TestCase):
    def test_classification(self):
        X, y = make_moons(noise=0.1, random_state=0)
        X = torch.tensor(X).float()
        y = torch.tensor(y).long()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        m = MLPClassifier(2, 2, 10)
        m.fit(X_train, y_train, lr=0.01, verbose=True)
        preds = m.predict(X_test)
        error = torch.abs(preds - y_test).sum() / len(preds)
        self.assertLess(error, 0.10)
