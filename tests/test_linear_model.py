import unittest

import torch
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from src.linear_model import LinearRegression, LogisticRegression

device = "cuda" if torch.cuda.is_available() else "cpu"


class TestLinearRegression(unittest.TestCase):
    def test_analytic(self):
        X, y, coef = make_regression(
            n_features=2, n_informative=2, coef=True, random_state=0
        )
        X, y, coef = (
            torch.tensor(X, dtype=torch.float32).to(device),
            torch.tensor(y, dtype=torch.float32).to(device),
            torch.tensor(coef, dtype=torch.float32).to(device),
        )

        m = LinearRegression(l2_reg=1).to(device)
        m.fit(X, y, verbose=True)

        self.assertTrue(torch.allclose(coef, m.beta, rtol=0.1))

    def test_gradient_decent(self):
        X, y, coef = make_regression(
            n_features=2, n_informative=2, coef=True, random_state=0
        )
        X, y, coef = (
            torch.tensor(X, dtype=torch.float32).to(device),
            torch.tensor(y, dtype=torch.float32).to(device),
            torch.tensor(coef, dtype=torch.float32).to(device),
        )

        m = LinearRegression(l1_reg=1).to(device)
        m.fit(X, y, verbose=True)

        self.assertTrue(torch.allclose(coef, m.beta, rtol=0.1))


class TestLogisticRegression(unittest.TestCase):
    def test_classification(self):
        X, y = make_classification(
            n_samples=500,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            random_state=0,
        )
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.LongTensor(y).to(device)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        m = LogisticRegression(2, 2).to(device)
        m.fit(X_train, y_train, verbose=True, lr=0.005, iterations=1000)
        preds = m.predict(X_test)
        error = torch.abs(preds - y_test).sum() / len(preds)
        self.assertLess(error, 0.2)
