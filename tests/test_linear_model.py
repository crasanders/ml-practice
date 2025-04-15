import unittest

import torch
from sklearn.datasets import make_regression

from src.linear_model import LinearRegression

device = "cuda" if torch.cuda.is_available() else "cpu"


class TestLinearRegression(unittest.TestCase):
    def test_ols(self):
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
