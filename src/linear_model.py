import torch


class LinearRegression(torch.nn.Module):
    def __init__(
        self,
        l1_reg: float = 0,
        l2_reg: float = 0,
    ):
        super().__init__()
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.beta = None
        self.bias = None

    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        return (
            torch.mean((y_true - y_pred) ** 2)
            + self.l1_reg * self.beta.weight.abs().sum()
            + self.l2_reg * (self.beta.weight**2).sum()
        )

    def _fit_analytic(self, X: torch.Tensor, y: torch.Tensor):
        self.bias = y.mean()
        y -= self.bias
        self.beta = (
            torch.inverse(X.T @ X + (self.l2_reg * torch.eye(X.shape[-1]).to(X)))
            @ X.T
            @ y
        )

    def _fit_gradient_descent(
        self, X: torch.Tensor, y: torch.Tensor, iterations: int, verbose: bool
    ):
        pass

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        iterations: int = 1000,
        verbose: bool = False,
    ):
        if self.l1_reg == 0:
            if verbose:
                print("Fitting analytically.")
            self._fit_analytic(X, y)

        else:
            self._fit_gradient_descent(X, y, iterations, verbose)

    def predict(self, X: torch.Tensor):
        if self.beta is None:
            raise RuntimeError("Cannot predict before the model is trained!")
        return X @ self.beta + self.bias
