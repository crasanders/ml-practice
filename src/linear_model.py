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

    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return (
            torch.mean((y_true - y_pred) ** 2)
            + self.l1_reg * self.beta.abs().sum()
            + self.l2_reg * (self.beta**2).sum()
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
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        iterations: int,
        step_size: float,
        verbose: bool,
    ):
        self.bias = y.mean()
        y -= self.bias
        self.beta = torch.rand(X.shape[-1], device=X.device, requires_grad=True)
        for i in range(iterations):
            pred = X @ self.beta
            loss = self.loss(y, pred)

            if verbose:
                print(f"Loss at iteration {i}: {loss}")

            loss.backward()
            with torch.no_grad():
                self.beta = self.beta - step_size * self.beta.grad
            self.beta.requires_grad = True

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        iterations: int = 1000,
        step_size: float = 0.01,
        verbose: bool = False,
    ):
        if self.l1_reg == 0:
            if verbose:
                print("Fitting analytically.")
            self._fit_analytic(X, y)

        else:
            self._fit_gradient_descent(X, y, iterations, step_size, verbose)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.beta is None:
            raise RuntimeError("Cannot predict before the model is trained!")
        return X @ self.beta + self.bias


class LogisticRegression(torch.nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        l1_reg: float = 0,
        l2_reg: float = 0,
    ):
        super().__init__()
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.beta = torch.nn.Linear(n_features, n_classes)

    def loss(self, pred, target) -> torch.Tensor:
        return (
            torch.nn.functional.cross_entropy(pred, target)
            + self.l1_reg * self.beta.weight.data.abs().sum()
            + self.l2_reg * self.beta.weight.data.pow(2).sum()
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.beta(X)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lr: float = 0.001,
        iterations: int = 1000,
        verbose: bool = False,
    ):
        self.train()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        for i in range(iterations):
            pred = self(X)
            loss = self.loss(pred, y)

            if verbose:
                print(f"Loss at iteration {i}: {loss}")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self(X)
        return torch.argmax(logits, dim=-1)
