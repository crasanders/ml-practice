import torch


class MLPClassifier:
    def __init__(self, n_features: int, n_classes: int, n_hidden_units: int):
        self.input_weights = torch.rand(
            (n_features + 1, n_hidden_units), requires_grad=True
        )
        self.output_weights = torch.rand(
            (n_hidden_units + 1, n_classes), requires_grad=True
        )

    def _add_ones_column(self, X: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(*X.shape[:-1], 1, dtype=X.dtype, device=X.device)
        out = torch.cat([X, ones], dim=-1)
        return out

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_biased = self._add_ones_column(X)
        hidden = torch.relu(X_biased @ self.input_weights)
        hidden_biased = self._add_ones_column(hidden)
        self.output = hidden_biased @ self.output_weights
        return self.output

    def backward(self, loss: torch.Tensor, lr: float):
        loss.backward()
        with torch.no_grad():
            self.input_weights = self.input_weights - lr * self.input_weights.grad
            self.output_weights = self.output_weights - lr * self.output_weights.grad
        self.input_weights.requires_grad = True
        self.output_weights.requires_grad = True

    def predict(self, X: torch.Tensor):
        logits = self.forward(X)
        return logits.argmax(-1)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lr: float = 0.01,
        iterations: int = 1000,
        verbose: bool = False,
    ):
        for i in range(iterations):
            logits = self.forward(X)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if verbose:
                print(f"Loss for iteration {i}: {loss}")
            self.backward(loss, lr)
