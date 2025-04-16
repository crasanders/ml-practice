import torch


class CategoricalNB:
    def __init__(self):
        self.train_X = None
        self.train_y = None

    def fit(self, X: torch.LongTensor, y: torch.LongTensor):
        self.train_X = X
        self.train_y = y
        self.n_classes = int(y.max().item()) + 1

    def _predict(self, x: torch.LongTensor) -> int:
        probs = [0] * self.n_classes
        for y in range(self.n_classes):
            X_cond = self.train_X[self.train_y == y]
            p_y = torch.mean((self.train_y == y).float())
            probs[y] += torch.log(p_y)
            for i in range(x.shape[-1]):
                p_x = torch.mean((X_cond[:, i] == x[i]).float())
                probs[y] += torch.log(p_x)
        return torch.argmax(torch.tensor(probs)).item()

    def predict(self, X: torch.LongTensor) -> torch.LongTensor:
        return torch.tensor([self._predict(x) for x in X])
