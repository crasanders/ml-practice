import random
from collections.abc import Callable

import torch


def euclidean_dist(a: torch.Tensor, b: torch.Tensor) -> torch.tensor:
    return torch.sqrt(((a - b) ** 2).sum(axis=-1))


class KNN:
    def __init__(
        self,
        k: int,
        num_classes: int,
        dist_fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = euclidean_dist,
    ):
        self.k = k
        self.num_classes = num_classes
        self.dist_fun = dist_fun
        self.train_X = None

    def train(self, X: torch.Tensor, y: torch.Tensor):
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of entries!")
        if len(X) < self.k:
            raise ValueError("Number of training samples is less than specified k!")
        self.train_X = X
        self.train_y = y

    def _predict(self, x: torch.Tensor) -> int:
        dists = self.dist_fun(x, self.train_X)
        votes = [0 for _ in range(self.num_classes)]
        sort_indices = torch.argsort(dists)
        for i in sort_indices[: self.k]:
            vote = self.train_y[i]
            votes[vote] += 1
        return torch.argmax(torch.tensor(votes)).item()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.train_X is None:
            raise RuntimeError("Cannot predict before the model has been trained!")
        return torch.tensor([self._predict(x) for x in X])


class KMeans:
    def __init__(
        self,
        k: int,
        dist_fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = euclidean_dist,
    ):
        self.k = k
        self.dist_fun = dist_fun
        self.means = None

    def _get_closest_mean(self, x: torch.Tensor) -> int:
        smallest_dist = torch.inf
        index = 0
        for i, m in enumerate(self.means):
            dist = self.dist_fun(x, m)
            if dist < smallest_dist:
                smallest_dist = dist
                index = i
        return index

    def train(self, X: torch.Tensor, iterations: int = 100):
        seeds = [x for x in X]
        random.shuffle(seeds)
        self.means = torch.stack(seeds[: self.k])
        for i in range(iterations):
            new_means = torch.zeros(self.means.size())
            counts = [0] * self.k
            for x in X:
                index = self._get_closest_mean(x)
                new_means[index] += x
                counts[index] += 1
            for i in range(self.k):
                new_means[i] /= counts[i]
            self.means = new_means

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.means is None:
            raise RuntimeError("Cannot predict before the model has been trained!")
        ans = [self._get_closest_mean(x) for x in X]
        return torch.tensor(ans)
