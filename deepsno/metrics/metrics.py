from abc import ABC, abstractmethod

from torch import nn


class Metric(ABC):
    @abstractmethod
    def update(self, predict: dict, truth: dict) -> None:
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class BatchedMetric(Metric):
    def __init__(self, metric_fn: nn.Module):
        self.metric_fn = metric_fn
        self.running_metric: float = 0
        self.n_points: int = 0

    def update(self, predict: dict, truth: dict) -> None:
        batch_size = len(next(iter(predict)))
        self.n_points += batch_size

        self.running_metric += batch_size * self.metric_fn(predict, truth).item()

    def compute(self) -> float:
        return self.running_metric / self.n_points

    def reset(self) -> None:
        self.running_metric = 0
        self.n_points = 0
