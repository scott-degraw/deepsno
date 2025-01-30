from abc import ABC, abstractmethod


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
