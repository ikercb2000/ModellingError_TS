# Other Modules

from abc import ABC, abstractmethod
import numpy as np

# Distribution Simulator Interface


class IDistSimulator(ABC):

    @abstractmethod
    def __init__(self, params) -> None:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def draw(self, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def cdf(self, **kwargs) -> float:
        pass

    @abstractmethod
    def pdf(self, *kwargs) -> float:
        pass

    @abstractmethod
    def quantile(self, *kwargs) -> float:
        pass

    @abstractmethod
    def theory(self, *kwargs) -> dict:
        pass
