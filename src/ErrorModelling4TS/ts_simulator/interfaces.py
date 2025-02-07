# Other Modules

from abc import ABC, abstractmethod

# Interface Time Series Simulator


class ITSSimulator(ABC):

    @abstractmethod
    def simulate_noise(self, **kwargs):
        pass

    @abstractmethod
    def get_ts(self, **kwargs):
        pass

# Distribution Simulator Interface


class IDistSimulator(ABC):

    @abstractmethod
    def __init__(self, params):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def draw(self, **kwargs):
        pass

    @abstractmethod
    def cdf(self, **kwargs):
        pass

    @abstractmethod
    def pdf(self, *kwargs):
        pass

    @abstractmethod
    def theory(self, *kwargs):
        pass
