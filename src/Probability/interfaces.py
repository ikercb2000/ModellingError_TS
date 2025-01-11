# Other Modules

from abc import ABC, abstractmethod

# Interface Time Series Simulator

class IDistSimulator(ABC):

    @abstractmethod
    def __init__(self, params):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def cdf(self, **kwargs):
        pass

    @abstractmethod
    def pdf(self,*kwargs):
        pass

    @abstractmethod
    def theory(self,*kwargs):
        pass