# Other Modules

from abc import ABC, abstractmethod
from enum import Enum

# Distributions Enum

DetermTerm = Enum("DeterTerm", ["Sin","Polynom"])

# Interface Time Series Simulator

class ISeriesSimulator(ABC):

    @abstractmethod
    def __init__(self, data):
        pass

    @abstractmethod
    def simulate(self, **kwargs):
        pass

    @abstractmethod
    def plot_sim(self,*kwargs):
        pass