# Other Modules

from abc import ABC, abstractmethod
from enum import Enum

# Distributions Enum

Distrib = Enum("Distrib",["Nothing","Normal","Cauchy","Gumbel","Lognormal","Pareto"])

# Loss Enum

Loss = Enum("Loss",["L1","L2","Huber"])

# Interface Plots

class IPlot(ABC):

    @abstractmethod
    def __init__(self, data):
        pass

    @abstractmethod
    def get_plot(self, **kwargs):
        pass