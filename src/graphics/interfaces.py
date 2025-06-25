# Other Modules

from abc import ABC, abstractmethod

# Interface Graphics


class ISimulationPlots(ABC):

    @abstractmethod
    def plot_sim_ts(self, **kwargs):
        pass

    @abstractmethod
    def plot_predictions(self, **kwargs):
        pass

    @abstractmethod
    def plot_dist_error(self, **kwargs):
        pass
