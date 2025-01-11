# Project Modules

from TimeSeries.interfaces import *
from Probability.interfaces import *
from TimeSeries.utils import *

# Other Modules

import matplotlib.pyplot as plt
import pandas as pd

# Time Series Simulator Class

class TimeSeriesSimulator(ISeriesSimulator):

    def __init__(self, determ, noise: IDistSimulator = None):
        
        self.determ = determ
        self.noise = noise
    
    def simulate(self, det_params: dict, n: int = 100):

        times = range(0, n)
        det_values = self.determ(det_params, times)
        if self.noise != None:
            noise_values = self.noise.cdf(size=n)
        else:
            noise_values = [0]*n
        series = det_values + noise_values
        sim_dict ={"Value": series, "Determ": det_values, "Noise": noise_values}
        self.series = pd.DataFrame(sim_dict)

        return self.series

    def plot_sim(self,scatter_plot: bool = False):

        plt.figure(figsize=(10, 6))
        if scatter_plot:
            plt.scatter(self.series.index, self.series['Value'], label='Time Series')
        else:
            plt.plot(self.series.index, self.series['Value'], label='Time Series')
        plt.plot(self.series.index, self.series['Determ'], label="Real Values", color="red", linestyle= "--")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.ylim([-5,5])
        plt.title('Simulated Time Series')
        plt.legend()
        plt.show()