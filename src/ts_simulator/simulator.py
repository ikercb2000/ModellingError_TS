# Project Modules

from src.ErrorModelling4TS.ts_simulator.interfaces import ITSSimulator
from src.ErrorModelling4TS.ts_simulator.distributions import IDistSimulator

# Other Modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Time Series Simulator Class


class TimeSeriesSimulator(ITSSimulator):

    def __str__(self):

        return "Time Series Simulator"

    def simulate_noise(self, n: int, dist: IDistSimulator, seed: int):

        np.random.seed(seed)

        return np.array(dist.draw(size=n))

    def get_ts(self, determ_series: pd.Series, noise_series: pd.Series, constant_determ: float = 0, constant_noise: float = 0):

        determ_series = pd.DataFrame(
            np.array(determ_series) + constant_determ, columns=["Determ"], index=determ_series.index)
        noise_series = pd.DataFrame(
            np.array(noise_series) + constant_noise, columns=["Noise"], index=noise_series.index)
        series_values = np.array(determ_series) + np.array(noise_series)
        ts = pd.DataFrame(series_values, index=determ_series.index,
                          columns=["Value"])
        ts_df = pd.concat([determ_series, noise_series, ts], axis=1)

        return ts_df
