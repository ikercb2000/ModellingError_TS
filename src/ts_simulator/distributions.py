# Project modules

from src.ts_simulator.interfaces import IDistSimulator
from src.ts_simulator.distributions import *

# Other modules

import numpy as np
from scipy.special import erf

# Distribution Simulation Classes


class NormalDist(IDistSimulator):

    def __init__(self, params: dict):

        self.params = params

    def __str__(self, for_title: bool = False):

        return "Gaussian" if for_title else "Gaussian Distribution"

    def draw(self, size=1000):

        loc = self.params["loc"]
        scale = self.params["scale"]
        samples = np.random.normal(loc=loc, scale=scale, size=size)

        return samples

    def cdf(self, x):

        loc = self.params["loc"]
        scale = self.params["scale"]
        return 0.5 * (1 + erf((x - loc) / (scale * np.sqrt(2))))

    def pdf(self, x):

        loc = self.params["loc"]
        scale = self.params["scale"]
        return (1 / (scale * np.sqrt(2 * np.pi))) * np.exp(-((x - loc) ** 2) / (2 * scale ** 2))

    def theory(self):

        loc = self.params["loc"]
        scale = self.params["scale"]

        theory = {
            "mean": loc,
            "std": scale,
            "median": loc,
        }

        return theory


class CauchyDist(IDistSimulator):

    def __init__(self, params: dict):

        self.params = params

    def __str__(self, for_title: bool = False):

        return "Cauchy" if for_title else "Cauchy Distribution"

    def draw(self, size=1000):

        loc = self.params["loc"]
        scale = self.params["scale"]
        u = np.random.uniform(0, 1, size)
        samples = loc + scale * np.tan(np.pi * (u - 0.5))

        return samples

    def cdf(self, x):

        loc = self.params["loc"]
        scale = self.params["scale"]
        return 0.5 + (1 / np.pi) * np.arctan((x - loc) / scale)

    def pdf(self, x):

        loc = self.params["loc"]
        scale = self.params["scale"]
        return 1 / (np.pi * scale * (1 + ((x - loc) / scale) ** 2))

    def theory(self):

        loc = self.params["loc"]

        theory = {
            "mean": 0,
            "std": 0,
            "median": 0,
        }

        return theory


class GumbelDist(IDistSimulator):

    def __init__(self, params: dict):

        self.params = params

    def __str__(self, for_title: bool = False):

        return "Gumbel" if for_title else "Gumbel Distribution"

    def draw(self, size=1000):

        loc = self.params["loc"]
        scale = self.params["scale"]
        u = np.random.uniform(0, 1, size)
        samples = loc - scale * np.log(-np.log(u))

        return samples

    def cdf(self, x):

        loc = self.params["loc"]
        scale = self.params["scale"]
        z = (x - loc) / scale
        return np.exp(-np.exp(-z))

    def pdf(self, x):

        loc = self.params["loc"]
        scale = self.params["scale"]
        z = (x - loc) / scale
        return (1 / scale) * np.exp(-(z + np.exp(-z)))

    def theory(self):

        loc = self.params["loc"]
        scale = self.params["scale"]
        mean = loc + scale * np.euler_gamma
        std = (scale * np.pi) / np.sqrt(6)
        median = loc - scale * np.log(np.log(2))

        theory = {
            "mean": mean,
            "std": std,
            "median": median,
        }

        return theory


class LogNormalDist(IDistSimulator):
    def __init__(self, params: dict):

        self.params = params

    def __str__(self, for_title: bool = False):

        return "LogNormal" if for_title else "LogNormal Distribution"

    def draw(self, size=1000):

        mean = self.params["loc"]
        sigma = self.params["scale"]
        samples = np.random.lognormal(mean=mean, sigma=sigma, size=size)

        return samples

    def cdf(self, x):

        mean = self.params["loc"]
        sigma = self.params["scale"]
        cdf_values = np.where(
            x > 0,
            0.5 + 0.5 * erf((np.log(x) - mean) / (sigma * np.sqrt(2))),
            0
        )

        return cdf_values

    def pdf(self, x):

        mean = self.params["loc"]
        sigma = self.params["scale"]

        pdf_values = np.where(
            x > 0,
            (1 / (x * sigma * np.sqrt(2 * np.pi))) *
            np.exp(-((np.log(x) - mean) ** 2) / (2 * sigma ** 2)),
            0
        )

        return pdf_values

    def theory(self):

        mean = self.params["loc"]
        sigma = self.params["scale"]

        theoretical_mean = np.exp(mean + (sigma**2) / 2)
        std = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mean + sigma**2))
        median = np.exp(mean)

        theory = {
            "mean": theoretical_mean,
            "std": std,
            "median": median
        }

        return theory


class ParetoDist(IDistSimulator):

    def __init__(self, params: dict):
        self.params = params

    def __str__(self, for_title: bool = False):
        return "Pareto" if for_title else "Pareto Distribution"

    def draw(self, size=1000):
        xm = self.params["xm"]
        alpha = self.params["alpha"]
        u = np.random.uniform(0, 1, size)
        samples = xm * (1 - u) ** (-1 / alpha)
        return samples

    def cdf(self, x):
        xm = self.params["xm"]
        alpha = self.params["alpha"]
        cdf_values = np.where(x >= xm, 1 - (xm / x) ** alpha, 0)
        return cdf_values

    def pdf(self, x):
        xm = self.params["xm"]
        alpha = self.params["alpha"]
        pdf_values = np.where(
            x >= xm, (alpha * xm**alpha) / (x**(alpha + 1)), 0)
        return pdf_values

    def theory(self):
        xm = self.params["xm"]
        alpha = self.params["alpha"]

        if alpha <= 1:
            mean = None
        else:
            mean = (xm * alpha) / (alpha - 1)

        if alpha <= 2:
            std = None
        else:
            std = xm * np.sqrt((alpha / ((alpha - 1) ** 2))
                               * (1 / (alpha - 2)))

        median = xm * 2**(1 / alpha)

        theory = {"mean": mean,
                  "std": std,
                  "median": median}

        return theory
