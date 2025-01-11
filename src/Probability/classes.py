# Project modules

from Probability.interfaces import *

# Other modules

import numpy as np

# Distribution Simulation Classes

class NormalDist(IDistSimulator):

    def __init__(self,params: dict):

        self.params = params

    def __str__(self, for_title: bool = False):

        if for_title:
            return "Gaussian"
        else:
            return "Gaussian Distribution"

    def cdf(self,size=1000):
        loc = self.params["loc"]
        scale = self.params["scale"]
        return np.random.normal(loc=loc, scale=scale, size=size)

    def pdf(self, x):
        loc = self.params["loc"]
        scale = self.params["scale"]
        return (1 / (scale * np.sqrt(2 * np.pi))) * np.exp(-((x - loc) ** 2) / (2 * scale ** 2))
    
    def theory(self):

        loc = self.params["loc"]
        scale = self.params["scale"]

        theory = {"mean": loc,
                  "std": scale,
                  "median": loc,
                  }
        
        return theory
    

class CauchyDist(IDistSimulator):

    def __init__(self,params: dict):

        self.params = params

    def __str__(self, for_title: bool = False):

        if for_title:
            return "Cauchy"
        else:
            return "Cauchy Distribution"

    def cdf(self,size=1000):
        loc = self.params["loc"]
        scale = self.params["scale"]
        u = np.random.uniform(0, 1, size)
        return loc + scale * np.tan(np.pi * (u - 0.5))

    def pdf(self, x):
        loc = self.params["loc"]
        scale = self.params["scale"]
        return 1 / (np.pi * scale * (1 + ((x - loc) / scale) ** 2))
    
    def theory(self):

        theory = {"mean": "None",
                  "std": "None",
                  "median": "None",
                  }
        
        return theory
    
class GumbelDist(IDistSimulator):

    def __init__(self,params: dict):

        self.params = params

    def __str__(self, for_title: bool = False):

        if for_title:
            return "Gumbel"
        else:
            return "Gumbel Distribution"

    def cdf(self,size=1000):
        loc = self.params["loc"]
        scale = self.params["scale"]
        u = np.random.uniform(0, 1, size)
        return loc - scale * np.log(-np.log(u))

    def pdf(self, x):
        loc = self.params["loc"]
        scale = self.params["scale"]
        z = (x - loc) / scale
        return (1 / scale) * np.exp(-(z + np.exp(-z)))
    
    def theory(self):

        loc = self.params["loc"]
        scale = self.params["scale"]

        theory = {
            "mean": loc + scale*np.euler_gamma,
            "std": (scale*np.pi)/np.sqrt(6),
            "median": loc - scale*np.log(np.log(2)),
        }
        
        return theory
    
class LogNormalDist(IDistSimulator):

    def __init__(self,params: dict):

        self.params = params

    def __str__(self, for_title: bool = False):

        if for_title:
            return "LogNormal"
        else:
            return "LogNormal Distribution"

    def cdf(self,size=1000):
        mean = self.params["mean"]
        sigma = self.params["sigma"]
        return np.random.lognormal(mean=mean, sigma=sigma, size=size)

    def pdf(self, x):
        mean = self.params["mean"]
        sigma = self.params["sigma"]
        return np.where(
            x > 0,
            (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mean) ** 2) / (2 * sigma ** 2)),
            0
        )
    
    def theory(self):

        mean = self.params["mean"]
        sigma = self.params["sigma"]

        theory = {
            "mean": np.exp(mean+(sigma**2)/2),
            "std": np.sqrt((np.exp(sigma**2)-1)*np.exp(2*mean+sigma**2)),
            "median": np.exp(mean),
        }
        
        return theory

class ParetoDist(IDistSimulator):

    def __init__(self,params: dict):

        self.params = params

    def __str__(self, for_title: bool = False):

        if for_title:
            return "Pareto"
        else:
            return "Pareto Distribution"

    def cdf(self,size=1000):
        xm = self.params["xm"]
        alpha = self.params["alpha"]
        u = np.random.uniform(0, 1, size)
        return xm * (1 - u) ** (-1 / alpha)

    def pdf(self, x):
        xm = self.params["xm"]
        alpha = self.params["alpha"]
        return np.where(x >= xm, (alpha * xm**alpha) / (x**(alpha + 1)), 0)
    
    def theory(self):

        xm = self.params["xm"]
        alpha = self.params["alpha"]

        theory = {
            "mean": (xm*alpha)/(alpha-1),
            "std": xm*(2**(1/alpha)),
            "median": xm*np.sqrt(alpha/((alpha-1)*(alpha-2))),
        }

        return theory