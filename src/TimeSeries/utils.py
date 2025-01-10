# Other modules

import numpy as np
import pandas as pd

# Deterministic term

def polinomForm(params: list, x: float):
    total = []
    for i in range(0,len(params)):
        result = params[i]*(x**(i))
        total.append(result)
    return np.sum(total)

def sinForm(params: dict, x: float):
    return params["A"]*np.sin(params["B"]*x+params["C"])

def cosForm(params: dict, x: float):
    return params["A"]*np.cos(params["B"]*x+params["C"])

# Distribution Simulation

def cauchy_dist(size=1000, loc=0, scale=1):
    u = np.random.uniform(0, 1, size)
    cauchy_samples = loc + scale * np.tan(np.pi * (u - 0.5))
    return cauchy_samples

def cauchy_pdf(x, loc, scale):
    return 1 / (np.pi * scale * (1 + ((x - loc) / scale) ** 2))

def gumbel_dist(size=1000, loc=0, scale=1):
    u = np.random.uniform(0, 1, size)
    gumbel_samples = loc - scale * np.log(-np.log(u))
    return gumbel_samples

def gumbel_pdf(x, loc, scale):
    z = (x - loc) / scale
    return (1 / scale) * np.exp(-(z + np.exp(-z)))

def lognormal_pdf(x, mu, sigma):
    return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu)**2) / (2 * sigma**2))

def pareto_dist(size, xm, alpha):
    u = np.random.uniform(0, 1, size)
    return xm * (1 - u) ** (-1 / alpha)

def pareto_pdf(x, xm, alpha):
    return np.where(x >= xm, (alpha * xm**alpha) / (x**(alpha + 1)), 0)
