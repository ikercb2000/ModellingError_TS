# Project Modules

from ErrorModelling.interfaces import *

# Other Modules

import numpy as np
from scipy.stats import skew, kurtosis

# Results function

def stats_results(errors: np.ndarray, theory: dict):

    results = {
        "mean": errors.mean(),
        "median": np.median(errors),
        "std": errors.std(),
        "skew": skew(errors),
        "kurtosis": kurtosis(errors),
        "max": errors.max(),
        "min": errors.min(),
    }
    
    print(f"Mean: {results["mean"]} vs. Theoretical Mean: {theory["mean"]}")
    print(f"Median: {results["median"]} vs. Theoretical Median: {theory["median"]}")
    print(f"Standard Dev.: {results["std"]} vs. Theoretical Std. Dev.: {theory["std"]}")
    print(f"Skew: {results["skew"]}")
    print(f"Kurtosis: {results["kurtosis"]}")
    print(f"Max: {results["max"]}")
    print(f"Min: {results["min"]}")
    print("\n")

    return results

# "get" functions

def get_loss(loss: Loss):

    if loss == Loss.L1:

        return "L1"
    
    elif loss == Loss.L2:

        return "L2"
    
    elif loss == Loss.Huber:

        return "Huber"
