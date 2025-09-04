# packages

import numpy as np

# Return functions


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    return np.log(prices[1:] / prices[:-1])


def compute_simple_returns(prices: np.ndarray) -> np.ndarray:
    return prices[1:] / prices[:-1] - 1


def compute_relative_returns(prices: np.ndarray) -> np.ndarray:
    return prices[1:] / prices[:-1]
