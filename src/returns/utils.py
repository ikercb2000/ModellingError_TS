# packages

import numpy as np

# log returns


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    return np.log(prices[1:] / prices[:-1])
