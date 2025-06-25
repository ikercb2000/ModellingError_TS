# Other Module

import numpy as np

# Sinusoidal Form


def sinForm(params: dict, x: np.ndarray) -> np.ndarray:
    return params['A'] * np.sin(params['B'] * x + params['C'])
