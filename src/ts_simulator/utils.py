# Other Module

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sinusoidal Form


def sinForm(params: dict, x: list):
    results = []
    for i in x:
        results.append(params["A"]*np.sin(params["B"]*x[i]+params["C"]))
    return results
