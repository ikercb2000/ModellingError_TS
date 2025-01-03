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