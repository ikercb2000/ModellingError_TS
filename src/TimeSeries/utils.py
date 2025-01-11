# Project modules

from TimeSeries.interfaces import *

# Other modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Deterministic term

def polinomForm(params: list, x: list):
    total = []
    for i in x:
        subtotal = []
        for j in range(0,len(params)):
            result = params[j]*(x[i]**(j))
            subtotal.append(result)
        total.append(np.sum(subtotal))
    return total            

def sinForm(params: dict, x: list):
    results = []
    for i in x:
        results.append(params["A"]*np.sin(params["B"]*x[i]+params["C"]))
    return results

def cosForm(params: dict, x: list):
    results = []
    for i in x:
        results.append(params["A"]*np.cos(params["B"]*x[i]+params["C"]))
    return results

def plot_estimations(predictions: np.ndarray, series: pd.DataFrame, model_name: str, use_total_series: bool = True):
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label=f"Predictions for {model_name}", linestyle="--")
    
    if use_total_series == False:
        plt.plot(series.index, series["Determ"], label="Real Deterministic Values")
    else:
        plt.plot(series.index, series["Value"], label="Real Overall Values")
    
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.ylim([-5,5])
    plt.title(f"Time Series {model_name} Model Predictions")
    plt.legend()
    plt.show()

def plot_multiple_estimations(pred_dict: dict, series: pd.DataFrame, use_total_series: bool = True):
    plt.figure(figsize=(10, 6))

    for k in pred_dict.keys():
        plt.plot(pred_dict[k], label=f"Predictions for {k}", linestyle="--")
    
    if use_total_series == False:
        plt.plot(series.index, series["Determ"], label="Real Deterministic Values")
    else:
        plt.plot(series.index, series["Value"], label="Real Overall Values")
    
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.ylim([-5,5])
    plt.title(f"Time Series Models' Predictions")
    plt.legend()
    plt.show()