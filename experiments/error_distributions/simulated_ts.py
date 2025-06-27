# project modules
# ---------------

from src.ts_simulator.distributions import NormalDist, CauchyDist, GumbelDist, LogNormalDist, ParetoDist
from src.ts_simulator.simulator import TimeSeriesSimulator
from src.ts_simulator.utils import sinForm
from src.graphics.classes import PlotSimulatedTS
from src.dnn.utils import train_and_predict

# libraries
# ---------

import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Time Series Simulation Parameters
# --------------------------------

n = 1000
params = {"A": 1, "B": 0.05, "C": 1}
seed = 12

errors = {
    "Normal": NormalDist({"loc": 0, "scale": 0.25}),
    "Cauchy": CauchyDist({"loc": 0, "scale": 0.01}),
    "Gumbel": GumbelDist({"loc": 3, "scale": 0.25}),
    "LogNormal": LogNormalDist({"loc": 0, "scale": 0.3}),
    "Pareto": ParetoDist({"xm": 0.01, "alpha": 1.25})
}

# Create time vector and deterministic component
times = np.arange(n)
det_series = sinForm(params=params, x=times)

# Initialize simulator and plotter
simul = TimeSeriesSimulator()
plotter = PlotSimulatedTS()
dict_series = {}

# Simulate original series (no noise)
# -----------------------------------

noise_zero = np.zeros(n)
ts_orig = simul.get_ts(
    determ_series=det_series,
    noise_series=noise_zero,
    constant_determ=0,
    constant_noise=0
)
dict_series["Original"] = ts_orig

# Prepare output directory for series plots
output_dir = os.path.join(
    os.getcwd(), "experiments/error_distributions/ts_plots")
os.makedirs(output_dir, exist_ok=True)

# Plot and save original series
fig = plotter.plot_sim_ts(
    times,
    ts_orig,
    det_series,
    errors=None,
    title="Time Series without Noise"
)
fig.savefig(os.path.join(output_dir, "series_no_noise.png"))
plt.close(fig)

# Simulate for each error distribution
# ------------------------------------

for name, dist_sim in errors.items():
    # Simulate noise and get theoretical mean
    noise = simul.simulate_noise(n=n, dist=dist_sim, seed=seed)
    theory = dist_sim.theory()

    # Generate series with noise (mean-adjusted)
    ts = simul.get_ts(
        determ_series=det_series,
        noise_series=noise,
        constant_determ=0,
        constant_noise=-theory["mean"]
    )
    dict_series[name] = ts

    # Plot and save
    fig = plotter.plot_sim_ts(
        times,
        ts,
        det_series,
        errors=noise,
        title=f"Time Series with {name} Noise"
    )
    fig.savefig(os.path.join(output_dir, f"series_{name}.png"))
    plt.close(fig)

# DNN Parameters
# --------------

loss_functions = {
    "L1": "mae",
    "L2": "mse",
    "Huber": keras.losses.Huber(delta=0.5)
}
seq_length = 1     # usar sólo un lag: x=z[:-1], y=z[1:]
batch_size = 16
epochs = 50
learning_rate = 0.01

dict_predictions = {}

# Training and Evaluation of DNN
# ------------------------------

for series_name, series_data in dict_series.items():
    for loss_name, loss_fn in loss_functions.items():
        print(
            f"\n\nTraining {series_name} with {loss_name} loss function...\n\n")
        results, preds = train_and_predict(
            series_data,
            loss_function=loss_fn,
            scaler=StandardScaler(),
            seq_length=seq_length,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate
        )
        dict_predictions[f"{series_name}_{loss_name}"] = preds

# Prediction Results
# ------------------

plot_dir = os.path.join(
    os.getcwd(), "experiments/error_distributions/prediction_plots")
os.makedirs(plot_dir, exist_ok=True)
offset = seq_length  # Align predictions axis

for key, pred_array in dict_predictions.items():
    series_name, loss_name = key.split("_", 1)
    original = dict_series[series_name]

    # Prepare time axes
    times_true = np.arange(len(original))
    times_pred = np.arange(offset, offset + len(pred_array))

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(times_true, original, label="Original (con ruido)", alpha=0.6)
    plt.plot(times_pred, pred_array,
             label=f"Predicción ({loss_name})", linestyle='--')

    plt.title(f"Predicción DNN - Serie: {series_name} | Loss: {loss_name}")
    plt.xlabel("Tiempo")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)

    # Save
    fname = f"pred_{series_name}_{loss_name}.png"
    plt.savefig(os.path.join(plot_dir, fname))
    plt.close()
