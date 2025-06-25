# project modules
# ---------------

from src.ts_simulator.distributions import NormalDist, CauchyDist, GumbelDist, LogNormalDist, ParetoDist
from src.ts_simulator.simulator import TimeSeriesSimulator
from src.ts_simulator.utils import sinForm
from src.graphics.classes import PlotSimulatedTS
from src.utils import train_and_predict

# libraries
# ---------

import keras
import os
import pandas as pd
import matplotlib.pyplot as plt

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

det_series = pd.DataFrame(sinForm(params=params, x=list(range(
    0, n))), index=range(0, n), columns=["Determ"])

# Time Series Simulators
# ----------------------

simul = TimeSeriesSimulator()
plotter = PlotSimulatedTS()

# Time Series Simulation
# ----------------------

det_series = pd.DataFrame(
    sinForm(params=params, x=list(range(0, n))), index=range(0, n), columns=["Determ"])
ts_orig = simul.get_ts(determ_series=det_series["Determ"], noise_series=pd.Series(
    [0]*len(det_series.index)), constant_determ=0, constant_noise=0)

dict_series = {"Original": ts_orig}

# Save the figure without noise for comparison

fig = plotter.plot_sim_ts(ts_orig, errors=None,
                          title="Time Series without Noise")
output_dir = os.path.join(
    os.getcwd(), "experiments/error_distributions/ts_plots")
os.makedirs(output_dir, exist_ok=True)
ruta_archivo = os.path.join(output_dir, f"series_no_noise.png")
fig.savefig(ruta_archivo)
plt.close(fig)

# Try for the different distributions

for dist in errors.keys():
    globals()[f"noise_{dist.lower()}"] = pd.DataFrame(
        simul.simulate_noise(n=n, dist=errors[dist], seed=12),
        index=range(0, n), columns=["Noise"])

    error_theory = errors[dist].theory()

    globals()[f"ts_{dist.lower()}"] = simul.get_ts(determ_series=det_series["Determ"], noise_series=globals()[
        f"noise_{dist.lower()}"], constant_determ=0, constant_noise=-error_theory["mean"])

    dict_series[dist] = globals()[f"ts_{dist.lower()}"]

    fig = plotter.plot_sim_ts(
        globals()[f"ts_{dist.lower()}"], errors=globals()[
            f"noise_{dist.lower()}"], title=f"Time Series with {dist} Noise")

    ruta_archivo = os.path.join(output_dir, f"series_{dist}.png")
    fig.savefig(ruta_archivo)
    plt.close(fig)

# DNN Parameters
# --------------------------------

loss_functions = {
    "L1": "mae",
    "L2": "mse",
    "Huber": keras.losses.Huber(delta=0.5)
}

seq_length = 10
batch_size = 16
epochs = 50
learning_rate = 0.01

# Training and Evaluation of DNN
# --------------------------------

dict_predictions = {}

for series in dict_series.keys():
    for loss in loss_functions.keys():
        print(f"\n\nTraining {series} with {loss} loss function...\n\n")
        globals()[f"results_{series}_{loss}"], globals()[
            f"predictions_{series}_{loss}"] = train_and_predict(dict_series[series], loss_function=loss_functions[loss], loss_name=loss, seq_length=seq_length, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)
        dict_predictions[f"{series}_{loss}"] = globals()[
            f"predictions_{series}_{loss}"]


# Prediction Results
# ------------------

plot_dir = os.path.join(
    os.getcwd(), "experiments/error_distributions/prediction_plots")
os.makedirs(plot_dir, exist_ok=True)

# Longitud de entrada usada en la red (por ejemplo 10)
offset = seq_length  # Para alinear los targets y predicciones

for key in dict_predictions.keys():
    series_name, loss_name = key.split("_", 1)

    # Obtener la serie original completa
    original_series = dict_series[series_name].values.flatten()

    # Obtener la predicción
    pred = dict_predictions[key].flatten()

    # Ajustar el eje x para predicciones (porque las secuencias usan desplazamiento)
    x_pred = list(range(offset, offset + len(pred)))
    x_true = list(range(len(original_series)))

    plt.figure(figsize=(12, 4))
    plt.plot(x_true, original_series, label="Original (con ruido)", alpha=0.6)
    plt.plot(x_pred, pred, label=f"Predicción ({loss_name})", linestyle='--')

    plt.title(f"Predicción DNN - Serie: {series_name} | Loss: {loss_name}")
    plt.xlabel("Tiempo")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)

    fname = f"pred_{series_name}_{loss_name}.png"
    plt.savefig(os.path.join(plot_dir, fname))
    plt.close()
