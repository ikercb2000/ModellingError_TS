# project modules
# ---------------

from src.graphics.classes import PlotSimulatedTS
from src.dnn.utils import train_and_predict, l_infinity_loss

# libraries
# ---------

import os
import random
import numpy as np
import tensorflow as tf
from scipy.stats import gumbel_r, lognorm, cauchy

# Seed
# ---------------

seed = 3254
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Loss Functions
# --------------

loss_functions = {
    "L1": "mae",
    "L2": "mse",
    "L_inf": l_infinity_loss,
}

# DNN Parameters
# --------------

seq_length = 500
num_examples = 100
batch_size = 16
epochs = 250
lr = 0.01
x = np.linspace(0, 1, seq_length)

# Sinoidal Time Series
# --------------------

y_clean_dict = {
    "U-shape":    np.sin(x + 3*np.pi),
    "Inverted U": np.sin(2 * np.pi * x),
    "Half U":     np.sin(np.pi * x),
}

# Error distributions
# -------------------

# Gumbel
theory_mean = np.euler_gamma  # para scale=1
x_pdf_g, pdf_g = np.linspace(-5, 10, 1000), gumbel_r.pdf(
    np.linspace(-5, 10, 1000), loc=-theory_mean, scale=1)

# LogNormal
mean, sigma = 0, 1
x_pdf_l, pdf_l = np.linspace(0, 10, 1000), lognorm.pdf(np.linspace(
    0, 10, 1000), s=sigma, scale=np.exp(mean), loc=-np.exp(mean + sigma**2/2))

# Cauchy
x_pdf_c, pdf_c = np.linspace(-50, 50,
                             1000), cauchy.pdf(np.linspace(-50, 50, 1000))


various_noises = {
    "Gumbel": np.random.gumbel(loc=0, scale=1, size=num_examples*seq_length).reshape(num_examples, seq_length),
    "LogNormal": np.random.lognormal(mean=mean, sigma=sigma, size=num_examples*seq_length).reshape(num_examples, seq_length),
    "Cauchy": np.random.standard_cauchy(size=num_examples*seq_length).reshape(num_examples, seq_length),
}

# Folders
# -------

base_out = "experiments/real_time_series"
dirs = {
    "pred":   os.path.join(base_out, "various_predictions"),
    "errs":   os.path.join(base_out, "various_errors"),
    "uniq_e": os.path.join(base_out, "unique_errors"),
    "scatter": os.path.join(base_out, "scatter")
}

for d in dirs.values():
    os.makedirs(d, exist_ok=True)


# Time Series Predictions & Evaluation
# ------------------------------------

plotter = PlotSimulatedTS()

for name, noise in various_noises.items():

    for shape, y_clean in y_clean_dict.items():
        print(f"\n=== Processing signal: {shape} ===")
        # Generar datos repetidos y ruido
        x_rep = np.tile(x, (num_examples, 1))
        np.random.seed(seed)

        # Construir dataset: y_clean + ruido centrado
        data = np.tile(y_clean, (num_examples, 1)) + noise - theory_mean
        target = np.tile(y_clean,   (num_examples, 1))

        # Train and predict
        predict, errors = {}, {}
        for title, loss_fn in loss_functions.items():
            print(f"  -> Training with {title} loss with {name} error")

            # Entrena y predice sobre el primer ejemplo
            results, preds = train_and_predict(
                series=data[0],
                loss_function=loss_fn,
                seq_length=seq_length,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=lr
            )

            x_pred = x[seq_length:]
            y_clean_pred = y_clean[seq_length:]

            predict[title] = preds
            errors[title] = y_clean_pred - preds

        fig = plotter.plot_various_predictions(
            x_pred,
            predict,
            y_clean_pred,
            shape
        )
        fig.savefig(os.path.join(dirs["pred"], f"various_pred_{shape}.png"))

        # Error distributions
        fig = plotter.plot_various_errors(
            errors,
            x_pdf_g, pdf_g,
            shape
        )
        fig.savefig(os.path.join(dirs["errs"], f"various_errs_{shape}.png"))

        # Differences L2-L1
        diff_series = predict["L2"] - predict["L1"]
        diff_mean = diff_series.mean()
        fig = plotter.plot_unique_errors(
            x_pred,
            diff_series,
            diff_mean,
            shape
        )
        fig.savefig(os.path.join(dirs["uniq_e"], f"uniq_err_{shape}.png"))

        # Scatter True/Predicted
        fig = plotter.plot_scatter_predictions(
            true_series=y_clean_pred,
            pred_series=predict["L2"],
            offset=1
        )
        fig.savefig(os.path.join(dirs["scatter"], f"scatter_{shape}.png"))
