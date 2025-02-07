# Package Modules

from src2.ErrorModelling4TS.graphics.interfaces import *

# Other Modules

import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

# Functions for Training Neural Networks


def create_sequences_from_ts(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)


def build_model(input_shape, units):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        keras.layers.Dense(units["1"], activation="relu"),
        keras.layers.Dense(units["output"], activation=None)  # Output layer
    ])
    return model


def get_series_transform(num=int):
    if num == 1:
        return "Original"
    elif num == 2:
        return "Mean"
    elif num == 3:
        return "Median"


def get_series_loss(num=int):
    if num == 1:
        return "Without"
    elif num == 2:
        return "Normal"
    elif num == 3:
        return "Cauchy"
    elif num == 4:
        return "Gumbel"
    elif num == 5:
        return "Lognorm"
    elif num == 6:
        return "Pareto"


def evaluate_predictions(y_true, y_pred):
    results = {}
    results["MAE"] = mean_absolute_error(y_true, y_pred)
    results["MSE"] = mean_squared_error(y_true, y_pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) /
                   np.clip(np.abs(y_true), 1e-8, None))) * 100
    results["MAPE"] = mape
    results["RMSE"] = np.sqrt(results["MSE"])
    results["MedAE"] = median_absolute_error(y_true, y_pred)
    results["R2"] = r2_score(y_true, y_pred)
    return results


def train_and_evaluate(list_series, series_names, loss_functions=None, seq_length=10, batch_size=16, epochs=50, learning_rate=0.01):
    if loss_functions is None:
        loss_functions = {
            "L1": "mae",
            "L2": "mse",
            "Huber": keras.losses.Huber(delta=0.5)
        }

    all_results = []
    predictions_dict = {}

    for idx, series in enumerate(list_series):
        print(f"\nProcessing Series {idx + 1}...\n")

        scaler = StandardScaler()
        series["Value"] = scaler.fit_transform(series[["Value"]])

        data = series["Value"].values
        series["Determ"] = series["Determ"].iloc[seq_length:].reset_index(
            drop=True)
        x, y = create_sequences_from_ts(data, seq_length)
        x = x.reshape(x.shape[0], seq_length)

        for loss_name, loss_fn in loss_functions.items():
            print(f"\nTraining with {loss_name} loss...\n")

            model = build_model(input_shape=(seq_length,),
                                units={"1": 128, "output": 1})
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=loss_fn,
                metrics=["mae", "mse", "mape"]
            )

            history = model.fit(
                x, y,
                batch_size=batch_size,
                epochs=epochs,
                shuffle=True,
                verbose=0
            )

            predictions = model.predict(x, batch_size=batch_size)
            predicted_values = scaler.inverse_transform(
                predictions.reshape(-1, 1))
            y_actual = scaler.inverse_transform(y.reshape(-1, 1))

            eval_metrics = evaluate_predictions(y_actual, predicted_values)

            result_dict = {
                "Series": f"{series_names(idx+1)}",
                "Loss Function": loss_name,
                "MAE": eval_metrics["MAE"],
                "MAPE": eval_metrics["MAPE"],
                "MedAE": eval_metrics["MedAE"],
                "MSE": eval_metrics["MSE"],
                "RMSE": eval_metrics["RMSE"],
                "R2": eval_metrics["R2"],
            }
            all_results.append(result_dict)

            predictions_dict[(f"Series {series_names(
                idx+1)}", loss_name)] = predicted_values

    results_df = pd.DataFrame(all_results)

    return results_df, predictions_dict


def generate_prediction_plots(list_series, predictions_dict, loss_functions, plotter, show_figs_grid, series_names, max_series=3, num_cols=None, num_rows=None):
    figs = []
    titles = []

    for series_idx in range(1, max_series + 1):
        ts_series = list_series[series_idx - 1]

        for loss_name in loss_functions.keys():
            predictions = predictions_dict.get(
                (f"Series {series_names(series_idx)}", loss_name))
            if predictions is None:
                continue

            predictions = np.array(predictions).reshape(-1, 1)
            fig = plotter.plot_predictions(
                model_name=loss_name,
                ts_df=ts_series,
                predictions=predictions,
                scatter_plot=False,
                show_ts=False
            )
            figs.append(fig)
            titles.append(f"Series {series_names(series_idx)} - {loss_name}")

    if figs:
        show_figs_grid(figs, titles, num_rows=num_rows, num_cols=num_cols)
    else:
        print("No valid figures to display.")
