import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

# Function to create sequences from a single series


def create_sequences_from_series(series, seq_length):
    X, y = [], []
    for t in range(seq_length, len(series)):
        X.append(series[t - seq_length:t])
        y.append(series[t])
    return np.array(X), np.array(y)

# Build a Dense model to learn the time series


def build_dense_model(input_shape, hidden_units=128):
    input_layer = keras.layers.Input(shape=input_shape)
    x = keras.layers.Flatten()(input_layer)
    x = keras.layers.Dense(hidden_units, activation="relu")(x)
    output = keras.layers.Dense(1, activation="linear")(x)
    model = keras.Model(inputs=input_layer, outputs=output)
    return model

# Function to train and obtain predictions for each loss function using multiple time series


def train_and_predict_multiple_series(list_series, loss_functions, seq_length=10, batch_size=16, epochs=50, learning_rate=0.01):

    all_X, all_y = [], []

    for series in list_series:

        series = series.reshape(-1, 1)
        scaler = StandardScaler()
        series_scaled = scaler.fit_transform(series).flatten()

        X, y = create_sequences_from_series(series_scaled, seq_length)
        all_X.append(X)
        all_y.append(y)

    # Concatenate all sequences from all series
    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    # Reshape X to have a single feature per time step
    all_X = all_X.reshape(-1, seq_length, 1)

    results = {}
    predictions_by_loss = {}

    for loss_name, loss_func in loss_functions.items():
        print(f"\n=== Training with loss function: {loss_name} ===\n")

        model = build_dense_model(input_shape=(
            seq_length, 1), hidden_units=128)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=loss_func, metrics=["mae", "mse", "mape"])

        # Train the model on the combined training set
        model.fit(all_X, all_y, batch_size=batch_size,
                  epochs=epochs, shuffle=True, verbose=1)

        # Obtain predictions on the training set
        predictions = model.predict(all_X, batch_size=batch_size)
        predictions = predictions.reshape(-1, 1)
        predictions_inv = scaler.inverse_transform(predictions)
        y_inv = scaler.inverse_transform(all_y.reshape(-1, 1))

        mae = mean_absolute_error(y_inv, predictions_inv)
        mse = mean_squared_error(y_inv, predictions_inv)
        rmse = np.sqrt(mse)
        medae = median_absolute_error(y_inv, predictions_inv)
        r2 = r2_score(y_inv, predictions_inv)

        results[loss_name] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MedAE": medae,
            "R2": r2,
        }
        predictions_by_loss[loss_name] = predictions_inv

    results_df = pd.DataFrame(results).T
    return results_df, predictions_by_loss

# Evaluate prediction performance


def evaluate_predictions(y_true, y_pred):
    results = {}
    results["MAE"] = mean_absolute_error(y_true, y_pred)
    results["MSE"] = mean_squared_error(y_true, y_pred)
    # Calculate MAPE while avoiding division by zero
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) /
                   np.clip(np.abs(y_true), 1e-8, None))) * 100
    results["MAPE"] = mape
    results["RMSE"] = np.sqrt(results["MSE"])
    results["MedAE"] = median_absolute_error(y_true, y_pred)
    results["R2"] = r2_score(y_true, y_pred)
    return results
