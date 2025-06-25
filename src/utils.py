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

# Function to train and obtain predictions for each loss function using multiple time series


def train_and_predict(series, loss_function, loss_name="loss", seq_length=10, batch_size=16, epochs=50, learning_rate=0.01):

    # Scale series
    series_values = series.values
    series = series_values.reshape(-1, 1)
    scaler = StandardScaler()
    series_scaled = scaler.fit_transform(series).flatten()

    # Create Sequences
    X, y = create_sequences_from_series(series_scaled, seq_length)
    X = X.reshape(-1, seq_length, 1)

    # Construct and compile the model
    model = build_dense_model(input_shape=(seq_length, 1), hidden_units=128)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss_function, metrics=["mae", "mse", "mape"])

    # Train the model
    model.fit(X, y, batch_size=batch_size,
              epochs=epochs, shuffle=True, verbose=1)

    # The model is trained, now we can make predictions
    predictions = model.predict(X, batch_size=batch_size).reshape(-1, 1)
    predictions_inv = scaler.inverse_transform(predictions)
    y_inv = scaler.inverse_transform(y.reshape(-1, 1))

    # Evaluation metrics
    results = evaluate_predictions(y_inv, predictions_inv)

    return results, predictions_inv
