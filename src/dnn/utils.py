import numpy as np
import tensorflow as tf
import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

# Function to create sequences from a single numpy series


def create_sequences_from_series(series: np.ndarray, seq_length: int):
    """
    Given a 1D numpy array, returns X of shape (n-seq_length, seq_length) and y of shape (n-seq_length,)
    where y[t] = series[t+seq_length] and X[t] = series[t:t+seq_length]
    """
    X, y = [], []
    for t in range(seq_length, len(series)):
        X.append(series[t - seq_length:t])
        y.append(series[t])
    return np.array(X), np.array(y)


def create_sequences_from_series_2d(series: np.ndarray, seq_length: int):
    """
    Given a 1D numpy array, returns X of shape (n-seq_length, seq_length) and y of shape (n-seq_length,)
    where y[t] = series[t+seq_length] and X[t] = series[t:t+seq_length]
    """
    X, y = [], []
    for t in range(seq_length, len(series)):
        X.append(series[t - seq_length:t, :])
        y.append(series[t, :])
    return np.array(X), np.array(y)

# Build a Dense model to learn the time series


def build_dense_model(input_shape, hidden_units=128):
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Dense(hidden_units, activation="relu")(inputs)
    out = keras.layers.Dense(1, activation="linear")(x)
    return keras.Model(inputs, out)

# Function to train and obtain predictions for each loss function using numpy arrays


def train_and_predict(
    X: np.ndarray,
    y: np.ndarray,
    loss_function,
    seq_length: int = 10,
    batch_size: int = 16,
    epochs: int = 50,
    learning_rate: float = 0.01
):
    """
    Trains a dense model on the given series (1D numpy array) and returns
    metrics, predictions, and the Keras training history.
    """

    model = build_dense_model(input_shape=(
        seq_length,), hidden_units=128)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function,
        metrics=["mae", "mse", "mape"]
    )

    history = model.fit(
        X, y,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        verbose=1      # optional, if you want val_loss
    )

    preds = model.predict(X, batch_size=batch_size).reshape(-1, 1)

    return preds, history

# Infinity loss function


def l_infinity_loss(y_true, y_pred):
    return tf.reduce_max(tf.abs(y_true - y_pred), axis=-1)

# Quantile loss function


def quantile_loss_fn(gamma):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(
            tf.maximum(gamma * error, (gamma - 1) * error),
            axis=-1
        )
    return loss
