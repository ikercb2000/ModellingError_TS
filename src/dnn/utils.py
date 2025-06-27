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

# Build a Dense model to learn the time series


def build_dense_model(input_shape, hidden_units=128):
    input_layer = keras.layers.Input(shape=input_shape)
    x = keras.layers.Flatten()(input_layer)
    x = keras.layers.Dense(hidden_units, activation="relu")(x)
    output = keras.layers.Dense(1, activation="linear")(x)
    model = keras.Model(inputs=input_layer, outputs=output)
    return model

# Evaluate prediction performance


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes common regression metrics between y_true and y_pred arrays.
    """
    results = {}
    results["MAE"] = mean_absolute_error(y_true, y_pred)
    results["MSE"] = mean_squared_error(y_true, y_pred)
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) /
                   np.clip(np.abs(y_true), 1e-8, None))) * 100
    results["MAPE"] = mape
    results["RMSE"] = np.sqrt(results["MSE"])
    results["MedAE"] = median_absolute_error(y_true, y_pred)
    results["R2"] = r2_score(y_true, y_pred)
    return results

# Function to train and obtain predictions for each loss function using numpy arrays


def train_and_predict(
    series: np.ndarray,
    loss_function,
    scaler,
    seq_length: int = 10,
    batch_size: int = 16,
    epochs: int = 50,
    learning_rate: float = 0.01
):
    """
    Trains a dense model on the given series (1D numpy array) and returns
    metrics, predictions, and the Keras training history.
    """
    # 1) Scale the data
    data = series.reshape(-1, 1)
    scaled = scaler.fit_transform(data).flatten()

    # 2) Build sequences
    X, y = create_sequences_from_series(scaled, seq_length)
    X = X.reshape(-1, seq_length, 1)

    # 3) Build & compile the model
    model = build_dense_model(input_shape=(seq_length, 1), hidden_units=128)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function,
        metrics=["mae", "mse", "mape"]
    )

    # 4) Train and capture the History
    history = model.fit(
        X, y,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        validation_split=0.1      # optional, if you want val_loss
    )

    # 5) Predict on the same data
    preds = model.predict(X, batch_size=batch_size).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds).flatten()
    y_inv = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    # 6) Evaluate
    results = evaluate_predictions(y_inv, preds_inv)

    # 7) Return metrics, predictions, and history
    return results, preds_inv, history

# Infinity loss


def l_infinity_loss(y_true, y_pred):
    return tf.reduce_max(tf.abs(y_true - y_pred), axis=-1)
