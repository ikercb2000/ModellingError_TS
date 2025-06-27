#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Load and filter data
data = pd.read_csv('all_stocks_5yr.csv', on_bad_lines='skip')
data['date'] = pd.to_datetime(data['date'])
apple = data[data['Name'] == 'AAPL'].sort_values('date').reset_index(drop=True)

# 2. Extract closing prices and compute daily returns
prices = apple['close'].values
dates = apple['date'].values
returns = (prices[1:] - prices[:-1]) / prices[:-1]  # shape (N-1,)

# 3. Prepare X and y: x_t = r_t, y_t = r_{t+1}
X = returns[:-1].reshape(-1, 1)
Y = returns[1:].reshape(-1, 1)

# 4. Train/test split (e.g. 95% train, 5% test)
split = int(len(X) * 0.95)
X_train, Y_train = X[:split], Y[:split]
X_test,  Y_test = X[split:], Y[split:]
last_train_price = prices[split]  # starting price for reconstruction

# 5. Define infinite-norm loss


def inf_norm_loss(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)
    return tf.reduce_max(diff, axis=-1)

# 6. Model builder


def build_model(loss_fn):
    model = keras.models.Sequential([
        keras.layers.Input(shape=(1,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss=loss_fn)
    return model


# 7. Train & evaluate for each loss
results = {}
for loss_name, loss_fn in [
    ('MSE', 'mean_squared_error'),
    ('MAE', 'mean_absolute_error'),
    ('Inf', inf_norm_loss),
]:
    print(f"\n=== Training with {loss_name} loss ===")
    model = build_model(loss_fn)
    history = model.fit(
        X_train, Y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    # Predict returns
    r_hat = model.predict(X_test)

    # Reconstruct prices
    pred_prices = [last_train_price]
    for r in r_hat.flatten():
        pred_prices.append(pred_prices[-1] * (1 + r))
    pred_prices = np.array(pred_prices)

    # True prices for test: start from last_train_price and apply true returns
    true_prices = [last_train_price]
    for r in Y_test.flatten():
        true_prices.append(true_prices[-1] * (1 + r))
    true_prices = np.array(true_prices)

    # Metrics on returns
    mse_r = mean_squared_error(Y_test, r_hat)
    mae_r = mean_absolute_error(Y_test, r_hat)
    maxerr_r = np.max(np.abs(Y_test - r_hat))

    # Metrics on reconstructed prices (skip the initial point)
    mse_p = mean_squared_error(true_prices[1:], pred_prices[1:])
    mae_p = mean_absolute_error(true_prices[1:], pred_prices[1:])

    results[loss_name] = {
        'r_hat': r_hat.flatten(),
        'pred_prices': pred_prices,
        'true_prices': true_prices,
        'metrics_returns': (mse_r, mae_r, maxerr_r),
        'metrics_prices': (mse_p, mae_p)
    }
    print(
        f"Returns → RMSE: {np.sqrt(mse_r):.6f}, MAE: {mae_r:.6f}, MaxErr: {maxerr_r:.6f}")
    print(f"Prices  → RMSE: {np.sqrt(mse_p):.6f}, MAE: {mae_p:.6f}")

# 8. Plot reconstructed price series (alineación dinámica)
plt.figure(figsize=(12, 6))
for loss_name, res in results.items():
    # Number of puntos en pred_prices
    L = len(res['pred_prices'])
    # Tomas las fechas desde el final del entrenamiento hasta completar L puntos
    x_dates = dates[split: split + L]
    plt.plot(
        x_dates,
        res['pred_prices'],
        label=f'Predicted ({loss_name})'
    )

# True series
true_L = len(results['MSE']['true_prices'])
x_dates_true = dates[split: split + true_L]
plt.plot(
    x_dates_true,
    results['MSE']['true_prices'],
    'k--', lw=2, label='True'
)

plt.title('Apple Close Price Reconstruction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# 9. Plot return predictions vs true (alineación dinámica)
plt.figure(figsize=(12, 6))
for loss_name, res in results.items():
    # r_hat tiene longitud M = len(X_test)
    M = len(res['r_hat'])
    t_r = dates[split+1: split+1 + M]
    plt.plot(t_r, res['r_hat'], label=f'Predicted r ({loss_name})')

# True returns
M_true = len(Y_test)
t_r_true = dates[split+1: split+1 + M_true]
plt.plot(t_r_true, Y_test.flatten(), 'k--', lw=2, label='True r')

plt.title('Daily Return: Prediction vs True')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.tight_layout()
plt.show()
