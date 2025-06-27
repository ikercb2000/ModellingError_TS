import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from src.dnn.utils import train_and_predict, l_infinity_loss

# 1) Genera o carga tu serie z de retornos (1D numpy array)
#    Por Ejemplo:
# z = compute_log_returns(prices)
# Aquí simulamos una onda senoidal con ruido:
N = 1000
x = np.linspace(0, 1, N)
z = np.sin(2*np.pi*x) + 0.1*np.random.randn(N)

# 2) Define las pérdidas a probar
losses = {
    "MSE": "mse",
    "MAE": "mae",
    "L_inf": l_infinity_loss
}

# 3) Longitud de ventana lag-1
seq_length = 1

# 4) Corre train_and_predict para cada pérdida
results = {}
predictions = {}

for name, loss_fn in losses.items():
    print(f"\n--- Training con pérdida {name} ---")
    res, preds = train_and_predict(
        series=z,
        loss_function=loss_fn,
        seq_length=seq_length,
        batch_size=16,
        epochs=50,
        learning_rate=0.001
    )
    results[name] = res
    predictions[name] = preds

# 5) Muestra métricas
for name, res in results.items():
    print(
        f"{name} -> MAE: {res['MAE']:.4f}, MSE: {res['MSE']:.4f}, RMSE: {res['RMSE']:.4f}")

# 6) Grafica comparativa de predicciones sobre la serie real

# Como seq_length=1, preds de longitud len(z)-1; alineemos en el eje x
t = np.arange(len(z))
plt.figure(figsize=(10, 6))
plt.plot(t, z, 'k-', alpha=0.3, label="Serie real (z)")
for name, pred in predictions.items():
    plt.plot(t[seq_length:], pred, label=f"Predicción {name}")
plt.xlabel("Índice de tiempo")
plt.ylabel("Valor de z")
plt.title("Comparativa: MSE vs MAE vs L∞")
plt.legend()
plt.tight_layout()
plt.show()
