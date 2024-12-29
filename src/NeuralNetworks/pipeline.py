# Project Modules
from classes import *
from utils import *

# Other Modules
from sklearn.model_selection import train_test_split
import pandas as pd
import keras

# Architecture

units = {"input": 64 , "1": 128, "output": 10}  # 64 input features, 128 in the first hidden layer, 10 outputs (for 10 classes)
activations = {"input": None, "1": "relu", "output": "softmax"}

# Parameters

loss = keras.losses.SparseCategoricalCrossentropy() 
optimizer = keras.optimizers.Adam()
metrics = [keras.metrics.AUC(), keras.metrics.Accuracy(), keras.metrics.Precision()]
n_batches = 32 
epochs = 10

# Parameters for GridSearch

param_grid = {
    'batch_size': [32, 64],
    'epochs': [10, 20],
    'optimizer': [keras.optimizers.Adam(), keras.optimizers.SGD()],
    'layers': [1, 2],
    'neurons': [64, 128, 256],
}

# Input Processing

(train_x, train_y), (test_x, test_y) = keras.datasets.fashion_mnist.load_data()

train_x = train_x.reshape(-1, 28 * 28)  # Flatten images to vectors of size 28*28
test_x = test_x.reshape(-1, 28 * 28)  # Flatten images to vectors of size 28*28
train_x = train_x.astype('float32') / 255.0  # Normalize to range [0, 1]
test_x = test_x.astype('float32') / 255.0  # Normalize to range [0, 1]

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3, random_state=42)

# Model Generation
nn_generator = VanillaNNGenerator()
nn_model = nn_generator.generate(input_shape=(28*28,), units=units, activation=activations)

# Model Training
nn_handler = NeuralNetworkHandler(model=nn_model)
nn_handler.compile(optimizer=optimizer, loss=loss, metrics=metrics, n_batches=n_batches)

# Training with hyperparameter optimization
nn_handler.train(
    train_x=train_x, 
    train_y=train_y, 
    batch_size=n_batches, 
    epochs=epochs, 
    opt_hyp=True, 
    param_grid=param_grid,
    callback=None, 
    shuffle=True, 
    y_weights=None, 
    x_weights=None
)

# Evaluate the model
loss, metrics = nn_handler.evaluate(
    train_x=train_x, 
    train_y=train_y, 
    batch_size=n_batches, 
    x_weights=None, 
    callbacks=None
)

print("\nLoss: ", loss)
print("\nMetrics:\n", pd.DataFrame(metrics))

# Model Predictions

preds = nn_handler.predict(test_x=test_x, batch_size=n_batches, callbacks=None)

for i in range(len(preds["pred"])):
    print("X=%s, Predicted=%s" % (preds["test_x"][i], preds["preds"][i]))

