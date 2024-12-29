# Project Modules

from interfaces import *
from utils import *

# Other Modules

import keras
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

# Neural Network Handler

class NeuralNetworkHandler():

    def __init__(self, model: keras.Model):

        self.model = model

    def __str__(self):
        return "Neural Network Handler"

    def compile(
        self,
        optimizer: keras.optimizers,
        loss: keras.Loss,
        metrics: list,
        n_batches: int,
    ):

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            steps_per_execution=n_batches,
        )

    def train(
        self,
        train_x,
        train_y,
        batch_size: int,
        epochs: int,
        callback: keras.callbacks,
        shuffle: bool,
        y_weights: dict,
        x_weights,
        opt_hyp: bool = True,
        param_grid: dict = None,
    ):
        if opt_hyp:

            model = KerasClassifier(model=self.model, verbose=0)
            grid = GridSearchCV(estimator=model,param_grid=param_grid,verbose=4)
            grid.fit(train_x, train_y)
            model = grid.best_estimator_
            self.model = model.model_
        
        else:

            self.model.fit(
                x=train_x,
                y=train_y,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callback,
                shuffle=shuffle,
                class_weight=y_weights,
                sample_weight=x_weights,
            )

    def evaluate(
        self,
        train_x,
        train_y,
        batch_size: int,
        x_weights,
        callbacks: keras.callbacks,
    ):

        return self.model.evaluate(
            x=train_x,
            y=train_y,
            batch_size=batch_size,
            sample_weight=x_weights,
            callbacks=callbacks,
            return_dict=True,
        )

    def predict(self, test_x, batch_size: int, callbacks: keras.callbacks):

        predictions = self.model.predict(
            x=test_x,
            batch_size=batch_size,
            callbacks=callbacks,
        )

        preds = {"test_x": [], "pred": []}

        for i in range(len(test_x)):

            preds["test_x"].append(test_x[i])

            if is_classif(self.model):

                pred_class = tf.argmax(predictions[i]).numpy()
                preds["preds"].append(pred_class)

            else:
                
                preds["preds"].append(predictions[i])

        return preds
    
# Neural Network Generators
    
class VanillaNNGenerator(IModelGenerator):

    def __str__(self):
        return "Vanilla Neural Network"
    
    def generate(self, input_shape: tuple, units: dict = {"input": 1 , "1": 4, "output": 4}, activation: dict = {"input": None, "1": "relu", "output": "sigmoid"}):

        model = keras.Sequential()

        model.add(keras.layers.Dense(units=units["input"],input_shape = input_shape))

        for k in list(units.keys())[1:-1]:

            model.add(keras.layers.Dense(units=units[k],activation=activation[k]))
        
        model.add(keras.layers.Dense(units=units["output"],activation=activation["output"]))

        return model