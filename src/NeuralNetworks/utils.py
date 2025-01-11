# Other modules

import keras
import numpy as np

# Functions

def is_classif(model: keras.Model):

    output_layer = model.layers[-1]
    activation = output_layer.activation

    if activation == keras.activations.softmax or activation == keras.activations.sigmoid:
        return True
    
    return False

def create_sequences_from_ts(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)
