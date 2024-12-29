# Other modules

import keras

# Functions

def is_classif(model: keras.Model):

    output_layer = model.layers[-1]
    activation = output_layer.activation

    if activation == keras.activations.softmax or activation == keras.activations.sigmoid:
        return True
    
    return False