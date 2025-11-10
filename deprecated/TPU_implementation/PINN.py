from definitions import *
from training import *

class PINN(tf.keras.Model):
    def __init__(self, input_params, output_params, hidden_layers, neurons_per_layer, activation, use_batch_norm, dropout_rate):
        super(PINN, self).__init__()
        self.model = tf.keras.Sequential()

        self.input_params = input_params
        self.output_params = output_params
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        
        # Adding the first layer (input layer to first hidden layer)
        self.model.add(tf.keras.layers.Dense(neurons_per_layer[0], input_shape=(len(input_params),), activation=activation))

        # Adding hidden layers
        for i in range(1, hidden_layers):
            if use_batch_norm:
                self.model.add(tf.keras.layers.BatchNormalization())
            if dropout_rate is not None and dropout_rate > 0:
                self.model.add(tf.keras.layers.Dropout(dropout_rate))
            self.model.add(tf.keras.layers.Dense(neurons_per_layer[i], activation=activation))

        # Adding the output layer
        self.model.add(tf.keras.layers.Dense(len(output_params)))

    def call(self, x):
        return self.model(x)

    def get_config(self):
        config = super(PINN, self).get_config()
        config.update({
            'input_params': self.input_params,
            'output_params': self.output_params,
            'hidden_layers': self.hidden_layers,
            'neurons_per_layer': self.neurons_per_layer,
            'activation': self.activation,
            'use_batch_norm': self.use_batch_norm,
            'dropout_rate': self.dropout_rate
        })
        return config