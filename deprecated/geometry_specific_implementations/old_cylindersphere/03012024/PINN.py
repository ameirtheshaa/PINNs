from definitions import *
from training import *
from plotting import *

class PINN(nn.Module):
    def __init__(self, input_params, output_params, hidden_layers, neurons_per_layer, activation, use_batch_norm, dropout_rate):
        super(PINN, self).__init__()
        input_size = len(input_params)
        output_size = len(output_params)
        layers = []
        layers.append(nn.Linear(input_size, neurons_per_layer[0]))
        for i in range(hidden_layers - 1):
            layers.append(activation())
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(neurons_per_layer[i]))
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(neurons_per_layer[i], neurons_per_layer[i + 1]))
        layers.append(activation())
        layers.append(nn.Linear(neurons_per_layer[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)