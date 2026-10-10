import numpy as np
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, in_dimension, out_dimension, hidden_layers, neurons_per_hidden_layer):
        super().__init__()

        layers = []
        hidden_layers = max(0, int(hidden_layers))
        neurons_per_hidden_layer = max(1, int(neurons_per_hidden_layer))

        if hidden_layers == 0:
            layers.append(nn.Linear(in_dimension, out_dimension))
        else:
            layers.append(nn.Linear(in_dimension, neurons_per_hidden_layer))
            layers.append(nn.ReLU())
            for _ in range(hidden_layers - 1):
                layers.append(nn.Linear(neurons_per_hidden_layer, neurons_per_hidden_layer))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(neurons_per_hidden_layer, out_dimension))

        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    