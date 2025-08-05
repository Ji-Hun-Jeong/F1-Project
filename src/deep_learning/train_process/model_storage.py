import copy
from Users.project.train_process.model import Model
import torch
from torch import nn
import pandas as pd

class ModelStorage:
    def __init__(self, num_of_model: int, model: Model):
        self.model_list: list[Model] = []
        for i in range(num_of_model):
            copy_model = copy.deepcopy(model)
            self._reinitialize_weights(copy_model.neural_network)
            self.model_list.append(copy_model)

    def print_model(self):
        for model in self.model_list:
            print(model)

    def _reinitialize_weights(self, network):
        for layer in network.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)