from Users.project.train_process.optimizer import AdamOptimizer, EOptimizerType, IOptimizer
import torch
from torch import nn

class ImprovedNN(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )
        

    def forward(self, x):
        return self.net(x)
    
class Model:
    def __init__(self, neural_network: nn.Module, optimizer: IOptimizer):
        self.neural_network = neural_network
        self.optimizer = optimizer
        self.evaluate_value = -1.0

    def set_evaluate_value(self, evaluate_value: float):
        self.evaluate_value = evaluate_value
    
