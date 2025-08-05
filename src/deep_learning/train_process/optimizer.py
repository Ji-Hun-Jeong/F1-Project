from torch import optim, nn
from enum import IntEnum

class EOptimizerType(IntEnum):
    Adam = 0


class IOptimizer:
    def __init__(self):
        pass

    def zero_grad(self):
        pass

    def step(self):
        pass

    def get_optimizer_type(self) -> EOptimizerType:
        pass

    def state_dict(self):
        pass

    def load_state_dict(self):
        pass

class AdamOptimizer(IOptimizer):
    def __init__(self, neural_network: nn.Module, learning_rate: float):
        self.optimizer = optim.Adam(neural_network.parameters(), lr=learning_rate)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def get_optimizer_type(self) -> EOptimizerType:
        return EOptimizerType.Adam

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
