from Users.project.train_process.model import ImprovedNN, Model
from Users.project.train_process.optimizer import AdamOptimizer
import torch

class ModelLoader:
    def __init__(self, absolute_path: str, in_dim: int):
        self.absolute_path = absolute_path
        self.in_dim = in_dim

    def save_model(self, model: Model, file_name: str):
        path = self.absolute_path + file_name
        torch.save({
            "neural_network": model.neural_network.state_dict(),
            "optimizer": model.optimizer.state_dict(),
        }, path)


    def load_model(self, file_name: str, device: torch.device) -> Model:
        full_path = self.absolute_path + file_name
        save_data = torch.load(full_path, map_location=device)
        # 이거 nn이랑 optimizer는 enum붙히고 정수값 따로 저장해서 불러오기
        nn = ImprovedNN(self.in_dim)
        optimizer = AdamOptimizer(nn, 0.001)
        loaded_model = Model(nn, optimizer)
        loaded_model.neural_network.load_state_dict(save_data["neural_network"])
        loaded_model.optimizer.load_state_dict(save_data["optimizer"])
        return loaded_model