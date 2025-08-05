from Users.project.train_process.loss_func import ILossFunc
from Users.project.train_process.model import Model
from Users.project.train_process.optimizer import IOptimizer

import torch
from torch import nn

class IModelTrainer:
    def train_regression_model(self, x_data: list, y_data: list, model: Model, device: torch.device
                               , loss_func: ILossFunc, optimizer: IOptimizer, epochs: int, batch_size: int = 32, verbose: bool = True):
        pass

class MyModelTrainer(IModelTrainer):
    def train_regression_model(self, x_data: list, y_data: list, model: Model, device: torch.device
                               , loss_func: ILossFunc, optimizer: IOptimizer, epochs: int, batch_size: int = 32, verbose: bool = True):
        model.neural_network.to(device)
        model.neural_network.train()

        # 데이터를 텐서로 변환 (한번만)
        x_tensor = torch.tensor(x_data, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1).to(device)

        dataset_size = len(x_data)

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            # 배치 단위로 학습
            for i in range(0, dataset_size, batch_size):
                end_idx = min(i + batch_size, dataset_size)

                x_batch = x_tensor[i:end_idx]
                y_batch = y_tensor[i:end_idx]

                # Forward pass
                predictions = model.neural_network(x_batch)
                loss_func.calculate_loss(predictions, y_batch)

                # Backward pass
                optimizer.zero_grad()
                loss_func.backward()
                optimizer.step()

                total_loss += loss_func.get_loss()
                num_batches += 1

            if num_batches == 0:
                continue  # 다음 epoch으로

            avg_loss = total_loss / num_batches

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

    