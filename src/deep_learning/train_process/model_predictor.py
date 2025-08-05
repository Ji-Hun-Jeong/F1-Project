from Users.project.train_process.model import Model
import torch
from torch import nn

class IModelPredictor:
    def predict_model(self, x_data: list, y_data: list,
                 model: Model, device: torch.device,
                 return_predictions: bool = False) -> list[float]:
        pass
    def evaluate_model(self, x_data: list, y_data: list,
                  model: Model, device: torch.device) -> float:
        pass

class MyModelPredictor(IModelPredictor):
    def predict_model(self, x_data: list, y_data: list,
                 model: Model, device: torch.device,
                 return_predictions: bool = False) -> list[float]:
        model.neural_network.to(device)
        model.neural_network.eval()
        
        predictions = []
        
        with torch.no_grad():
            # 전체 데이터를 한번에 텐서로 변환
            x_tensor = torch.tensor(x_data, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y_data, dtype=torch.float32).to(device)
    
            # 배치로 예측 (더 효율적)
            pred_tensor = model.neural_network(x_tensor).squeeze()
            predictions = pred_tensor.cpu().numpy().tolist()
            
            # 결과 출력
            for i in range(len(x_data)):
                actual = y_tensor[i].item()
                predicted = predictions[i] if isinstance(predictions, list) else predictions
                print(f"샘플 {i+1:>2}: 실제 = {actual:.2f}, 예측 = {predicted:.2f}, "
                      f"오차 = {abs(actual - predicted):.2f}")
        
        if return_predictions:
            return predictions

    def predict_one_sample(self, x_data, y_data, model: Model, device: torch.device):
        # 모델 세팅
        model.neural_network.to(device)
        model.neural_network.eval()
    
        with torch.no_grad():
            # 1. 입력 데이터를 [1, feature_dim] 형태의 텐서로 변환
            #    x_data가 [f1, f2, ..., fn] 형태의 리스트라면 unsqueeze(0)으로 배치 차원 추가
            x_tensor = (
                torch.tensor(x_data, dtype=torch.float32, device=device)
                     .unsqueeze(0)
            )
    
            # 2. forward 호출 -> [1, *] 형태 출력
            pred_tensor = model.neural_network(x_tensor)
    
            # 3. [1, *] -> 스칼라 값으로 변환
            pred_value = pred_tensor.squeeze().item()
    
            # 4. 실제값 준비 (y_data가 단일 값일 때)
            actual_value = float(y_data)
    
            # 5. 결과 출력
            error = abs(actual_value - pred_value)
            print(f"실제 = {actual_value:.2f}, 예측 = {pred_value:.2f}, 오차 = {error:.2f}")

    def evaluate_model(self, x_data: list, y_data: list,
                  model: Model, device: torch.device) -> float:
        model.neural_network.to(device)
        model.neural_network.eval()

        with torch.no_grad():
            x_tensor = torch.tensor(x_data, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1).to(device)

            predictions = model.neural_network(x_tensor)
            
            mae = nn.L1Loss()(predictions, y_tensor).item()

        return mae