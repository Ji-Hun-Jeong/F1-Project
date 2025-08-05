from typing import Dict

from Users.project.data_container.data_container import FolderAccess
from Users.project.train_process.data_state import DataFeaturing, IDatasetCreator
from Users.project.train_process.file_name_storage import FileNameStorage
from Users.project.train_process.loss_func import ILossFunc
from Users.project.train_process.model import Model
from Users.project.train_process.model_predictor import IModelPredictor
from Users.project.train_process.model_storage import ModelStorage
from Users.project.train_process.model_trainer import IModelTrainer

import pandas as pd
import torch

class ProcessUnit:
    def __init__(self, model_storage: ModelStorage, file_name_storage: FileNameStorage):
        self.model_storage: ModelStorage = model_storage
        self.file_name_storage: FileNameStorage = file_name_storage
        self.greatest_model: Model = None

    def add_file(self, file_name: str):
        self.file_name_storage.add_file_name(file_name)

    def train_models(self, x_data: list[float], y_data: list[float], model_trainer: IModelTrainer, loss_func: ILossFunc, device: torch.device):
        for model in self.model_storage.model_list:
            model_trainer.train_regression_model(x_data, y_data, model, device, loss_func, model.optimizer, 100, verbose=False)

    def decide_greatest_model(self, model_predictor: IModelPredictor, dataset_creator: IDatasetCreator, folder_access: FolderAccess
                       , featuring: DataFeaturing, device: torch.device, avg_count: int):
        great_evaluate_values: list[list[float]] = [[] for _ in range(len(self.model_storage.model_list))]
        for count in range(avg_count):
            file_name: str = self.file_name_storage.get_random_file_name()
            data_frame: pd.DataFrame = folder_access.read_csv_by_data_frame(file_name)
            x_data, y_data = dataset_creator.create_dataset_from_dataframe(data_frame, featuring)

            for i in range(len(self.model_storage.model_list)):
                model: Model = self.model_storage.model_list[i]
                evaluate_value = model_predictor.evaluate_model(x_data, y_data, model, device)
                model.evaluate_value = evaluate_value
                great_evaluate_values[i].append(evaluate_value)

        greatest_value = 1000000000000.0
        greatest_index = 0
        for i in range(len(great_evaluate_values)):
            one_model_evaluate_datas = great_evaluate_values[i]
            avg = sum(one_model_evaluate_datas) / len(one_model_evaluate_datas)
            self.model_storage.model_list[i].set_evaluate_value(avg)
            if avg < greatest_value:
                greatest_value = avg
                greatest_index = i
        
        self.greatest_model = self.model_storage.model_list[greatest_index]
            

    def predict_models(self, model_predictor: IModelPredictor, dataset_creator: IDatasetCreator, folder_access: FolderAccess
                       , featuring: DataFeaturing, device: torch.device):
        file_name = self.file_name_storage.get_random_file_name()
        data_frame = folder_access.read_csv_by_data_frame(file_name)
        x_data, y_data = dataset_creator.create_dataset_from_dataframe(data_frame, featuring)
        
        for model in self.model_storage.model_list:
            model_predictor.predict_model(x_data, y_data, model, device)

