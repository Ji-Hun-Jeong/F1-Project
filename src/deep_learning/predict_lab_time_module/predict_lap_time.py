from typing import Dict

from Users.project.data_container.data_container import AzureStorageAccess, FolderAccess
from Users.project.train_process import model_predictor
from Users.project.train_process.data_state import DataFeaturing, IDatasetCreator
from Users.project.train_process.file_name_storage import FileNameStorage
from Users.project.train_process.loss_func import ILossFunc, MSELoss
from Users.project.train_process.model import ImprovedNN, Model
from Users.project.train_process.model_loader import ModelLoader
from Users.project.train_process.model_predictor import IModelPredictor, MyModelPredictor
from Users.project.train_process.model_storage import ModelStorage
from Users.project.train_process.model_trainer import IModelTrainer, MyModelTrainer
from Users.project.train_process.my_utils import extract_track_from_path
from Users.project.train_process.optimizer import AdamOptimizer
from Users.project.train_process.process_unit import ProcessUnit
from Users.project.predict_lab_time_module.create_lap_time_dataset import LapTimePredictDatasetCreator

import pandas as pd
import torch


class PredictLapTime:
    def __init__(self, device: torch.device):
        self.track_name_unit_dict: Dict[str, ProcessUnit] = {}
        self.folder_access: FolderAccess = AzureStorageAccess()
        self.featuring: DataFeaturing = DataFeaturing()
        self.dataset_creator: IDatasetCreator = LapTimePredictDatasetCreator()
        self.model_trainer: IModelTrainer = MyModelTrainer()
        self.model_predictor: IModelPredictor = MyModelPredictor()
        self.device: torch.device = device
        self.model_loader: ModelLoader = None
        
    def filtering_file_name(self, file_name: str) -> bool:
        if ".csv" not in file_name or "car_data_all.csv" not in file_name:
            return False
        return True
    
    def make_unit(self, model: Model, in_dim: int, track_name: str):
        model_storage = ModelStorage(5, model)
        file_name_storage = FileNameStorage()
        process_unit = ProcessUnit(model_storage, file_name_storage)
        self.track_name_unit_dict[track_name] = process_unit

    def train(self):
        in_dim = self.get_data_state()
        self.model_loader = ModelLoader("./Users/project/model_data/", in_dim)

        neural_network = ImprovedNN(in_dim)
        optimizer = AdamOptimizer(neural_network, 0.001)
        model = Model(neural_network, optimizer)
        loss_func: ILossFunc = MSELoss()

        for file in self.folder_access.get_all_file():
            file_name = file.name
            
            if self.filtering_file_name(file_name) == False:
                print("FileNameFiltering: ", file_name)
                continue

            data_frame = self.folder_access.read_csv_by_data_frame(file_name)
            data_frame_length = len(data_frame)
            if data_frame_length <= 5:
                print("DataFrameLength: ", data_frame_length)
                continue

            speed_zero_ratio = (data_frame["Speed"] == 0).mean()
            if speed_zero_ratio > 0.8:
                print("SpeedZeroRatio > 0.8: ", speed_zero_ratio)
                continue

            track_name = extract_track_from_path(file_name)

            if track_name not in self.track_name_unit_dict:
                print("MakeUnit: ", f"{track_name}_Unit")
                self.make_unit(model, in_dim, track_name)

            x_data, y_data = self.dataset_creator.create_dataset_from_dataframe(data_frame, self.featuring)
            if x_data == None or y_data == None:
                print("CreateDatasetFailed: ", file_name)
                print(data_frame)
                continue
            
            unit: ProcessUnit = self.track_name_unit_dict[track_name]
            unit.train_models(x_data, y_data, self.model_trainer, loss_func, self.device)
            unit.add_file(file_name)

    def save_all_track_greatest_models(self):
        for track_name, unit in self.track_name_unit_dict.items():
            self.model_loader.save_model(unit.greatest_model, f"{track_name}_model.pth")

    def decide_greatest_model(self):
        for track_name, unit in self.track_name_unit_dict.items():
            unit.decide_greatest_model(self.model_predictor, self.dataset_creator, self.folder_access, self.featuring, self.device, 20)

    def predict_one_sample(self, x_data, y_data, track_name):
        model = self.model_loader.load_model(f"{track_name}_model.pth")
        self.model_predictor.predict_model(x_data, y_data, model, self.device)
        print(model.evaluate_value)

    def predict_file(self, file_name: str):
        if self.filtering_file_name(file_name) == False:
            print("FileNameFiltering: ", file_name)
            return
        data_frame = self.folder_access.read_csv_by_data_frame(file_name)
        data_frame_length = len(data_frame)
        if data_frame_length <= 5:
            print("DataFrameLength: ", data_frame_length)
            return
        speed_zero_ratio = (data_frame["Speed"] == 0).mean()
        if speed_zero_ratio > 0.8:
            print("SpeedZeroRatio > 0.8: ", speed_zero_ratio)
            return
        track_name = extract_track_from_path(file_name)
        if track_name not in self.track_name_unit_dict:
            print("TrackNameNotFound: ", track_name)
            return
        
        x_data, y_data = self.dataset_creator.create_dataset_from_dataframe(data_frame, self.featuring)
        if x_data == None or y_data == None:
            print("CreateDatasetFailed: ", file_name)
            print(data_frame)
            return
        
        unit = self.track_name_unit_dict[track_name]
        model = unit.greatest_model

        self.model_predictor.predict_model(x_data, y_data, model, self.device)

    def get_data_state(self) -> int:
        column_names = ["Date",	"RPM",	"Speed",	"nGear",	"Throttle",	"Brake",	"DRS",	"Source",	"Time",	"SessionTime",	"Distance",	"LapNumber"]
        test_data = [
            ["03:18.8",	11009,	2,	1,	29,	True,	1,	"car",	"0 days 00:00:00.145000",	"0 days 01:02:17.008000",	0,	1],
            ["03:18.8",	11009,	2,	1,	29,	True,	1,	"car",	"0 days 00:00:00.145000",	"0 days 01:02:17.008000",	0,	1],
            ["03:18.8",	11009,	2,	1,	29,	True,	1,	"car",	"0 days 00:00:00.145000",	"0 days 01:02:17.008000",	0,	1],
            ["03:18.8",	11009,	2,	1,	29,	True,	1,	"car",	"0 days 00:00:00.145000",	"0 days 01:02:17.008000",	0,	2],
            ["03:18.8",	11009,	2,	1,	29,	True,	1,	"car",	"0 days 00:00:00.145000",	"0 days 01:02:17.008000",	0,	2],
            ["03:18.8",	11009,	2,	1,	29,	True,	1,	"car",	"0 days 00:00:00.145000",	"0 days 01:02:17.008000",	0,	2],
            ["03:18.8",	11008,	2,	1,	29,	True,	1,	"car",	"0 days 00:00:00.145000",	"0 days 01:02:17.008000",	0,	3],
            ["03:18.8",	11005,	2,	1,	29,	True,	1,	"car",	"0 days 00:00:00.145000",	"0 days 01:02:17.008000",	0,	3],
            ["03:18.8",	11005,	2,	1,	29,	True,	1,	"car",	"0 days 00:00:00.145000",	"0 days 01:02:17.008000",	0,	3],
        ]
        df = pd.DataFrame(test_data, columns=column_names)
        test_x, test_y = self.dataset_creator.create_dataset_from_dataframe(df, self.featuring)
        self.model_loader = ModelLoader("./Users/project/model_data/", len(test_x[0]))
        return len(test_x[0])
