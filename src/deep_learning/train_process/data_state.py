import pandas as pd
import numpy as np
from datetime import datetime
from enum import IntFlag

def time_to_seconds(time_str):
    """
    시간 문자열(형식: HH:MM:SS.ssssss)을 초 단위로 변환합니다.
    
    Args:
        time_str (str): 시간 문자열 (예: '00:00:00.208000')
    
    Returns:
        float: 초 단위로 변환된 시간
    """
    # 시간 문자열을 datetime 객체로 파싱
    time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
    # 초 단위로 변환 (시, 분, 초, 마이크로초를 고려)
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1_000_000
    return total_seconds

class EFeatureType(IntFlag):
    Basic = 1
    Change = 2
    Boolean = 4
    ZeroRatio = 8

class DataFeaturing:
    def __init__(self):
        pass

    def feature_by_list(self, data_list: list, feature_type: EFeatureType):
        feature_list = []
        
        if feature_type & EFeatureType.Basic:
            feature_list += self.feature_by_basic_statistic(data_list)
            
        if feature_type & EFeatureType.Change:
            feature_list += self.feature_by_change_statistic_improved(data_list)

        if feature_type & EFeatureType.Boolean:
            feature_list += self.feature_by_boolean_statistic(data_list)

        if feature_type & EFeatureType.ZeroRatio:
            feature_list += self.feature_by_zero_ratio(data_list)

        return feature_list

    def feature_by_basic_statistic(self, data_list: list):
        pandas_data = pd.Series(data_list)
        avg = pandas_data.mean().item()
        max = pandas_data.max().item()
        min = pandas_data.min().item()
        std = pandas_data.std().item()
        mid = pandas_data.median().item()
        _25 = pandas_data.quantile(0.25).item()
        _75 = pandas_data.quantile(0.75).item()
        return [avg, max, min, std, mid, _25, _75]

    def feature_by_change_statistic_improved(self, data_list: list[float]):
        if len(data_list) <= 1:
            return [0.0, 0.0, 0.0]

        s = pd.Series(data_list)
        changes = s.diff().abs().dropna()  # NaN 제거

        if changes.empty:
            return [0.0, 0.0, 0.0]

        # 변화가 0이 아닌 비율
        rate = ((changes != 0).sum() / len(changes)).item()

        # 모든 변화량의 평균과 표준편차 (당신 방식과 동일)
        mean_change = changes.mean().item()
        std_change = changes.std(ddof=1).item()

        return [rate, mean_change, std_change]

    def feature_by_boolean_statistic(self, data_list: list):
        pandas_data = pd.Series(data_list, dtype=bool)

        change_to_true_count = 0
        true_count = 0
        prev_value = False
        for index, data in pandas_data.items():
            if data == True:
                if prev_value == False:
                    change_to_true_count += 1
                true_count += 1
            prev_value = data
        
        if change_to_true_count > 0:
            rate_of_true_maintain = true_count / change_to_true_count
        else:
            rate_of_true_maintain = 0.0

        rate_of_true = pandas_data.mean().item()

        return [rate_of_true, change_to_true_count, rate_of_true_maintain]

    def feature_by_zero_ratio(self, data_list: list):
        pandas_data = pd.Series(data_list)
        zero_ratio = (pandas_data == 0).mean().item()
        return [zero_ratio]

    def trend_feature(self, data_list):
        if len(data_list) < 2:
            return np.float64(0.0)
        x = np.arange(len(data_list))
        slope = np.polyfit(x, data_list, 1)[0]  # 선형 기울기
        return np.float64(slope)
    def skewness_feature(self, data_list):
        s = pd.Series(data_list)
        return s.skew()

class IDatasetCreator:
    def create_dataset_from_dataframe(self, data_frame: pd.DataFrame, featuring: DataFeaturing) -> tuple[list, list]:
        pass