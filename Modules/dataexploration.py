import pandas as pd
import numpy as np

class DataExploration():
    """"""
    def __init__(self, dataframe):
        self._dataframe = dataframe.copy()

    def show_nans_or_zeroes(self, label: str, filter=[]):
        if list(filter):
            features = list(filter)
        else:
            features = list(self._dataframe.columns)
        total_records = self._dataframe.shape[0]
        if label == 'nans':
            label_data = (self._dataframe[features].isna()).sum()
        else:
            label_data = (self._dataframe[features] == 0).sum()
        label_percent = label_data / total_records * 100
        data_types = self._dataframe[features].dtypes
        return(
            pd.DataFrame({f'{label} Count'.title(): label_data, 
                f'{label} Percentage (%)'.title(): label_percent,
                'Data Types': data_types
            })
        )