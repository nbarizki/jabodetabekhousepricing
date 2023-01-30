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
        if label == 'nans':
            label_data = (self._dataframe[features].isna())
        elif label == 'zeroes':
            label_data = (self._dataframe[features] == 0)
        else:
            raise ValueError('Wrong argument for "label"')
        label_count = label_data.sum()
        label_percent = label_data.mean() * 100
        data_types = self._dataframe[features].dtypes
        return(
            pd.DataFrame({f'{label} Count'.title(): label_count, 
                f'{label} Percentage (%)'.title(): label_percent,
                'Data Types': data_types})
            )