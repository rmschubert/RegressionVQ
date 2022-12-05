import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset

## source
## https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Prognostic%29
## deleted columns 0, 1 and last


class BreastCancer(TensorDataset):

    def __init__(self, target: int = 3):
        ds = pd.read_csv('./prototorch_relvq/datasets/wpbc.csv', sep=',')
        data = pd.DataFrame(ds.drop(ds.columns[target], axis=1))
        targets = ds.iloc[:, target]
        scaler = MinMaxScaler()
        self.data = torch.Tensor(scaler.fit_transform(data.values))
        self.targets = torch.squeeze(torch.Tensor(targets))

        super().__init__(self.data, self.targets)
