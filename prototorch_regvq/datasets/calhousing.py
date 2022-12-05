import torch
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset


class CalHousing(TensorDataset):

    def __init__(self):
        data, targets = fetch_california_housing(return_X_y=True)
        scaler = MinMaxScaler()
        self.data = torch.Tensor(scaler.fit_transform(data))
        self.targets = torch.Tensor(targets)
        super().__init__(self.data, self.targets)
