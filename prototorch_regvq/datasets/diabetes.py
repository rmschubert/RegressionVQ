import torch
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset


class Diabetes(TensorDataset):
    def __init__(self, ):
        data, targets = load_diabetes(return_X_y=True)
        self.targets = torch.Tensor(targets)
        scaler = MinMaxScaler()
        self.data = torch.Tensor(scaler.fit_transform(data))
        super().__init__(self.data, self.targets)
