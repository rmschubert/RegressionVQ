import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset

## source https://archive.ics.uci.edu/ml/datasets/Wine+Quality
## type: red


class WineQuality(TensorDataset):

    def __init__(self, target):
        dataset = pd.read_csv(
            'prototorch_regvq/datasets/winequality-red.csv', sep=';')
        vals = dataset.columns.values.tolist()
        vals.remove(target)
        data = dataset.loc[:, vals]
        targets = dataset.loc[:, target]
        scaler = MinMaxScaler()
        self.data = torch.Tensor(scaler.fit_transform(data.values))
        self.targets = torch.squeeze(torch.Tensor(targets.values))
        super().__init__(self.data, self.targets)
