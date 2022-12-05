import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset


class Toy_Sin(TensorDataset):
    def __init__(self, samples, dim, seed):
        rng = np.random.default_rng(seed=seed)
        rand_data = rng.standard_exponential(size=(samples, dim))
        rand_data[:int(samples/2), :] *= -1
        self.data = torch.Tensor(rand_data)
        t = np.linspace(-np.pi, np.pi, samples)
        self.targets = torch.Tensor(np.sin(t))

        super().__init__(self.data, self.targets)
