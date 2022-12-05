from typing import Callable, List, Tuple
from warnings import warn

import numpy as np
import pytorch_lightning as pl
import torch
from prototorch.core.distances import squared_euclidean_distance
from prototorch.core.initializers import MCI, AbstractComponentsInitializer
from prototorch.models import NeuralGas, VisNG2D
from prototorch.utils.utils import parse_data_arg
from pytorch_lightning.callbacks import Callback, EarlyStopping
from torch_kmeans import KMeans

warning_pruned = lambda x, y : f""" \n
--------------------------------------------\n
    Attention: {x - y} Prototypes have been pruned!
    Original size was {x}.\n
----------------------------------------------"""


class KMeans_Initializer(AbstractComponentsInitializer):
    ## uses https://pypi.org/project/torch-kmeans/
    def __init__(
            self,
            hparams,
            data: torch.Tensor,
            noise: float = 0.0,
            transform: Callable = torch.nn.Identity(),
    ):
        self.data = data
        self.noise = noise
        self.transform = transform
        self.model = KMeans(**hparams)
    
    def generate_end_hook(self, center):
        drift = torch.rand_like(center) * self.noise
        components = self.transform(center + drift)
        return components

    def prune_loser(self, prototypes):
        dists = squared_euclidean_distance(self.data, prototypes)
        _, winners = torch.min(dists, 1)
        plabels = torch.linspace(0, len(prototypes) - 1, len(prototypes))
        loser_mask = [False if w not in winners else True for w in plabels]
        new_protos = prototypes[loser_mask, :]
        
        if len(new_protos) != len(prototypes):
            warn(warning_pruned(len(prototypes), len(new_protos)))
        
        return new_protos

    
    def generate(self):
        model_result = self.model(self.data.unsqueeze(0))
        centers = model_result.centers.squeeze()
        components = self.prune_loser(centers)
        components = self.generate_end_hook(components)
        return components


class NeuralGasInitializer(AbstractComponentsInitializer):
    def __init__( 
        self, 
        hparams,
        t_loader,
        data: Tuple[torch.Tensor, torch.Tensor],
        callbacks: List[Callback],
        early_stop: bool = True, 
        max_epochs: int = 1000,
        noise: float=0.0, 
        transform: Callable=torch.nn.Identity(), 
        ):

        self.hparams = hparams
        self.train_loader = t_loader,
        self.data, self.y = data
        
        self.callbacks = callbacks
        #self.callbacks.append(VisNG2D(data))
        if early_stop:
            es = EarlyStopping(
                    monitor="loss",
                    min_delta=0.0001,
                    patience=200,
                    mode="min",
                    verbose=False,
                    check_on_train_epoch_end=True,
                )
            self.callbacks.append(es)
        
        self.max_epochs = max_epochs
        self.noise = noise
        self.transform = transform
        self.model = NeuralGas(hparams=self.hparams, prototypes_initializer=MCI(self.data))
    
    def generate_end_hook(self, center):
        components = self.transform(center)
        return components
    
    def prune_loser(self, prototypes):
        dists = squared_euclidean_distance(self.data, prototypes)
        _, winners = torch.min(dists, 1)
        plabels = torch.LongTensor(range(len(prototypes)))
        loser_mask = [True if w in winners else False for w in plabels]
        new_protos = prototypes[loser_mask, :]
        
        if len(new_protos) != len(prototypes):
            warn(warning_pruned(len(prototypes), len(new_protos)))
        
        return new_protos
    
    def generate(self):
        trainer = pl.Trainer( 
            max_epochs=self.max_epochs,
            callbacks=self.callbacks,
            detect_anomaly=True,
        )
        trainer.fit(self.model, self.train_loader)
        components = self.prune_loser(self.model.prototypes)
        components = self.generate_end_hook(components)
        return components
        

class SigmaInitializer(AbstractComponentsInitializer):
    def __init__( 
        self, 
        data, 
        prototypes, 
        transform: Callable = torch.nn.Identity()):
        self. data = data
        self.prototypes = prototypes
        self.transform = transform
    
    def generate_end_hook(self, sigmas):
        components = self.transform(sigmas)
        return components
    
    def generate(self):
        dists = squared_euclidean_distance(self.data, self.prototypes)
        _, winners = torch.min(dists, 1)
        labels = torch.unique(winners)
        x_and_l = list(zip(self.data, winners))
        grouped_x = [np.array([x[0].numpy() for x in x_and_l if x[1] == l]) for l in labels]
        sigmas = [np.std(X) for X in grouped_x]
        return self.generate_end_hook(torch.Tensor(sigmas))




def init_parameters( 
    y: torch.Tensor, 
    indim: int, 
    outdim: int, 
    ordr: int, 
    norm: bool, 
    bias: bool, 
    ):

    t = y
    if norm:
        t = t / torch.max(t)

    pref = torch.sum(torch.rand(indim * ordr,
                                outdim)) / torch.mean(t)
    
    if bias:
        params = (1 / pref) * torch.rand(indim * ordr + 1, outdim)
    else:
        params = (1 / pref) * torch.rand(indim * ordr, outdim)
    
    if outdim == 1:
        params = params.squeeze()

    return params
