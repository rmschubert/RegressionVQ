from itertools import cycle, product
from typing import List
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from torch import relu
from torch.nn.parameter import Parameter

warning_pruned = lambda x, y : f""" \n
--------------------------------------------\n
    Attention: {x - y} Prototypes have been pruned!
    Original size was {x}.\n
----------------------------------------------"""


class vis_Callback(Callback):

    def __init__(self, batch, name=None):
        self.x, self.y = batch
        self.name = name

    def on_train_end(self, trainer, pl_module):
        x, y = self.x, self.y

        device = pl_module.device

        with torch.no_grad():
            x = torch.Tensor(x).to(device)
            preds = pl_module.predict((x, y)).numpy()
            prototypes = pl_module.prototypes
            prototypes = torch.Tensor(prototypes).to(device)

        pro_to_data_ratio = prototypes.shape[0] / x.shape[0]

        x_vals = np.linspace(0, len(y), len(y))
        plt.plot(x_vals, sorted(y), c='b', label='targets')
        plt.plot(x_vals, sorted(preds), c='c', label='predictions')
        plt.legend()
        plt.title( 
            f"ratio proto/data = {pro_to_data_ratio:.5f}.{prototypes.shape[0]} Prototypes. {trainer.max_epochs} epochs."
            )
        if self.name is not None:
            plt.savefig('./' + self.name, dpi=600)
        plt.show()
        plt.close()


class RegNGParameterCallback(Callback):
    ## Neural Gas is very strict for small values
    ## of lambda, therefore leaving lambda > ~ 0.1 
    ## increases flexibility of the prototypes
    ## Additionally lambda' for regression
    ## should be smaller, to gain the effect
    ## of finding 'regression groups'
    ## Otherwise this might result in an
    ## adiabatic optimization

    def __init__(self, lmbda: float, beta: float = 1., decay_lmbda: float = 0.99, decay_beta: float = 0.99):
        self.l = lmbda
        self.b = beta
        self.dl = decay_lmbda
        self.db = decay_beta
        super(RegNGParameterCallback, self).__init__()

    def new_lmbda(self):
        return torch.Tensor([self.l * self.dl])

    def new_beta(self):
        return torch.Tensor([self.b * self.db])

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        state_dict = pl_module.state_dict()
        x = (trainer.current_epoch + 1) / trainer.max_epochs
        if 'lmbda' in state_dict.keys():
            state_dict['lmbda'] = self.new_lmbda()
        else:
            lm = pl_module.energy_layer.lm
            pl_module.energy_layer.lm = lm * ((1e-2 / lm) ** x)

        if 'beta' in state_dict.keys():
            state_dict['beta'] = self.new_beta()

        pl_module.load_state_dict(state_dict)


class InitWeightsLinLayer(Callback):
    ## Used for a 'good guess' of initial parameters
    ## in a linear layer
    def __init__(self, batch):
        self.x, self.y = batch

    def on_train_start(self, trainer, pl_module) -> None:
        state_dict = pl_module.state_dict()
        with torch.no_grad():
            weights = torch.abs(pl_module.linLayer.weights)
            bias = torch.abs(pl_module.linLayer.bias)

            n_y = self.y / pl_module.max_y

            pref = torch.sum(weights) / torch.mean(n_y)
            paras = (1 / pref) * weights
            biases = (1 / pref) * bias

            state_dict['linLayer.weights'] = paras
            pl_module.load_state_dict(state_dict)
            state_dict['linLayer.bias'] = biases
            pl_module.load_state_dict(state_dict)


class PruneLosers(Callback):

    def __init__(
        self,
        data: torch.Tensor,
        at_train_end_only: bool=True,
        tol_epochs: int = 10,
    ):

        self.x = data
        self.at_train_end = at_train_end_only
        self.tol_epochs = tol_epochs
        self.loser_counts: List[int] = [0]

    def _get_loser_mask(self, pl_module):
        winners = pl_module.predict_winner(self.x)
        plabels = pl_module.prototype_labels
        return [False if label in winners else True for label in plabels]

    def _prune_parameters(self, mask, pl_module):
        prototypes = pl_module.prototypes
        weights = pl_module.weights
        plabels = pl_module.prototype_labels

        new_protos = prototypes[mask, :]
        new_weights = weights[:, mask]
        rgp = prototypes.requires_grad

        altered_params = list([
            ('prototypes', Parameter(new_protos, requires_grad=rgp)),
            ('weights', Parameter(new_weights)),
        ], )

        if pl_module.bias_on:
            s = len(pl_module.biases)
            if s > 1:
                new_biases = new_weights[-1, :]
            else:
                new_biases = new_weights[-1]

            altered_params.append(('biases', Parameter(new_biases)))

        pl_module.alter_parameters(altered_params)
        pl_module.prototype_labels = plabels[mask]

        warn(warning_pruned(len(prototypes), len(new_protos)))
    
    def on_train_epoch_end(self, trainer, pl_module):
        if not self.at_train_end:
            with torch.no_grad():
                losers = self.get_mask(pl_module)

                old_counts = self.loser_counts
                current_counts = [
                    sum(p) for p in zip(losers, cycle(old_counts))
                ]
                winner_mask = [
                    True if count < self.tol_epochs else False
                    for count in current_counts
                ]

                self.loser_counts = [
                    c for c, w in zip(current_counts, winner_mask) if w
                ]

                if not all(winner_mask):
                    self._prune_parameters(winner_mask, pl_module)
                    return True

    def on_train_end(self, trainer, pl_module):
        if self.at_train_end:
            with torch.no_grad():
                losers = self._get_loser_mask(pl_module)
                winner_mask = [not x for x in losers]
                self._prune_parameters(winner_mask, pl_module)
                return True
        
        


class HardRLVQCallback(Callback):

    def __init__(self, batch):
        self.x, self.y = batch

    def on_train_batch_end(self, trainer, pl_module):
        state_dict = pl_module.state_dict()

        p = pl_module.prototypes
        distances = pl_module.compute_distances(self.x, p)
        probabilities = pl_module._compute_probabilities(distances)
        normed_probs = probabilities / torch.sum(probabilities, 0)

        state_dict['approximations'] = self.y @ normed_probs
        pl_module.load_state_dict(state_dict)


class SoftRLVQCallback(Callback):

    def __init__(self, train_size: int) -> None:
        super(SoftRLVQCallback, self).__init__()
        self.anneal_size = train_size
    
    def new_gamma(self, current_e):
        return torch.Tensor([5. * (5 * self.anneal_size) / ((5 * self.anneal_size) + current_e)])
    
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        state_dict = pl_module.state_dict()
        state_dict['gamma'] = self.new_gamma(trainer.current_epoch)
        pl_module.load_state_dict(state_dict)
        
class RBFCallback(Callback):

    def __init__(self) -> None:
        super(RBFCallback, self).__init__()

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        sd = pl_module.state_dict()
        s = sd['sigma']
        sd['sigma'] = torch.where(s <= 5e-2, s + 0.1, s)
        pl_module.load_state_dict(sd)
        

class OmegaCallback(Callback):
    def __init__(self) -> None:
        super(OmegaCallback, self).__init__()

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        sd = pl_module.state_dict()
        o = sd['omega']
        sd['omega'] = o / (torch.trace(o.T @ o))
        pl_module.load_state_dict(sd)