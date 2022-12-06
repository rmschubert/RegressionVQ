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
            preds = pl_module.predict(x).numpy()
            prototypes = pl_module.prototypes
            prototypes = torch.Tensor(prototypes).to(device)

        pro_to_data_ratio = prototypes.shape[0] / x.shape[0]

        x_vals = np.linspace(0, len(y), len(y))
        plt.plot(x_vals, sorted(y), c='b', label='targets')
        plt.plot(x_vals, sorted(preds), c='c', label='predictions')
        plt.legend()
        plt.title( 
            f"""ratio proto/data = {pro_to_data_ratio:.5f}.
            {prototypes.shape[0]} Prototypes.
            {trainer.max_epochs} epochs.""" 
            )
        if self.name is not None:
            plt.savefig('./' + self.name, dpi=600)
        plt.show()
        plt.close()


class LmbdaCallback(Callback):
    ## Neural Gas is very strict for small values
    ## of lambda, therefore leaving lambda > ~ 0.1 
    ## increases flexibility of the prototypes
    ## Additionally lambda' for regression
    ## should be smaller, to gain the effect
    ## of finding 'regression groups'
    ## Otherwise this might result in an
    ## adiabatic optimization

    def __init__(self):
        super(LmbdaCallback, self).__init__()

    def new_lmbda(self, current_e, max_e):
        x = current_e / max_e
        return torch.Tensor([0.15 ** x])


    def new_beta(self, max_e, current_e):
        x = current_e / max_e
        return torch.Tensor([0.01**(x)])


    def on_train_epoch_start(self, trainer, pl_module) -> None:
        state_dict = pl_module.state_dict()
        if 'lmbda' in state_dict.keys():
            state_dict['lmbda'] = self.new_lmbda(trainer.current_epoch,
                                                 trainer.max_epochs)
        else:
            pl_module.energy_layer.lm = self.new_lmbda(trainer.current_epoch, 
                                                        trainer.max_epochs)

        if 'beta' in state_dict.keys():
            state_dict['beta'] = self.new_beta(trainer.max_epochs,
                                               trainer.current_epoch)

        pl_module.load_state_dict(state_dict)


class ParameterCallback(Callback):
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
        tol_epochs: int = 10,
    ):

        self.x = data
        self.tol_epochs = tol_epochs
        self.loser_counts: List[int] = [0]

    def on_train_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            prototypes = pl_module.prototypes
            weights = pl_module.weights

            distances = pl_module.compute_distances(self.x)
            _, winners = torch.min(distances, 1)
            plabels = pl_module.prototype_labels
            losers = [False if label in winners else True for label in plabels]
            
            old_counts = self.loser_counts
            current_counts = [sum(p) for p in zip(losers, cycle(old_counts))]
            winner_mask = [True if count < self.tol_epochs else False for count in current_counts]

            self.loser_counts = [c for c, w in zip(current_counts, winner_mask) if w]

            if not all(winner_mask):

                new_protos = prototypes[winner_mask, :]
                new_weights = weights[:, winner_mask]
                rgp = prototypes.requires_grad

                altered_params = list( 
                    [ 
                        ('prototypes', Parameter(new_protos, requires_grad=rgp)), 
                        ('weights', Parameter(new_weights)), 
                    ], 
                    )

                if pl_module.bias_on:
                    s = len(pl_module.biases)
                    if s > 1:
                        new_biases = new_weights[-1, :]
                    else:
                        new_biases = new_weights[-1]
                    
                    altered_params.append(('biases', Parameter(new_biases)))
                
                pl_module.alter_parameters(altered_params)
                pl_module.prototype_labels = plabels[winner_mask]

                warn(warning_pruned(len(prototypes), len(new_protos)))
            
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

    def __init__(self) -> None:
        super(SoftRLVQCallback, self).__init__()

    def new_gamma(self, current_e):
        return torch.Tensor([1.5 * 50 / (50 + current_e)])
    
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        state_dict = pl_module.state_dict()
        state_dict['gamma'] = self.new_gamma(trainer.current_epoch)
        pl_module.load_state_dict(state_dict)
        
