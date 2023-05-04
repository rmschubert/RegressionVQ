from typing import List
from warnings import warn

import numpy as np
import torch
from matplotlib import pyplot as plt
from prototorch.models.vis import Vis2DAbstract

normalize = lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x)))
warning_method = lambda x, y: f""" \n
--------------------------------------------\n
{x} is not included in {y} switching to method mean.\n
----------------------------------------------"""

class VisReg2D(Vis2DAbstract):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xlabel = "Data dimension 1"
        self.ylabel = "Data dimension 2"

    def plot_data(self, ax, x, label_preds):
        ax.scatter(
            x[:, 0],
            x[:, 1],
            c=label_preds,
            cmap=self.cmap,
            edgecolor="k",
            marker="o",
            s=30,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.precheck(trainer):
            return True
        
        with torch.no_grad():

            protos = pl_module.prototypes.numpy()
            plabels = pl_module.prototype_labels.numpy()
            x_train = self.x_train
            ax = self.setup_ax()
            with torch.no_grad():
                y_preds = pl_module.predict_winner(x_train).numpy()
            x1 = x_train.numpy()
            self.plot_data(ax, x1, y_preds)
            self.plot_protos(ax, protos, plabels)

            self.log_and_display(trainer, pl_module)
            if trainer.current_epoch == trainer.max_epochs - 1:
                plt.show()
                plt.close()


class VisRegPCA(Vis2DAbstract):

    def __init__(self, epoch_step: int = 10, method: str = "mean", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xlabel = "Data dimension 1"
        self.ylabel = "Data dimension 2"
        self.step = epoch_step
        self.p_container = torch.empty(1)
        
        self.methods = { 
            "mean": self._mean_prototypes,
            "step": self._step_prototypes,
        }
        self.method = self.methods.get(method, self._warn_methoderror)
        self.u_method = method

        x_pca, _, v = torch.pca_lowrank(self.x_train, 2)
        self.x = normalize(x_pca.numpy())
    
    def _warn_methoderror(self, p1, p2):
        warn(warning_method(self.u_method, self.methods.keys()))
        return self._mean_prototypes(p1, p2)
    
    def _mean_prototypes(self, p1, p2):
        return 0.5 * (p1 + p2)
    
    def _step_prototypes(self, p1, p2):
        return p2

    def plot_data(self, ax, x, label_preds):
        ax.scatter(
            x[:, 0],
            x[:, 1],
            c=label_preds,
            cmap=self.cmap,
            edgecolor="k",
            marker="o",
            s=30,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.precheck(trainer):
            return True
        
        ce, me = trainer.current_epoch + 1, trainer.max_epochs
        with torch.no_grad():
            if ce % self.step == 0 or ce == me:
                    prototypes = self.method(self.p_container, pl_module.prototypes)
                    
                    x_and_p = torch.cat((self.x_train, prototypes), 0)
                    pcaxp, _, _ = torch.pca_lowrank(x_and_p, 2)
                    normpca = normalize(pcaxp.numpy())
                    p1 = normpca[-len(prototypes):, :]
                    
                    y_preds = pl_module.predict_winner(self.x_train).numpy()
                    plabels = pl_module.prototype_labels.numpy()
                    ax = self.setup_ax()
                    self.plot_data(ax, self.x, y_preds)
                    self.plot_protos(ax, p1, plabels)

                    self.log_and_display(trainer, pl_module)
                    if trainer.current_epoch == trainer.max_epochs - 1:
                        plt.show()
                        plt.close()
            else:
                self.p_container = self.method(self.p_container, pl_module.prototypes)