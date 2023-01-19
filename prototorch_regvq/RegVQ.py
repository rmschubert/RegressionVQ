from functools import partial
from typing import Callable, Iterable, List, Optional, Tuple, Union

import torch
from prototorch.core.competitions import WTAC
from prototorch.core.distances import squared_euclidean_distance
from prototorch.models.abstract import ProtoTorchBolt
from prototorch.nn import LossLayer
from torch import nn
from torch.nn.parameter import Parameter

from prototorch_regvq.misc.competition import WTAC_RLVQ, WTAC_regression
from prototorch_regvq.misc.initializer import init_parameters
from prototorch_regvq.misc.losses import ngtsp_loss, softRLVQ, supervised_RegNG
from prototorch_regvq.misc.metrics import err10, r_squared


class RegVQ(ProtoTorchBolt):

    def __init__(
        self,
        hparams,
        targets: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
        order: Optional[int] = None,
        warm_start: bool = True,
        norm_targets: bool = True,
        bias_on: bool = True,
        **kwargs,
    ):

        super(RegVQ, self).__init__(hparams, **kwargs)

        self.targets = targets
        self.warm_start = warm_start
        self.norm_y = norm_targets
        self.bias_on = bias_on

        ## if the targets should be normalized during training
        ## the corresponding max_value will be saved
        if self.norm_y:
            if self.targets is None:
                raise ValueError(
                    "Targets cannot be None when norm_targets is True" 
                    )
            else:
                self.max_y = torch.max(self.targets)

        ## prototypes are allowed to be None
        ## but if given, requires_grad is by default False        
        if prototypes is not None:
            self.prototypes = Parameter(prototypes, requires_grad=False)
            self.prototype_labels = torch.LongTensor(range(len(prototypes)))
            self.competition_layer = WTAC()

        idim = self.hparams.input_dim
        odim = self.hparams.latent_dim

        if order is None:
            self.order = 1
        else:
            self.order = order

        ## warm_start allows for custom initialized weights
        if not self.warm_start:
            self.linLayer = nn.Linear( 
                idim * self.order, 
                odim, 
                self.bias_on, 
                )
        else:
            fun_paras = init_parameters(
                y=self.targets,
                indim=idim,
                outdim=odim,
                ordr=self.order,
                norm=self.norm_y,
                bias=self.bias_on,
            )
            self.weights = Parameter(fun_paras, requires_grad=True)

            if self.bias_on:
                if odim == 1:
                    b = fun_paras[-1]
                else:
                    b = fun_paras[-1, :]
                self.biases = Parameter(b, requires_grad=True)
        
        ## prediction layer
        self.regval_prediction_layer = nn.Module()
    
    def alter_parameters(self, altered_parameters: Iterable[Tuple[str, Parameter]]):
        f"""
        Custom function to register added and removed parameters,
        i.e. weights, biases, prototypes etc.
        This is ONLY to REGISTER the altered parameters!

        The actual altering of parameters needs to be done in a Callback and calling
        this function.

        Args:
            altered_parameters (Iterable[Tuple[str, Parameter]]): {List} or {Iterable} Type
            containing parameter-name as {str} and the {Parameter} itself
        """
        for n, p in altered_parameters:
            self.register_parameter(n, p)
        self.reconfigure_optimizers()
    
    def forward(self, x):
        """
        forward includes a prediction_layer and a pre_step function,
        which both can easily be altered without the necessity of 
        defining a new forward. Note that pre_step must return kwargs 
        as a dictionary, which is then used in the regval_prediction_layer.

        Args:
            x (torch.Tensor): Input

        Returns:
            torch.Tensor: The predictions
        """
        kwargs = self.pre_step(x)
        predictions = self.regval_prediction_layer(**kwargs)
        if self.norm_y:
            predictions = predictions * self.max_y
        return predictions
 
    def check_dims(self, x):
        """
        Add additional column of ones to x if bias is enabled.

        Args:
            x (torch.Tensor): Input

        Returns:
            torch.Tensor: transformed x or x itself
        """
        if self.bias_on:
            add_ones = torch.ones(x.shape[0], 1).to(self.device)
            return torch.cat((x, add_ones), 1)
        else:
            return x

    def compute_distances(self, x):
        if self.prototypes is None:
            return squared_euclidean_distance(x, x)
        else:
            protos = self.prototypes
            return squared_euclidean_distance(x, protos)
    
    def _log_metric(self, fun: Callable, batch, tag: str):
        """
        Allows to log any custom metric which depends
        on predictions and targets.
        
        Args:
            fun (Callable): A custom metric m, s.t. m(predictions, targets)
            batch (Any): Current batch
            tag (str): Name of the metric
        """
        x, y = batch
        with torch.no_grad():
            predictions = self.forward(x)
            result = fun(predictions, y)
        
        self.log( 
            tag,
            result,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
 
    def training_step(self, batch):
        train_loss = self.shared_step(batch)
        self.log("loss", train_loss)
        self._log_metric(r_squared, batch, tag="RSq")
        self._log_metric(err10, batch, tag="Err10")
        return train_loss

    def predict_winner(self, x):
        """
        Predicts the closest Prototype/Center

        Args:
            x (torch.Tensor): Input

        Returns:
            torch.Tensor: Labels of the winning prototypes
        """
        p_labels = self.prototype_labels
        with torch.no_grad():
            distances = self.compute_distances(x)
            labels = self.competition_layer(distances, p_labels)
        return labels

    def predict(self, batch):
        x, _ = batch
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions
    
    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self.predict(batch)
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            test_loss = self.shared_step(batch)
        self._log_metric(r_squared, batch, tag="test_RSq")
        self._log_metric(err10, batch, tag="test_Err10")
        return test_loss



class RBFNetwork(RegVQ):
    
    f"""
    
    RBF Network implementation in prototorch inspired by
    https://github.com/michaelshiyu/kerNET

    Args:
        RegVQ (PrototorchBolt): Mother Class

    Raises:
        ValueError: When prototypes are given and sigma is of type {torch.Tensor}, 
        but the number of prototypes does not match the number of sigmas.
        TypeError: When sigma is neither of type {float} nor of type {torch.Tensor}.
        ValueError: When supervised is set {True}, but prototypes are {None}.

    """
    
    def __init__( 
        self, 
        hparams, 
        sigma: Union[float, torch.Tensor] = 0.65,
        sigma_grad: bool = False,
        supervised: bool = False, 
        **kwargs, 
    ):
        super(RBFNetwork, self).__init__(hparams, **kwargs)
        
        ## sigma Parameter for gaussians
        ## if sigma is a tensor, the length
        ## is expected to be equal to the 
        ## number of prototypes
        if isinstance(sigma, float):
            s = torch.Tensor([sigma])
        elif isinstance(sigma, torch.Tensor):
            if len(sigma) != len(self.prototypes):
                raise ValueError(
                    f"Shape of Sigma-Tensor needs to match "
                    f"number of Prototypes, but got {len(self.prototypes)} "
                    f"Prototypes and len(sigma) = {len(sigma)}"
                    )
            else:
                s = sigma
        else:
            raise TypeError(
                f"Expected sigma to be of type {float} or {torch.Tensor},"
                f" but got type {sigma.type}."
                )
        
        self.sigma = Parameter(s, requires_grad=sigma_grad)
        
        ## supervised parameter decides
        ## about the trainability of the
        ## prototypes
        if self.prototypes is None:
            if supervised:
                raise ValueError(
                    "Prototypes cannot be None "
                    "when supervised is True"
                    )
        
        else:
            self.prototypes.requires_grad = supervised
        
        self.regval_prediction_layer = nn.Identity()

        self.loss = nn.MSELoss(reduction='sum')
    
    def rbf_kernel(self, distances, sigma):
        """
        RBF-Kernel implemented as Gaussian-type, i.e.

        rbf(x) = exp(-(x - w)**2 / (2*sigma**2))

        With (x - w)**2 here as the distances Argument
        
        Args:
            distances (torch.Tensor): Distances of X and Prototypes
            sigma (torch.Tensor): Scaling of the Gaussians

        Returns:
            torch.Tensor: RBF-Kernel
        """
        s = sigma.to(self.device)
        pref = -1 / (2 * s.pow(2))
        rbf = torch.exp(pref * distances)
        return rbf

    def approximations(self, d, s):
        if not self.warm_start:
            return self.linLayer(self.rbf_kernel(d, s)).squeeze()
        else:
            return self.check_dims(self.rbf_kernel(d, s)) @ self.weights
    
    def pre_step(self, x):
        distances = self.compute_distances(x)
        approximations = self.approximations(distances, self.sigma)
        return dict(input=approximations)
    

    def shared_step(self, batch):
        x, y = batch
        if self.norm_y:
            y = y / self.max_y

        dists = self.compute_distances(x)
        preds = self.approximations(dists, self.sigma)
        loss = self.loss(preds, y)
        return loss
    

class RLVQ(RegVQ):

    """

    Implementation of Regression-LVQ
    based on the paper by Grbovic and Vucetic (2009).

    Args:
        RegVQ (PrototorchBolt): Base Class
    
    """
    
    def __init__(self, hparams, loss: Optional[Callable] = None, soft: bool = True, **kwargs):
        super(RLVQ, self).__init__(hparams, **kwargs)

        ## set trainable prototypes
        self.prototypes.requires_grad = True

        ## scaling parameter for gaussians
        ## needs to be handled by callback!
        self.gamma = Parameter(torch.empty(1), requires_grad=False)

        ## approximation per prototype
        ## if soft is False this needs to be handled by callback!
        self.soft = soft
        self.cplabel = Parameter(torch.rand(len(self.prototypes)), requires_grad=self.soft)

        self.softLayer = nn.Softmin(dim=1)

        if loss is None:
            loss = softRLVQ
        self.loss = LossLayer(loss, name=loss.__name__)

        ## approximation competition
        self.regval_prediction_layer = WTAC_RLVQ()
    
    def compute_probabilities(self, distances):
        pref = 1 / (2 * self.gamma.pow(2))
        return self.softLayer(pref * distances)
    
    def pre_step(self, x):
        distances = self.compute_distances(x)
        probs = self.compute_probabilities(distances)
        return dict( 
            probabilities=probs,
            approximations=self.cplabel,
            soft=self.soft,
        )
    
    def shared_step(self, batch):
        x, y = batch

        if self.norm_y:
            y = y / self.max_y

        distances = self.compute_distances(x)
        probabilities = self.compute_probabilities(distances)
        loss = self.loss(probabilities, self.cplabel, y)
        return loss


class RegNG(RegVQ):
    
    """
    
    (Supervised) Regression Neural Gas.

    Args:
        RegVQ (PrototorchBolt): Base Class
    
    """
    
    def __init__(self, hparams, loss: Optional[Callable] = None, **kwargs):
        super(RegNG, self).__init__(hparams, **kwargs)

        ## set trainable prototypes
        self.prototypes.requires_grad = True

        ## neighborhood range lambda and balancing beta,
        ## needs to be handled by a callback!
        self.lmbda = Parameter(torch.empty(1), requires_grad=False)
        self.beta = Parameter(torch.empty(1), requires_grad=False)

        ## competition for approximation
        self.regval_prediction_layer = WTAC_regression()

        ## loss layer
        if loss is None:
            loss = supervised_RegNG
        
        self.loss = LossLayer( 
            partial( 
                loss, self.lmbda, self.beta
            ),
            name=loss.__name__,
        )
    

    def polynomial_transform(self, x):
        ## create polynomial-type features
        assert self.order > 0, "order must be positive"
        if self.order == 1:
            return x
        else:
            parts = [x**i for i in range(1, self.order + 1)]
            return torch.cat(tuple(parts), dim=1)

    def approximations(self, x):
        pol_x = self.polynomial_transform(x)
        if not self.warm_start:
            return self.linLayer(pol_x).squeeze()
        else:
            return self.check_dims(pol_x) @ self.weights
    
    def pre_step(self, x):
        return dict( 
                reg_vals=self.approximations(x),
                distances=self.compute_distances(x), 
                )

    def shared_step(self, batch):
        x, y = batch
        if self.norm_y:
            y = y / self.max_y

        distances = self.compute_distances(x)
        apps = self.approximations(x)
        loss = self.loss(apps, y, distances)
        return loss


class NGTSP(RegNG):
    
    """
    
    Implementation of NG for Time-Series Prediction
    by Thomas Martinetz (1993)

    Args:
        RegNG (RegVQ): Base Class
    
    """
    
    def __init__(self, hparams, **kwargs):
        super(NGTSP, self).__init__(hparams, **kwargs)

        ## non-trainable prototypes
        self.prototypes.requires_grad = False

        ## loss-layer
        self.loss = LossLayer(ngtsp_loss, name=ngtsp_loss.__name__)
    
    def pre_step(self, x):
        dists = self.compute_distances(x)
        apps = self.approximations(dists)
        return dict( 
            reg_vals=apps,
            distances=dists,
        )

    def shared_step(self, batch):
        x, y = batch
        if self.norm_y:
            y = y / self.max_y

        distances = self.compute_distances(x)
        apps = self.approximations(distances)
        loss = self.loss(apps, distances, y, self.lmbda)
        return loss


class RNGTSP(NGTSP):

    """
    
    Extension to NGTSP.
    Datapoints instead of distance vectors.


    Args:
        NGTSP (RegVQ): Base Class
    
    """
    
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
    
    def pre_step(self, x):
        dists = self.compute_distances(x)
        apps = self.approximations(x)
        return dict( 
            reg_vals=apps,
            distances=dists,
        )
    
    def shared_step(self, batch):
        x, y = batch
        if self.norm_y:
            y = y / self.max_y

        distances = self.compute_distances(x)
        apps = self.approximations(x)
        loss = self.loss(apps, distances, y, self.lmbda)
        return loss



