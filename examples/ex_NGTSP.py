import os

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from prototorch_regvq.datasets import (BreastCancer, CalHousing, Diabetes,
                                       Toy_Sin, WineQuality)
from prototorch_regvq.misc.callbacks import RegNGParameterCallback
from prototorch_regvq.misc.initializer import NeuralGasInitializer
from prototorch_regvq.misc.metrics import err10, r_squared
from prototorch_regvq.RegVQ import NGTSP, RNGTSP

n_prototypes = 5


if __name__ == "__main__":
    ## Dataset
    
    training_set = WineQuality(target='residual sugar')
    #training_set = Toy_Sin(samples=500, dim=2, seed=42)
    #training_set = Diabetes()
    #training_set = BreastCancer()


    ## Split into train and test

    X, y = training_set.data, training_set.targets

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=42,
    )

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    ## Dataloaders
    train_loader = DataLoader(train_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=len(y_test))

    ## Hyperparameters
    hparams = dict(
        input_dim=X_train.shape[1],
        latent_dim=n_prototypes,
        device='cuda:0',
        lr=1e-3,
    )

    ## Prototype Initializer
    init_centers = NeuralGasInitializer( 
        hparams=dict(num_prototypes=n_prototypes),
        t_loader=DataLoader(X_train, batch_size=64),
        data = (torch.Tensor(X_train), y_train), 
        callbacks=[RegNGParameterCallback(end_lmbda=0.25)], 
        early_stop=True,
        ).generate()

    ## Model

    # model = NGTSP(
    #     hparams,
    #     optimizer=torch.optim.Adam,
    #     prototypes=init_centers,
    #     targets=y_train,
    #     warm_start=False,
    #     norm_targets=True,
    # )


    model = RNGTSP(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes=init_centers,
        targets=y_train,
        warm_start=False,
        norm_targets=True,
    )

    ## train Model

    trainer = pl.Trainer(
        callbacks=[ 
            RegNGParameterCallback(end_lmbda=0.2),
        ],
        max_epochs=1000,
        detect_anomaly=True,
    )

    ## Training loop
    trainer.fit(model, train_loader)

    ## Predict and print performance
    pd = trainer.predict(model, test_loader)
    print(f"Test Results: \n R**2 = {r_squared(*pd, y_test):.4f} \n MEC10 = {err10(*pd, y_test):.4f}")
