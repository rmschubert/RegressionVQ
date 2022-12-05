import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from prototorch_regvq.datasets import (BreastCancer, CalHousing, Diabetes,
                                       Toy_Sin, WineQuality)
from prototorch_regvq.misc.callbacks import (LmbdaCallback, ParameterCallback,
                                             PruneLosers, vis_Callback)
from prototorch_regvq.misc.initializer import KMeans_Initializer
from prototorch_regvq.misc.losses import hybrid_RegNG, supervised_RegNG
from prototorch_regvq.misc.metrics import err10, r_squared
from prototorch_regvq.misc.visualization import VisReg2D, VisRegPCA
from prototorch_regvq.RegVQ import RegNG

n_prototypes = 5
ordr = 1

if __name__ == "__main__":

    ## Dataset
    
    training_set = WineQuality(target='residual sugar')
    #training_set = Toy_Sin(samples=500, dim=2, seed=42)
    #training_set = Diabetes()
    #training_set = BreastCancer()
    #training_set = CalHousing()

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
    
    
    # Hyperparameters
    hparams = dict(
        input_dim=X_train.shape[1],
        latent_dim=n_prototypes,
        device='cuda:0',
        lr=1e-3,
    )

    ## Hyperparameters for K-Means Initialization
    init_params = dict(
        n_clusters=n_prototypes,
        max_iter=100,
        tol=1e-7,
    )

    init_protos = KMeans_Initializer(init_params, data=X_train).generate()

    model = RegNG(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes=init_protos,
        targets=y_train,
        warm_start=True,
        order=ordr,
        norm_targets=True,
        loss=supervised_RegNG,
        #loss=hybrid_RegNG,
    )

    # Visualization
    vis = VisReg2D(training_set,
                   filename='lks_trained.pdf',
                   show_last_only=False,
                   block=False)

    # vis = VisRegPCA( 
    #         epoch_step=20, 
    #         method="step", 
    #         data=training_set, 
    #         show_last_only=False, 
    #         block=False 
    #         )

    # Setup trainer
    trainer = pl.Trainer(
        callbacks=[
            vis,
            LmbdaCallback(),
            PruneLosers(X_train, tol_epochs=50),
        ],
        max_epochs=1000,
        detect_anomaly=True,
    )

    # Training loop
    trainer.fit(model, train_loader)

    ## Predict and print performance
    pd = trainer.predict(model, test_loader)
    print(f"Test Results: \n R**2 = {r_squared(*pd, y_test):.4f} \n MEC10 = {err10(*pd, y_test):.4f}")

