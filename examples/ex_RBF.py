import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from prototorch_regvq.datasets import (BreastCancer, CalHousing, Diabetes,
                                       Toy_Sin, WineQuality)
from prototorch_regvq.misc.initializer import (KMeans_Initializer,
                                               SigmaInitializer)
from prototorch_regvq.misc.metrics import err10, r_squared
from prototorch_regvq.RegVQ import RBFNetwork

num_centers = 100

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

    ## Hyperparameters
    hparams = dict(
        input_dim=num_centers,
        latent_dim=1,
        device='cuda:0',
        lr=1e-3,
    )

    ## Hyperparameters for K-Means Initialization
    init_params = dict(
        n_clusters=num_centers, 
        max_iter=100, 
        tol=1e-7, 
    )

    ## Initialize Centers
    init_centers = KMeans_Initializer(init_params, data=X_train).generate()
    
    ## initialize scaling of gaussian-rbfs
    sigmas = SigmaInitializer(X_train, init_centers).generate()

    ## Model
    model = RBFNetwork(
        hparams,
        optimizer=torch.optim.Adam,
        supervised=False,
        prototypes=init_centers,
        targets=y_train,
        norm_targets=False,
        warm_start=False,
        sigma=sigmas,
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        detect_anomaly=True,
    )

    ## Training loop
    trainer.fit(model, train_loader)
    
    ## Predict and print performance
    pd = trainer.predict(model, test_loader)
    print(f"Test Results: \n R**2 = {r_squared(*pd, y_test):.4f} \n MEC10 = {err10(*pd, y_test):.4f}")