import os
from typing import Dict, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from prototorch_regvq.datasets import (BreastCancer, CalHousing, Diabetes,
                                       WineQuality)
from prototorch_regvq.misc.callbacks import (RBFCallback,
                                             RegNGParameterCallback,
                                             SoftRLVQCallback)
from prototorch_regvq.misc.initializer import (KMeans_Initializer,
                                               NeuralGasInitializer,
                                               SigmaInitializer)
from prototorch_regvq.RegVQ import (NGTSP, RLVQ, RNGTSP, RBFNetwork, RegNG,
                                    RegSeNG)

models = [NGTSP, RLVQ, RNGTSP, RBFNetwork, RegNG, RegSeNG]
datasets = [WineQuality(target="alcohol"), BreastCancer(), Diabetes(), CalHousing()]
kfold = KFold(shuffle=True, random_state=21)

if __name__ == "__main__":

    cwd = os.getcwd()
    results_path = os.path.join(cwd, "results")
    model_path = os.path.join(results_path, "models")
    value_path = os.path.join(results_path, "comparison")
    if not os.path.exists(results_path):
        os.mkdir(results_path)
        os.mkdir(model_path)
        os.mkdir(value_path)

    seed_everything(seed=46556)

    ks = ["Model", "Fold", "Prototypes", "Dataset", "test_RSq", "test_Err10", "test_SEP"]
    results: Dict =  dict([(k, []) for k in ks])

    for ds in datasets:
        for num_p in [5, 10, 15]:
            for i, (train_idx, test_idx) in enumerate(kfold.split(np.arange(len(ds)))):

                if type(ds).__name__ == "CalHousing":
                    meps=10
                else:
                    meps=100

                train_sampler = SubsetRandomSampler(train_idx)
                train_loader = DataLoader(ds, batch_size=64, sampler=train_sampler)
                test_sampler = SubsetRandomSampler(test_idx)
                test_loader = DataLoader(ds, batch_size=len(test_idx), sampler=test_sampler)

                X_train, y_train = ds[train_idx]

                kmeans_params = dict(
                    n_clusters=num_p, 
                    max_iter=100, 
                    tol=1e-7, 
                )

                kmeans_protos = KMeans_Initializer(kmeans_params, data=X_train).generate()
                kmeans_sigmas = SigmaInitializer(X_train, kmeans_protos).generate()

                rbfn = RBFNetwork( 
                    hparams=dict( 
                    input_dim=num_p,
                    latent_dim=1,
                    lr=1e-2,
                    ),
                    optimizer=torch.optim.Adam,
                    supervised=True,
                    prototypes=kmeans_protos,
                    sigma=kmeans_sigmas,
                    targets=y_train,
                    sigma_grad=True,
                    lr_scheduler=ExponentialLR,
                    lr_scheduler_kwargs=dict(gamma=0.999, verbose=False),
                )

                rbftrainer = pl.Trainer( 
                    callbacks=[ 
                    RBFCallback(),
                    ],
                    max_epochs=meps,
                    detect_anomaly=True,
                    enable_checkpointing=False,
                )

                rbftrainer.fit(rbfn, train_loader)
                rbfres = rbftrainer.test(rbfn, test_loader)[0]
                results["Model"].append(type(rbfn).__name__)
                results["Fold"].append(i)
                results["Prototypes"].append(num_p)
                results["Dataset"].append(type(ds).__name__)
                for k, v in rbfres.items():
                    results[k].append(v)
                
                rbfname = type(rbfn).__name__ + "_" + type(ds).__name__ + "_p_" + str(num_p) + "_f_" + str(i)
                torch.save(rbfn, os.path.join(model_path, rbfname))
                
                rlvq = RLVQ( 
                    hparams=dict( 
                    input_dim=X_train.shape[1],
                    latent_dim=1,
                    lr=1e-2,
                    ),
                    optimizer=torch.optim.Adam,
                    prototypes=kmeans_protos,
                    targets=y_train,
                    soft=True,
                    lr_scheduler=ExponentialLR,
                    lr_scheduler_kwargs=dict(gamma=0.999, verbose=False),
                )

                rlvqtrainer = pl.Trainer( 
                    callbacks=[ 
                    SoftRLVQCallback(train_size=len(X_train)),
                    ],
                    max_epochs=meps,
                    detect_anomaly=True,
                    enable_checkpointing=False,
                )

                rlvqtrainer.fit(rlvq, train_loader)
                rlvqres = rlvqtrainer.test(rlvq, test_loader)[0]
                results["Model"].append(type(rlvq).__name__)
                results["Fold"].append(i)
                results["Prototypes"].append(num_p)
                results["Dataset"].append(type(ds).__name__)
                for k, v in rlvqres.items():
                    results[k].append(v)
                
                rlvqname = type(rlvq).__name__ + "_" + type(ds).__name__ + "_p_" + str(num_p) + "_f_" + str(i)
                torch.save(rlvq, os.path.join(model_path, rlvqname))

                regng = RegNG( 
                    hparams=dict( 
                    input_dim=X_train.shape[1],
                    latent_dim=num_p,
                    lr=1e-2,
                    ),
                    optimizer=torch.optim.Adam,
                    prototypes=kmeans_protos,
                    targets=y_train,
                    lr_scheduler=ExponentialLR,
                    lr_scheduler_kwargs=dict(gamma=0.999, verbose=False),
                )

                regseng = RegSeNG( 
                    hparams=dict( 
                    input_dim=X_train.shape[1],
                    latent_dim=num_p,
                    lr=1e-2,
                    ),
                    optimizer=torch.optim.Adam,
                    prototypes=kmeans_protos,
                    targets=y_train,
                    lr_scheduler=ExponentialLR,
                    lr_scheduler_kwargs=dict(gamma=0.999, verbose=False),
                )

                ng_prototypes = NeuralGasInitializer( 
                    hparams=dict(num_prototypes=num_p),
                    t_loader=DataLoader(X_train, batch_size=64),
                    data = (X_train, y_train), 
                    callbacks=[RegNGParameterCallback(lmbda=10., decay_lmbda=0.95)], 
                    early_stop=True,
                    max_epochs=100,
                    ).generate()

                ngtsp = NGTSP( 
                    hparams=dict( 
                    input_dim=num_p,
                    latent_dim=num_p,
                    lr=1e-2,
                    ),
                    optimizer=torch.optim.Adam,
                    prototypes=ng_prototypes,
                    targets=y_train,
                    lr_scheduler=ExponentialLR,
                    lr_scheduler_kwargs=dict(gamma=0.999, verbose=False),
                )

                rngtsp = RNGTSP( 
                    hparams=dict( 
                    input_dim=X_train.shape[1],
                    latent_dim=num_p,
                    lr=1e-2,
                    ),
                    prototypes=ng_prototypes,
                    targets=y_train,
                    lr_scheduler=ExponentialLR,
                    lr_scheduler_kwargs=dict(gamma=0.999, verbose=False),
                )

                for model in  [regng, regseng, ngtsp, rngtsp]:
                    ngtrainer = pl.Trainer( 
                        callbacks=[ 
                        RegNGParameterCallback(lmbda=1., beta=1., decay_lmbda=0.999, decay_beta=0.99),
                        ],
                        max_epochs=meps,
                        detect_anomaly=True,
                        enable_checkpointing=False,
                    )

                    ngtrainer.fit(model, train_loader)
                    ngresults = ngtrainer.test(model, test_loader)[0]
                    results["Model"].append(type(model).__name__)
                    results["Fold"].append(i)
                    results["Prototypes"].append(num_p)
                    results["Dataset"].append(type(ds).__name__)
                    for k, v in ngresults.items():
                        results[k].append(v)
                    
                    modelname = type(model).__name__ + "_" + type(ds).__name__ + "_p_" + str(num_p) + "_f_" + str(i)
                    torch.save(model, os.path.join(model_path, modelname))


    resdf = pd.DataFrame.from_dict(results, orient='columns')
    resdf.to_csv(os.path.join(value_path, "results.csv"))
