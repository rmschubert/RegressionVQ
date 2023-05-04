# Description
This repository is used for experiments considering the Regression Neural Gas and Regression-Sensitive Neural Gas [[source]](https://www.techfak.uni-bielefeld.de/~fschleif/mlr/mlr_01_2023.pdf).

There are also other models implemented: Radial-Basis-Function Network ([[__RBFN__]](https://ieeexplore.ieee.org/document/6796477)), Regression Learning Vector Quantization ([[__RLVQ__]](https://ieeexplore.ieee.org/document/5360312)), Neural Gas for time-series prediction ([[__NGTSP__]](https://ieeexplore.ieee.org/document/238311)) and  a variant of it, in which the predictor is altered to be dependent on the data samples rather than on the distance vector (__xNGTSP__ / __RNGTSP__)

## Experiments
---
In the file __experiments.py__ the code can be found used to produce the results of a comparison of these models. This section deals with the setting. A comprehensive overview of the results is available in the folder __results__ including a summary.

### Datasets
---
The datasets included are WineQuality-red, California Housing, Breastcancer Prognostics and Diabetes. All Datasets are normalized in range [0, 1]. Furthermore, different targets can be chosen for Winequality and Breastcancer. For Winequality we chose for experiments.py the target alcohol. For Breastcancer we went with the mean perimeter as the target and removed the columns ID and Outcome (for the sake of normalization) and the column Lymph Node Status, due to missing values.

### Parameter Setting and Modelling
---
For the visibility parameter in Neural Gas we used $\lambda(t) = 10 \cdot (0.95) ^ t$ for training time $t$ (here the epoch number which was in total 100 epochs) to initialize the Neural Gas prototypes for the models NGTSP and RNGTSP. For the regression setting we chose $\lambda_{reg}(t) = 0.999^t$ and $\hat{\lambda}_{Reg}(t) = 0.5 \cdot \lambda_{Reg}(t)$. As for the learning rate $\epsilon(t)$ we used an exponential decay with $\epsilon(t) = 0.01^t$. Furthermore, the RBFs are modelled as 

$$g_{RBF}(\sigma, x, p_i) = exp(- \sigma_i ||x - p_i||^2)$$

for the RBFN with prototype/center $p_i$ and deviation $\sigma_i$. And as

$$g_{Reg(Se)NG}(\hat{\lambda}_{Reg}(t), x, p_i) = exp\left(- \frac{||x - p_i||^2}{\hat{\lambda}_{Reg}(t)}\right)$$

for RegNG and RegSeNG.
Further for the parameter $\sigma_P$ in RLVQ we decided for $\sigma_P(0) = 5$ and for a similar schedule as in [[RLVQ]](https://ieeexplore.ieee.org/document/5360312).

Furthermore, a batch-normlization layer was applied to accelerate training and enhance reproducability.

### Validation and Measures
---
We used a 5-fold Cross-Validation for each 5, 10 and 15 prototypes. For validation measures we used the coefficient of determination $r^2$ and the standard error $sep$ (both are provided by [[scipy]](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html)).

Note that there is an additional measure $err10$ which is used to evaluate the percentage of the predictions having less or equal than $10\%$ deviation (in $L_1$ Norm) to the targets.

In the __summary.csv__ also the maximum achieved values for each measure are recorded.