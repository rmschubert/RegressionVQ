import torch
from scipy.stats import linregress


def r_squared(p, y):
    _, _, r, _, _ = linregress(p, y)
    return r**2

def std_error(p, y):
    return linregress(p, y)[-1]

def err10(p, y):
    err = torch.abs(p - y)
    err10 = torch.sum(torch.where(err <= 0.1 * y, 1, 0)) / len(y)
    return err10

