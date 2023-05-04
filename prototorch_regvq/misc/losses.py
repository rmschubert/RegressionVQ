import torch

ranks_fun = lambda d: torch.argsort(torch.argsort(d, 1), 1)
ngh_fun = lambda r, l: torch.exp(-r / l)
sgd = lambda d, l: 1/ (1 + ngh_fun(-d,l))


def calc_regVal_distances(x, y):
    target_matrix = y.repeat_interleave(x.shape[1]).reshape(
        x.shape[0], x.shape[1])
    reg_val_distances = (x - target_matrix)**2

    return reg_val_distances

def hybrid_RegNG(lmbda, beta, reg_vals, y, proto_distances):
    reg_val_distances = calc_regVal_distances(reg_vals, y)
    class_ranks = ranks_fun(proto_distances)
    cc = ngh_fun(class_ranks, lmbda)

    loss = torch.mean(torch.sum(cc * proto_distances, 0)) + torch.mean(torch.sum(cc * reg_val_distances, 0))

    return loss

def supervised_RegNG(lmbda, beta, reg_vals, y, proto_distances):
    reg_val_distances = calc_regVal_distances(reg_vals, y)
    class_ranks = ranks_fun(proto_distances)
    cc = ngh_fun(class_ranks, lmbda)
    fc = ngh_fun(proto_distances, 0.5 * lmbda)

    loss = beta * torch.mean(torch.sum(cc * proto_distances, 0)) + (1 - beta) * torch.mean(torch.sum(fc * reg_val_distances, 0))

    return loss

def regsensitive_RegNG(lmbda, beta, reg_vals, y, proto_distances):
    reg_val_distances = calc_regVal_distances(reg_vals, y)
    reg_ranks = ranks_fun(reg_val_distances)
    rsc = ngh_fun(reg_ranks, lmbda)
    fc = ngh_fun(proto_distances, 0.5 * lmbda)

    loss = beta * torch.mean(torch.sum(rsc * proto_distances, 0)) + (1 - beta) * torch.mean(torch.sum(fc * reg_val_distances))

    return loss

def hardRLVQ(probabilities, reg_vals, targets):
    r, c = probabilities.shape
    approxs = reg_vals.repeat_interleave(r).reshape(c, r)
    target_matrix = targets.repeat(c).reshape(c, r)
    reg_val_distances = (approxs - target_matrix)**2

    loss = torch.sum(torch.mean(probabilities * reg_val_distances.T, 0))

    return loss

def softRLVQ(probabilities, reg_vals, targets):
    r, c = probabilities.shape
    approxs = reg_vals.repeat_interleave(r).reshape(c, r)
    target_matrix = targets.repeat(c).reshape(c, r)
    reg_val_distances = (approxs - target_matrix)**2
    p_tj = ngh_fun(reg_val_distances.T, torch.mean(reg_val_distances.detach()))

    loss = -torch.sum(torch.log(torch.sum(probabilities * p_tj, 0)))

    return loss

def ngtsp_loss(reg_vals, distances, y, lmbda):
    reg_val_distances = calc_regVal_distances(reg_vals, y)
    class_ranks = ranks_fun(distances)
    cc = ngh_fun(class_ranks, lmbda)

    loss = torch.mean(torch.sum(cc * reg_val_distances, 0))

    return loss
