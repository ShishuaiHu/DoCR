import torch


def ent_loss(output, ita=2.0):
    P = torch.sigmoid(output)
    logP = torch.log2(P+1e-6)
    PlogP = P * logP
    ent = -1.0 * PlogP.sum(dim=1)
    ent = ent ** 2.0 + 1e-8
    ent = ent ** ita
    return ent.mean()
