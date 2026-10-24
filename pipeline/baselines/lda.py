"""Control: the shrinkage-LDA discriminant, the most *discriminative* linear direction."""

import torch

from pipeline.directions.base import Direction
from pipeline.operator import Steer


class LDA(Direction):
    PARAM_SCHEMA = {"shrinkage": float}

    def fit(self, pos, neg, ref, params, heldout=None):
        pos, neg = pos.float(), neg.float()
        X = torch.cat([pos, neg])
        mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
        zp, zn = (pos - mu) / sd, (neg - mu) / sd
        Sw = torch.cov(torch.cat([zp - zp.mean(0), zn - zn.mean(0)]).T)
        lam = params["shrinkage"]
        Sw = (1 - lam) * Sw + lam * torch.eye(Sw.shape[0]) * Sw.diagonal().mean()
        w = torch.linalg.solve(Sw, zp.mean(0) - zn.mean(0)) / sd   # back to raw coordinates
        return Steer(w, 0.0)


DIRECTION = LDA
