"""Control: the kernel direction field d(h) ∝ ∇ log p̂₊(h)/p̂₋(h) = m₊(h) − m₋(h).

m±(h) are softmax-weighted (Nadaraya-Watson) class means around h; distances are measured in
per-dimension standardized space so a few massive-activation dimensions cannot dominate. As
the bandwidth grows the weights become uniform and d(h) becomes DIM exactly, so the bandwidth
sweep interpolates between the linear hypothesis and a nonlinear one.
"""

import torch

from pipeline.directions.base import Direction
from pipeline.operator import FieldSteer


class KdeField:
    def __init__(self, pos, neg, gamma: float):
        pos, neg = pos.float(), neg.float()
        X = torch.cat([pos, neg])
        self.mu, self.sd = X.mean(0), X.std(0).clamp(min=1e-6)
        z = (X - self.mu) / self.sd
        self.sigma = float(gamma * torch.cdist(z, z).flatten().median())
        self.pos, self.neg = pos, neg

    def to(self, device):
        for k in ("mu", "sd", "pos", "neg"):
            setattr(self, k, getattr(self, k).to(device))
        self.zp, self.zn = (self.pos - self.mu) / self.sd, (self.neg - self.mu) / self.sd
        self.zp2, self.zn2 = (self.zp ** 2).sum(-1), (self.zn ** 2).sum(-1)
        return self

    def _weights(self, z, zc, zc2):
        d2 = (z ** 2).sum(-1, keepdim=True) + zc2 - 2 * z @ zc.T
        return torch.softmax(-d2 / (2 * self.sigma ** 2), dim=-1)

    def __call__(self, h):
        if not hasattr(self, "zp"):
            self.to(h.device)
        z = (h - self.mu) / self.sd
        d = self._weights(z, self.zp, self.zp2) @ self.pos - self._weights(z, self.zn, self.zn2) @ self.neg
        return d / d.norm(dim=-1, keepdim=True).clamp(min=1e-8)


class KdeFieldDirection(Direction):
    PARAM_SCHEMA = {"gamma": float}

    def fit(self, pos, neg, ref, params, heldout=None):
        return FieldSteer(KdeField(pos, neg, params["gamma"]))


DIRECTION = KdeFieldDirection
