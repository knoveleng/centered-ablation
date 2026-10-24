"""Control: the gate-optimal (whitened) direction u ∝ M_λ⁻¹ δ in RMS-normalized space.

M is the uncentered second moment of normalized benign states (scaled to trace d), δ the
normalized class-mean difference; λ = 1 is normalized-space DIM.
"""

import torch

from pipeline.directions.base import Direction
from pipeline.operator import Steer, rms_normalize


class Rayleigh(Direction):
    PARAM_SCHEMA = {"lam": float}

    def fit(self, pos, neg, ref, params, heldout=None):
        p, n = rms_normalize(pos.float()), rms_normalize(neg.float())
        M = n.T @ n / len(n)
        M = M / M.trace() * M.shape[0]
        lam = params["lam"]
        return Steer(torch.linalg.solve((1 - lam) * M + lam * torch.eye(M.shape[0]), p.mean(0) - n.mean(0)), 0.0)


DIRECTION = Rayleigh
