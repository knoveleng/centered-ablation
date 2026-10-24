"""Centered Ablation (ours), the direction route: remove the benign common mode from DIM."""

from pipeline.directions.base import Direction, common_mode, dim
from pipeline.operator import Steer, unit


class CA(Direction):
    def fit(self, pos, neg, ref, params, heldout=None):
        u, mu = dim(pos, neg), common_mode(ref)
        return Steer(unit(u - (u @ mu) * mu), 0.0)


DIRECTION = CA
