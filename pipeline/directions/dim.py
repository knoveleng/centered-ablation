"""DIM ablation: the standard difference-in-means direction, origin at 0."""

from pipeline.directions.base import Direction, dim
from pipeline.operator import Steer


class DIM(Direction):
    def fit(self, pos, neg, ref, params, heldout=None):
        return Steer(dim(pos, neg), 0.0)


DIRECTION = DIM
