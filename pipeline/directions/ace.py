"""ACE, the offset route (Marshall et al., 2024, at alpha = 0): keep DIM, move the origin to μ₋."""

from pipeline.directions.base import Direction, dim
from pipeline.operator import Steer


class ACE(Direction):
    def fit(self, pos, neg, ref, params, heldout=None):
        u = dim(pos, neg)
        return Steer(u, float(neg.float().mean(0) @ u))


DIRECTION = ACE
