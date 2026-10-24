"""Control: the length-reweighted DIM fixed point u = unit(m₊(u) − m₋(u)), m± = E±[h / rms(P_u h)].

The stationary point of linear-kernel MMD erasure after the model's normalization
(Proposition B.5). Its gain over DIM on Llama is entirely implicit centering.
"""

import torch

from pipeline.directions.base import Direction, dim
from pipeline.operator import Steer


class FixedPoint(Direction):
    PARAM_SCHEMA = {"iters": int}

    def fit(self, pos, neg, ref, params, heldout=None):
        pos, neg = pos.float(), neg.float()
        u = dim(pos, neg)
        for _ in range(params["iters"]):
            w = lambda x: 1.0 / (x - (x @ u)[:, None] * u).pow(2).mean(-1).add(1e-6).sqrt()
            g = (pos * w(pos)[:, None]).mean(0) - (neg * w(neg)[:, None]).mean(0)
            new = g / g.norm()
            done = 1 - float(new @ u) < 1e-7
            u = new
            if done:
                break
        return Steer(u, 0.0)


DIRECTION = FixedPoint
