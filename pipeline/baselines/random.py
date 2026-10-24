"""Control: an isotropic random unit direction (noise floor)."""

import torch

from pipeline.directions.base import Direction
from pipeline.operator import Steer


class Random(Direction):
    PARAM_SCHEMA = {"seed": int}

    def fit(self, pos, neg, ref, params, heldout=None):
        g = torch.Generator().manual_seed(params["seed"])
        return Steer(torch.randn(pos.shape[1], generator=g), 0.0)


DIRECTION = Random
