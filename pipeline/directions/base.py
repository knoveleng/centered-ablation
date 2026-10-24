"""Direction interface, shared by pipeline/directions/ (the family) and pipeline/baselines/.

`fit` receives one calibration draw and returns a Steer. `pos`/`neg` are the behaviour and
benign calibration states; `ref` are the benign reference states that define the common mode
(for refusal they are `neg`; for emotion they are token states of the other emotions).
"""

import importlib
import pkgutil
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from pipeline.operator import Steer, rms_normalize, unit


class Direction(ABC):
    PARAM_SCHEMA: dict[str, type] = {}   # every param; defaults come from config defaults.direction.<name>

    @abstractmethod
    def fit(self, pos: torch.Tensor, neg: torch.Tensor, ref: torch.Tensor, params: dict,
            heldout: tuple[torch.Tensor, torch.Tensor] | None = None) -> Steer:
        """heldout: the pool rows outside this draw (positive, negative), in pool order; only
        directions with early stopping use it."""


def dim(pos, neg) -> torch.Tensor:
    return unit(pos.float().mean(0) - neg.float().mean(0))


def common_mode(ref) -> torch.Tensor:
    """μ̂₋: the unit mean of RMS-normalized benign reference states."""
    return unit(rms_normalize(ref.float()).mean(0))


def discover(package: str) -> dict[str, type[Direction]]:
    """name -> class for every module in the package that defines `DIRECTION`."""
    path = Path(importlib.import_module(package).__file__).parent
    out = {}
    for _, name, _ in pkgutil.iter_modules([str(path)]):
        if name != "base":
            cls = getattr(importlib.import_module(f"{package}.{name}"), "DIRECTION", None)
            if cls is not None:
                out[name] = cls
    return out
