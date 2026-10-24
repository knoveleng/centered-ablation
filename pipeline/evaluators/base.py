"""Evaluator interface. `score` returns one dict per item; the dict's `label` (bool) is the
evaluator's headline outcome, and any other keys are stored per item as extra detail.

NEEDS_REFERENCE evaluators also receive the unsteered completions of the same items (from the
strength-0 steer entry), e.g. to measure how much a benign answer changed.
"""

import importlib
import pkgutil
from abc import ABC, abstractmethod
from pathlib import Path


class Evaluator(ABC):
    NEEDS_REFERENCE = False
    NEEDS_GPU = False
    SETS: set[str] | None = None   # evaluation sets this evaluator applies to; None = all

    @classmethod
    def applies(cls, eval_set: str) -> bool:
        return cls.SETS is None or eval_set in cls.SETS

    def __init__(self, params: dict, runtime: dict | None = None):
        """params: config defaults.evaluator.<name> merged with the run's (in the cache key);
        runtime: resources.judges.<name>, vLLM memory knobs that do not change labels."""
        self.params, self.runtime = params, runtime or {}

    @abstractmethod
    def score(self, items: list[dict], completions: list[str], reference: list[str] | None = None) -> list[dict]:
        ...


def discover() -> dict[str, type[Evaluator]]:
    path = Path(__file__).parent
    out = {}
    for _, name, _ in pkgutil.iter_modules([str(path)]):
        if name != "base":
            cls = getattr(importlib.import_module(f"pipeline.evaluators.{name}"), "EVALUATOR", None)
            if cls is not None:
                out[name] = cls
    return out


def vllm_judge(model_id: str, runtime: dict):
    from vllm import LLM

    return LLM(model=model_id, dtype="bfloat16", max_model_len=runtime["max_model_len"],
               gpu_memory_utilization=runtime["gpu_memory"])


def truncate(tok, text: str, n: int) -> str:
    ids = tok.encode(text, add_special_tokens=False)
    return text if len(ids) <= n else tok.decode(ids[:n])
