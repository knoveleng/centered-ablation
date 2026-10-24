"""Repetition collapse ("Here Here Here…"): an output of at least `min_bytes` bytes whose
zlib-compressed size is below `max_ratio` of its length (config: 200 and 0.2; normal prose of
this length sits at 0.4-0.6)."""

import zlib

from pipeline.evaluators.base import Evaluator


def is_degenerate(text: str, min_bytes: int, max_ratio: float) -> bool:
    raw = text.encode()
    return len(raw) >= min_bytes and len(zlib.compress(raw)) / len(raw) < max_ratio


class Degeneration(Evaluator):
    def score(self, items, completions, reference=None):
        return [{"label": is_degenerate(c, self.params["min_bytes"], self.params["max_ratio"])} for c in completions]


EVALUATOR = Degeneration
