"""GSM8K: exact match between the last number in the completion and the gold answer."""

import re

from pipeline.evaluators.base import Evaluator


def last_number(text: str) -> str | None:
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None


class GSM8K(Evaluator):
    SETS = {"gsm8k"}

    def score(self, items, completions, reference=None):
        return [{"label": last_number(c) == it["reference"], "prediction": last_number(c)}
                for it, c in zip(items, completions)]


EVALUATOR = GSM8K
