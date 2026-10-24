"""Benign faithfulness: label = the answer is *badly changed* (difflib similarity to the
unsteered answer below `threshold`)."""

from difflib import SequenceMatcher

from pipeline.evaluators.base import Evaluator


class Faithfulness(Evaluator):
    NEEDS_REFERENCE = True
    SETS = {"alpaca", "alpaca50"}

    def score(self, items, completions, reference=None):
        th = self.params["threshold"]
        out = []
        for c, r in zip(completions, reference):
            ratio = SequenceMatcher(None, r, c).ratio()
            out.append({"label": ratio < th, "similarity": ratio})
        return out


EVALUATOR = Faithfulness
