"""Paired comparisons on HarmBench: CA - DIM and CA - ACE per model and judge, a 95% bootstrap
interval over prompts (draws averaged inside each replicate) and, for CA - DIM, the number of
draws on which an exact McNemar test gives p < .05.

    python -m pipeline.analysis.paired           # -> results/analysis/paired.{json,md}

cells.<model>.<judge>.<comparison> = {est, lo, hi, significant[, mcnemar_p05]}; judges hb, lg
(HarmBench classifier, Llama-Guard) and hbf, lgf (fluent-only). Statistics: pipeline/stats.py,
bootstrap n and seed: config study.bootstrap.
"""

import torch

from pipeline.analysis._output import save
from pipeline.analysis.study import BOOT, DRAWS, MODELS, NAMES, labels, point
from pipeline.stats import mcnemar_p, paired_bootstrap

JUDGES = {"hb": ("harmbench", False), "hbf": ("harmbench", True), "lg": ("llamaguard", False), "lgf": ("llamaguard", True)}


def outcomes(m, d, judge):
    evaluator, fluent = JUDGES[judge]
    rows = []
    for s in DRAWS:
        p = point(m, f"direction.name={d}", f"calibration_seed={s}")
        y = labels(p, evaluator)
        if fluent:
            y = [a and not g for a, g in zip(y, labels(p, "degeneration"))]
        rows.append(y)
    return torch.tensor(rows, dtype=torch.float)


def compute():
    cells = {}
    for m in MODELS:
        cells[m] = {}
        for judge in JUDGES:
            Y = outcomes(m, "ca", judge)
            cells[m][judge] = {}
            for other in ("dim", "ace"):
                X = outcomes(m, other, judge)
                est, lo, hi = paired_bootstrap(X, Y, BOOT["n"], BOOT["seed"])
                c = {"est": est, "lo": lo, "hi": hi, "significant": lo > 0 or hi < 0}
                if other == "dim":
                    c["mcnemar_p05"] = sum(mcnemar_p(x, y) < 0.05 for x, y in zip(X, Y))
                cells[m][judge][f"ca-{other}"] = c
    return {"unit": "difference in attack-success rate (fraction)", "n": {"prompts": 240, "draws": len(DRAWS)},
            "bootstrap": BOOT, "cells": cells}


def render(r):
    print("CA minus DIM / ACE, HarmBench, 95% prompt-bootstrap interval; McN: draws with McNemar p<.05")
    for m, row in r["cells"].items():
        for judge, c in row.items():
            a, b = c["ca-dim"], c["ca-ace"]
            print(f"{NAMES['models'][m]:14s} {judge:4s}  CA-DIM {a['est']:+.3f} [{a['lo']:+.3f}, {a['hi']:+.3f}] McN {a['mcnemar_p05']}/6"
                  f"   CA-ACE {b['est']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]")


if __name__ == "__main__":
    save("paired", compute(), render)
