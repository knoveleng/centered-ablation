"""Results kept from the pre-refactor runs that the cache cannot reproduce offline, in the
output format of every other analysis (frozen inputs: legacy/results/).

    python -m pipeline.analysis.frozen_tables    # -> results/analysis/{reduction,mmlu,quality}.{json,md}

reduction  RQ1: agreement between the closed-form first token and the served one, 144 cells
           (geometry/validity_all8.json; needs a GPU to recompute)
mmlu       logit-level MMLU accuracy, 1000 questions, mean over draws (all8/mmlu.md, printed at
           3 decimals; recomputable by the diagnose stage once the checkpoints are available)
quality    pairwise preference of steered vs unsteered Alpaca answers (quality.json: only the
           counts were kept, no per-item verdicts)
"""

import numpy as np

from pipeline.analysis._output import save
from pipeline.analysis.study import DIRECTIONS, MODELS, NAMES, corr, frozen

ID = {"dim": "dim", "dimc": "ca", "ace": "ace", "baseline": "none"}


def reduction():
    val = frozen("geometry/validity_all8.json")
    # rows are written per model, then per draw 0..5, then dim, dimc (= ca), ace (legacy/ks/all8.py: validity)
    cells, seen = [], {}
    for m, pred, served, agree in val:
        i = seen.get(m, 0); seen[m] = i + 1
        cells.append({"model": m, "draw": i // 3, "cond": ["dim", "ca", "ace"][i % 3],
                      "predicted_refusal": pred, "served_refusal": served, "agreement": agree})
    return {"unit": "fraction of prompts", "n": len(val), "agreement": float(np.mean([r[3] for r in val])),
            **corr([r[1] for r in val], [r[2] for r in val]),
            "per_model": {m: float(np.mean([r[3] for r in val if r[0] == m])) for m in MODELS}, "cells": cells}


def mmlu():
    mm = {}
    for line in frozen("all8/mmlu.md").splitlines()[1:]:
        p = line.split()
        mm[p[0]] = dict(zip(["none", "dim", "ca", "ace"], [float(p[1]), float(p[3]), float(p[5]), float(p[7])]))
    return {"unit": "accuracy", "n": 1000, "mmlu": mm,
            "mmlu_maxdev": max(abs(mm[m][c] - mm[m]["none"]) for m in mm for c in DIRECTIONS)}


def quality():
    q = frozen("quality.json")
    out = {}
    for k, v in q.items():
        m, c = k.split("/")
        out[f"{m}/{ID[c]}"] = {"net": (v["steered_win"] - v["steered_loss"]) / sum(v.values()),
                               "win": v["steered_win"] / sum(v.values()), "loss": v["steered_loss"] / sum(v.values()),
                               "tie": v["tie"] / sum(v.values()), "n": sum(v.values())}
    return {"unit": "fraction of pairs", "quality": out}


def render_reduction(r):
    print(f"agreement over {r['n']} cells: {r['agreement']:.3f}; predicted vs served refusal rate Pearson {r['pearson']:.3f}")
    for m, v in r["per_model"].items():
        print(f"  {NAMES['models'][m]:14s} {v:.3f}")


def render_mmlu(r):
    print(f"MMLU (logit level, n={r['n']}); largest deviation from unsteered {r['mmlu_maxdev']:.3f}")
    for m, v in r["mmlu"].items():
        print(f"  {m:14s} " + "  ".join(f"{NAMES['directions'][c]} {x:.3f}" for c, x in v.items()))


def render_quality(r):
    print("Steered vs unsteered Alpaca answer, judged both orders: net = wins - losses")
    for k, v in r["quality"].items():
        print(f"  {k:20s} net {v['net']:+.3f}  win {v['win']:.3f}  loss {v['loss']:.3f}  tie {v['tie']:.3f}  (n={v['n']})")


if __name__ == "__main__":
    save("reduction", reduction(), render_reduction)
    save("mmlu", mmlu(), render_mmlu)
    save("quality", quality(), render_quality)
