"""The single-draw ablations on the study models (draw 0, HarmBench): control directions,
directions fitted to generated text, and the strength sweep on Gemma-2-9B.

    python -m pipeline.analysis.controls         # -> results/analysis/controls.{json,md}

cells.<model>.<condition> = {asr, degen, n}: HarmBench-classifier ASR and degeneration rate.
Conditions are "<id>@s<strength>". DIM and CA at s=1 are the separate single-draw run that the
ablation tables report (study.last_sample). legacy/<tag>: directions the pipeline does not
express (fitted to generation-time states or the residual deflection), read from
legacy/results/gen with the degeneration evaluator's rule.
"""

import json

from pipeline.analysis._output import save
from pipeline.analysis.study import FROZEN, NAMES, STUDY, is_degenerate, labels, last_sample, point

MMD = ["direction.name=mmd_erasure", "direction.params.steps=400", "direction.params.normalize=true", "direction.params.pca_k=0"]
CONTROLS = {   # id -> overrides (the parameters of scripts/run_baselines.sh)
    "none": ["steering.strength=0"],
    "dim": ["direction.name=dim"],
    "ca": ["direction.name=ca"],
    "random": ["direction.name=random", "direction.params.seed=0"],
    "lda": ["direction.name=lda", "direction.params.shrinkage=0.9"],
    "fixed_point": ["direction.name=fixed_point", "direction.params.iters=50"],
    "mmd_linear": [*MMD, "direction.params.kernel=linear", "direction.params.gamma=1.0"],
    "mmd_rbf": [*MMD, "direction.params.kernel=rbf", "direction.params.gamma=1.0"],
    "mmd_rbf_g0.5": [*MMD, "direction.params.kernel=rbf", "direction.params.gamma=0.5"],
    "mmd_rbf_g2": [*MMD, "direction.params.kernel=rbf", "direction.params.gamma=2.0"],
    "mmd_rbf_raw": [*MMD, "direction.params.kernel=rbf", "direction.params.gamma=1.0", "direction.params.normalize=false"],
    "mmd_rbf_pca64": [*MMD, "direction.params.kernel=rbf", "direction.params.gamma=1.0", "direction.params.pca_k=64"],
    **{f"kde_g{g:g}": ["direction.name=kde_field", f"direction.params.gamma={g}"] for g in (1.0, 0.5, 0.25, 0.125, 0.0625)},
}
LEGACY = ["erase_traj_s1.0", "trajkde_g0.25_s1.0", "trajkde_g0.5_s1.0", "erase_defl_s1.0", "erase_mix_s1.0",
          "switch_dimc+defl_s1.0"]
STRENGTH = {"model": "gemma-2-9b", "ids": ["dim", "kde_g0.25", "kde_g0.125"], "s": [0.5, 0.75, 1.0, 1.25]}


def cell(hb, deg):
    return {"asr": sum(hb) / len(hb), "degen": sum(deg) / len(deg), "n": len(hb)}


def cached(m, cid, s=1.0):
    p = point(m, *CONTROLS[cid]) if cid == "none" else point(m, *CONTROLS[cid], f"steering.strength={s}")
    if cid in ("dim", "ca") and s == 1.0:
        p = last_sample(p)
    return cell(labels(p, "harmbench"), labels(p, "degeneration"))


def legacy(m, tag):
    it = json.load(open(FROZEN / "gen" / m / "harmbench" / f"{tag}.json"))["items"]
    return cell([bool(x["harmful"]) for x in it], [is_degenerate(x["generation"]) for x in it])


def compute():
    cells = {}
    for m in STUDY:
        cells[m] = {(cid if cid == "none" else f"{cid}@s1"): cached(m, cid) for cid in CONTROLS}
        cells[m].update({f"legacy/{tag}": legacy(m, tag) for tag in LEGACY})
    m = STRENGTH["model"]
    for cid in STRENGTH["ids"]:
        for s in STRENGTH["s"]:
            cells[m][f"{cid}@s{s:g}"] = cached(m, cid, s)
    return {"unit": "fraction of prompts", "eval": "harmbench", "draw": 0, "cells": cells}


def render(r):
    print("HarmBench ASR / degeneration, draw 0")
    ids = list(next(iter(r["cells"].values())))
    print(f"{'condition':32s} " + "  ".join(f"{NAMES['models_short'][m]:>11s}" for m in r["cells"]))
    for cid in sorted({c for row in r["cells"].values() for c in row}, key=lambda c: (c not in ids, c)):
        print(f"{cid:32s} " + "  ".join(f"{row[cid]['asr']:.3f} / {row[cid]['degen']:.2f}" if cid in row else f"{'':>11s}"
                                        for row in r["cells"].values()))


if __name__ == "__main__":
    save("controls", compute(), render)
