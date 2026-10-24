"""The collapse index β (RQ3), from the frozen 144-cell table (8 models x {DIM, CA, ACE} x 6
draws; legacy/results/geometry/beta_refusal.json): per-model means, the β+ = 1 threshold, and
how β- relates to ω and to benign faithfulness.

    python -m pipeline.analysis.collapse         # needs main_matrix -> results/analysis/collapse.{json,md}
"""

import numpy as np

from pipeline.analysis._output import load, save
from pipeline.analysis.study import DIRECTIONS, MODELS, NAMES, corr, frozen, omega_eta

COND = {"dim": "dim", "dimc": "ca", "ace": "ace"}   # the frozen table's direction names -> config ids


def compute():
    faith = load("main_matrix")["faith"]
    omega, _ = omega_eta()
    cells = [{**r, "cond": COND[r["cond"]]} for r in frozen("geometry/beta_refusal.json")]
    beta = {}
    for m in MODELS:
        beta[m] = {}
        for c in DIRECTIONS:
            rows = [r for r in cells if r["model"] == m and r["cond"] == c]
            beta[m][c] = {k: {"mean": float(np.mean([r[k] for r in rows])), "sd": float(np.std([r[k] for r in rows], ddof=1))}
                          for k in ("beta_harm", "beta_benign", "degen")}
    bh, dg, bb = (np.array([r[k] for r in cells]) for k in ("beta_harm", "degen", "beta_benign"))
    threshold = {
        "n_cells": len(cells),
        "above1": {"n": int((bh > 1).sum()), "mean_degen": float(dg[bh > 1].mean()), "max_degen": float(dg[bh > 1].max()),
                   "models": sorted({(r["model"], r["cond"]) for r in cells if r["beta_harm"] > 1})},
        "below1": {"n": int((bh <= 1).sum()), "mean_degen": float(dg[bh <= 1].mean()), "max_degen": float(dg[bh <= 1].max()),
                   "max_beta": float(bh[bh <= 1].max())},
        "beta_above1_mean": float(bh[bh > 1].mean()), "beta_above1_sd": float(bh[bh > 1].std(ddof=1)),
        "corr_betaharm_degen": corr(bh, dg), "corr_betabenign_degen": corr(bb, dg),
    }
    om = [omega[m] for m in MODELS]
    dbeta = [beta[m]["dim"]["beta_benign"]["mean"] - beta[m]["ca"]["beta_benign"]["mean"] for m in MODELS]
    dfaith = [faith[m]["dim"] - faith[m]["ca"] for m in MODELS]
    return {"unit": "β dimensionless; degen a fraction", "n": len(cells), "cells": cells, "beta": beta,
            "beta_threshold": threshold,
            "omega_beta": {c: corr(om, [beta[m][c]["beta_benign"]["mean"] for m in MODELS]) for c in DIRECTIONS},
            "dbeta_dfaith": corr(dbeta, dfaith),
            "beta_level_faith": corr([beta[m][c]["beta_benign"]["mean"] for m in MODELS for c in DIRECTIONS],
                                     [faith[m][c] for m in MODELS for c in DIRECTIONS])}


def render(r):
    t = r["beta_threshold"]
    print(f"beta+ > 1 in {t['above1']['n']} of {t['n_cells']} cells ({t['above1']['models']}): mean degeneration "
          f"{t['above1']['mean_degen']:.3f}; the others at most {t['below1']['max_degen']:.3f} (beta+ <= {t['below1']['max_beta']:.3f})")
    print(f"Pearson(drop in beta-, faithfulness gain) {r['dbeta_dfaith']['pearson']:+.3f}\n")
    print(f"{'model':14s} " + "  ".join(f"{NAMES['directions'][c]:>13s}" for c in DIRECTIONS) + "   (beta+ / beta-)")
    for m, row in r["beta"].items():
        print(f"{NAMES['models'][m]:14s} " + "  ".join(f"{v['beta_harm']['mean']:.3f} / {v['beta_benign']['mean']:.3f}" for v in row.values()))


if __name__ == "__main__":
    save("collapse", compute(), render)
