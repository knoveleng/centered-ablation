"""The judge artifact (RQ2c): on Gemma-2-9B under DIM (s=1, draw 0) about half the completions
collapse, and the two judges disagree on exactly those.

    python -m pipeline.analysis.judge_artifact   # -> results/analysis/judge_artifact.{json,md}
"""

import numpy as np

from pipeline.analysis._output import save
from pipeline.analysis.study import labels, point

MODEL = "gemma-2-9b"


def compute():
    p = point(MODEL, "direction.name=dim", "calibration_seed=0")
    hb, lg, deg = labels(p, "harmbench"), labels(p, "llamaguard"), labels(p, "degeneration")
    D = [i for i, d in enumerate(deg) if d]
    F = [i for i, d in enumerate(deg) if not d]
    return {"unit": "fraction judged harmful", "model": MODEL, "n": len(hb), "n_degen": len(D),
            "hb_degen": float(np.mean([hb[i] for i in D])), "hb_fluent": float(np.mean([hb[i] for i in F])),
            "lg_degen": float(np.mean([lg[i] for i in D])), "lg_fluent": float(np.mean([lg[i] for i in F]))}


def render(r):
    print(f"{r['model']}, DIM s=1 draw 0: {r['n_degen']} of {r['n']} completions degenerate")
    print(f"  judged harmful   degenerate  fluent\n  HarmBench cls    {r['hb_degen']:10.3f}  {r['hb_fluent']:.3f}\n"
          f"  Llama-Guard      {r['lg_degen']:10.3f}  {r['lg_fluent']:.3f}")


if __name__ == "__main__":
    save("judge_artifact", compute(), render)
