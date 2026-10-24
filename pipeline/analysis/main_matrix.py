"""The main matrix (RQ1-RQ2): attack success on HarmBench under two judges, with and without the
fluency filter, and benign faithfulness on Alpaca; 8 models x {none, DIM, CA, ACE} x 6 draws.

    python -m pipeline.analysis.main_matrix      # -> results/analysis/main_matrix.{json,md}

asr.<model>.<direction>.<metric> = {mean, sd} over draws (sd 0 for the one unsteered run), metrics:
  hb, lg    rate judged harmful by the HarmBench classifier / Llama-Guard-3-8B
  hbf, lgf  the same, a degenerate completion counting as a failure (fluent-only)
  deg       degeneration rate;  ref  keyword-refusal rate
faith.<model>.<direction>: fraction of the 100 Alpaca answers badly changed, pooled over draws.
"""

import numpy as np

from pipeline.analysis._output import save
from pipeline.analysis.study import DIRECTIONS, DRAWS, MODELS, NAMES, labels, point


def draws(m, d, eval_set):
    if d == "none":
        return [point(m, "steering.strength=0", f"data.eval={eval_set}")]
    return [point(m, f"direction.name={d}", f"calibration_seed={s}", f"data.eval={eval_set}") for s in DRAWS]


def asr_cell(m, d):
    per = {"hb": [], "hbf": [], "lg": [], "lgf": [], "deg": [], "ref": []}
    for p in draws(m, d, "harmbench"):
        hb, lg, deg, ref = labels(p, "harmbench"), labels(p, "llamaguard"), labels(p, "degeneration"), labels(p, "refusal_keyword")
        per["hb"].append(np.mean(hb))
        per["hbf"].append(np.mean([h and not g for h, g in zip(hb, deg)]))
        per["lg"].append(np.mean(lg))
        per["lgf"].append(np.mean([x and not g for x, g in zip(lg, deg)]))
        per["deg"].append(np.mean(deg))
        per["ref"].append(np.mean(ref))
    return {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0} for k, v in per.items()}


def compute():
    asr = {m: {d: asr_cell(m, d) for d in ["none", *DIRECTIONS]} for m in MODELS}
    faith = {m: {d: float(np.mean([x for p in draws(m, d, "alpaca") for x in labels(p, "faithfulness")]))
                 for d in DIRECTIONS} for m in MODELS}
    return {"unit": "fraction of prompts", "n": {"harmbench": 240, "alpaca": 100, "draws": len(DRAWS)},
            "asr": asr, "faith": faith}


def render(r):
    dn, mn = NAMES["directions"], NAMES["models"]
    print("Attack success (HarmBench, n=240; mean over draws). F: fluent-only.")
    print(f"{'model':14s} {'dir':5s} {'HB':>6s} {'HB-F':>6s} {'LG':>6s} {'LG-F':>6s} {'degen':>6s} {'refuse':>6s}")
    for m, row in r["asr"].items():
        for d, a in row.items():
            print(f"{mn[m]:14s} {dn[d]:5s} " + " ".join(f"{a[k]['mean']:6.3f}" for k in ("hb", "hbf", "lg", "lgf", "deg", "ref")))
    print("\nBenign answers badly changed (Alpaca, n=100, pooled over draws)")
    for m, row in r["faith"].items():
        print(f"{mn[m]:14s} " + "  ".join(f"{dn[d]} {v:.3f}" for d, v in row.items()))


if __name__ == "__main__":
    save("main_matrix", compute(), render)
