"""Refusal amplification (RQ4): DIM and CA at s < 0 against held-out jailbreak templates, with
over-refusal on XSTest-safe and JailbreakBench-benign; the three study models, draw 0.

    python -m pipeline.analysis.amplification    # -> results/analysis/amplification.{json,md}

amplification.<model>.<condition>, condition "none", "<direction>/s<strength>", or
"dim/s-1/gated" (KDE-gated DIM-Amp: not a pipeline direction, read from legacy/results/gen):
  hb, lg        fluent-only ASR averaged over the held-out templates (None: no fluent output)
  xstest, benign  keyword refusal on the fluent outputs of the two benign sets
  degen         degeneration averaged over every set;  per_template_hb
"""

import json

import numpy as np

from pipeline.analysis._output import save
from pipeline.analysis.study import FROZEN, NAMES, S, STUDY, has, is_degenerate, labels, point

HELD = S["amplification_templates"]
SETS = HELD + ["xstest_safe", "jbb_benign"]
CONDITIONS = [("none", 0)] + [("ca", s) for s in (-0.5, -1, -1.5, -2, -3)] + [("dim", s) for s in (-0.5, -1, -1.5, -2)]


def cached(m, d, s):
    """Per set: (degenerate, harmbench or None, llamaguard or None, refusal) labels."""
    out = {}
    for ds in SETS:
        p = point(m, "steering.strength=0", f"data.eval={ds}") if d == "none" else \
            point(m, f"direction.name={d}", f"steering.strength={s}", f"data.eval={ds}")
        out[ds] = (labels(p, "degeneration"), labels(p, "harmbench") if has(p, "harmbench") else None,
                   labels(p, "llamaguard") if has(p, "llamaguard") else None, labels(p, "refusal_keyword"))
    return out


def legacy(m, tag):
    out = {}
    for ds in SETS:
        it = json.load(open(FROZEN / "gen" / m / ds / f"{tag}.json"))["items"]
        out[ds] = ([is_degenerate(x["generation"]) for x in it], [bool(x["harmful"]) for x in it] if "harmful" in it[0] else None,
                   [bool(x.get("lg_unsafe")) for x in it] if "lg_unsafe" in it[0] else None, [bool(x["refusal"]) for x in it])
    return out


def row(sets):
    fluent = lambda ds, y: [v for v, g in zip(y, sets[ds][0]) if not g]
    hb = [np.mean(fluent(ds, sets[ds][1])) for ds in HELD if fluent(ds, sets[ds][1])]
    lg = [np.mean(fluent(ds, sets[ds][2])) for ds in HELD if sets[ds][2] is not None and fluent(ds, sets[ds][2])]
    r = {"hb": float(np.mean(hb)) if hb else None, "lg": float(np.mean(lg)) if lg else None}
    for ds, k in (("xstest_safe", "xstest"), ("jbb_benign", "benign")):
        f = fluent(ds, sets[ds][3])
        r[k] = float(np.mean(f)) if f else None
    r["degen"] = float(np.mean([np.mean(sets[ds][0]) for ds in SETS]))
    r["per_template_hb"] = {ds: float(np.mean(fluent(ds, sets[ds][1]))) for ds in HELD if fluent(ds, sets[ds][1])}
    return r


def compute():
    amp = {}
    for m in STUDY:
        amp[m] = {("none" if d == "none" else f"{d}/s{s:g}"): row(cached(m, d, s)) for d, s in CONDITIONS}
        amp[m]["dim/s-1/gated"] = row(legacy(m, "refusal_dim_s-1_gated"))
    return {"unit": "fraction of fluent outputs", "templates": HELD, "amplification": amp}


def render(r):
    g = lambda v: "  -- " if v is None else f"{v:.3f}"
    print("Fluent-only ASR over held-out templates (HB, LG); over-refusal (XSTest, JBB-benign); degeneration")
    for m, rows in r["amplification"].items():
        for c, v in rows.items():
            print(f"{NAMES['models'][m]:14s} {c:15s} HB {g(v['hb'])} LG {g(v['lg'])} XS {g(v['xstest'])} BN {g(v['benign'])} deg {v['degen']:.3f}")


if __name__ == "__main__":
    save("amplification", compute(), render)
