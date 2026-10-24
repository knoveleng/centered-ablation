"""The second concept (RQ4): emotion removal.

    python -m pipeline.analysis.emotion          # -> results/analysis/emotion.{json,md}

emotion_diag          the generation-free diagnostics of all 32 model x emotion pairs (frozen:
                      emotion/diagnostics_all8.json, which needs the unembeddings to recompute)
emotion_strength      the strength sweep's per-cell table as reported (frozen: emotion/strength.md)
emotion_strength_agg  the sweep's averages over the study models x study.emotion_strength_emotions,
                      from the cache: <direction>/s<strength> -> relative change in emotion-word
                      rate, change in judged P(emotion) (points and ratio), degeneration mean / max
"""

import re

import numpy as np
from scipy import stats

from pipeline.analysis._output import save
from pipeline.analysis.study import MODELS, S, STUDY, completions, corr, frozen, labels, point, scores, unsteered
from pipeline.evaluators.emotion_judge import LEXICON


def diagnostics():
    ed = frozen("emotion/diagnostics_all8.json")
    t = lambda k: np.array([r[k] for r in ed])
    return {
        "n_pairs": len(ed),
        "pooled_omega_reduction": corr(t("omega"), t("other_dim") - t("other_dimc"))["spearman"],
        "beta_collateral": corr(np.r_[t("beta_dim"), t("beta_dimc")], np.r_[t("other_dim"), t("other_dimc")]),
        "dbeta_dcollateral": corr(t("beta_dim") - t("beta_dimc"), t("other_dim") - t("other_dimc")),
        "within_model_spearman": {m: float(stats.spearmanr([r["omega"] for r in ed if r["model"] == m],
                                                           [r["other_dim"] - r["other_dimc"] for r in ed if r["model"] == m])[0])
                                  for m in MODELS},
        "eta_range": [float(t("eta_dim").min()), float(t("eta_dim").max())],
        "beta_dimc_range": [float(t("beta_dimc").min()), float(t("beta_dimc").max())],
        "beta_dim_max": float(t("beta_dim").max()),
        "rows": ed,
    }


def strength_table():
    out, cur = [], None
    for line in frozen("emotion/strength.md").splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
        elif line.startswith("| ") and cur and not line.startswith("| E ") and not line.startswith("|---"):
            p = [c.strip() for c in line.strip("|").split("|")]
            try:
                out.append({"model": cur, "emotion": p[0], "cond": p[1], "s": int(p[2]), "pe": float(p[3]),
                            "vocab": float(p[4]), "degen": float(p[5]), "alpaca": None if p[6] in ("–", "-") else float(p[6])})
            except (ValueError, IndexError):
                pass
    return out


def strength_agg():
    lex = lambda texts, e: 100 * np.mean([len(re.findall(LEXICON[e], g.lower())) / max(1, len(g.split())) for g in texts])
    pe = lambda p, e: np.mean([x["emotion"] == e for x in scores(p, "emotion_judge")])
    agg = {}
    for c in ("dim", "ca"):
        for s in (1, 2, 3):
            dv, dp, dgs, pr = [], [], [], []
            for m in STUDY:
                for e in S["emotion_strength_emotions"]:
                    b = unsteered(m, f"ask_{e}", concept="emotion")
                    x = point(m, f"concept.target={e}", f"direction.name={c}", f"steering.strength={s}", f"data.eval=ask_{e}",
                              concept="emotion")
                    dv.append(lex(completions(x), e) / lex(completions(b), e) - 1)
                    dp.append(pe(x, e) - pe(b, e))
                    dgs.append(np.mean(labels(x, "degeneration")))
                    pr.append(pe(x, e) / pe(b, e))
            agg[f"{c}/s{s}"] = {"vocab_rel": float(np.mean(dv)), "pe_pts": float(np.mean(dp)) * 100,
                                "pe_rel": float(np.mean(pr)), "degen_mean": float(np.mean(dgs)), "degen_max": float(np.max(dgs))}
    return agg


def compute():
    return {"unit": "fractions unless named *_pts (points) or vocab (words per 100)",
            "emotion_diag": diagnostics(), "emotion_strength": strength_table(), "emotion_strength_agg": strength_agg()}


def render(r):
    d = r["emotion_diag"]
    print(f"{d['n_pairs']} model x emotion pairs: pooled Spearman(omega, collateral reduction) {d['pooled_omega_reduction']:.3f}; "
          f"eta {d['eta_range'][0]:.3f}-{d['eta_range'][1]:.3f}")
    print("\nStrength sweep (study models x happy/sad/afraid):")
    for k, v in r["emotion_strength_agg"].items():
        print(f"  {k:7s} words {100 * v['vocab_rel']:+5.0f}%  P(E) {v['pe_pts']:+5.1f} pts  degen max {v['degen_max']:.2f}")


if __name__ == "__main__":
    save("emotion", compute(), render)
