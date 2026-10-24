"""The two routes of origin correction (Proposition: family), all 8 models, draw 0: ACE's
harmful-gate ratio over CA against 1/sqrt(1 - ω²), and the benign gate spread with its
common-mode decomposition; joined with the ACE - CA outcome gaps of the main matrix.

    python -m pipeline.analysis.family           # needs main_matrix -> results/analysis/family.{json,md}
"""

import numpy as np
from scipy import stats

from pipeline.analysis._output import load, save
from pipeline.analysis.study import MODELS, NAMES, direction, refusal_states


def compute():
    mm = load("main_matrix")
    rows = []
    for m in MODELS:
        st = refusal_states(m)
        D, C, A = direction(m, "dim"), direction(m, "ca"), direction(m, "ace")
        d, dc, c = D["u"].float(), C["u"].float(), float(A["c"])
        d, dc = d / d.norm(), dc / dc.norm()
        Hc = st["harmless_calib"].float()
        mu = (Hc / Hc.norm(dim=-1, keepdim=True)).mean(0)
        mu = mu / mu.norm()
        if float(d @ mu) < 0:
            mu = -mu
        w = float(d @ mu)
        Hh, Hb = st["harmbench"].float(), st["harmless_heldout"].float()
        gate = lambda H, u, off=0.0: ((H @ u) - off) / H.norm(dim=-1)
        g_ace_h, g_ca_h = gate(Hh, d, c).mean().item(), gate(Hh, dc).mean().item()
        g_ace_b, g_ca_b = gate(Hb, d, c), gate(Hb, dc)
        a = Hb @ mu   # benign states h = a*mu + r'
        rows.append({"model": m, "omega": w, "pred_ratio": 1 / np.sqrt(1 - w * w), "gate_ratio": g_ace_h / g_ca_h,
                     "gate_harm_ace": g_ace_h, "gate_harm_ca": g_ca_h,
                     "benign_sd_ace": float(g_ace_b.std()), "benign_sd_ca": float(g_ca_b.std()),
                     "benign_mean_ace": float(g_ace_b.mean()), "benign_mean_ca": float(g_ca_b.mean()),
                     "var_a": float(a.var()), "var_r": float((Hb @ dc).var()),
                     "ace_excess_rewrite": mm["faith"][m]["ace"] - mm["faith"][m]["ca"],
                     "asr_ace_minus_ca": mm["asr"][m]["ace"]["hbf"]["mean"] - mm["asr"][m]["ca"]["hbf"]["mean"]})
    t = lambda k: np.array([r[k] for r in rows])
    return {"unit": "gates are cosines; gaps are fractions", "n": len(rows), "rows": rows,
            "ratio_corr": stats.pearsonr(t("pred_ratio"), t("gate_ratio"))[0],
            "ratio_max_abs_err": float(np.max(np.abs(t("pred_ratio") - t("gate_ratio")))),
            "sd_gap_vs_rewrite": [stats.pearsonr(t("benign_sd_ace") - t("benign_sd_ca"), t("ace_excess_rewrite"))[0],
                                  stats.spearmanr(t("benign_sd_ace") - t("benign_sd_ca"), t("ace_excess_rewrite"))[0]],
            "ratio_vs_asr_gap": [stats.pearsonr(t("gate_ratio"), t("asr_ace_minus_ca"))[0],
                                 stats.spearmanr(t("gate_ratio"), t("asr_ace_minus_ca"))[0]]}


def render(r):
    print(f"gate ratio ACE/CA vs 1/sqrt(1-omega^2): Pearson {r['ratio_corr']:.3f}, max error {r['ratio_max_abs_err']:.3f}")
    for x in r["rows"]:
        print(f"  {NAMES['models'][x['model']]:14s} omega {x['omega']:.3f}  ratio {x['pred_ratio']:.3f} / {x['gate_ratio']:.3f}  "
              f"benign sd ACE {x['benign_sd_ace']:.3f} CA {x['benign_sd_ca']:.3f}")


if __name__ == "__main__":
    save("family", compute(), render)
