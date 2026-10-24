"""The two laws (RQ2): ω predicts CA's faithfulness gain over DIM, DIM's η predicts its attack
success gain; plus the locked out-of-sample prediction's simulated token changes.

    python -m pipeline.analysis.laws             # needs main_matrix -> results/analysis/laws.{json,md}

ω, η: frozen inputs (study.omega_eta). Gains are CA minus DIM, fractions.
"""

from pipeline.analysis._output import load, save
from pipeline.analysis.study import MODELS, NAMES, corr, frozen, omega_eta

ID = {"dimc": "ca"}   # the locked file's direction names -> config ids


def compute():
    mm = load("main_matrix")
    asr, faith = mm["asr"], mm["faith"]
    omega, eta = omega_eta()
    om = [omega[m] for m in MODELS]
    gain_hb = [asr[m]["ca"]["hb"]["mean"] - asr[m]["dim"]["hb"]["mean"] for m in MODELS]
    gain_lgf = [asr[m]["ca"]["lgf"]["mean"] - asr[m]["dim"]["lgf"]["mean"] for m in MODELS]
    dfaith = [faith[m]["dim"] - faith[m]["ca"] for m in MODELS]
    ace_excess = [faith[m]["ace"] - faith[m]["ca"] for m in MODELS]
    locked = frozen("ladder_prediction.locked.json")
    laws = {
        "omega": omega, "eta": eta,
        "faithfulness_law": corr(om, dfaith),
        "efficacy_law_hb": corr([eta[m] for m in MODELS], gain_hb),
        "efficacy_law_lgf": corr([eta[m] for m in MODELS], gain_lgf),
        "omega_vs_asr_gain": corr(om, gain_hb),
        "ace_excess_vs_omega": corr(om, ace_excess),
        "gain_hb": dict(zip(MODELS, gain_hb)), "dfaith": dict(zip(MODELS, dfaith)), "ace_excess": dict(zip(MODELS, ace_excess)),
    }
    return {"unit": "fractions; correlations over the 8 models", "n": len(MODELS), "laws": laws,
            "locked_sim": {m: {ID.get(c, c): v for c, v in locked[m]["sim"].items()} for m in locked},
            "locked_regime": {m: locked[m]["predicted_regime"] for m in locked}}


def render(r):
    L = r["laws"]
    for k in ("faithfulness_law", "efficacy_law_hb", "efficacy_law_lgf", "omega_vs_asr_gain", "ace_excess_vs_omega"):
        print(f"{k:22s} Pearson {L[k]['pearson']:+.3f}  Spearman {L[k]['spearman']:+.3f}")
    print(f"\n{'model':14s} {'omega':>6s} {'eta':>6s} {'faith gain':>10s} {'ASR gain':>9s}")
    for m in MODELS:
        print(f"{NAMES['models'][m]:14s} {L['omega'][m]:6.3f} {L['eta'][m]:6.3f} {L['dfaith'][m]:+10.3f} {L['gain_hb'][m]:+9.3f}")


if __name__ == "__main__":
    save("laws", compute(), render)
