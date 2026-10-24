"""Per-prompt gates for the main figure: one model (study.gates_model), calibration draw 0, the
gate cos(h, u) of DIM and CA and ACE's shifted gate, on HarmBench and held-out Alpaca states.

    python -m pipeline.analysis.gates            # -> results/analysis/gates.{json,md}
"""

from pipeline.analysis._output import save
from pipeline.analysis.study import S, direction, refusal_states


def compute():
    m = S["gates_model"]
    st = refusal_states(m)
    D, C, A = direction(m, "dim"), direction(m, "ca"), direction(m, "ace")
    d, dc, c = D["u"] / D["u"].norm(), C["u"] / C["u"].norm(), float(A["c"])
    mu = st["harmless_calib"] / st["harmless_calib"].norm(dim=-1, keepdim=True)
    mu = mu.mean(0)
    mu = mu / mu.norm()
    out = {"unit": "cosine gate per prompt", "model": m, "omega": float((d @ mu).abs()), "ace_offset": c}
    for name, H in (("harmful", st["harmbench"]), ("benign", st["harmless_heldout"])):
        H = H.float()
        n = H.norm(dim=-1)
        g = {"dim": (H @ d) / n, "ca": (H @ dc) / n, "ace": ((H @ d) - c) / n}
        out[name] = {k: {"mean": float(v.mean()), "values": [round(float(x), 4) for x in v]} for k, v in g.items()}
    return out


def render(r):
    print(f"{r['model']}, draw 0: omega {r['omega']:.3f}, ACE offset {r['ace_offset']:.3f}")
    for name in ("harmful", "benign"):
        print(f"  mean gate on {name:7s} " + "  ".join(f"{k} {v['mean']:+.3f}" for k, v in r[name].items()))


if __name__ == "__main__":
    save("gates", compute(), render)
