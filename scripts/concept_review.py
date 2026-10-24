#!/usr/bin/env python3
"""Build the review page with two drafts of the paper's main figure (real gate data, draw 0).

    python scripts/concept_review.py <out.html>     # reads results/analysis/{gates,main_matrix}.json
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.analysis._output import load  # noqa: E402

G = load("gates")
MM = load("main_matrix")
M = G["model"]
ASR = {k: MM["asr"][M][k]["hbf"]["mean"] for k in ("dim", "ace", "ca")}
FAITH = {k: MM["faith"][M][k] for k in ("dim", "ace", "ca")}

BLUE, BLUE_L, ORANGE, GRAY, INK, MUTED, RULE = "#2B6CB0", "#8FB8DE", "#D68A3C", "#7A8494", "#1F2733", "#5B6573", "#D9DEE5"
COL = {"dim": GRAY, "ace": ORANGE, "ca": BLUE}
NAME = {"dim": "DIM ablation", "ace": "ACE (Marshall et al.)", "ca": "CA (ours)"}
LO, HI = -0.45, 0.95


def hist(vals, x0, y0, w, h, color, bins=28, fill_op=0.85):
    """Histogram of gate values on [LO, HI], drawn upward from baseline y0+h; density-normalized per panel."""
    edges = np.linspace(LO, HI, bins + 1)
    c, _ = np.histogram(np.clip(vals, LO, HI - 1e-9), edges)
    c = c / c.max()
    bw = w / bins
    out = []
    for i, v in enumerate(c):
        if v == 0:
            continue
        bh = v * h
        out.append(f'<rect x="{x0 + i * bw + 0.4:.1f}" y="{y0 + h - bh:.1f}" width="{bw - 0.8:.1f}" height="{bh:.1f}" fill="{color}" opacity="{fill_op}"/>')
    return "".join(out)


def sx(v, x0, w):
    return x0 + (v - LO) / (HI - LO) * w


def gate_strip(k, x0, y0, w, h, label=True):
    """Two overlaid histograms (benign gray outline-ish, harmful in method color) with mean ticks and a zero line."""
    hv, bv = np.array(G["harmful"][k]["values"]), np.array(G["benign"][k]["values"])
    s = []
    z = sx(0, x0, w)
    s.append(f'<line x1="{z:.1f}" y1="{y0 - 2}" x2="{z:.1f}" y2="{y0 + h + 2}" stroke="{INK}" stroke-width="0.6" stroke-dasharray="2 2"/>')
    s.append(hist(bv, x0, y0, w, h, "#B8BFC9", fill_op=0.9))
    s.append(hist(hv, x0, y0, w, h, COL[k], fill_op=0.75))
    s.append(f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" stroke="{INK}" stroke-width="0.6"/>')
    for v, c, lab in ((bv.mean(), "#5B6573", "benign"), (hv.mean(), COL[k], "harmful")):
        x = sx(v, x0, w)
        s.append(f'<path d="M{x:.1f},{y0 + h + 1} l-3,5 h6 z" fill="{c}"/>')
        if label:
            s.append(f'<text x="{x:.1f}" y="{y0 + h + 14}" font-size="7.2" text-anchor="middle" fill="{c}">{v:+.2f}</text>')
    return "".join(s)


def axis(x0, y, w):
    s = []
    for t in (-0.4, 0, 0.4, 0.8):
        x = sx(t, x0, w)
        s.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y + 2.5}" stroke="{INK}" stroke-width="0.5"/>')
        s.append(f'<text x="{x:.1f}" y="{y + 10}" font-size="6.8" text-anchor="middle" fill="{MUTED}">{t:g}</text>')
    return "".join(s)


def arrow_defs():
    return "".join(f'<marker id="ah-{n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" fill="{c}"/></marker>'
                   for n, c in (("ink", INK), ("gray", GRAY), ("blue", BLUE), ("orange", ORANGE), ("muted", MUTED)))


def geometry(x0, y0, scale=1.0, show_ace=True):
    """Origin, benign/harmful clouds, common mode, DIM, CA, and ACE's moved origin."""
    S = scale
    ox, oy = x0 + 14 * S, y0 + 118 * S
    bx, by = x0 + 150 * S, y0 + 34 * S        # benign mean
    hx, hy = x0 + 186 * S, y0 + 74 * S        # harmful mean
    s = [f'<ellipse cx="{bx}" cy="{by}" rx="{30 * S}" ry="{15 * S}" fill="#D7DCE3"/>',
         f'<ellipse cx="{hx}" cy="{hy}" rx="{26 * S}" ry="{13 * S}" fill="{BLUE_L}" opacity="0.55"/>',
         f'<text x="{bx - 4 * S}" y="{by + 26 * S}" font-size="7.5" text-anchor="middle" fill="{MUTED}">benign states</text>',
         f'<text x="{hx + 4 * S}" y="{hy + 25 * S}" font-size="7.5" text-anchor="middle" fill="{BLUE}">harmful states</text>',
         f'<circle cx="{ox}" cy="{oy}" r="2.2" fill="{INK}"/>',
         f'<text x="{ox - 4}" y="{oy + 12}" font-size="7.5" fill="{INK}">origin</text>',
         f'<line x1="{ox}" y1="{oy}" x2="{bx - 3}" y2="{by + 2}" stroke="{MUTED}" stroke-width="1" marker-end="url(#ah-muted)"/>',
         f'<text x="{ox + 40 * S}" y="{oy - 58 * S}" font-size="8" fill="{MUTED}" font-style="italic">μ̂₋ (common mode)</text>']
    # DIM: from origin, partly along the common mode
    dx, dy = ox + 104 * S, oy - 26 * S
    s.append(f'<line x1="{ox}" y1="{oy}" x2="{dx}" y2="{dy}" stroke="{GRAY}" stroke-width="2" marker-end="url(#ah-gray)"/>')
    s.append(f'<text x="{dx + 4}" y="{dy + 3}" font-size="8" fill="{GRAY}" font-weight="600">DIM</text>')
    # CA: orthogonal to common mode
    cx, cy = ox + 52 * S, oy + 6 * S
    s.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + 64 * S}" y2="{oy - 3 * S}" stroke="{BLUE}" stroke-width="2.2" marker-end="url(#ah-blue)"/>')
    s.append(f'<text x="{ox + 66 * S}" y="{oy + 9 * S}" font-size="8" fill="{BLUE}" font-weight="700">CA: u ⟂ μ̂₋</text>')
    if show_ace:
        # ACE: same DIM direction, origin moved to the benign mean
        s.append(f'<circle cx="{bx}" cy="{by}" r="2.6" fill="white" stroke="{ORANGE}" stroke-width="1.3"/>')
        s.append(f'<line x1="{bx}" y1="{by}" x2="{bx + 46 * S}" y2="{by - 22 * S}" stroke="{ORANGE}" stroke-width="2" marker-end="url(#ah-orange)"/>')
        s.append(f'<text x="{bx - 6 * S}" y="{by - 26 * S}" font-size="8" fill="{ORANGE}" font-weight="600">ACE: DIM from μ₋</text>')
    return "".join(s)


def reduction_box(x0, y0, w):
    """h -> gate x footprint -> biased logits, with the equation."""
    s = []
    def box(x, y, bw, bh, t1, t2, fill, stroke):
        return (f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>'
                f'<text x="{x + bw / 2}" y="{y + 13}" font-size="8" text-anchor="middle" fill="{INK}" font-weight="600">{t1}</text>'
                f'<text x="{x + bw / 2}" y="{y + 25}" font-size="7.8" text-anchor="middle" fill="{INK}" font-style="italic">{t2}</text>')
    s.append(box(x0, y0, 40, 32, "state", "h", "#F1F3F6", RULE))
    s.append(box(x0 + 58, y0, 78, 32, "gate (scalar)", "κ = cos(h, u)", "#E8F0FA", BLUE_L))
    s.append(box(x0 + 58, y0 + 46, 78, 32, "footprint (fixed)", "u = W_U G u ∈ ℝ^V", "#E8F0FA", BLUE_L))
    s.append(box(x0 + 150, y0 + 23, w - 150, 32, "logits", "ℓ(h) − s√d κ u", "#F1F3F6", RULE))
    s.append(f'<line x1="{x0 + 40}" y1="{y0 + 16}" x2="{x0 + 57}" y2="{y0 + 16}" stroke="{INK}" stroke-width="0.8" marker-end="url(#ah-ink)"/>')
    s.append(f'<line x1="{x0 + 136}" y1="{y0 + 16}" x2="{x0 + 149}" y2="{y0 + 34}" stroke="{INK}" stroke-width="0.8" marker-end="url(#ah-ink)"/>')
    s.append(f'<line x1="{x0 + 136}" y1="{y0 + 62}" x2="{x0 + 149}" y2="{y0 + 46}" stroke="{INK}" stroke-width="0.8" marker-end="url(#ah-ink)"/>')
    return "".join(s)


def svg(w, h, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="STIX Two Text, Times New Roman, serif" '
            f'role="img"><defs>{arrow_defs()}</defs><rect width="{w}" height="{h}" fill="white"/>{body}</svg>')


# ------------------------------------------------------------------ Version A: pipeline (geometry -> reduction -> outcome)
def version_a():
    W, H = 675, 210
    b = []
    b.append(f'<text x="6" y="14" font-size="9.5" font-weight="700" fill="{INK}">(a) Where the gate is measured from</text>')
    b.append(geometry(4, 22))
    b.append(f'<text x="236" y="14" font-size="9.5" font-weight="700" fill="{INK}">(b) Ablation is a gated logit bias</text>')
    b.append(reduction_box(236, 34, 214))
    b.append(f'<text x="236" y="134" font-size="7.6" fill="{MUTED}">Benign text is untouched iff κ ≈ 0.</text>')
    b.append(f'<text x="236" y="146" font-size="7.6" fill="{MUTED}">s√d·|κ|·sd(u) &gt; logit spread ⇒ collapse (β &gt; 1).</text>')
    b.append(f'<text x="236" y="158" font-size="7.6" fill="{MUTED}">κ ≈ 0 on a behaviour ⇒ nothing removes it.</text>')
    x0 = 470
    b.append(f'<text x="{x0}" y="14" font-size="9.5" font-weight="700" fill="{INK}">(c) Llama-3.2-3B: gate on each prompt</text>')
    rows = [("dim", 30), ("ace", 82), ("ca", 134)]
    for k, y in rows:
        b.append(f'<text x="{x0}" y="{y - 3}" font-size="7.8" font-weight="600" fill="{COL[k]}">{NAME[k]}</text>')
        b.append(gate_strip(k, x0, y, 130, 24))
        b.append(f'<text x="{x0 + 140}" y="{y + 8}" font-size="7.4" fill="{INK}">ASR {ASR[k]:.2f}</text>')
        b.append(f'<text x="{x0 + 140}" y="{y + 19}" font-size="7.4" fill="{INK}">rewrites {FAITH[k]:.2f}</text>')
    b.append(axis(x0, 184, 130))
    b.append(f'<text x="{x0 + 65}" y="203" font-size="7.4" text-anchor="middle" fill="{MUTED}">gate κ  (gray: benign, color: harmful)</text>')
    return svg(W, H, "".join(b))


# ------------------------------------------------------------------ Version B: contrast (one row per method)
def version_b():
    W, H = 675, 268
    b = []
    b.append(f'<rect x="0" y="0" width="{W}" height="40" fill="#F4F6F9"/>')
    b.append(f'<text x="10" y="17" font-size="9.5" font-weight="700" fill="{INK}">Last-layer ablation = gated logit bias:</text>')
    b.append(f'<text x="200" y="17" font-size="10" fill="{INK}" font-style="italic">ℓ(Aₛᵘh) ∝ ℓ(h) − s√d · κᵤ(h) · u ,   κᵤ(h) = cos(h, u),   u = W_U G u</text>')
    b.append(f'<text x="10" y="32" font-size="7.8" fill="{MUTED}">The gate κ decides how hard the edit fires; the footprint u decides which tokens move. Because κ is a cosine, it is measured from the origin.</text>')
    cols = [(10, "direction and origin"), (170, "gate on 240 harmful / 160 benign prompts"), (430, "outcome (6 draws)")]
    for x, t in cols:
        b.append(f'<text x="{x}" y="56" font-size="7.6" fill="{MUTED}" letter-spacing="0.4">{t.upper()}</text>')
    rows = [("dim", 66, "fires on benign text (−0.28), weak on harmful (0.32)"),
            ("ace", 126, "subtracts an offset: benign mean 0, spread kept"),
            ("ca", 186, "removes the common mode: benign mean 0, tight")]
    for k, y, note in rows:
        b.append(f'<line x1="6" y1="{y - 4}" x2="{W - 6}" y2="{y - 4}" stroke="{RULE}" stroke-width="0.6"/>')
        b.append(f'<text x="10" y="{y + 10}" font-size="9" font-weight="700" fill="{COL[k]}">{NAME[k]}</text>')
        glyph = {"dim": "u = DIM, origin 0", "ace": "u = DIM, origin μ₋", "ca": "u = DIM ⟂ μ̂₋, origin 0"}[k]
        b.append(f'<text x="10" y="{y + 23}" font-size="7.8" fill="{INK}" font-style="italic">{glyph}</text>')
        b.append(f'<text x="10" y="{y + 36}" font-size="7.2" fill="{MUTED}">{note}</text>')
        b.append(gate_strip(k, 170, y + 2, 230, 30))
        # outcome bars
        for j, (lab, v, good_low) in enumerate((("attack success", ASR[k], False), ("benign rewritten", FAITH[k], True))):
            yy = y + 6 + j * 18
            b.append(f'<text x="430" y="{yy + 8}" font-size="7.4" fill="{INK}">{lab}</text>')
            b.append(f'<rect x="505" y="{yy}" width="120" height="9" fill="#EEF1F5"/>')
            b.append(f'<rect x="505" y="{yy}" width="{120 * v:.1f}" height="9" fill="{COL[k]}" opacity="{0.95 if not good_low else 0.5}"/>')
            b.append(f'<text x="{630}" y="{yy + 8}" font-size="7.4" fill="{INK}">{v:.2f}</text>')
    b.append(axis(170, 240, 230))
    b.append(f'<text x="285" y="261" font-size="7.4" text-anchor="middle" fill="{MUTED}">gate κ  (gray: benign prompts; color: harmful prompts; ▲ = mean)</text>')
    return svg(W, H, "".join(b))


page = f"""<title>Main Figure Drafts</title>
<style>
:root {{ --bg:#F3F5F8; --fg:#1F2733; --muted:#5B6573; --card:#FFFFFF; --rule:#D9DEE5; --accent:#2B6CB0;
  --serif:"Source Serif 4", Georgia, serif; --sans:"IBM Plex Sans", system-ui, sans-serif; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#14181E; --fg:#E6EAF0; --muted:#A3ACB9; --card:#1D232B; --rule:#2E3640; --accent:#8FB8DE; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#14181E; --fg:#E6EAF0; --muted:#A3ACB9; --card:#1D232B; --rule:#2E3640; --accent:#8FB8DE; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--sans); }}
main {{ max-width:1080px; margin:0 auto; padding-inline:16px; padding-block:28px 48px; display:grid; gap:28px; }}
h1 {{ font-family:var(--serif); font-weight:600; font-size:1.7rem; margin:0; text-wrap:balance; }}
p {{ margin:0; line-height:1.55; max-width:72ch; color:var(--muted); }}
section {{ display:grid; gap:12px; }}
h2 {{ font-family:var(--serif); font-weight:600; font-size:1.2rem; margin:0; }}
.tag {{ font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; color:var(--accent); font-weight:600; }}
.paper {{ background:#fff; border:1px solid var(--rule); border-radius:6px; padding:14px; overflow-x:auto; }}
.paper svg {{ display:block; width:100%; min-width:640px; height:auto; }}
ul {{ margin:0; padding-left:1.1rem; color:var(--muted); line-height:1.55; max-width:72ch; }}
</style>
<main>
  <header style="display:grid;gap:8px">
    <div class="tag">Figure 2 review · data: Llama-3.2-3B, calibration draw 0</div>
    <h1>Main figure: two drafts</h1>
    <p>Both drafts use the real per-prompt gates of the worked example (240 HarmBench and 160 held-out Alpaca prompt-final states). They differ in one decision: Draft A is organized around the mechanism, Draft B around the contrast between the three methods. Both place ACE as the offset route to the same correction, credited to its authors.</p>
  </header>
  <section>
    <div class="tag">Draft A</div><h2>Pipeline: geometry → reduction → what it does on real prompts</h2>
    <div class="paper">{version_a()}</div>
    <ul><li>Reads left to right like the method section; the reduction is a diagram.</li><li>The data panel is compact; the ACE-vs-CA spread difference is visible but small.</li></ul>
  </section>
  <section>
    <div class="tag">Draft B</div><h2>Contrast: one row per method, same prompts</h2>
    <div class="paper">{version_b()}</div>
    <ul><li>The equation sits on top as the frame; each row answers “where is the origin, what does the gate do, what does it buy”.</li><li>The key contrast (DIM fires on benign text; ACE and CA both centre it; CA is tighter, ACE fires harder) gets most of the width.</li></ul>
  </section>
</main>
"""
Path(sys.argv[1]).write_text(page)
print("written", sys.argv[1])
