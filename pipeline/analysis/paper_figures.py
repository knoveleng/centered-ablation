"""Draw every data figure of the paper from results/analysis/*.json.

    python -m pipeline.analysis.paper_figures    # PDFs -> paper/figures/, PNG previews -> results/previews/,
                                                 # what was plotted -> results/analysis/figure_data.json

Layout only: every plotted value comes from an analysis file through load(). Rates are plotted in percent and
differences in points (the data files hold fractions; the conversion happens here, at display time).

Colour means method everywhere: gray = DIM ablation, orange = DIM+offset, blue = CA (ours). Where one method is
shown on two prompt sets, the fill means the set (hollow / hatched = benign, filled = harmful).
"""
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.analysis._output import OUT, ROOT, jsonable, load
from pipeline.analysis.study import MODELS, NAMES

FIG = ROOT / "paper" / "figures"
PREV = ROOT / "results" / "previews"
FIG.mkdir(parents=True, exist_ok=True); PREV.mkdir(parents=True, exist_ok=True)
MM, LAWS, COL = load("main_matrix"), load("laws"), load("collapse")

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "legend.fontsize": 6.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "pdf.fonttype": 42, "hatch.linewidth": 0.5,
})
OURS, OFFSET, DIM, TEXT, GRID, NEG = "#2B6CB0", "#D68A3C", "#8A93A2", "#4A5568", "#E6EAF0", "#B5473A"
COND = {"dim": ("DIM ablation", DIM), "ca": ("CA (ours)", OURS), "ace": ("DIM+offset", OFFSET)}
# Fig. (RQ4) only: colour = model, hues grouped by family (Qwen greens, Llama purples, Gemma reds)
MCOL = {"qwen2.5-0.5b": "#9DBF3A", "qwen3-4b": "#4FB477", "qwen2.5-7b": "#1F7A5C", "qwen2.5-3b": "#2BA3A0",
        "llama-3.1-8b": "#5E3C99", "llama-3.2-3b": "#B07CC6", "gemma-2-2b": "#E8926B", "gemma-2-9b": "#B5473A"}
MSHAPE = {"dim": "o", "ace": "s", "ca": "^"}
SHORT = NAMES["models_short"]
L = LAWS["laws"]
OM, ETA = L["omega"], L["eta"]
pct = lambda v: 100 * v
PLOTTED = {}


def save(fig, name, width, height=None):
    """Write the PDF so that its tight bounding box is exactly `width` (and, if given, `height`) inches:
    figures that sit side by side in the one-column paper are made the same height here, not by scaling."""
    for _ in range(6):
        buf = io.BytesIO(); fig.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0.02)
        bb = fig.get_tightbbox(fig.canvas.get_renderer()); w, h = bb.width + 0.04, bb.height + 0.04
        if abs(w - width) < 0.005 and (height is None or abs(h - height) < 0.005):
            break
        fw, fh = fig.get_size_inches(); fig.set_size_inches(fw * width / w, fh if height is None else fh * height / h)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PREV / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.5); ax.set_axisbelow(True)


def ylabels_by_omega(ax, y):
    ax.set_yticks(y); ax.set_yticklabels([f"{SHORT[m]}  " + f"{OM[m]:.2f}"[1:] for m in MODELS], fontsize=6.0)
    ax.tick_params(axis="y", length=0)


def label(ax, text, xy, dx, dy, ha="left", color=TEXT, size=5.6, va="center"):
    ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points", fontsize=size, color=color, ha=ha, va=va)


def law_points(ax, xs, ys):
    """One point per model, in the model's colour (MCOL); names come from the figure legend."""
    for m, x, y in zip(MODELS, xs, ys):
        ax.scatter([x], [y], s=22, marker="o", color=MCOL[m], edgecolor="white", lw=0.5, zorder=4)


def fit_line(ax, xs, ys, color=TEXT, pad=0.08):
    """Least-squares line with its 95% confidence band for the mean (t with n-2 degrees of freedom)."""
    from scipy import stats
    x, yv = np.asarray(xs, float), np.asarray(ys, float)
    n = len(x); slope, icpt = np.polyfit(x, yv, 1)
    resid = yv - (slope * x + icpt); s2 = resid @ resid / (n - 2)
    lo, hi = x.min(), x.max(); span = hi - lo
    g = np.linspace(lo - pad * span, hi + pad * span, 100)
    se = np.sqrt(s2 * (1 / n + (g - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum()))
    t = stats.t.ppf(0.975, n - 2)
    ax.fill_between(g, slope * g + icpt - t * se, slope * g + icpt + t * se, color=color, alpha=0.10, lw=0, zorder=1)
    ax.plot(g, slope * g + icpt, color=color, lw=0.9, zorder=2)
    return float(slope), float(icpt)


A, F = MM["asr"], MM["faith"]
y = np.arange(len(MODELS))[::-1]          # smallest omega at the top

# ---------------------------------------------------------------- Fig. 1 teaser: what CA changes, DIM -> CA, per model
fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.25, 1.8), gridspec_kw={"wspace": 0.12}, sharey=True)
for ax, key, title, xlab in ((a1, "asr", "(a) CA fixes under-firing", "ASR (%), HarmBench, fluent-only"),
                             (a2, "faith", "(b) CA fixes collateral", "benign rewrite rate (%), Alpaca")):
    for m, yy in zip(MODELS, y):
        if key == "asr":
            d, c, base = pct(A[m]["dim"]["hbf"]["mean"]), pct(A[m]["ca"]["hbf"]["mean"]), pct(A[m]["none"]["hbf"]["mean"])
            ax.plot([base], [yy], marker="|", ms=5, color="black", mew=0.8, zorder=2)
        else:
            d, c = pct(F[m]["dim"]), pct(F[m]["ca"])
        ax.plot([d, c], [yy, yy], color=OURS if abs(c - d) >= 10 else "#C5CBD5", lw=1.3, zorder=1, solid_capstyle="round")
        ax.scatter([d], [yy], s=16, color=DIM, zorder=3)
        ax.scatter([c], [yy], s=16, color=OURS, zorder=4)
        if abs(c - d) >= 10:
            ax.annotate(f"{c - d:+.0f}", ((c + d) / 2, yy), xytext=(0, 4.2), textcoords="offset points", ha="center", fontsize=5.6, color=OURS, fontweight="bold")
    ax.set_title(title, loc="left", pad=3, fontsize=6.8)
    ax.set_xlabel(xlab, fontsize=6.3); grid(ax, "x")
    ax.spines["left"].set_visible(False)
ylabels_by_omega(a1, y)
a1.set_xlim(0, 80); a2.set_xlim(0, 100)
a1.text(-0.02, 1.0, r"model   $\omega$", transform=a1.transAxes, fontsize=5.6, color=TEXT, ha="right", va="bottom")
g9 = y[MODELS.index("gemma-2-9b")]
a1.annotate("collapse 50% " + r"$\to$" + " 1%", (pct(A["gemma-2-9b"]["ca"]["hbf"]["mean"]), g9), xytext=(8, 0), textcoords="offset points",
            fontsize=5.4, color=TEXT, va="center")
h = [plt.Line2D([], [], marker="o", ls="", mfc=DIM, mec=DIM, ms=3.8, label="DIM ablation (standard)"),
     plt.Line2D([], [], marker="o", ls="", mfc=OURS, mec=OURS, ms=3.8, label="CA (ours)"),
     plt.Line2D([], [], marker="|", ls="", color="black", ms=5, mew=0.8, label="unsteered")]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.55, 0.99), ncol=3, frameon=False, handletextpad=0.2, columnspacing=0.9, fontsize=6.0)
save(fig, "fig_teaser", 3.25)
PLOTTED["fig_teaser"] = {"asr_pct": {m: {c: pct(A[m][c]["hbf"]["mean"]) for c in ["none", "dim", "ca"]} for m in MODELS},
                         "rewrite_pct": {m: {c: pct(F[m][c]) for c in ["dim", "ca"]} for m in MODELS}, "omega": OM}

# ---------------------------------------------------------------- Fig. (RQ1): the closed form predicts the served first token
RED = load("reduction")
agree = [pct(RED["per_model"][m]) for m in MODELS]
fig, (v1, v2) = plt.subplots(1, 2, figsize=(3.25, 1.7), gridspec_kw={"wspace": 0.55, "width_ratios": [1.0, 1.0]})
# (a) one point per cell: refusal rate predicted from cached states vs. the served model's refusal rate
v1.plot([0, 100], [0, 100], color=TEXT, lw=0.7, ls=(0, (3, 1.6)), zorder=1)
for cnd in ["dim", "ace", "ca"]:
    pts_ = [c for c in RED["cells"] if c["cond"] == cnd]
    v1.scatter([pct(c["predicted_refusal"]) for c in pts_], [pct(c["served_refusal"]) for c in pts_], s=11, color=COND[cnd][1],
               edgecolor="white", lw=0.3, alpha=0.9, zorder=3, label=COND[cnd][0])
g9 = [c for c in RED["cells"] if c["model"] == "gemma-2-9b"]
v1.scatter([pct(c["predicted_refusal"]) for c in g9], [pct(c["served_refusal"]) for c in g9], s=13, facecolor=[COND[c["cond"]][1] for c in g9],
           edgecolor=NEG, lw=0.8, zorder=4)
v1.scatter([], [], s=13, facecolor="white", edgecolor=NEG, lw=0.8, label="G-9B (RQ5)")
v1.set_xlabel("predicted refusal rate (%)"); v1.set_ylabel("served refusal rate (%)")
v1.set_xlim(-3, 95); v1.set_ylim(-3, 95); v1.set_xticks([0, 50]); v1.set_yticks([0, 50]); v1.set_aspect("equal")
v1.text(0.97, 0.04, f"Pearson {RED['pearson']:.2f}", transform=v1.transAxes, va="bottom", ha="right", fontsize=6, color=TEXT)
v1.legend(loc="upper left", frameon=False, fontsize=5.0, handletextpad=0.1, borderaxespad=0.0, labelspacing=0.15, bbox_to_anchor=(-0.02, 1.02), markerscale=1.1)
grid(v1)
# (b) per-prompt agreement of the predicted and served first token, one point per model
for m, yy, a in zip(MODELS, y, agree):
    v2.plot([75, a], [yy, yy], color=GRID, lw=1.0, zorder=1)
    v2.scatter([a], [yy], s=16, color=NEG if a < 90 else OURS, zorder=3)
    v2.text(101.3, yy, f"{a:.1f}", va="center", ha="left", fontsize=5.4, color=NEG if a < 90 else TEXT, clip_on=False)
v2.axvline(pct(RED["agreement"]), color=TEXT, lw=0.7, ls=(0, (3, 1.6)), zorder=2)
v2.text(pct(RED["agreement"]) - 0.6, y[0] + 0.65, f"all cells: {pct(RED['agreement']):.1f}%", ha="right", va="bottom", fontsize=5.4, color=TEXT)
ylabels_by_omega(v2, y); v2.spines["left"].set_visible(False)
v2.set_xlim(75, 100.6); v2.set_xticks([80, 90, 100]); v2.set_ylim(-0.6, len(MODELS) - 0.4 + 0.8)
v2.set_xlabel("first-token agreement (%)")
# titles on one baseline: (a) is square (set_aspect) and so shorter than (b); anchor both to the top of (b)
for ax, t in ((v1, "(a) Refusal rate, per cell"), (v2, "(b) Agreement, per model")):
    ax.annotate(t, xy=(0, 1), xycoords=("axes fraction", v2.transAxes), xytext=(0, 3),
                textcoords="offset points", ha="left", va="bottom", fontsize=plt.rcParams["axes.titlesize"])
v2.text(-0.02, 1.0, r"model   $\omega$", transform=v2.transAxes, fontsize=5.6, color=TEXT, ha="right", va="bottom")
grid(v2, "x")
save(fig, "fig_rq1", 3.25)
PLOTTED["fig_rq1"] = {"agreement_pct": dict(zip(MODELS, agree)), "all_pct": pct(RED["agreement"]), "pearson": RED["pearson"],
                      "cells": RED["cells"]}

# ---------------------------------------------------------------- Fig. (RQ2): two routes, gains over DIM
fig, (b1, b2) = plt.subplots(1, 2, figsize=(3.15, 1.95), gridspec_kw={"wspace": 0.12}, sharey=True)
w = 0.36
gain_asr = {c: [pct(A[m][c]["hbf"]["mean"] - A[m]["dim"]["hbf"]["mean"]) for m in MODELS] for c in ("ca", "ace")}
gain_fai = {c: [pct(F[m]["dim"] - F[m][c]) for m in MODELS] for c in ("ca", "ace")}
for ax, g, title, xlab in ((b1, gain_asr, "(a) ASR gain over DIM", "ASR gain (points)"),
                           (b2, gain_fai, "(b) Rewrites avoided vs. DIM", "rewrites avoided (points)")):
    ax.barh(y + w / 2, g["ace"], w * 0.92, color=OFFSET, label="DIM+offset", lw=0)
    ax.barh(y - w / 2, g["ca"], w * 0.92, color=OURS, label="CA (ours)", lw=0)
    ax.axvline(0, color=TEXT, lw=0.6)
    ax.set_title(title, loc="left", pad=3, fontsize=6.8); ax.set_xlabel(xlab, fontsize=6.3); grid(ax, "x")
    ax.spines["left"].set_visible(False)
ylabels_by_omega(b1, y)
b1.text(-0.02, 1.0, r"model   $\omega$", transform=b1.transAxes, fontsize=5.6, color=TEXT, ha="right", va="bottom")
b2.annotate("", xy=(-46, y[MODELS.index("qwen2.5-0.5b")] + 0.05), xytext=(-8, y[MODELS.index("qwen2.5-0.5b")] + 0.05), arrowprops=dict(arrowstyle="->", color=TEXT, lw=0.6))
b2.text(-47, y[MODELS.index("qwen2.5-0.5b")] - 0.25, "more rewrites\nthan DIM", fontsize=5.2, color=TEXT, ha="left", va="top", linespacing=1.0)
b1.set_xlim(-5, 62); b2.set_xlim(-50, 45)
fig.legend(*b1.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.55, 0.99), ncol=2, frameon=False, columnspacing=1.0, fontsize=6.0)
save(fig, "fig_rq2", 3.15, 2.05)      # paired with fig_routes
PLOTTED["fig_rq2"] = {"asr_gain_pts": {c: dict(zip(MODELS, v)) for c, v in gain_asr.items()},
                      "rewrites_avoided_pts": {c: dict(zip(MODELS, v)) for c, v in gain_fai.items()}}

# ---------------------------------------------------------------- Fig. (RQ3): the two routes on prompt-final states
FAM = {r["model"]: r for r in load("family")["rows"]}
fig, (r1, r2) = plt.subplots(1, 2, figsize=(3.15, 1.8), gridspec_kw={"wspace": 0.62, "width_ratios": [1.0, 1.12]})
oms = np.linspace(0, 0.5, 200)
r1.plot(oms, 1 / np.sqrt(1 - oms ** 2), color=TEXT, lw=1.0, ls=(0, (3, 1.6)), zorder=2, label=r"predicted $1/\sqrt{1-\omega^2}$")
r1.scatter([FAM[m]["omega"] for m in MODELS], [FAM[m]["gate_ratio"] for m in MODELS], s=18, color=OFFSET, edgecolor="white", lw=0.5, zorder=3, label="measured")
for m, dx, dy, ha in (("gemma-2-9b", 2, -5, "right"), ("llama-3.2-3b", -4, 2, "right"), ("llama-3.1-8b", 4, -2, "left"), ("gemma-2-2b", -4, 3, "right")):
    label(r1, SHORT[m], (FAM[m]["omega"], FAM[m]["gate_ratio"]), dx, dy, ha)
r1.set_xlabel(r"overlap $\omega$"); r1.set_ylabel("harmful gate, DIM+offset / CA")
r1.set_title("(a) Harmful gate ratio", loc="left", pad=3)
r1.set_xlim(-0.02, 0.52); r1.set_ylim(0.98, 1.22); r1.set_yticks([1.0, 1.1, 1.2])
r1.legend(loc="upper left", frameon=False, fontsize=5.4, handlelength=1.6, handletextpad=0.4, borderaxespad=0.2, labelspacing=0.3, bbox_to_anchor=(-0.02, 1.0))
for m, yy in zip(MODELS, y):
    o, c = FAM[m]["benign_sd_ace"], FAM[m]["benign_sd_ca"]
    r2.plot([c, o], [yy, yy], color="#C5CBD5", lw=1.3, zorder=1, solid_capstyle="round")
    r2.scatter([o], [yy], s=16, color=OFFSET, zorder=3); r2.scatter([c], [yy], s=16, color=OURS, zorder=4)
ylabels_by_omega(r2, y); r2.spines["left"].set_visible(False)
r2.text(-0.02, -0.055, r"model   $\omega$", transform=r2.transAxes, fontsize=5.6, color=TEXT, ha="right", va="top")
r2.set_xlim(0.03, 0.10); r2.set_xticks([0.04, 0.06, 0.08, 0.10]); r2.set_xlabel("sd of the gate on benign states")
r2.set_title("(b) Benign gate spread", loc="left", pad=3)
h = [plt.Line2D([], [], marker="o", ls="", mfc=OFFSET, mec=OFFSET, ms=3.8, label="DIM+offset"),
     plt.Line2D([], [], marker="o", ls="", mfc=OURS, mec=OURS, ms=3.8, label="CA (ours)")]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.55, 0.99), ncol=2, frameon=False, fontsize=6.0, handletextpad=0.2, columnspacing=1.0)
grid(r1); grid(r2, "x")
save(fig, "fig_routes", 3.15, 2.05)   # paired with fig_rq2
PLOTTED["fig_routes"] = {m: {k: FAM[m][k] for k in ("omega", "pred_ratio", "gate_ratio", "benign_sd_ace", "benign_sd_ca")} for m in MODELS}

# ---------------------------------------------------------------- Fig. (RQ4): diagnostics predict before generation
om = [OM[m] for m in MODELS]; eta = [ETA[m] for m in MODELS]
df = [pct(L["dfaith"][m]) for m in MODELS]; dg = [pct(L["gain_hb"][m]) for m in MODELS]
B, cells = COL["beta"], COL["cells"]
db = [100 * (B[m]["dim"]["beta_benign"]["mean"] - B[m]["ca"]["beta_benign"]["mean"]) for m in MODELS]
fig, axs = plt.subplots(2, 2, figsize=(6.5, 3.3), gridspec_kw={"wspace": 0.28, "hspace": 0.62})
(c1, c2), (c3, c4) = axs
# (a) omega -> benign answers preserved: a line law
fit_line(c1, om, df)
law_points(c1, om, df)
c1.set_xlabel(r"overlap $\omega$ of DIM with the common mode"); c1.set_ylabel("benign answers preserved (pts)")
c1.set_title(r"(a) $\omega$ predicts the faithfulness gain", loc="left", pad=3)
c1.text(0.97, 0.05, f"Spearman {L['faithfulness_law']['spearman']:.2f}, Pearson {L['faithfulness_law']['pearson']:.2f}", transform=c1.transAxes, va="bottom", ha="right", fontsize=6, color=TEXT)
c1.set_xlim(-0.02, 0.52); c1.set_ylim(-6, 46)
# (b) eta -> ASR gain: two regimes split at eta = 0.6
c2.axvspan(-0.05, 0.6, color=NEG, alpha=0.07, lw=0, zorder=0)
c2.axvline(0.6, color=TEXT, lw=0.7, ls=(0, (3, 1.6)), zorder=1)
c2.text(0.37, 55, "DIM under-fires\n" + r"($\eta<0.6$): CA gains", ha="center", va="center", fontsize=5.6, color=NEG)
c2.text(0.83, 55, "DIM already fires\n" + r"($\eta>0.6$): no gain", ha="center", va="center", fontsize=5.6, color=TEXT)
fit_line(c2, eta, dg, color="#A0A8B5")
law_points(c2, eta, dg)
c2.set_xlabel(r"efficacy $\eta$ of DIM"); c2.set_ylabel("ASR gain (pts)")
c2.set_title(r"(b) $\eta$ predicts the ASR gain", loc="left", pad=3)
c2.text(0.03, 0.05, f"Pearson {L['efficacy_law_hb']['pearson']:.2f} (line)", transform=c2.transAxes, va="bottom", ha="left", fontsize=6, color=TEXT)
c2.set_xlim(-0.02, 1.05); c2.set_ylim(-13, 62)
# (c) beta+ -> collapse: a threshold at one (symlog y so that the 138 cells near zero are visible)
agg = {}
for r in cells:
    agg.setdefault((r["model"], r["cond"]), []).append((r["beta_harm"], r["degen"]))
c3.axvspan(1.0, 1.12, color=NEG, alpha=0.07, lw=0, zorder=0)
c3.axvline(1.0, color=TEXT, lw=0.7, ls=(0, (3, 1.6)), zorder=1)
for (m, cnd), v in agg.items():
    bh, dg_ = np.array([a for a, _ in v]), np.array([pct(b) for _, b in v])
    c3.plot([bh.min(), bh.max()], [dg_.mean()] * 2, color=MCOL[m], lw=0.8, alpha=0.6, zorder=2)
    c3.scatter([bh.mean()], [dg_.mean()], s=18, marker=MSHAPE[cnd], color=MCOL[m], edgecolor="white", lw=0.3, zorder=3)
for cnd in ["dim", "ace", "ca"]:
    c3.scatter([], [], s=18, marker=MSHAPE[cnd], color="#9AA1AD", label=COND[cnd][0])
c3.set_yscale("symlog", linthresh=1.0, linscale=0.6); c3.set_yticks([0, 1, 10, 50]); c3.set_yticklabels(["0", "1", "10", "50"])
c3.set_ylim(-0.25, 90); c3.set_xlim(0.25, 1.12)
c3.set_xlabel(r"collapse index $\beta^{+}$ on harmful states (bar: range of draws)"); c3.set_ylabel("collapse rate (%, symlog)")
c3.set_title(r"(c) $\beta^{+}>1$ marks collapse", loc="left", pad=3)
c3.legend(loc="upper left", frameon=False, fontsize=5.6, handletextpad=0.1, borderaxespad=0.0, labelspacing=0.2, bbox_to_anchor=(0.01, 0.99), title="marker = direction", title_fontsize=5.6, alignment="left")
# (d) drop in beta- -> the same benign answers preserved: a line law
fit_line(c4, db, df)
law_points(c4, db, df)
c4.set_xlabel(r"drop in $\beta^{-}$ from DIM to CA ($\times$100)"); c4.set_ylabel("benign answers preserved (pts)")
c4.set_title(r"(d) $\Delta\beta^{-}$ predicts the same gain", loc="left", pad=3)
c4.text(0.97, 0.05, f"Pearson {COL['dbeta_dfaith']['pearson']:.2f}, Spearman {COL['dbeta_dfaith']['spearman']:.2f}", transform=c4.transAxes, va="bottom", ha="right", fontsize=6, color=TEXT)
c4.set_ylim(-6, 46)
for ax in (c1, c2, c4):
    grid(ax); ax.axhline(0, color=GRID, lw=0.6, zorder=0)
grid(c3)
h = [plt.Line2D([], [], marker="o", ls="", mfc=MCOL[m], mec="white", mew=0.4, ms=4.6, label=f"{SHORT[m]} ($\\omega$ " + f"{OM[m]:.2f}"[1:] + ")") for m in MODELS]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=8, frameon=False, fontsize=6.0, handletextpad=0.1, columnspacing=1.0,
           title="colour = model, ordered by $\\omega$", title_fontsize=6.0)
save(fig, "fig_rq3", 6.5)
PLOTTED["fig_rq3"] = {"omega": OM, "eta": ETA, "faith_gain_pts": dict(zip(MODELS, df)), "asr_gain_pts": dict(zip(MODELS, dg)),
                      "dbeta_x100": dict(zip(MODELS, db)),
                      "cells_mean": {f"{m}/{c}": {"beta_harm_mean": float(np.mean([a for a, _ in v])), "degen_pct_mean": float(np.mean([pct(b) for _, b in v]))}
                                     for (m, c), v in agg.items()}}

# ---------------------------------------------------------------- Fig. (RQ5): where the operator stops
E = load("emotion")["emotion_strength_agg"]; AMP = load("amplification")["amplification"]["gemma-2-9b"]
fig, (d1, d2) = plt.subplots(1, 2, figsize=(3.15, 1.8), gridspec_kw={"wspace": 0.55, "width_ratios": [0.9, 1.25]})
ss = [0, 1, 2, 3]
voc = [100] + [100 * (1 + E[f"ca/s{s}"]["vocab_rel"]) for s in (1, 2, 3)]
pel = [100] + [100 * E[f"ca/s{s}"]["pe_rel"] for s in (1, 2, 3)]
d1.plot(ss, pel, color=OURS, lw=2.0, marker="o", ms=3, mec="white", mew=0.3)
d1.plot(ss, voc, color="#7FA7D6", lw=1.2, ls=(0, (3, 1.5)), marker="o", ms=2.6, mec="white", mew=0.3)
d1.text(3.15, pel[-1], "judged\nemotion", fontsize=5.6, color=OURS, va="center")
d1.text(3.15, voc[-1], "emotion\nwords", fontsize=5.6, color="#5B8AC4", va="center")
d1.set_xticks(ss); d1.set_xlim(-0.2, 4.6); d1.set_ylim(0, 108)
d1.set_xlabel("strength $s$ of CA"); d1.set_ylabel("% of unsteered level")
d1.set_title("(a) Emotion only erodes", loc="left", pad=3)
grid(d1)

STR = ["-0.5", "-1", "-1.5", "-2", "-3"]
base = AMP["none"]


def series(direction):
    """(|s|, over-refusal %, jailbreak ASR %, collapse fraction) at every strength with a fluent output to score."""
    out = [(0, pct(base["xstest"]), pct(base["hb"]), 0.0)]
    for s in STR:
        v = AMP.get(f"{direction}/s{s}")
        if v is not None and v["xstest"] is not None and v["hb"] is not None:
            out.append((abs(float(s)), pct(v["xstest"]), pct(v["hb"]), v["degen"]))
    return out


ca, dm = series("ca"), series("dim")
for pts_, col, lw, name in ((dm, DIM, 1.4, "DIM-Amp"), (ca, OURS, 2.0, "CA-Amp (ours)")):
    d2.plot([p[1] for p in pts_], [p[2] for p in pts_], color=col, lw=lw, zorder=2, label=name)
    for sv, x, yv, dgn in pts_[1:]:
        collapsed = dgn >= 0.10
        d2.scatter([x], [yv], s=16, facecolor="white" if collapsed else col, edgecolor=col, lw=0.9, zorder=3)
        if col == OURS:   # left of the CA-Amp curve; the DIM-Amp line runs on its right
            d2.annotate(f"{sv:g}", (x, yv), xytext=(-4, 0), textcoords="offset points", fontsize=5.2, color=col, ha="right", va="center")
        else:
            lab = f"{sv:g}" if dgn < 0.10 else f"$|s|={sv:g}$: {pct(dgn):.0f}% collapsed"
            if dgn < 0.10:
                d2.annotate(lab, (x, yv), xytext=(3, 4), textcoords="offset points", fontsize=5.2, color=col, ha="left", va="bottom")
            else:   # under the DIM-Amp line, left of the collapsed point
                d2.annotate(lab, (x, yv), xytext=(-5, 0), textcoords="offset points", fontsize=5.2, color=col, ha="right", va="center")
d2.scatter([ca[0][1]], [ca[0][2]], marker="*", color="black", s=30, zorder=4, label="unsteered")
d2.text(0.97, 0.80, "DIM-Amp at $|s|\\geq1.5$:\n95–100% collapsed,\nnothing to score", transform=d2.transAxes, fontsize=5.2, color=TEXT, ha="right", va="top")
d2.set_xlabel("over-refusal on XSTest (%)"); d2.set_ylabel("jailbreak ASR (%)")
d2.set_title("(b) Amplification trade-off, G-9B", loc="left", pad=3)
d2.set_xlim(5, 100); d2.set_ylim(-2, 42); d2.set_xticks([25, 50, 75, 100])
grid(d2)
h = [plt.Line2D([], [], marker="*", ls="", color="black", ms=5, label="unsteered"),
     plt.Line2D([], [], color=DIM, lw=1.4, marker="o", ms=3, label="DIM-Amp"),
     plt.Line2D([], [], color=OURS, lw=2.0, marker="o", ms=3, label="CA-Amp (ours)"),
     plt.Line2D([], [], marker="o", ls="", mfc="white", mec=TEXT, mew=0.9, ms=3.4, label=r"hollow: $\geq$10% collapsed")]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False, fontsize=5.5, handlelength=1.3, columnspacing=0.7, handletextpad=0.3)
save(fig, "fig_rq4", 3.15, 2.0)      # paired with fig_judges
PLOTTED["fig_rq4"] = {"emotion": {"s": ss, "vocab_pct": voc, "judged_pct": pel},
                      "tradeoff_ca": ca, "tradeoff_dim": dm, "unsteered": base}

# ---------------------------------------------------------------- Fig. (judges): two judges on the same collapsed text
JA = load("judge_artifact")
fig, ax = plt.subplots(figsize=(3.15, 1.5))
groups = [("degenerate\n(%d outputs)" % JA["n_degen"], JA["hb_degen"], JA["lg_degen"]), ("fluent\n(%d outputs)" % (JA["n"] - JA["n_degen"]), JA["hb_fluent"], JA["lg_fluent"])]
yy = np.array([1, 0])
ax.barh(yy + 0.19, [pct(g[1]) for g in groups], 0.34, color=TEXT, lw=0, label="HarmBench classifier")
ax.barh(yy - 0.19, [pct(g[2]) for g in groups], 0.34, color="white", edgecolor=TEXT, hatch="////", lw=0.6, label="Llama-Guard-3-8B")
for i, g in enumerate(groups):
    ax.text(pct(g[1]) + 1.5, yy[i] + 0.19, f"{pct(g[1]):.1f}%", va="center", fontsize=5.6, color=TEXT)
    ax.text(pct(g[2]) + 1.5, yy[i] - 0.19, f"{pct(g[2]):.1f}%", va="center", fontsize=5.6, color=TEXT)
ax.set_yticks(yy); ax.set_yticklabels([g[0] for g in groups], fontsize=6.0); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.set_xlim(0, 118); ax.set_xticks([0, 50, 100]); ax.set_xlabel("judged harmful (%), Gemma-2-9B under DIM ablation", fontsize=6.3)
ax.legend(loc="lower right", frameon=False, fontsize=5.4, handlelength=1.4, handletextpad=0.4, labelspacing=0.2, borderaxespad=0.1)
grid(ax, "x")
save(fig, "fig_judges", 3.15, 2.0)   # paired with fig_rq4
PLOTTED["fig_judges"] = {k: JA[k] for k in ("n", "n_degen", "hb_degen", "hb_fluent", "lg_degen", "lg_fluent")}

# ---------------------------------------------------------------- Fig. (method): gate per prompt and outcome, one row per method
GT = load("gates")
M32 = GT["model"]
rows = [("dim", "DIM ablation", DIM), ("ace", "DIM+offset", OFFSET), ("ca", "CA (ours)", OURS)]
fig, (g1, g2) = plt.subplots(1, 2, figsize=(3.64, 1.85), gridspec_kw={"wspace": 0.10, "width_ratios": [1.55, 1.0]}, sharey=True)
PLOTTED["fig_concept"] = {}
for i, (k, name, col) in enumerate(rows):
    y0 = 2 - i
    hv, bv = np.array(GT["harmful"][k]["values"]), np.array(GT["benign"][k]["values"])
    for vals, dy, lab in ((bv, +0.19, "benign"), (hv, -0.19, "harmful")):
        lo, hi = np.percentile(vals, [5, 95])
        g1.plot([lo, hi], [y0 + dy, y0 + dy], color=col, lw=2.4, solid_capstyle="round", alpha=0.35 if lab == "benign" else 0.55, zorder=2)
        g1.scatter([vals.mean()], [y0 + dy], s=20, facecolor="white" if lab == "benign" else col, edgecolor=col, lw=0.9, zorder=4)
        if hi < 0:   # range left of zero: label on its left, clear of the zero line
            g1.text(lo - 0.03, y0 + dy, f"{vals.mean():+.2f}", va="center", ha="right", fontsize=5.6, color=col)
        else:
            g1.text(hi + 0.03, y0 + dy, f"{vals.mean():+.2f}", va="center", ha="left", fontsize=5.6, color=col)
    a, f = pct(MM["asr"][M32][k]["hbf"]["mean"]), pct(MM["faith"][M32][k])
    g2.barh([y0 + 0.19], [f], 0.3, facecolor="white", edgecolor=col, hatch="////", lw=0.6)
    g2.barh([y0 - 0.19], [a], 0.3, color=col, lw=0)
    g2.text(f + 2.5, y0 + 0.19, f"{f:.0f}% rewritten", va="center", fontsize=5.6, color=col)
    g2.text(a + 2.5, y0 - 0.19, f"{a:.0f}% ASR", va="center", fontsize=5.6, color=col)
    PLOTTED["fig_concept"][k] = {"gate_harm_mean": float(hv.mean()), "gate_benign_mean": float(bv.mean()),
                                 "gate_benign_sd": float(bv.std()), "asr_pct": a, "rewrite_pct": f}
g1.axvline(0, color=TEXT, lw=0.7, ls=(0, (2.5, 2)), zorder=1)
g1.text(-0.04, 2.62, "benign untouched", ha="right", va="bottom", fontsize=5.4, color=TEXT)
g1.set_yticks([2, 1, 0]); g1.set_yticklabels([n for _, n, _ in rows], fontsize=6.2)
for t, (_, _, c) in zip(g1.get_yticklabels(), rows):
    t.set_color(c); t.set_fontweight("bold")
g1.tick_params(axis="y", length=0); g1.spines["left"].set_visible(False)
g1.set_xlim(-0.72, 1.08); g1.set_ylim(-0.55, 2.75)
g1.set_xlabel(r"gate $\kappa_u(h)=\cos\angle(h,u)$ per prompt")
g1.set_title("(a) Where the edit fires (gate per prompt)", loc="left", pad=3)
g2.set_xlim(0, 125); g2.set_xticks([0, 50, 100]); g2.spines["left"].set_visible(False); g2.tick_params(axis="y", length=0)
g2.set_xlabel("outcome (%)"); g2.set_title("(b) What it does", loc="left", pad=3)
grid(g1, "x"); grid(g2, "x")
h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=TEXT, mew=0.9, ms=4, label="hollow / hatched: benign prompts"),
     plt.Line2D([], [], marker="o", ls="", mfc=TEXT, mec=TEXT, ms=4, label="filled: harmful prompts")]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False, fontsize=5.6, handlelength=1.0, columnspacing=1.2, handletextpad=0.4)
save(fig, "fig_concept", 3.64)      # 0.56 of the 6.5 in text width, centred
(OUT / "figure_data.json").write_text(json.dumps(jsonable(PLOTTED), indent=1, default=str))
print("figures ok")
