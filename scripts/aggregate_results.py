#!/usr/bin/env python3
"""Fold every cached evaluate / diagnose entry into flat tables (the only producer of tables).

    python scripts/aggregate_results.py      # -> results/analysis/cache_tables.{json,md}

results.<table> is a list of row objects (pandas.DataFrame(rows) reads it):

  runs             one row per evaluate entry of the family (DIM, ACE, CA) and the unsteered model:
                   rate, the degeneration rate of the same completions, and the fluent-only rate
                   (a degenerate completion counts as a failure)
  controls         the same for baseline/control directions (never mixed into runs)
  summary          runs averaged over calibration draws (mean, sd, n_draws)
  paired           CA − DIM and CA − ACE per model × eval × evaluator: paired bootstrap 95% CI over
                   prompts (draws averaged) on fluent-only labels, and draws with McNemar p < .05
  diagnostics      one row per diagnose entry: ω, and per diagnostic set gate, token change, β
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import cache  # noqa: E402
from pipeline.cli import steer_key  # noqa: E402
from pipeline.analysis._output import save  # noqa: E402
from pipeline.stats import mcnemar_p, paired_bootstrap  # noqa: E402

TABLES = {}
FAMILY = {"dim", "ace", "ca"}
BOOT = yaml.safe_load(open(cache.ROOT / "config" / "default.yaml"))["study"]["bootstrap"]


def entries(stage):
    for rc in sorted((cache.CACHE / stage).glob("*/run_config.json")):
        art = rc.parent / cache.ARTIFACT[stage]
        if art.exists():
            yield json.load(open(rc)), art


def write(name, rows):
    TABLES[name] = rows


def render(r):
    print("Tables folded from the cache (row lists in cache_tables.json):")
    for name, rows in r.items():
        print(f"  {name:12s} {len(rows):5d} rows")


def main():
    # degeneration labels per steer entry, to derive the fluent-only rate of every other evaluator
    degen = {}
    scored = []
    for cfg, art in entries("evaluate"):
        sc = json.load(open(art))
        labels = [bool(x["label"]) for x in sc["items"]]
        sk = cache.key_hash(steer_key(cfg))
        if cfg["evaluator"]["name"] == "degeneration":
            degen[sk] = labels
        scored.append((cfg, sk, labels))

    runs, controls, per_item = [], [], {}
    for cfg, sk, labels in scored:
        unsteered = cfg["steering"]["strength"] == 0
        d = degen.get(sk)
        row = {"model": cfg["model"]["model_id"], "concept": cfg["concept"]["name"], "target": cfg["concept"].get("target"),
               "direction": "none" if unsteered else cfg["direction"]["name"],
               "params": "" if unsteered else json.dumps(cfg["direction"]["params"], sort_keys=True),
               "seed": "" if unsteered else cfg["calibration_seed"], "strength": cfg["steering"]["strength"],
               "sample": cfg["generation"].get("sample", 0),
               "eval": cfg["data"]["eval"], "evaluator": cfg["evaluator"]["name"], "n": len(labels),
               "rate": sum(labels) / len(labels),
               "degeneration": sum(d) / len(d) if d else "",
               "fluent_rate": sum(l and not g for l, g in zip(labels, d)) / len(labels) if d else ""}
        (runs if unsteered or row["direction"] in FAMILY else controls).append(row)
        fluent = [l and not g for l, g in zip(labels, d)] if d else labels
        per_item[(row["model"], row["concept"], row["target"], row["direction"], row["params"], row["seed"],
                  row["strength"], row["sample"], row["eval"], row["evaluator"])] = fluent
    write("runs", runs)
    write("controls", controls)

    groups = defaultdict(list)
    for r in runs:
        groups[tuple(r[k] for k in ("model", "concept", "target", "direction", "strength", "sample", "eval", "evaluator"))].append(r)
    summary = []
    for k, rs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        row = dict(zip(("model", "concept", "target", "direction", "strength", "sample", "eval", "evaluator"), k))
        for m in ("rate", "degeneration", "fluent_rate"):
            v = torch.tensor([float(r[m]) for r in rs if r[m] != ""], dtype=torch.float64)
            row[m] = v.mean().item() if len(v) else ""
            row[f"{m}_sd"] = v.std().item() if len(v) > 1 else ""
        row["n_draws"] = len(rs)
        summary.append(row)
    write("summary", summary)

    paired = []
    keyset = {(m, c, t, s, sm, e, ev) for (m, c, t, d, p, sd, s, sm, e, ev) in per_item if d == "ca"}
    for m, c, t, s, sm, e, ev in sorted(keyset, key=str):
        def stack(direction):
            rows = [(sd, v) for (m2, c2, t2, d2, p2, sd, s2, sm2, e2, ev2), v in per_item.items()
                    if (m2, c2, t2, d2, s2, sm2, e2, ev2) == (m, c, t, direction, s, sm, e, ev)]
            return dict(sorted(rows))
        ca = stack("ca")
        for other in ("dim", "ace"):
            ot = stack(other)
            seeds = sorted(set(ca) & set(ot))
            if not seeds:
                continue
            X = torch.tensor([ot[sd] for sd in seeds], dtype=torch.float)
            Y = torch.tensor([ca[sd] for sd in seeds], dtype=torch.float)
            diff, lo, hi = paired_bootstrap(X, Y, BOOT["n"], BOOT["seed"])
            paired.append({"model": m, "concept": c, "target": t, "strength": s, "sample": sm, "eval": e, "evaluator": ev,
                           "comparison": f"ca-{other}", "n_draws": len(seeds), "diff": diff, "ci_low": lo, "ci_high": hi,
                           "significant": lo > 0 or hi < 0,
                           "mcnemar_draws_p05": sum(mcnemar_p(x, y) < 0.05 for x, y in zip(X, Y))})
    write("paired", paired)

    diag = []
    for cfg, art in entries("diagnose"):
        dg = json.load(open(art))
        row = {"model": cfg["model"]["model_id"], "concept": cfg["concept"]["name"], "target": cfg["concept"].get("target"),
               "direction": cfg["direction"]["name"], "params": json.dumps(cfg["direction"]["params"], sort_keys=True),
               "seed": cfg["calibration_seed"], "strength": dg["strength"], "omega": dg["omega"], "omega_dim": dg["omega_dim"]}
        for sname, v in dg["sets"].items():
            for k in ("gate_mean", "gate_sd", "token_change", "beta_mean"):
                row[f"{sname}.{k}"] = v[k]
        if "mmlu" in dg:
            row["mmlu_unsteered"], row["mmlu_steered"] = dg["mmlu"]["unsteered"], dg["mmlu"]["steered"]
        diag.append(row)
    write("diagnostics", diag)
    save("cache_tables", TABLES, render)


if __name__ == "__main__":
    main()
