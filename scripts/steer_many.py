#!/usr/bin/env python3
"""Run the steer stage for a grid of points on ONE model, loading the model once.

Each point is an ordinary steer cache entry, identical to what
`python -m pipeline.cli steer --config ... --set ...` writes; this only saves the per-point model
load, which dominates the cost of a sweep.

    python scripts/steer_many.py --config config/refusal.yaml \
        --set model.model_id=google/gemma-2-9b-it --set model.attention_backend=TRITON_ATTN \
        --grid direction.name=dim,ace,ca --grid calibration_seed=0,1,2,3,4,5 \
        --grid data.eval=harmbench,alpaca [--unsteered]

--unsteered also generates the strength-0 completions for every data.eval in the grid.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import cache  # noqa: E402
from pipeline.cli import do_steer, resolve, steer_key, unsteered, validate  # noqa: E402


def points(args):
    axes = [(g.partition("=")[0], g.partition("=")[2].split(",")) for g in args.grid]
    for combo in itertools.product(*[vals for _, vals in axes]):
        over = args.overrides + [f"{k}={v}" for (k, _), v in zip(axes, combo)]
        # put direction.name first so its params (set later in the list) are not cleared
        over.sort(key=lambda o: o.split("=")[0] != "direction.name")
        yield resolve(args.config, over)


def estimate(pending) -> float | None:
    """GPU-seconds for the pending points, from the ledger's seconds per item on this model."""
    from pipeline import data

    model = pending[0]["model"]["model_id"]
    past = [json.loads(l) for l in open(cache.LEDGER)] if cache.LEDGER.exists() else []
    past = [p for p in past if p["stage"] == "steer" and p["model"] == model and p["n_items"]]
    if not past:
        return None
    per_item = sum(p["gpu_seconds"] for p in past) / sum(p["n_items"] for p in past)
    n = sum(len(data.eval_items(c["data"]["eval"], data.set_params(c, c["data"]["eval"]))) for c in pending)
    return per_item * n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--grid", action="append", default=[], help="key=v1,v2,... (cartesian product)")
    ap.add_argument("--unsteered", action="store_true")
    args = ap.parse_args()
    todo = []
    for cfg in points(args):
        validate(cfg)
        if args.unsteered:
            todo.append(unsteered(cfg))
        todo.append(cfg)
    seen, pending = set(), []
    for cfg in todo:
        k = cache.key_hash(steer_key(cfg))
        if k not in seen and not cache.done("steer", steer_key(cfg)):
            seen.add(k)
            pending.append(cfg)
    print(f"{len(pending)} steer points to run ({len(todo) - len(pending)} cached or duplicate)", flush=True)
    if not pending:
        return
    cache.compute_allowed("steer", steer_key(pending[0]), pending[0])   # before the model loads
    est = estimate(pending)
    if est is not None:
        print(f"estimated {est / 3600:.2f} GPU-hours from the ledger's past runs of this model", flush=True)
    from pipeline.model import load_llm

    llm = load_llm(pending[0])
    for i, cfg in enumerate(pending, 1):
        do_steer(cfg, llm=llm)
        print(f"[{i}/{len(pending)}] {cfg['direction']['name']} seed {cfg['calibration_seed']} "
              f"s={cfg['steering']['strength']} {cfg['data']['eval']}", flush=True)


if __name__ == "__main__":
    main()
