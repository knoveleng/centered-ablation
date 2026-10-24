#!/usr/bin/env python3
"""Score every cached steer entry that lacks an evaluate entry for one evaluator, loading the
judge once. The entries written are ordinary evaluate cache entries.

    python scripts/evaluate_pending.py --evaluator harmbench [--where concept.name=refusal] \
        [--where calibration_seed=0,1]

Only steer entries whose data.eval the evaluator applies to are scored; evaluators that need
the unsteered reference skip the unsteered entries themselves.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import cache  # noqa: E402
from pipeline.cli import do_evaluate, evaluate_key, make_evaluator, rebase, steer_key, unsteered, with_defaults  # noqa: E402
from pipeline.evaluators.base import discover  # noqa: E402


def get(cfg, path):
    for k in path.split("."):
        cfg = cfg.get(k) if isinstance(cfg, dict) else None
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluator", required=True)
    ap.add_argument("--param", action="append", default=[], help="evaluator param key=value")
    ap.add_argument("--where", action="append", default=[], help="config.path=v1,v2 filter")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="cost controls for this invocation, e.g. cache_only=true, budget.max_gpu_hours=10")
    args = ap.parse_args()
    cls = discover()[args.evaluator]
    params = {k: yaml.safe_load(v) for k, _, v in (p.partition("=") for p in args.param)}
    filters = [(k, v.split(",")) for k, _, v in (w.partition("=") for w in args.where)]
    todo = []
    for rc in sorted((cache.CACHE / "steer").glob("*/run_config.json")):
        cfg = rebase(json.load(open(rc)), args.overrides)
        if not cls.applies(cfg["data"]["eval"]):
            continue
        if cls.NEEDS_REFERENCE and cfg["steering"]["strength"] == 0:
            continue
        if any(str(get(cfg, k)) not in vals for k, vals in filters):
            continue
        cfg = copy.deepcopy(cfg)
        cfg["evaluator"] = {"name": args.evaluator, "params": params}
        cfg = with_defaults(cfg)
        if cache.done("evaluate", evaluate_key(cfg)):
            continue
        if cls.NEEDS_REFERENCE and not cache.done("steer", steer_key(unsteered(cfg))):
            print(f"skip (no unsteered reference): {rc.parent.name}")
            continue
        todo.append(cfg)
    print(f"{args.evaluator}: {len(todo)} entries to score", flush=True)
    if not todo:
        return
    cache.compute_allowed("evaluate", evaluate_key(todo[0]), todo[0], gpu=cls.NEEDS_GPU)   # before the judge loads
    ev = make_evaluator(todo[0])
    for i, cfg in enumerate(todo, 1):
        out = do_evaluate(cfg, ev)
        print(f"[{i}/{len(todo)}] {cfg['model']['model_id']} {cfg['direction']['name']} seed {cfg['calibration_seed']} "
              f"s={cfg['steering']['strength']} {cfg['data']['eval']}: {out['rate']:.3f}", flush=True)


if __name__ == "__main__":
    main()
