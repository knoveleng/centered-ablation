#!/usr/bin/env python3
"""One-off migration: re-key the cache after the move to config/default.yaml.

The keys gained the build parameters of every set a stage reads (`sets.<name>`), the calibration
size, `generation.sample` and the evaluators' merged default params. Nothing is recomputed: for
each entry the OLD key is rebuilt from its run_config.json with the old key functions (frozen
below) and must reproduce the entry's directory name; the new key comes from the current code.

    python scripts/maintenance/rekey_config_defaults.py           # dry run: counts only
    python scripts/maintenance/rekey_config_defaults.py --apply   # cache/ -> cache_before_rekey/, new tree -> cache/

Never loads a model or calls a judge.
"""

import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline import cache  # noqa: E402
from pipeline.cli import _merge, with_defaults  # noqa: E402
from pipeline.cli import diagnose_key, direction_key, evaluate_key, extract_key, steer_key  # noqa: E402

STAGES = ["extract", "direction", "diagnose", "steer", "evaluate"]


# ------------------------------------------------------------------ the key functions before the migration (frozen)
def old_extract_key(cfg):
    c = cfg["concept"]
    pools = [c["positive"], c["negative"]] if c["name"] == "refusal" else []
    return {"model": cfg["model"]["model_id"], "concept": c["name"], "pools": pools, "extraction": cfg["extraction"]}


def old_direction_key(cfg):
    return {**old_extract_key(cfg), "target": cfg["concept"].get("target"), "direction": cfg["direction"],
            "seed": cfg["calibration_seed"]}


def old_diagnose_key(cfg):
    return {**old_direction_key(cfg), "diagnose": cfg["diagnose"]}


def old_steer_key(cfg):
    base = {"model": cfg["model"]["model_id"], "eval": cfg["data"]["eval"], "generation": cfg["generation"]}
    s = cfg["steering"]["strength"]
    return base if s == 0 else {**base, **old_direction_key(cfg), "strength": s}


def old_evaluate_key(cfg):
    return {**old_steer_key(cfg), "evaluator": cfg["evaluator"]}


OLD = {"extract": old_extract_key, "direction": old_direction_key, "diagnose": old_diagnose_key,
       "steer": old_steer_key, "evaluate": old_evaluate_key}
NEW = {"extract": extract_key, "direction": direction_key, "diagnose": diagnose_key,
       "steer": steer_key, "evaluate": evaluate_key}


def new_config(old: dict) -> dict:
    """The old resolved config, re-resolved under the current config files (its values win)."""
    base = _merge(yaml.safe_load(open(ROOT / "config" / "default.yaml")),
                  yaml.safe_load(open(ROOT / "config" / f"{old['concept']['name']}.yaml")))
    return with_defaults(_merge(base, old))


def main(apply: bool):
    src, dst = cache.CACHE, cache.CACHE.parent / "cache_rekeyed"
    if apply and dst.exists():
        sys.exit(f"{dst} exists; remove it first")
    moves, bad = {}, []
    for stage in STAGES:
        moves[stage] = {}
        for d in sorted(p for p in (src / stage).glob("*") if p.is_dir()) if (src / stage).exists() else []:
            old = json.load(open(d / "run_config.json"))
            if cache.key_hash(OLD[stage](old)) != d.name:
                bad.append(f"{stage}/{d.name}")
                continue
            new = new_config(old)
            h = cache.key_hash(NEW[stage](new))
            if h in moves[stage].values():
                sys.exit(f"collision: two {stage} entries map to {h}")
            moves[stage][d.name] = h
            if apply:
                (dst / stage / h).mkdir(parents=True)
                shutil.copy2(d / cache.ARTIFACT[stage], dst / stage / h / cache.ARTIFACT[stage])
                (dst / stage / h / "run_config.json").write_text(json.dumps(new, indent=2, sort_keys=True))
        print(f"{stage:9s} {len(moves[stage]):5d} entries re-keyed")
    if bad:
        print(f"{len(bad)} entries whose old key does not reproduce their name (NOT moved):", *bad[:20], sep="\n  ")
        sys.exit(1)
    lines = [json.loads(l) for l in open(src / "index.jsonl")]
    unknown = [l for l in lines if l["hash"] not in moves[l["stage"]]]
    print(f"index.jsonl: {len(lines)} lines, {len(unknown)} pointing at no entry (dropped)")
    if not apply:
        print("dry run; pass --apply to write")
        return
    with open(dst / "index.jsonl", "w") as f:
        for l in lines:
            if l["hash"] in moves[l["stage"]]:
                f.write(json.dumps({**l, "hash": moves[l["stage"]][l["hash"]]}) + "\n")
    src.rename(src.parent / "cache_before_rekey")
    dst.rename(src)
    print(f"done: old tree kept at {src.parent / 'cache_before_rekey'}")


if __name__ == "__main__":
    main("--apply" in sys.argv)
