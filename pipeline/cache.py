"""Content-addressed cache: cache/<stage>/<hash>/.

A stage's entry is keyed by a hash of exactly the inputs that change its output (the
`*_key` functions in cli.py). An entry existing is the only record that the stage ran; there
is no status file. Each entry carries run_config.json (the full resolved config), and every
write appends one human-greppable line to cache/index.jsonl, which nothing reads back.

GPU stages also append to cache/ledger.jsonl: wall time, GPU type and engine versions, which
are recorded but never keyed (an upgrade must not discard the cache). The ledger's total is
what budget.max_gpu_hours caps.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
LEDGER = CACHE / "ledger.jsonl"
BUDGET_EXIT = 75   # exit code when the GPU budget is spent, so a resumable wrapper stops

ARTIFACT = {
    "extract": "activations.pt",
    "direction": "direction.pt",
    "diagnose": "diagnostics.json",
    "steer": "completions.jsonl",
    "evaluate": "scores.json",
}


def key_hash(key: dict) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def entry(stage: str, key: dict) -> Path:
    return CACHE / stage / key_hash(key)


def artifact(stage: str, key: dict) -> Path:
    return entry(stage, key) / ARTIFACT[stage]


def done(stage: str, key: dict) -> bool:
    return artifact(stage, key).exists()


def record(stage: str, key: dict, cfg: dict) -> None:
    """Write run_config.json next to a finished artifact and log it in index.jsonl."""
    d = entry(stage, key)
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    line = {"stage": stage, "hash": key_hash(key), "model": cfg["model"]["model_id"],
            "concept": cfg["concept"]["name"], "target": cfg["concept"].get("target"),
            "direction": cfg["direction"]["name"], "params": cfg["direction"]["params"],
            "seed": cfg["calibration_seed"], "strength": cfg["steering"]["strength"],
            "eval": cfg["data"]["eval"], "evaluator": cfg["evaluator"]["name"],
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(CACHE / "index.jsonl", "a") as f:
        f.write(json.dumps(line) + "\n")


class CacheMiss(RuntimeError):
    pass


def compute_allowed(stage: str, key: dict, cfg: dict, gpu: bool = True) -> None:
    """Call before computing a missing entry: refuses under cache_only, and stops a GPU stage
    once the ledger has reached budget.max_gpu_hours."""
    if cfg["cache_only"]:
        raise CacheMiss(f"cache_only: no {stage} entry {key_hash(key)} ({cfg['model']['model_id']}, "
                        f"{cfg['direction']['name']}, seed {cfg['calibration_seed']}, s={cfg['steering']['strength']}, "
                        f"{cfg['data']['eval']}, {cfg['evaluator']['name']})")
    cap = cfg["budget"]["max_gpu_hours"]
    if gpu and cap is not None and gpu_hours() >= cap:
        print(f"GPU budget spent: {gpu_hours():.2f} of {cap} GPU-hours in {LEDGER}", file=sys.stderr)
        sys.exit(BUDGET_EXIT)


def gpu_hours() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["gpu_seconds"] for l in open(LEDGER)) / 3600


def engine() -> dict:
    """Versions and hardware of this process (recorded, not keyed)."""
    import importlib.metadata as md

    import torch

    out = {p: md.version(p) for p in ("torch", "transformers", "vllm") if _installed(p)}
    if torch.cuda.is_available():
        out["gpu"], out["n_gpus"] = torch.cuda.get_device_name(0), torch.cuda.device_count()
    return out


def _installed(package: str) -> bool:
    import importlib.metadata as md

    try:
        md.version(package)
        return True
    except md.PackageNotFoundError:
        return False


def ledger(stage: str, key: dict, cfg: dict, seconds: float, n_items: int) -> None:
    eng = engine()
    line = {"stage": stage, "hash": key_hash(key), "model": cfg["model"]["model_id"],
            "evaluator": cfg["evaluator"]["name"] if stage == "evaluate" else None, "n_items": n_items,
            "wall_seconds": round(seconds, 1), "gpu_seconds": round(seconds * eng.get("n_gpus", 1), 1), "engine": eng,
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(line) + "\n")
