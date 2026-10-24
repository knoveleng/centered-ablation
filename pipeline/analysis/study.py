"""What every analysis shares: the study settings (config `study` block), the reader of cached
results by experiment point, and the frozen legacy inputs.

Analyses never call a model or a judge: every point is resolved with cache_only=true, so a
missing entry is a CacheMiss, not a computation.

Points are named the way the run scripts name them, as --set overrides on top of a concept:

    p = point("llama-3.1-8b", "direction.name=ca", "calibration_seed=3")
    labels(p, "harmbench")      # per-prompt bools of the cached evaluate entry
    completions(p)              # the cached steer entry's texts

Frozen inputs (legacy/results/) are results of the pre-refactor code the pipeline cannot
reproduce offline (they need the checkpoints' unembeddings, a GPU, or were saved only as
aggregates); `frozen()` is the one place that reads them.
"""

import json
from functools import cache as memo
from pathlib import Path

import yaml

from pipeline import cache
from pipeline.cli import evaluate_key, resolve, steer_key, with_defaults

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(open(ROOT / "config" / "default.yaml"))
S = CONFIG["study"]
MODELS, STUDY, HELDOUT = S["models"], S["study_models"], S["heldout_models"]
DRAWS, DIRECTIONS, BOOT = S["draws"], S["directions"], S["bootstrap"]
NAMES = S["names"]
IDS = {name: m["id"] for name, m in CONFIG["resources"]["models"].items()}
FROZEN = ROOT / "legacy" / "results"


@memo
def _point(model: str, concept: str, overrides: tuple[str, ...]) -> str:
    cfg = resolve(str(ROOT / "config" / f"{concept}.yaml"), [f"model.model_id={IDS[model]}", "cache_only=true", *overrides])
    return json.dumps(cfg)


def point(model: str, *overrides: str, concept: str = "refusal") -> dict:
    """The resolved config of one experiment point (a fresh copy)."""
    return json.loads(_point(model, concept, tuple(overrides)))


def unsteered(model: str, eval_set: str, concept: str = "refusal") -> dict:
    return point(model, "steering.strength=0", f"data.eval={eval_set}", concept=concept)


def completions(cfg: dict) -> list[str]:
    key = steer_key(cfg)
    if not cache.done("steer", key):
        raise cache.CacheMiss(f"no steer entry {cache.key_hash(key)}")
    return [json.loads(l)["completion"] for l in open(cache.artifact("steer", key))]


def scores(cfg: dict, evaluator: str) -> list[dict]:
    """The per-item scores of one cached evaluate entry."""
    ecfg = with_defaults({**cfg, "evaluator": {"name": evaluator, "params": {}}})
    key = evaluate_key(ecfg)
    if not cache.done("evaluate", key):
        raise cache.CacheMiss(f"no {evaluator} entry {cache.key_hash(key)} for {cfg['model']['model_id']} "
                              f"{cfg['direction']['name']} seed {cfg['calibration_seed']} s={cfg['steering']['strength']} "
                              f"{cfg['data']['eval']} sample {cfg['generation']['sample']}")
    return json.load(open(cache.artifact("evaluate", key)))["items"]


def labels(cfg: dict, evaluator: str) -> list[bool]:
    return [bool(x["label"]) for x in scores(cfg, evaluator)]


def has(cfg: dict, evaluator: str | None = None) -> bool:
    if evaluator is None:
        return cache.done("steer", steer_key(cfg))
    return cache.done("evaluate", evaluate_key(with_defaults({**cfg, "evaluator": {"name": evaluator, "params": {}}})))


def last_sample(cfg: dict) -> dict:
    """The point's highest cached generation.sample: the single-draw ablation runs were generated
    separately from the main matrix and are cached as sample 1 where they differ from it."""
    out = json.loads(json.dumps(cfg))
    while True:
        out["generation"]["sample"] += 1
        if not has(out):
            out["generation"]["sample"] -= 1
            return out


def frozen(rel: str):
    """A frozen legacy input: parsed JSON, or the text of anything else."""
    path = FROZEN / rel
    return json.load(open(path)) if path.suffix == ".json" else path.read_text()


def is_degenerate(text: str) -> bool:
    """The degeneration evaluator's rule with the config's thresholds, for the legacy-only
    generations (directions the pipeline does not express)."""
    from pipeline.evaluators.degeneration import is_degenerate as rule

    p = CONFIG["defaults"]["evaluator"]["degeneration"]
    return rule(text, p["min_bytes"], p["max_ratio"])


def omega_eta() -> tuple[dict, dict]:
    """ω and DIM's η per model: the held-out models from the locked prediction file (full
    precision), the study models from their printed values (frozen inputs, both)."""
    locked = frozen("ladder_prediction.locked.json")
    printed = frozen("geometry/omega_eta_study_models.json")
    omega = {m: locked[m]["omega"] for m in locked}
    eta = {m: locked[m]["sim"]["dim"]["harm_token_change"] for m in locked}
    omega.update(printed["omega"])
    eta.update(printed["eta"])
    return omega, eta


def corr(a, b) -> dict:
    import numpy as np
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    return {"pearson": float(stats.pearsonr(a, b)[0]), "spearman": float(stats.spearmanr(a, b)[0])}


def refusal_states(model: str) -> dict:
    """Draw-0 states of the refusal concept from the cached extract entry: HarmBench prompts, the
    benign calibration rows and the benign held-out rows (pool order)."""
    from pipeline.cli import calibration, do_extract

    cfg = point(model)
    art = do_extract(cfg)
    _, neg, _, (_, rest) = calibration(cfg, art)
    return {"harmbench": art["sets"]["harmbench"], "harmless_calib": neg, "harmless_heldout": rest}


def direction(model: str, name: str, seed: int = 0) -> dict:
    """A cached direction entry: u, c, u_dim, mu."""
    from pipeline.cli import do_direction

    return do_direction(point(model, f"direction.name={name}", f"calibration_seed={seed}"))
