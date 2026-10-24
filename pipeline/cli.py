"""pipeline.cli — one entry point, five cached stages, one override mechanism.

    python -m pipeline.cli {extract,direction,diagnose,steer,evaluate,run} \
        --config config/refusal.yaml --set key.path=value [--set ...]

One invocation is one point: one model, direction, calibration draw, strength, evaluation set
and evaluator. Loops belong in scripts/. Everything is validated before a model loads.

The resolved config is config/default.yaml, then the concept file (--config), then the --set
overrides, then each component's defaults (defaults.direction.<name>, defaults.evaluator.<name>)
merged under its explicit params, all before any cache key is computed.
"""

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import torch
import yaml

from pipeline import cache, data
from pipeline.directions.base import common_mode, dim, discover
from pipeline.evaluators.base import discover as discover_evaluators
from pipeline.operator import load_steer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


# ------------------------------------------------------------------ config
def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def resolve(config_path: str, overrides: list[str]) -> dict:
    cfg = _merge(yaml.safe_load(open(DEFAULT_CONFIG)), yaml.safe_load(open(config_path)))
    for item in overrides:
        path, _, raw = item.partition("=")
        keys = path.split(".")
        if keys == ["direction", "name"]:
            cfg["direction"]["params"] = {}   # a new direction has its own parameter schema
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = yaml.safe_load(raw) if raw != "" else None
    return with_defaults(cfg)


def rebase(stored: dict, overrides: list[str] = ()) -> dict:
    """A cache entry's run_config under the current config files: fields added since it was
    written get their defaults, and the cost controls (cache_only, budget) are this
    invocation's, not the stored ones."""
    current = resolve(str(ROOT / "config" / f"{stored['concept']['name']}.yaml"), list(overrides))
    out = _merge(current, stored)
    out["cache_only"], out["budget"] = current["cache_only"], current["budget"]
    return with_defaults(out)


def with_defaults(cfg: dict) -> dict:
    """Merge each component's defaults under its explicit params (explicit values win), and make
    numbers that a --set may spell as ints (strength=1, gamma=1) floats, so 1 and 1.0 share a key."""
    out = copy.deepcopy(cfg)
    for family in ("direction", "evaluator"):
        block = out[family]
        block["params"] = {**out["defaults"][family].get(block["name"], {}), **(block["params"] or {})}
    for block in ("steering", "diagnose"):
        if isinstance(out[block]["strength"], int) and not isinstance(out[block]["strength"], bool):
            out[block]["strength"] = float(out[block]["strength"])
    schema = getattr(directions().get(out["direction"]["name"]), "PARAM_SCHEMA", {})
    for k, t in schema.items():
        v = out["direction"]["params"].get(k)
        if t is float and isinstance(v, int) and not isinstance(v, bool):
            out["direction"]["params"][k] = float(v)
    return out


def directions() -> dict:
    return {**discover("pipeline.directions"), **discover("pipeline.baselines")}


def validate(cfg: dict) -> None:
    """Fail before any model loads, naming exactly what is wrong."""
    if not cfg["model"]["model_id"]:
        raise ValueError("set model.model_id")
    concept = cfg["concept"]
    if concept["name"] == "refusal":
        for side in ("positive", "negative"):
            if concept[side] not in data.POOLS:
                raise ValueError(f"concept.{side}='{concept[side]}' is not a pool; known: {sorted(data.POOLS)}")
    elif concept["name"] == "emotion":
        if concept.get("target") not in data.EMOTIONS:
            raise ValueError(f"concept.target must be one of {data.EMOTIONS}")
    else:
        raise ValueError(f"unknown concept '{concept['name']}' (refusal | emotion)")
    known = directions()
    name, params = cfg["direction"]["name"], cfg["direction"]["params"] or {}
    if name not in known:
        raise ValueError(f"unknown direction '{name}'; known: {sorted(known)}")
    schema = known[name].PARAM_SCHEMA
    if set(schema) != set(params):
        raise ValueError(f"direction '{name}' expects params {schema}, got {params}")
    for k, t in schema.items():
        if not isinstance(params[k], t) and not (t is float and isinstance(params[k], int)):
            raise ValueError(f"direction '{name}': param '{k}' must be {t.__name__}, got {params[k]!r}")
    if cfg["data"]["eval"] not in data.EVAL_SETS:
        raise ValueError(f"unknown data.eval '{cfg['data']['eval']}'; known: {sorted(data.EVAL_SETS)}")
    evaluators = discover_evaluators()
    if cfg["evaluator"]["name"] not in evaluators:
        raise ValueError(f"unknown evaluator '{cfg['evaluator']['name']}'; known: {sorted(evaluators)}")
    if not evaluators[cfg["evaluator"]["name"]].applies(cfg["data"]["eval"]):
        raise ValueError(f"evaluator '{cfg['evaluator']['name']}' does not apply to data.eval='{cfg['data']['eval']}'")


# ------------------------------------------------------------------ cache keys (only what changes the output)
def extract_sets(cfg) -> list[str]:
    """Every set the extract stage reads."""
    c = cfg["concept"]
    if c["name"] == "refusal":
        return [c["positive"], c["negative"], *cfg["extraction"]["sets"]]
    return ["stories", "neutral"]


def extract_key(cfg):
    c = cfg["concept"]
    pools = [c["positive"], c["negative"]] if c["name"] == "refusal" else []
    return {"model": cfg["model"]["model_id"], "concept": c["name"], "pools": pools, "extraction": cfg["extraction"],
            "sets": {s: data.set_params(cfg, s) for s in extract_sets(cfg)}}


def direction_key(cfg):
    return {**extract_key(cfg), "target": cfg["concept"].get("target"), "direction": cfg["direction"],
            "seed": cfg["calibration_seed"], "size": cfg["calibration_size"]}


def diagnose_key(cfg):
    return {**direction_key(cfg), "diagnose": cfg["diagnose"]}


def steer_key(cfg):
    e = cfg["data"]["eval"]
    base = {"model": cfg["model"]["model_id"], "eval": e, "set": data.set_params(cfg, e), "generation": cfg["generation"]}
    s = cfg["steering"]["strength"]
    return base if s == 0 else {**base, **direction_key(cfg), "strength": s}


def evaluate_key(cfg):
    return {**steer_key(cfg), "evaluator": cfg["evaluator"]}


def unsteered(cfg):
    """The strength-0 point of cfg: the reference of NEEDS_REFERENCE evaluators. A repeated
    steered generation (generation.sample > 0) still compares with the one unsteered run."""
    out = copy.deepcopy(cfg)
    out["steering"]["strength"] = 0
    out["generation"]["sample"] = 0
    return out


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ stages
def do_extract(cfg):
    key = extract_key(cfg)
    if cache.done("extract", key):
        return torch.load(cache.artifact("extract", key), weights_only=False)
    cache.compute_allowed("extract", key, cfg)
    from pipeline.model import capture_last, capture_tokens, load_hf

    t0 = time.time()
    model, tok = load_hf(cfg["model"]["model_id"])
    c, ex, sp = cfg["concept"], cfg["extraction"], lambda name: data.set_params(cfg, name)
    if c["name"] == "refusal":
        last = lambda items: capture_last(model, tok, [x["prompt"] for x in items], ex["batch_size"])
        art = {"pools": {p: last(data.pool_items(p, sp(p))) for p in (c["positive"], c["negative"])},
               "sets": {s: last(data.eval_items(s, sp(s))) for s in ex["sets"]}}
    else:
        stories = data.emotion_stories(sp("stories"))
        art = {"stories": {}}
        for e in data.EMOTIONS:
            m, t = capture_tokens(model, tok, stories[e], ex["skip"], ex["keep"], ex["max_len"])
            art["stories"][e] = {"mean": m, "tok": t}
        art["neutral"] = capture_tokens(model, tok, data.neutral_dialogues(), 0, ex["keep"], ex["max_len"])[1]
    del model
    _free_gpu()
    path = cache.artifact("extract", key)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(art, path)
    cache.record("extract", key, cfg)
    cache.ledger("extract", key, cfg, time.time() - t0, 0)
    return art


def calibration(cfg, art):
    """(pos, neg, ref, heldout) for this draw."""
    c = cfg["concept"]
    if c["name"] == "refusal":
        P, N = art["pools"][c["positive"]], art["pools"][c["negative"]]
        ip, in_ = data.draw((c["positive"], data.set_params(cfg, c["positive"])),
                            (c["negative"], data.set_params(cfg, c["negative"])),
                            cfg["calibration_seed"], cfg["calibration_size"], len(P), len(N))
        rest = lambda X, idx: X[[i for i in range(len(X)) if i not in set(idx)]]
        return P[ip].float(), N[in_].float(), N[in_].float(), (rest(P, ip), rest(N, in_))
    S, others = art["stories"], [e for e in data.EMOTIONS if e != c["target"]]
    return (S[c["target"]]["mean"].float(), torch.cat([S[e]["mean"] for e in others]).float(),
            torch.cat([S[e]["tok"].float() for e in others]), None)


def do_direction(cfg, art=None):
    key = direction_key(cfg)
    if cache.done("direction", key):
        return torch.load(cache.artifact("direction", key), weights_only=False)
    cache.compute_allowed("direction", key, cfg, gpu=False)
    art = art or do_extract(cfg)
    pos, neg, ref, heldout = calibration(cfg, art)
    steer = directions()[cfg["direction"]["name"]]().fit(pos, neg, ref, cfg["direction"]["params"] or {}, heldout)
    out = {**steer.state(), "u_dim": dim(pos, neg), "mu": common_mode(ref)}
    path = cache.artifact("direction", key)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    cache.record("direction", key, cfg)
    return out


def diagnostic_sets(cfg, art) -> tuple[dict, str]:
    c = cfg["concept"]
    if c["name"] == "refusal":
        N = art["pools"][c["negative"]]
        benign = N[[i for i in range(len(N)) if i >= cfg["calibration_size"]]]   # rows outside draw 0
        return {"harmbench": art["sets"]["harmbench"], "benign_heldout": benign}, "harmbench"
    S, others = art["stories"], [e for e in data.EMOTIONS if e != c["target"]]
    return {"target": S[c["target"]]["tok"], "other": torch.cat([S[e]["tok"] for e in others]),
            "neutral": art["neutral"]}, "target"


def do_diagnose(cfg):
    key = diagnose_key(cfg)
    if cache.done("diagnose", key):
        return json.load(open(cache.artifact("diagnose", key)))
    cache.compute_allowed("diagnose", key, cfg, gpu=False)
    from transformers import AutoTokenizer

    from pipeline.diagnostics import diagnose
    from pipeline.model import Head

    art, d = do_extract(cfg), do_direction(cfg)
    head, tok = Head(cfg["model"]["model_id"]), AutoTokenizer.from_pretrained(cfg["model"]["model_id"])
    sets, behaviour = diagnostic_sets(cfg, art)
    mmlu = None
    if "mmlu" in art.get("sets", {}):
        answers = torch.tensor([x["reference"] for x in data.eval_items("mmlu", data.set_params(cfg, "mmlu"))])
        mmlu = (art["sets"]["mmlu"], answers, [tok.encode(L, add_special_tokens=False)[0] for L in "ABCD"])
    out = diagnose(head, tok, load_steer(d), cfg["diagnose"]["strength"], sets, d["u_dim"], d["mu"], behaviour, mmlu,
                   cfg["diagnose"]["footprint_k"])
    path = cache.artifact("diagnose", key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    cache.record("diagnose", key, cfg)
    return out


def do_steer(cfg, llm=None):
    key = steer_key(cfg)
    if cache.done("steer", key):
        return [json.loads(l) for l in open(cache.artifact("steer", key))]
    cache.compute_allowed("steer", key, cfg)
    from pipeline.model import generate, load_llm

    t0 = time.time()
    items, s = data.eval_items(cfg["data"]["eval"], data.set_params(cfg, cfg["data"]["eval"])), cfg["steering"]["strength"]
    steer = load_steer(do_direction(cfg)) if s != 0 else None
    own = llm is None
    llm = llm or load_llm(cfg)
    completions = generate(llm, [x["prompt"] for x in items], cfg["generation"], steer, s)
    if own:
        del llm
        _free_gpu()
    rows = [{**it, "completion": c} for it, c in zip(items, completions)]
    path = cache.artifact("steer", key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    cache.record("steer", key, cfg)
    cache.ledger("steer", key, cfg, time.time() - t0, len(rows))
    return rows


def do_evaluate(cfg, evaluator=None):
    """`evaluator`: an already-loaded instance, so scripts/evaluate_pending.py can score many
    cached runs with one judge load."""
    key = evaluate_key(cfg)
    if cache.done("evaluate", key):
        return json.load(open(cache.artifact("evaluate", key)))
    cls = discover_evaluators()[cfg["evaluator"]["name"]]
    cache.compute_allowed("evaluate", key, cfg, gpu=cls.NEEDS_GPU)
    t0 = time.time()
    evaluator = evaluator or make_evaluator(cfg)
    if not cache.done("steer", steer_key(cfg)):
        raise RuntimeError("no cached completions for this point; run the steer stage first")
    rows = do_steer(cfg)
    reference = None
    if cls.NEEDS_REFERENCE:
        if not cache.done("steer", steer_key(unsteered(cfg))):
            raise RuntimeError("this evaluator compares with the unsteered completions; run steer with steering.strength=0")
        reference = [r["completion"] for r in do_steer(unsteered(cfg))]
    scores = evaluator.score(rows, [r["completion"] for r in rows], reference)
    out = {"evaluator": cfg["evaluator"]["name"], "n": len(scores),
           "rate": sum(bool(x["label"]) for x in scores) / len(scores), "items": scores}
    path = cache.artifact("evaluate", key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    cache.record("evaluate", key, cfg)
    if cls.NEEDS_GPU:
        cache.ledger("evaluate", key, cfg, time.time() - t0, len(scores))
    return out


def make_evaluator(cfg):
    """An evaluator instance: its params (in the key) and its vLLM memory knobs (not in the key)."""
    name = cfg["evaluator"]["name"]
    return discover_evaluators()[name](cfg["evaluator"]["params"], cfg["resources"]["judges"].get(name, {}))


def do_run(cfg):
    """extract → direction → diagnose → steer (+ unsteered reference if needed) → evaluate."""
    do_extract(cfg)
    do_direction(cfg)
    do_diagnose(cfg)
    if discover_evaluators()[cfg["evaluator"]["name"]].NEEDS_REFERENCE:
        do_steer(unsteered(cfg))
    do_steer(cfg)
    return do_evaluate(cfg)


STAGES = {"extract": do_extract, "direction": do_direction, "diagnose": do_diagnose,
          "steer": do_steer, "evaluate": do_evaluate, "run": do_run}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    args = ap.parse_args()
    cfg = resolve(args.config, args.overrides)
    validate(cfg)
    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return
    out = STAGES[args.stage](cfg)
    if args.stage in ("evaluate", "run"):
        print(f"{cfg['evaluator']['name']}: {out['rate']:.3f} (n={out['n']})")


if __name__ == "__main__":
    main()
