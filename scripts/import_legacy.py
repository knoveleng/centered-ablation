#!/usr/bin/env python3
"""Turn the pre-refactor outputs in legacy/results/ into ordinary cache entries. Nothing is
generated or judged: the legacy files ARE the record (vLLM is not bit-reproducible).

    python scripts/import_legacy.py                 # activations -> extract entries
    python scripts/import_legacy.py --generations   # also every generation the pipeline can express,
                                                    # with its judge labels (harmbench, llamaguard, emotion_judge)

Activations: the pools are rebuilt in the order the data module uses (draw 0 = first 256
rows), which is exactly calib + held-out in the legacy files.

Generations: each legacy file name is mapped to the pipeline point that produced it (REFUSAL
and EMOTION below; every direction was checked against the legacy one). An entry is written
only if its prompts equal the pipeline's evaluation items, in order. A legacy file that maps to
an already cached point with different completions (the same point generated twice, which
vLLM does not reproduce) is kept as generation.sample=1. Files that map to no pipeline point
(directions not ported, gated amplification, calibration-set generations) are listed and
stay in legacy/results/.
"""

import json
import re
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import cache, data  # noqa: E402
from pipeline.cli import evaluate_key, extract_key, resolve, steer_key  # noqa: E402
from pipeline.evaluators.emotion_judge import vocab_rate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LEGACY = ROOT / "legacy" / "results"
ACTS = LEGACY / "acts"
IDS = {name: m["id"] for name, m in yaml.safe_load(open(ROOT / "config" / "default.yaml"))["resources"]["models"].items()}
JB = ["concept.positive=jb_calib", "concept.negative=jb_calib_plain", "extraction.sets=[harmbench]"]


def put(cfg, art):
    key = extract_key(cfg)
    path = cache.artifact("extract", key)
    if path.exists():
        return "exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(art, path)
    cache.record("extract", key, cfg)
    return "imported"


# ------------------------------------------------------------------ legacy file name -> pipeline overrides
MMD = "direction.name=mmd_erasure", "direction.params.steps=400"
ERASE = {   # legacy erase_<tag> (directions saved in acts/<model>.erase_<tag>.pt) -> pipeline control
    "dimc": ["direction.name=ca"],
    "lda": ["direction.name=lda", "direction.params.shrinkage=0.9"],
    "fixedpt": ["direction.name=fixed_point", "direction.params.iters=50"],
    **{tag: [*MMD, f"direction.params.kernel={k}", f"direction.params.normalize={n}", f"direction.params.gamma={g}",
             f"direction.params.pca_k={p}"]
       for tag, (k, n, g, p) in {"kNone": ("rbf", "true", 1.0, 0), "g0.5": ("rbf", "true", 0.5, 0), "g2": ("rbf", "true", 2.0, 0),
                                 "linrms": ("linear", "true", 1.0, 0), "rbfraw": ("rbf", "false", 1.0, 0),
                                 "k64": ("rbf", "true", 1.0, 64)}.items()},
}
REFUSAL = [   # (pattern over the file stem, overrides from the match)
    (r"rob_(dim|dimc|ace)_seed(\d)", lambda d, s: [f"direction.name={'ca' if d == 'dimc' else d}", f"calibration_seed={s}"]),
    (r"baseline", lambda: ["steering.strength=0"]),
    (r"dim_s(-?[\d.]+)", lambda s: ["direction.name=dim", f"steering.strength={s}"]),
    (r"erase_(dimc|lda|fixedpt|kNone|g0\.5|g2|linrms|rbfraw|k64)_s(-?[\d.]+)", lambda t, s: [*ERASE[t], f"steering.strength={s}"]),
    (r"random_s(-?[\d.]+)", lambda s: ["direction.name=random", "direction.params.seed=0", f"steering.strength={s}"]),
    (r"kde_g([\d.]+)_s(-?[\d.]+)", lambda g, s: ["direction.name=kde_field", f"direction.params.gamma={g}", f"steering.strength={s}"]),
    (r"refusal_dim_s(-[\d.]+)", lambda s: ["direction.name=dim", f"steering.strength={s}"]),
    (r"refusalc_dim_s(-[\d.]+)", lambda s: ["direction.name=ca", f"steering.strength={s}"]),
    (r"jb_dim_s(\d+)", lambda s: [*JB, "direction.name=dim", f"steering.strength={s}"]),
]
EMOTION = [
    (r"(ask_\w+|alpaca)__baseline", lambda ds: [f"data.eval={ds}", "steering.strength=0"]),
    (r"(ask_\w+|alpaca)__abl_(\w+?)_(dim|dimc|ace)(?:_s(\d))?",
     lambda ds, e, d, s: [f"data.eval={ds}", f"concept.target={e}", f"direction.name={'ca' if d == 'dimc' else d}",
                          f"steering.strength={s or 1.0}"]),
    (r"ask_none__amp_(\w+?)_(dim|dimc)", lambda e, d: ["data.eval=ask_none", f"concept.target={e}",
                                                        f"direction.name={'ca' if d == 'dimc' else d}", "steering.strength=-2"]),
]
EMOTION_SETS = {"alpaca": "alpaca50"}   # legacy emotion set name -> pipeline set


def overrides(patterns, stem):
    for pat, f in patterns:
        m = re.fullmatch(pat, stem)
        if m:
            return f(*m.groups())
    return None


# ------------------------------------------------------------------ writing entries
def write_steer(cfg, completions, stats):
    """One steer entry from legacy completions; returns the config it was written under, or None."""
    items = data.eval_items(cfg["data"]["eval"], data.set_params(cfg, cfg["data"]["eval"]))
    if len(items) != len(completions):
        stats["prompt mismatch"].append(cfg["data"]["eval"])
        return None
    for sample in range(4):
        cfg["generation"]["sample"] = sample
        key = steer_key(cfg)
        if not cache.done("steer", key):
            break
        cached = [json.loads(l)["completion"] for l in open(cache.artifact("steer", key))]
        if cached == completions:
            stats["already cached"] += 1
            return cfg
    else:
        raise RuntimeError(f"four different samples already cached for {cache.key_hash(key)}")
    path = cache.artifact("steer", key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps({**it, "completion": c}) + "\n" for it, c in zip(items, completions))
    cache.record("steer", key, cfg)
    stats[f"steer sample {sample}"] += 1
    return cfg


def write_labels(cfg, evaluator, scores, stats):
    cfg = resolve_evaluator(cfg, evaluator)
    key = evaluate_key(cfg)
    if cache.done("evaluate", key):
        return
    path = cache.artifact("evaluate", key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"evaluator": evaluator, "n": len(scores),
                                "rate": sum(bool(x["label"]) for x in scores) / len(scores), "items": scores}))
    cache.record("evaluate", key, cfg)
    stats[f"labels {evaluator}"] += 1


def resolve_evaluator(cfg, evaluator):
    from pipeline.cli import with_defaults

    return with_defaults({**cfg, "evaluator": {"name": evaluator, "params": {}}})


def check_prompts(cfg, legacy_prompts, stats):
    items = data.eval_items(cfg["data"]["eval"], data.set_params(cfg, cfg["data"]["eval"]))
    if [it["prompt"] for it in items] != legacy_prompts:
        stats["prompt mismatch"].append(f"{cfg['model']['model_id']} {cfg['data']['eval']}")
        return False
    return True


def import_refusal(name, mid, stats):
    for ds_dir in sorted((LEGACY / "gen" / name).glob("*")):
        ds = ds_dir.name
        for f in sorted(ds_dir.glob("*.json")):
            over = overrides(REFUSAL, f.stem)
            if over is None or ds not in data.EVAL_SETS or ds == "gsm8k":
                stats["not a pipeline point"].append(f"{ds}/{f.stem}")
                continue
            cfg = resolve(str(ROOT / "config" / "refusal.yaml"), [f"model.model_id={mid}", f"data.eval={ds}", *over])
            items = json.load(open(f))["items"]
            if not check_prompts(cfg, [it["prompt"] for it in items], stats):
                continue
            cfg = write_steer(cfg, [it["generation"] for it in items], stats)
            if cfg is None or ds not in data.HARMFUL_SETS:
                continue
            for ev, field in (("harmbench", "harmful"), ("llamaguard", "lg_unsafe")):
                if field in items[0]:
                    write_labels(cfg, ev, [{"label": bool(it[field])} for it in items], stats)


def import_emotion(name, mid, stats):
    for f in sorted((LEGACY / "emotion" / "gen" / name).glob("*.json")):
        over = overrides(EMOTION, f.stem)
        if over is None:
            stats["not a pipeline point"].append(f"emotion/{f.stem}")
            continue
        over = [o if not o.startswith("data.eval=") else f"data.eval={EMOTION_SETS.get(o[10:], o[10:])}" for o in over]
        cfg = resolve(str(ROOT / "config" / "emotion.yaml"), [f"model.model_id={mid}", *over])
        items = json.load(open(f))["items"]
        legacy_prompts = [data.story_prompt(it["topic"], it["asked"]) if it["kind"] == "story" else it["topic"] for it in items]
        if not check_prompts(cfg, legacy_prompts, stats):
            continue
        cfg = write_steer(cfg, [it["generation"] for it in items], stats)
        if cfg is None or not cfg["data"]["eval"].startswith("ask_") or "label" not in items[0]:
            continue
        # emotion_judge's own item format, from the legacy judged label
        write_labels(cfg, "emotion_judge", [{"label": it["label"] == it["asked"], "emotion": it["label"],
                                             **({"vocab": vocab_rate(it["generation"], it["asked"])} if it["asked"] else {})}
                                            for it in items], stats)


def main():
    from collections import Counter, defaultdict

    stats = defaultdict(int, {"prompt mismatch": [], "not a pipeline point": []})
    for name, mid in IDS.items():
        A = ACTS / f"{name}.pt"
        if not A.exists():
            continue
        L = torch.load(A, weights_only=False)
        cfg = resolve(str(ROOT / "config" / "refusal.yaml"), [f"model.model_id={mid}"])
        art = {"pools": {"harmful": torch.cat([L["harmful_calib"], L["harmful_heldout"]]),
                         "harmless": torch.cat([L["harmless_calib"], L["harmless_heldout"]])},
               "sets": {"harmbench": L["harmbench"]}}
        if (ACTS / f"{name}.mmlu.pt").exists():
            art["sets"]["mmlu"] = torch.load(ACTS / f"{name}.mmlu.pt", weights_only=False)
        else:
            cfg["extraction"]["sets"] = ["harmbench"]
        print(f"{name:14s} refusal  {put(cfg, art)}")
        if "jb_calib" in L:   # the jailbreak-direction concept: train-template vs plain harmful prompts
            jcfg = resolve(str(ROOT / "config" / "refusal.yaml"), [f"model.model_id={mid}", *JB])
            print(f"{name:14s} jailbreak {put(jcfg, {'pools': {'jb_calib': L['jb_calib'], 'jb_calib_plain': L['jb_calib_plain']}, 'sets': {'harmbench': L['harmbench']}})}")
        E = ACTS / f"{name}.emotion.pt"
        if E.exists():
            S = torch.load(E, weights_only=False)
            ecfg = resolve(str(ROOT / "config" / "emotion.yaml"), [f"model.model_id={mid}"])
            eart = {"stories": {e: S[e] for e in ("happy", "sad", "angry", "afraid")}, "neutral": S["neutral"]["tok"]}
            print(f"{name:14s} emotion  {put(ecfg, eart)}")
        if "--generations" in sys.argv:
            if (LEGACY / "gen" / name).exists():
                import_refusal(name, mid, stats)
            if (LEGACY / "emotion" / "gen" / name).exists():
                import_emotion(name, mid, stats)
    if "--generations" in sys.argv:
        for k, v in stats.items():
            if isinstance(v, int):
                print(f"{k}: {v}")
        print(f"prompt mismatch: {len(stats['prompt mismatch'])}", *sorted(set(stats["prompt mismatch"])), sep="\n  ")
        print("not a pipeline point (kept in legacy/results/):",
              *(f"{n:3d} x {t}" for t, n in sorted(Counter(stats["not a pipeline point"]).items())), sep="\n  ")


if __name__ == "__main__":
    main()
