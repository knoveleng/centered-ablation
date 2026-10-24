"""Prompt sources: calibration pools per concept, and named evaluation sets.

Every item is a dict with `prompt` (the user turn), `behavior` (what a judge is shown as the
request; the unwrapped request for jailbreak-templated prompts) and optional `reference`,
`asked` (emotion requested) and `response` (an assistant turn whose tokens are read).

How each set is built (shuffle seeds, slices, sizes) is config (`sets.<name>` in
config/default.yaml); every builder takes that dict as `params`, and `set_params` hands it out.

Calibration draws (refusal): draw 0 is the first `calibration_size` prompts of each shuffled
pool; draw k >= 1 samples that many per class from the pool, never touching the excluded rows
(the Alpaca evaluation slice).
"""

import json
import random
from functools import cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

EMOTIONS = ["happy", "sad", "angry", "afraid"]
# 40 story topics written for this study, disjoint from the emotion dataset's own topics.
EVAL_TOPICS = [
    "a lighthouse keeper during the last night before automation", "a bakery on a rainy morning",
    "a lost dog finding its way home", "a train journey through the mountains",
    "a grandmother teaching a child to fish", "a scientist whose experiment finally runs",
    "a musician playing on a street corner", "a family moving into a new apartment",
    "a firefighter's first day on the job", "a hiker caught in a sudden storm",
    "a chess player facing an old rival", "a nurse on a night shift",
    "a gardener whose roses bloom early", "a pilot landing in heavy fog",
    "two friends meeting after twenty years", "a child's first day at a new school",
    "a fisherman returning with an empty net", "a painter finishing a mural",
    "a student waiting for exam results", "a chef cooking for a food critic",
    "a mail carrier on the last route before retirement", "a runner at the end of a marathon",
    "a beekeeper checking the hives in spring", "an astronaut looking at Earth",
    "a librarian closing the library for the night", "a farmer watching the first harvest",
    "a small-town election count", "a violinist before an audition",
    "a boy building a raft on a river", "a doctor giving test results",
    "a woman repairing her late father's clock", "a crowded ferry crossing",
    "a village festival at dusk", "a mechanic fixing an old truck",
    "a writer receiving a letter from a publisher", "a sailor on a calm sea",
    "a teenager learning to drive", "a museum guard alone at night",
    "a potter at the wheel", "a couple planting a tree together",
]


def _items(prompts, behaviors=None, **extra):
    return [{"prompt": p, "behavior": b, **{k: v[i] for k, v in extra.items()}}
            for i, (p, b) in enumerate(zip(prompts, behaviors or prompts))]


def set_params(cfg: dict, name: str) -> dict:
    """The build parameters of one set (empty for sets without any)."""
    return cfg["sets"].get(name) or {}


@cache
def _shuffled(file: str, key: str, seed: int) -> tuple[str, ...]:
    rows = [r[key] for r in json.load(open(DATA / file))]
    random.Random(seed).shuffle(rows)
    return tuple(rows)


def _extra(name: str) -> list[dict]:
    rows = json.load(open(DATA / f"extra_{name}.json"))
    return _items([r["prompt"] for r in rows], [r["behavior"] for r in rows])


# ------------------------------------------------------------------ refusal-shaped concepts
# Pools whose calibration states are prompt-final. The full pool is extracted once; a draw
# picks rows from it. `excluded` rows are evaluation prompts and are never drawn.
POOLS = {
    "harmful": lambda p: list(_shuffled("advbench.json", "prompt", p["shuffle_seed"])),
    "harmless": lambda p: list(_shuffled("alpaca.json", "instruction", p["shuffle_seed"])),
    "jb_calib": lambda p: [x["prompt"] for x in _extra("jb_calib")],
    "jb_calib_plain": lambda p: [x["prompt"] for x in _extra("jb_calib_plain")],
}


def pool_items(name: str, params: dict) -> list[dict]:
    return _items(POOLS[name](params))


def draw(pos: tuple[str, dict], neg: tuple[str, dict], seed: int, size: int, n_pos: int, n_neg: int):
    """Row indices into each pool for one calibration draw (legacy-exact). pos, neg: (pool, params)."""
    import torch

    def allowed(pool, n):
        ex = set(range(*pool[1]["excluded"])) if "excluded" in pool[1] else set()
        return [i for i in range(n) if i not in ex]

    a_pos = allowed(pos, n_pos)
    a_neg = allowed(neg, n_neg)
    kp, kn = min(size, len(a_pos)), min(size, len(a_neg))   # each class on its own
    if seed == 0:
        return a_pos[:kp], a_neg[:kn]
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(len(a_pos), generator=g)[:kp].tolist()
    q = torch.randperm(len(a_neg), generator=g)[:kn].tolist()
    return [a_pos[i] for i in p], [a_neg[i] for i in q]


# ------------------------------------------------------------------ emotion
def story_prompt(topic: str, emotion: str | None = None) -> str:
    if emotion:
        return f"Write a short story (about 150 words) about: {topic}. The main character feels {emotion}."
    return f"Write a short story about: {topic}."


def emotion_stories(params: dict) -> dict[str, list[dict]]:
    """Calibration stories per emotion: the first `per_emotion` of a seeded shuffle, emotions in
    EMOTIONS order (the shuffles share one generator, as in the original extraction)."""
    from datasets import load_dataset

    ds = load_dataset("knoveleng/emotion-datasets", "stories", split="train")
    rng = random.Random(params["shuffle_seed"])
    out = {}
    for emo in EMOTIONS:
        rows = [r for r in ds if r["emotion"] == emo]
        rng.shuffle(rows)
        out[emo] = [{"prompt": story_prompt(r["topic"]), "response": r["text"]} for r in rows[:params["per_emotion"]]]
    return out


def neutral_dialogues() -> list[dict]:
    from datasets import load_dataset

    out = []
    for r in load_dataset("knoveleng/emotion-datasets", "neutral_dialogues", split="train"):
        turns = r["turns"]
        if isinstance(turns, str):
            import ast
            turns = ast.literal_eval(turns)
        user = next(t["content"] for t in turns if t["role"] == "human")
        asst = next(t["content"] for t in turns if t["role"] == "assistant")
        out.append({"prompt": user, "response": asst})
    return out


# ------------------------------------------------------------------ evaluation sets
def _harmbench(p):
    rows = [x for x in json.load(open(DATA / "harmbench_test.json")) if x["semantic_category"] != "copyright"]
    return _items([x["prompt"] for x in rows])


def _alpaca(p):
    rows = list(_shuffled("alpaca.json", "instruction", p["shuffle_seed"]))[slice(*p["slice"])]
    return _items(rows[:p["n"]] if "n" in p else rows)


def _gsm8k(p):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(p["n"]))
    qs = [r["question"] + "\nSolve step by step and end with 'The answer is <number>.'" for r in ds]
    gold = [r["answer"].split("####")[-1].strip().replace(",", "") for r in ds]
    return _items(qs, reference=gold)


def _mmlu(p):
    from datasets import load_dataset

    d = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=p["shuffle_seed"]).select(range(p["n"]))
    q = [r["question"] + "\n" + "\n".join(f"{L}. {c}" for L, c in zip("ABCD", r["choices"])) +
         "\nAnswer with only the letter." for r in d]
    return _items(q, reference=list(d["answer"]))


def _ask(emotion):
    return [{"prompt": story_prompt(t, emotion), "behavior": story_prompt(t, emotion), "asked": emotion} for t in EVAL_TOPICS]


EVAL_SETS = {
    "harmbench": _harmbench,
    "alpaca": _alpaca,
    "alpaca50": _alpaca,
    "gsm8k": _gsm8k,
    "mmlu": _mmlu,
    "ask_none": lambda p: _ask(None),
    **{f"ask_{e}": (lambda p, e=e: _ask(e)) for e in EMOTIONS},
    **{n: (lambda p, n=n: _extra(n)) for n in
       ["xstest_safe", "jbb_benign", "jbb_plain", "jbb_aim", "jbb_dev_mode", "jbb_evil_confidant",
        "jbb_few_shot_compliance", "jbb_fiction", "jbb_prefix_injection", "jbb_refusal_suppression", "jbb_research"]},
}

# Sets judged for attack success (harmful requests); the rest are benign or capability sets.
HARMFUL_SETS = {"harmbench", "jbb_plain", "jbb_aim", "jbb_dev_mode", "jbb_evil_confidant", "jbb_few_shot_compliance",
                "jbb_fiction", "jbb_prefix_injection", "jbb_refusal_suppression", "jbb_research"}


def eval_items(name: str, params: dict) -> list[dict]:
    if name not in EVAL_SETS:
        raise ValueError(f"unknown evaluation set '{name}'; known: {sorted(EVAL_SETS)}")
    return EVAL_SETS[name](params)
