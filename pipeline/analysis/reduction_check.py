"""RQ1: does the closed form predict the served model's steered first token?

For one model, compares each cached diagnose entry's predicted first tokens on HarmBench with
the served completions of the same direction and draw: a prompt agrees when the predicted
token is a refusal onset exactly when the served completion is a keyword refusal. Refusal
onsets are the predicted unsteered first tokens of the prompts the unsteered model refuses.

    python -m pipeline.analysis.reduction_check --model meta-llama/Llama-3.2-3B-Instruct
"""

import argparse
import json

from pipeline import cache
from pipeline.cli import steer_key, unsteered
from pipeline.evaluators.refusal_keyword import is_refusal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    rows = []
    for rc in sorted((cache.CACHE / "diagnose").glob("*/run_config.json")):
        cfg = json.load(open(rc))
        if cfg["model"]["model_id"] != args.model or cfg["concept"]["name"] != "refusal":
            continue
        dg = json.load(open(rc.parent / cache.ARTIFACT["diagnose"]))
        cfg = {**cfg, "data": {**cfg["data"], "eval": "harmbench"}, "steering": {"strength": dg["strength"]}}
        if not (cache.done("steer", steer_key(cfg)) and cache.done("steer", steer_key(unsteered(cfg)))):
            continue
        head = cfg["defaults"]["evaluator"]["refusal_keyword"]["head_chars"]
        served = [is_refusal(json.loads(l)["completion"], head) for l in open(cache.artifact("steer", steer_key(cfg)))]
        base = [is_refusal(json.loads(l)["completion"], head) for l in open(cache.artifact("steer", steer_key(unsteered(cfg))))]
        onset = {t for t, r in zip(dg["first"]["unsteered"], base) if r}
        predicted = [t in onset for t in dg["first"]["steered"]]
        agree = sum(p == s for p, s in zip(predicted, served)) / len(served)
        rows.append((cfg["direction"]["name"], cfg["calibration_seed"], agree, sum(predicted) / len(predicted), sum(served) / len(served)))
        print(f"{cfg['direction']['name']:5s} draw {cfg['calibration_seed']}: agreement {agree:.3f}  "
              f"predicted refusal {rows[-1][3]:.3f}  served refusal {rows[-1][4]:.3f}")
    if rows:
        print(f"mean agreement over {len(rows)} cells: {sum(r[2] for r in rows) / len(rows):.3f}")


if __name__ == "__main__":
    main()
