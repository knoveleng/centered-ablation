#!/bin/bash
# The other sign (RQ4): amplify the refusal direction (DIM-Amp) or its centered version (CA-Amp)
# against 8 jailbreak templates (4 held out) and two over-refusal sets; and ablate the jailbreak
# direction (train-template vs plain harmful states).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/models.sh
CFG=config/refusal.yaml
SETS=jbb_plain,jbb_prefix_injection,jbb_refusal_suppression,jbb_fiction,jbb_aim,jbb_dev_mode,jbb_research,jbb_evil_confidant,jbb_few_shot_compliance,jbb_benign,xstest_safe
for m in "${STUDY[@]}"; do
  F=$(model_flags "$m")
  python scripts/steer_many.py --config $CFG $F --unsteered --grid direction.name=ca \
    --grid steering.strength=-0.5,-1,-1.5,-2,-3 --grid data.eval=$SETS
  python scripts/steer_many.py --config $CFG $F --grid direction.name=dim \
    --grid steering.strength=-0.5,-1,-1.5,-2 --grid data.eval=$SETS
  python -m pipeline.cli extract --config $CFG $F --set concept.positive=jb_calib --set concept.negative=jb_calib_plain \
    --set "extraction.sets=[harmbench]"
  python scripts/steer_many.py --config $CFG $F --set concept.positive=jb_calib --set concept.negative=jb_calib_plain \
    --set "extraction.sets=[harmbench]" --grid direction.name=dim --grid steering.strength=1,2 --grid data.eval=$SETS
done
for ev in degeneration refusal_keyword harmbench llamaguard; do
  python scripts/evaluate_pending.py --evaluator $ev --where concept.name=refusal
done
python scripts/aggregate_results.py
