#!/bin/bash
# A second concept (RQ4): for every model and emotion, DIM/ACE/CA removal at s=1 on prompts asking
# for each emotion and on benign prompts; s=2,3 on the study models; amplification (s=-2) on
# emotion-free story prompts.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/models.sh
CFG=config/emotion.yaml
EMO=(happy sad angry afraid)
for m in "${ALL[@]}"; do
  F=$(model_flags "$m")
  python -m pipeline.cli extract --config $CFG $F
  for e in "${EMO[@]}"; do
    for d in dim ace ca; do
      python -m pipeline.cli diagnose --config $CFG $F --set concept.target=$e --set direction.name=$d
    done
    python scripts/steer_many.py --config $CFG $F --set concept.target=$e --unsteered --grid direction.name=dim,ace,ca \
      --grid data.eval=ask_happy,ask_sad,ask_angry,ask_afraid,alpaca50
    python scripts/steer_many.py --config $CFG $F --set concept.target=$e --unsteered --grid direction.name=dim,ca \
      --grid steering.strength=-2 --grid data.eval=ask_none
  done
done
for m in "${STUDY[@]}"; do
  for e in "${EMO[@]}"; do
    python scripts/steer_many.py --config $CFG $(model_flags "$m") --set concept.target=$e --grid direction.name=dim,ca \
      --grid steering.strength=2,3 --grid data.eval=ask_$e,alpaca50
  done
done
for ev in degeneration emotion_judge faithfulness; do
  python scripts/evaluate_pending.py --evaluator $ev --where concept.name=emotion
done
python scripts/aggregate_results.py
