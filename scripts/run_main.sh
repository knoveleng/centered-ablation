#!/bin/bash
# Main matrix (RQ1-RQ3): 8 models x {DIM, ACE, CA} x 6 calibration draws on HarmBench and the
# benign Alpaca set, diagnostics for every direction, GSM8K on the study models, then every judge.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/models.sh
CFG=config/refusal.yaml

for m in "${ALL[@]}"; do
  python -m pipeline.cli extract --config $CFG $(model_flags "$m")
  for d in dim ace ca; do
    for s in 0 1 2 3 4 5; do
      python -m pipeline.cli diagnose --config $CFG $(model_flags "$m") --set direction.name=$d --set calibration_seed=$s
    done
  done
  python scripts/steer_many.py --config $CFG $(model_flags "$m") --unsteered \
    --grid direction.name=dim,ace,ca --grid calibration_seed=0,1,2,3,4,5 --grid data.eval=harmbench,alpaca
done
for m in "${STUDY[@]}"; do
  python scripts/steer_many.py --config $CFG $(model_flags "$m") --set generation.max_tokens=512 --unsteered \
    --grid direction.name=dim,ace,ca --grid data.eval=gsm8k
done

for ev in degeneration refusal_keyword faithfulness gsm8k harmbench llamaguard; do
  python scripts/evaluate_pending.py --evaluator $ev --where concept.name=refusal
done
python scripts/evaluate_pending.py --evaluator pairwise_quality --where concept.name=refusal --where calibration_seed=0,1
python scripts/aggregate_results.py
