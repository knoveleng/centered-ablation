#!/bin/bash
# Strength vs locality (ablation): DIM and two kernel fields on Gemma-2-9B across strengths.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/models.sh
CFG=config/refusal.yaml
m=gemma-2-9b
python scripts/steer_many.py --config $CFG $(model_flags $m) \
  --grid direction.name=dim --grid steering.strength=0.5,0.75,1.0,1.25 --grid data.eval=harmbench,alpaca
python scripts/steer_many.py --config $CFG $(model_flags $m) \
  --grid direction.name=kde_field --grid direction.params.gamma=0.25,0.125 \
  --grid steering.strength=0.5,0.75,1.0,1.25 --grid data.eval=harmbench
for ev in degeneration refusal_keyword faithfulness harmbench; do
  python scripts/evaluate_pending.py --evaluator $ev --where concept.name=refusal
done
python scripts/aggregate_results.py
