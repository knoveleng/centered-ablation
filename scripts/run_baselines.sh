#!/bin/bash
# Control directions on the three study models (draw 0): random, LDA, kernel fields, MMD erasure
# variants, the reweighted fixed point; the whitened (Rayleigh) directions are diagnosed only.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/models.sh
CFG=config/refusal.yaml
for m in "${STUDY[@]}"; do
  F=$(model_flags "$m")
  python scripts/steer_many.py --config $CFG $F --grid direction.name=random --grid direction.params.seed=0 --grid data.eval=harmbench,alpaca
  python scripts/steer_many.py --config $CFG $F --grid direction.name=lda --grid direction.params.shrinkage=0.9 --grid data.eval=harmbench,alpaca
  python scripts/steer_many.py --config $CFG $F --grid direction.name=fixed_point --grid direction.params.iters=50 --grid data.eval=harmbench,alpaca
  python scripts/steer_many.py --config $CFG $F --grid direction.name=kde_field \
    --grid direction.params.gamma=1.0,0.5,0.25,0.125,0.0625 --grid data.eval=harmbench,alpaca
  E="--set direction.name=mmd_erasure --set direction.params.steps=400 --set direction.params.pca_k=0"
  for v in "rbf true 1.0" "rbf true 0.5" "rbf true 2.0" "linear true 1.0" "rbf false 1.0"; do
    read -r k n g <<< "$v"
    python scripts/steer_many.py --config $CFG $F $E --set direction.params.kernel=$k \
      --set direction.params.normalize=$n --set direction.params.gamma=$g --grid data.eval=harmbench,alpaca
  done
  python scripts/steer_many.py --config $CFG $F $E --set direction.params.kernel=rbf --set direction.params.normalize=true \
    --set direction.params.gamma=1.0 --set direction.params.pca_k=64 --grid data.eval=harmbench,alpaca
  for lam in 1.0 0.5 0.2 0.05 0.01; do
    for s in 0.5 1.0 2.0; do
      python -m pipeline.cli diagnose --config $CFG $F --set direction.name=rayleigh --set direction.params.lam=$lam --set diagnose.strength=$s
    done
  done
done
for ev in degeneration refusal_keyword faithfulness harmbench; do
  python scripts/evaluate_pending.py --evaluator $ev --where concept.name=refusal
done
python scripts/aggregate_results.py
