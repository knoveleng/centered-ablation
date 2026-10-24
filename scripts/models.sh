# Sourced by scripts/run_*.sh. Model short names and per-model --set flags, read from
# config/default.yaml: resources.models (checkpoint, vLLM memory share, attention backend) and
# study.study_models / study.heldout_models.
_config() { python -c "import sys, yaml; c = yaml.safe_load(open('config/default.yaml')); print(eval(sys.argv[1]))" "$1"; }

STUDY=($(_config "' '.join(c['study']['study_models'])"))
HELDOUT=($(_config "' '.join(c['study']['heldout_models'])"))
ALL=("${STUDY[@]}" "${HELDOUT[@]}")

# --set flags for one model: checkpoint, vLLM memory share and (Gemma-2) the attention backend
model_flags() {
  _config "' '.join(f'--set model.{k}={v}' for k, v in
           {'model_id': c['resources']['models']['$1']['id'], **{k: v for k, v in c['resources']['models']['$1'].items() if k != 'id'}}.items())"
}
