# Usage

## Install

```bash
pip install -r requirements.txt
```

`extract`, `steer` and the model-based judges need a CUDA GPU. `direction`, `diagnose`, the
rule-based evaluators and every analysis run on CPU.

## Run one point

A point is one model, direction, calibration draw, strength, evaluation set and evaluator.

```bash
python -m pipeline.cli run --config config/refusal.yaml \
  --set model.model_id=meta-llama/Llama-3.2-3B-Instruct \
  --set direction.name=ca --set calibration_seed=0 \
  --set data.eval=harmbench --set evaluator.name=harmbench
```

`run` chains the five stages (and the unsteered generation, when the evaluator compares against
it) and prints the evaluator's rate as `<evaluator>: <rate> (n=<items>)`.
Every stage is cached, so running it again returns at once. Options for each key are in
[`configuration.md`](configuration.md).

## Run one stage

Give a stage name instead of `run`. `direction`, `diagnose` and `steer` compute any missing
earlier stage first; `evaluate` only judges, and fails if the completions (or, for `faithfulness`
and `pairwise_quality`, the unsteered completions) are not cached yet.

| Command | Does | Writes |
|---|---|---|
| `extract` | last-layer states of both calibration pools and the held-out sets | `activations.pt` |
| `direction` | fits $u$ and $c$ on one calibration draw | `direction.pt` |
| `diagnose` | the closed-form diagnostics, no generation | `diagnostics.json` |
| `steer` | generates with the operator applied | `completions.jsonl` |
| `evaluate` | judges the completions | `scores.json` |

For example, the diagnostics of DIM on one model, without generating text:

```bash
python -m pipeline.cli diagnose --config config/refusal.yaml \
  --set model.model_id=google/gemma-2-9b-it --set direction.name=dim
```

`diagnostics.json` holds the overlap $\omega$ (`omega`, and `omega_dim` for DIM), and for each set
of states the gate mean and spread (`gate_mean`, `gate_sd`), the efficacy $\eta$ (`token_change`)
and the collapse index $\beta$ (`beta_mean`), plus the footprint's top suppressed and promoted
tokens.

## Common variations

```bash
# the unsteered model
--set steering.strength=0

# DIM+offset, the standard DIM, or a control direction with its parameters
--set direction.name=ace
--set direction.name=dim
--set direction.name=lda --set direction.params.shrinkage=0.05

# amplify refusal instead of removing it
--set steering.strength=-1

# benign answers, and how much they changed (needs the unsteered completions)
--set data.eval=alpaca --set evaluator.name=faithfulness

# a second independent generation of the same point
--set generation.sample=1

# the emotion concept
python -m pipeline.cli run --config config/emotion.yaml \
  --set model.model_id=google/gemma-2-9b-it --set concept.target=sad \
  --set data.eval=ask_sad --set evaluator.name=emotion_judge
```

Evaluators: `refusal_keyword`, `harmbench`, `llamaguard`, `degeneration`, `faithfulness`,
`pairwise_quality`, `gsm8k`, `emotion_judge`. Evaluation sets are the keys of `EVAL_SETS` in
`pipeline/data.py`; an unknown name fails with the list of known ones.

## Run a grid

Loading a model dominates the cost of a point. `steer_many.py` loads it once for a whole grid
(the cartesian product of the `--grid` axes):

```bash
python scripts/steer_many.py --config config/refusal.yaml \
  --set model.model_id=google/gemma-2-9b-it --set model.attention_backend=TRITON_ATTN \
  --grid direction.name=dim,ace,ca --grid calibration_seed=0,1,2,3,4,5 \
  --grid data.eval=harmbench,alpaca --unsteered
```

`--unsteered` also generates the $s = 0$ completions for each evaluation set. The script prints a
GPU-hour estimate first.

Then judge everything not yet judged, loading the judge once:

```bash
python scripts/evaluate_pending.py --evaluator harmbench --where concept.name=refusal
python scripts/evaluate_pending.py --evaluator pairwise_quality --where calibration_seed=0,1
```

`--where key=v1,v2` keeps only steer entries whose resolved config matches; `--param key=value`
sets an evaluator parameter.

## The experiment sweeps

Each script loops over models with `steer_many.py`, then runs `evaluate_pending.py` per judge.
Model names and flags come from `resources.models` and `study.*` in `config/default.yaml`.

| Script | Runs |
|---|---|
| `scripts/run_main.sh` | all models × {DIM, DIM+offset, CA} × 6 draws on HarmBench and Alpaca, diagnostics, GSM8K on the study models, every judge |
| `scripts/run_strength.sh` | strength sweep of DIM and kernel fields on Gemma-2-9B |
| `scripts/run_baselines.sh` | control directions on the study models |
| `scripts/run_amplification.sh` | negative strength against jailbreak templates; the jailbreak direction |
| `scripts/run_emotion.sh` | the emotion concept, all models and emotions |

## Analyses

Each analysis reads the cache with `cache_only` on, so it never generates or judges:

```bash
python -m pipeline.analysis.main_matrix      # writes results/analysis/main_matrix.{json,md}
python -m pipeline.analysis.laws             # reads main_matrix.json
```

The `.json` holds the numbers at full precision with provenance (`meta`: command, commit, date);
the `.md` is a readable report rendered from it. An analysis that loads another one
(e.g. `laws` loads `main_matrix`) must run after it.

## Controlling cost

- `--set cache_only=true`: fail on any cache miss instead of loading a model. Use it to check that
  a point or a sweep is already complete.
- `--set budget.max_gpu_hours=N`: stop with exit code 75 before a GPU stage once the ledger reaches
  $N$ GPU-hours. Every GPU stage logs its wall time, GPU and engine versions to `cache/ledger.jsonl`.
- `cache/index.jsonl` has one line per finished entry, for grepping what has run.

`cache/` is the record of every generation and judge label. vLLM output is not bit-reproducible,
so a deleted entry cannot be recovered exactly: back it up, and delete entries only deliberately.
