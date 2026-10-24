# Configuration

Every tunable value lives in YAML. A run's configuration is resolved in this order, each layer
overriding the one before:

1. `config/default.yaml`: everything shared by all runs.
2. A concept file, `config/refusal.yaml` or `config/emotion.yaml`, passed with `--config`.
3. Each `--set key.path=value`, in order.

Then the component defaults (`defaults.direction.<name>`, `defaults.evaluator.<name>`) are merged
into the run's `direction.params` and `evaluator.params`. To see the result:

```bash
python -m pipeline.cli run --config config/refusal.yaml --set model.model_id=Qwen/Qwen2.5-7B-Instruct --print-config
```

The same resolved config is written next to every cache entry as `run_config.json`.

## Overrides

`--set` takes a dotted path and a YAML value, so types are inferred:

```bash
--set steering.strength=1.5                 # float
--set calibration_seed=3                    # int
--set direction.params.normalize=true       # bool
--set extraction.sets='[harmbench, mmlu]'   # list
--set budget.max_gpu_hours=null             # null
```

Setting `direction.name` clears `direction.params`, because each direction has its own parameters.
Put `--set direction.params.<p>=...` after it.

## `config/default.yaml`

### `resources`

What the hardware needs to know. Not part of any cache key unless noted.

| Key | Meaning |
|---|---|
| `resources.models.<short>` | `id` (Hugging Face checkpoint), `gpu_memory` (vLLM memory share), optional `attention_backend` (Gemma-2 uses `TRITON_ATTN`) |
| `resources.judges.<name>` | `max_model_len` and `gpu_memory` of each vLLM judge |

The `scripts/run_*.sh` sweeps turn a short name into `--set model.model_id=... --set model.gpu_memory=...`
flags (`scripts/models.sh`). A single `pipeline.cli` run takes the checkpoint directly in `model.model_id`.

### `sets`

How each calibration pool and evaluation set is built. A set's parameters are part of the key of
every stage that reads the set, so changing one re-runs those stages.

| Set | Parameters |
|---|---|
| `harmful` | `shuffle_seed` (AdvBench, 416 prompts) |
| `harmless` | `shuffle_seed`, `excluded` (Alpaca rows kept out of calibration) |
| `alpaca` | `shuffle_seed`, `slice` (the benign evaluation slice) |
| `alpaca50` | as `alpaca`, plus `n` |
| `gsm8k` | `n` |
| `mmlu` | `n`, `shuffle_seed` |
| `stories` | `per_emotion`, `shuffle_seed` (emotion calibration) |

Sets without parameters (the JailbreakBench and XSTest files, `harmbench`, `ask_<emotion>`) are not
listed.

### Cost controls

Never part of a key.

| Key | Default | Meaning |
|---|---|---|
| `cache_only` | `false` | `true`: any cache miss raises `CacheMiss` before a model or judge loads |
| `budget.max_gpu_hours` | `null` | exit with code 75 before a GPU stage once `cache/ledger.jsonl` reaches this |

### One run

The point a single process computes.

| Key | Default | Meaning |
|---|---|---|
| `model.model_id` | `null` (required) | Hugging Face checkpoint |
| `model.max_model_len`, `model.gpu_memory`, `model.attention_backend` | `2048`, `0.80`, `null` | vLLM settings for the steer stage |
| `calibration_seed` | `0` | which calibration draw (the paper uses 0 to 5) |
| `calibration_size` | `256` | prompts per class in a draw |
| `direction.name` | `ca` | a file in `pipeline/directions/` or `pipeline/baselines/` |
| `direction.params` | `{}` | that direction's parameters |
| `diagnose.strength` | `1.0` | the strength $s$ the closed-form diagnostics assume |
| `diagnose.footprint_k` | `12` | tokens listed at each end of the footprint |
| `steering.strength` | `1.0` | $s$; `0` is the unsteered model, negative values amplify |
| `generation.temperature`, `generation.max_tokens` | `0.0`, `256` | decoding |
| `generation.sample` | `0` | index of a repeated generation of the same point |
| `evaluator.name` | `refusal_keyword` | a file in `pipeline/evaluators/` |
| `evaluator.params` | `{}` | merged over `defaults.evaluator.<name>` |

### `defaults`

Each component's parameters. They are merged into the run before keying, so editing one is a
cache miss for that component only.

| Component | Defaults |
|---|---|
| `direction.mmd_erasure` | `lr`, `n_stop` (held-out rows per class for early stopping), `init_seed` |
| `evaluator.degeneration` | `min_bytes`, `max_ratio` (zlib-compressed size ratio below which an output is degenerate) |
| `evaluator.refusal_keyword` | `head_chars` (characters searched for a refusal phrase) |
| `evaluator.faithfulness` | `threshold` |
| `evaluator.harmbench`, `evaluator.llamaguard` | `model_id`, `truncate` (tokens of the completion judged) |
| `evaluator.emotion_judge`, `evaluator.pairwise_quality` | `model_id`, `clip_chars` |

Directions without defaults need their parameters on the command line. The schema is checked
before anything loads:

| Direction | Parameters |
|---|---|
| `dim`, `ace`, `ca` | none |
| `random` | `seed` (int) |
| `lda` | `shrinkage` (float) |
| `rayleigh` | `lam` (float) |
| `kde_field` | `gamma` (float) |
| `fixed_point` | `iters` (int) |
| `mmd_erasure` | `kernel`, `normalize`, `gamma`, `pca_k`, `steps`, … (see `PARAM_SCHEMA` in its file) |

### `study`

Read only by the analyses, never by a stage.

| Key | Meaning |
|---|---|
| `models` | every reported model, in table order |
| `study_models`, `heldout_models` | the models the analysis was developed on, and those added later |
| `draws`, `directions` | the calibration draws and directions the analyses aggregate |
| `bootstrap` | `n` and `seed` of every bootstrap interval |
| `amplification_templates` | jailbreak templates held out from the jailbreak direction |
| `emotion_strength_emotions` | emotions in the emotion strength sweep |
| `gates_model` | the model whose per-prompt gates are plotted |
| `names` | display labels for models and directions |

## Concept files

### `config/refusal.yaml`

| Key | Default | Meaning |
|---|---|---|
| `concept.positive` | `harmful` | pool that shows the behaviour; `jb_calib` for the jailbreak direction |
| `concept.negative` | `harmless` | benign pool; also defines the common mode $\hat\mu_-$ |
| `extraction.sets` | `[harmbench, mmlu]` | held-out states the diagnose stage reads |
| `extraction.batch_size` | `8` | prompts per forward pass |
| `data.eval` | `harmbench` | the set the steer stage generates on |

States are read at the last prompt token.

### `config/emotion.yaml`

| Key | Default | Meaning |
|---|---|---|
| `concept.target` | `happy` | `happy`, `sad`, `angry` or `afraid`; the other three are the reference |
| `extraction.skip` | `8` | first story tokens ignored |
| `extraction.keep` | `16` | token states kept per story |
| `extraction.max_len` | `1024` | tokens read per story |
| `data.eval` | `ask_happy` | `ask_<emotion>`, `ask_none` or `alpaca50` |

States are read over the story tokens: the mean per story calibrates DIM, and the kept token
states define the common mode.

## What changing a value re-runs

| You change | Re-runs |
|---|---|
| an evaluator or its params | `evaluate` |
| `steering.strength`, `data.eval`, `generation.*` | `steer`, `evaluate` (GPU) |
| `direction.*`, `calibration_seed`, `calibration_size` | `direction` and everything after |
| `diagnose.*` | `diagnose` |
| `model.model_id`, a concept, `extraction.*`, a set's params | everything |
| `resources.*`, `cache_only`, `budget`, `study.*` | nothing |

Editing a prompt template or judge prompt in code changes outputs without changing a key. Bump a
parameter of that component, or delete its cache entries deliberately.
