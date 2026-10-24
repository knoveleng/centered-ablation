# Architecture

The code answers one question: what does rank-1 ablation at the last residual stream do, and
which correction of it (none, offset or direction) should a given model use? It is built so that
every number in the results traces back to one cached computation, and so that nothing expensive
(a GPU generation or a judge call) ever runs by accident.

## Principles

1. **One config holds every tunable value:** `config/default.yaml`. `config/refusal.yaml` and
   `config/emotion.yaml` add only a concept's shape (which pools, which token states).
2. **One override mechanism:** `--set key.path=value`, in the CLI and in every script.
3. **One process computes one point:** one model × direction × calibration draw × strength ×
   evaluation set × evaluator × sample. Loops live in `scripts/`.
4. **Staged by cost, cached by content.** Each stage's cache key holds exactly the inputs that
   change its output. A cache entry existing is the only record that work ran.
5. **The cache is the record.** vLLM is not bit-reproducible, so completions and judge labels are
   kept like paid API responses. A point generated twice keeps both (`generation.sample` 0 and 1).
6. **The operator is fixed:** `h ← h − s (hᵀu − c) u` after the last decoder block, at every
   position. A direction only supplies `u` and `c`.
7. **Every number has one source.** Analyses write `results/analysis/<name>.json`; tables and
   figures only lay those files out.

## Layout

```
config/
  default.yaml       resources, sets, cost controls, run defaults, component defaults, study
  refusal.yaml       concept: harmful vs benign pools, prompt-final states
  emotion.yaml       concept: target emotion vs the other three, story-token states
data/                prompt sources (AdvBench, Alpaca, HarmBench, JailbreakBench, XSTest)
pipeline/
  cli.py             stages, --set, defaults merge, validation, cache keys
  cache.py           content-addressed stage cache, index, GPU ledger, cache_only / budget
  data.py            calibration pools and evaluation sets
  model.py           HF extraction and vLLM steering
  operator.py        the operator and Steer (u, c)
  diagnostics.py     ω, η, β and the closed-form first token
  stats.py           bootstrap and paired tests
  directions/        the studied family: dim, ace (DIM+offset), ca
  baselines/         control directions: random, lda, kde_field, mmd_erasure, fixed_point, rayleigh
  evaluators/        harmbench, llamaguard, refusal_keyword, degeneration, faithfulness,
                     pairwise_quality, emotion_judge, gsm8k
  analysis/          one module per analysis; paper_tables.py, paper_figures.py lay them out
scripts/
  run_*.sh           the experiment sweeps (the only scripts that generate or judge)
  steer_many.py      loads a model once for a grid of steer points
  evaluate_pending.py  loads a judge once for every unjudged completion
  aggregate_results.py, import_legacy.py, concept_review.py, models.sh, maintenance/
tests/               test_legacy_equivalence.py
cache/               stage cache (git-ignored; expensive, back it up)
results/analysis/    <name>.json + <name>.md for every analysis (versioned)
legacy/results/      frozen inputs from the pre-refactor code that the cache cannot reproduce
```

## Stages

A run is five stages. Each one is cached under `cache/<stage>/<hash>/` with its artifact and the
resolved `run_config.json`.

| Stage | Loads | Cache key | Artifact |
|---|---|---|---|
| `extract` | HF model (GPU) | model, concept and pools, parameters of every set read | `activations.pt` |
| `direction` | nothing | extract key, direction and params, calibration draw and size | `direction.pt`: u, c, u_DIM, μ̂₋ |
| `diagnose` | W_U and G only | direction key, `diagnose` | `diagnostics.json` |
| `steer` | vLLM model (GPU) | model, evaluation set and params, generation (incl. sample); direction key and strength if s ≠ 0 | `completions.jsonl` |
| `evaluate` | the judge (GPU or none) | steer key, evaluator and merged params | `scores.json` |

What re-runs what: an evaluator re-runs `evaluate`; a strength, evaluation set or sample re-runs
`steer` and `evaluate`; a direction or draw re-runs `direction` and everything after it; a model
or a set's build parameters re-run everything.

Editing a prompt template or a judge prompt changes outputs without changing a key. When you do,
bump a parameter of that component or delete its entries deliberately.

## Configuration

`config/default.yaml` holds resources, set builders, cost controls, the defaults of one run, each
component's parameters and the analysis settings; a concept file adds that concept's shape, and
`--set` overrides the rest. Component defaults are merged into the run before keying. Every key
is described in [`configuration.md`](configuration.md).

## Cost controls

- `--set cache_only=true`: any miss raises `CacheMiss` before a model or judge loads. Every
  analysis runs this way.
- GPU stages append wall time, GPU type and engine versions to `cache/ledger.jsonl`.
- `--set budget.max_gpu_hours=N` exits with code 75 before a GPU stage once the ledger reaches N.
- `steer_many.py` prints a GPU-hour estimate from the ledger before a sweep starts.

## Plugins

Directions, baselines and evaluators are discovered from their directories: one file per
implementation, exporting `DIRECTION` or `EVALUATOR`.

```python
class Direction(ABC):                       # pipeline/directions/base.py, also used by baselines/
    PARAM_SCHEMA: dict[str, type] = {}      # defaults in config defaults.direction.<name>
    def fit(self, pos, neg, ref, params, heldout=None) -> Steer: ...

class Evaluator(ABC):                       # pipeline/evaluators/base.py
    NEEDS_REFERENCE = False                 # also receives the unsteered completions
    NEEDS_GPU = False
    SETS = None                             # evaluation sets it applies to; None = all
    def score(self, items, completions, reference=None) -> list[dict]: ...
```

`pipeline/directions/` holds the studied family; every other direction is a control in
`pipeline/baselines/` and is reported separately.

## Outputs

Each analysis is a `compute() -> dict` and a `render(dict)`, saved by
`pipeline/analysis/_output.save(name, results, render)`:

- `results/analysis/<name>.json`: `{"meta": {analysis, command, commit, dirty, date, study},
  "results": ...}` at full precision. Proportions stay fractions; keys use config ids.
- `results/analysis/<name>.md`: the readable report, rendered from the written JSON, so it cannot
  show a number the JSON lacks.

`paper_tables.py` and `paper_figures.py` read only these files (`_output.load(name)`), and write
`paper/tables/appendix/*.tex` and `paper/figures/*.pdf` (PNG previews go to `results/previews/`).

## Legacy inputs

A few results come from the pre-refactor code in `legacy/` and are not expressible in the pipeline:
directions fitted on generation-time states and the KDE-gated amplification. Their outputs are
read as frozen inputs from `legacy/results/` (`study.frozen`). `scripts/import_legacy.py` imports
every legacy generation the pipeline can express into the cache, and
`tests/test_legacy_equivalence.py` checks that the direction stage reproduces every legacy
direction.
