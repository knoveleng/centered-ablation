# Extending

Each extension is one file or one config entry. New components are picked up automatically, and
their parameters become part of the cache key, so existing results are never overwritten. See
[`architecture.md`](architecture.md) for how the pieces fit.

| To add | Do |
|---|---|
| a model | one entry in `resources.models`; add it to `study.models` if analyses should report it |
| a direction in the family | one file in `pipeline/directions/`, defaults in `defaults.direction` |
| a control direction | one file in `pipeline/baselines/` |
| an evaluator | one file in `pipeline/evaluators/`, defaults in `defaults.evaluator` |
| an evaluation set | one builder in `EVAL_SETS` (`pipeline/data.py`), params in `sets.<name>` |
| a repeated generation | `--set generation.sample=1` |
| an analysis | one module in `pipeline/analysis/` |
| a sweep | one `scripts/run_*.sh` |

## A model

```yaml
# config/default.yaml
resources:
  models:
    mistral-7b: {id: mistralai/Mistral-7B-Instruct-v0.3, gpu_memory: 0.80}
```

`attention_backend` is optional (Gemma-2 needs `TRITON_ATTN`). `scripts/models.sh` reads these
entries, so sweeps pick the model up. To have analyses report it, add the short name to
`study.models` and to `study.heldout_models` or `study.study_models`.

## A direction

A direction maps one calibration draw to a `Steer(u, c)`. CA, in full:

```python
# pipeline/directions/ca.py
from pipeline.directions.base import Direction, common_mode, dim
from pipeline.operator import Steer, unit


class CA(Direction):
    def fit(self, pos, neg, ref, params, heldout=None):
        u, mu = dim(pos, neg), common_mode(ref)
        return Steer(unit(u - (u @ mu) * mu), 0.0)


DIRECTION = CA
```

- `pos`, `neg`: behaviour and benign calibration states, one row per prompt.
- `ref`: benign reference states that define the common mode (equal to `neg` for refusal).
- `heldout`: the pool rows outside this draw; only directions with early stopping use it.
- The file name is the direction's name: `--set direction.name=ca`.

With parameters, declare them and give defaults in the config:

```python
class MyDirection(Direction):
    PARAM_SCHEMA = {"shrinkage": float}
    def fit(self, pos, neg, ref, params, heldout=None):
        lam = params["shrinkage"]
        ...
```

```yaml
defaults:
  direction:
    my_direction: {shrinkage: 0.05}
```

Put it in `pipeline/directions/` only if it is a member of the studied family (one `u`, one `c`
from class means); anything else is a control and goes in `pipeline/baselines/`.

## An evaluator

```python
# pipeline/evaluators/my_judge.py
from pipeline.evaluators.base import Evaluator


class MyJudge(Evaluator):
    NEEDS_GPU = True                 # loads a model
    NEEDS_REFERENCE = False          # True: also receive the unsteered completions
    SETS = {"harmbench"}             # None: every evaluation set

    def score(self, items, completions, reference=None):
        threshold = self.params["threshold"]
        return [{"label": ..., "score": ...} for c in completions]


EVALUATOR = MyJudge
```

Each returned dict needs a boolean `label` (the headline outcome); other keys are stored as
per-item detail. `self.params` comes from `defaults.evaluator.my_judge` merged with the run's
params and is part of the cache key. A GPU judge's memory settings go in `resources.judges.my_judge`
and reach `self.runtime`; they are not keyed, since they do not change labels. To judge an
existing sweep without regenerating, run
`python scripts/evaluate_pending.py --evaluator my_judge` (`--where` filters the steer entries).

## An evaluation set

```python
# pipeline/data.py
def _my_set(p):
    rows = json.load(open(DATA / "my_set.json"))[: p["n"]]
    return _items([r["prompt"] for r in rows])     # one {"prompt", "behavior"} dict per item

EVAL_SETS = {
    ...,
    "my_set": _my_set,
}
```

```yaml
sets:
  my_set: {n: 200}
```

Add the name to `HARMFUL_SETS` if completions should be judged for attack success.

## An analysis

```python
# pipeline/analysis/my_analysis.py
"""What question this answers, and what it reads."""
from pipeline.analysis._output import load, save


def compute():
    mm = load("main_matrix")              # read other analyses, or the cache via study.py
    return {"unit": "fraction", "n": ..., "value": ...}


def render(r):
    print(f"value {r['value']:.3f}")


if __name__ == "__main__":
    save("my_analysis", compute(), render)
```

Run it with `python -m pipeline.analysis.my_analysis`, after any analysis it loads. Read cache entries through `pipeline/analysis/study.py` (`point`, `completions`, `labels`, `has`)
so the analysis stays offline. A table or figure built from it goes in `paper_tables.py` or
`paper_figures.py` and reads only `load("my_analysis")`.

## A sweep

Copy the closest `scripts/run_*.sh`. The pattern is one `steer_many.py` call per model (one model
load for the whole grid of directions, draws and strengths), then one `evaluate_pending.py` call
per judge. Before a long sweep, check the GPU-hour estimate `steer_many.py` prints, and consider
`--set budget.max_gpu_hours=N`.
