# Centered Ablation

Code and results for *Centered Ablation: Why Last-Layer Directional Ablation Under-Fires or
Collapses, and a Closed-Form Fix*.

**[Project page](https://knoveleng.github.io/centered-ablation/)**

<p align="center">
  <img src="docs/assets/centered_ablation_mechanism.gif" width="720"
       alt="Animation: standard (DIM) ablation moves benign states; Centered Ablation moves only harmful states">
  <br><sub>Schematic 2D picture, not real activations.</sub>
</p>

Directional ablation removes the component of a hidden state along a behaviour direction (e.g.
refusal). At the last residual stream it is exactly a logit bias scaled by a gate, the cosine
$\kappa_u(h) = h^\top u / \lVert h\rVert$ between the state and the direction. Last-layer states share a large common mode, so the standard
difference-in-means (DIM) direction fires on benign text and under-fires on its targets.
**Centered Ablation (CA)** projects the benign common mode out of the direction, in closed form.

Every method here is one operator at one site, the output of the last decoder block, applied at
every position:

$$
h \leftarrow h - s\,\big(h^\top u - c\big)\,u .
$$

| Direction | $u$ | $c$ |
|---|---|---|
| DIM (standard) | $u_{\mathrm{DIM}} = \mathrm{unit}(\mu_+ - \mu_-)$ | $0$ |
| DIM+offset (`ace`, ACE's affine correction at $\alpha = 0$) | $u_{\mathrm{DIM}}$ | $\mu_-^\top u_{\mathrm{DIM}}$ |
| **CA** (ours) | $u_{\mathrm{CA}} = \mathrm{unit}\big((I - \hat\mu_-\hat\mu_-^\top)\,u_{\mathrm{DIM}}\big)$ | $0$ |

Here $\mu_+$ and $\mu_-$ are the mean last-layer states of behaviour and benign prompts,
$\mathrm{unit}(x) = x/\lVert x\rVert$, and $\hat\mu_- = \mathrm{unit}\big(\mathbb{E}_{\mathcal{D}_-}[h/\mathrm{rms}(h)]\big)$
is the benign common mode.

## Install

```bash
pip install -r requirements.txt     # a CUDA GPU for extract / steer / judges; everything else runs on CPU
```

## Run one point

```bash
python -m pipeline.cli run --config config/refusal.yaml \
  --set model.model_id=meta-llama/Llama-3.2-3B-Instruct \
  --set direction.name=ca --set calibration_seed=0 \
  --set data.eval=harmbench --set evaluator.name=harmbench
```

`run` chains `extract → direction → diagnose → steer → evaluate`; name a stage instead of `run`
to stop there. `--print-config` shows the resolved configuration. Every stage is cached under
`cache/`, so re-running a finished point does nothing. Strength $s = 0$ is the unsteered model.

**Cost controls:** `--set cache_only=true` refuses any cache miss before a model loads;
`--set budget.max_gpu_hours=N` stops (exit 75) once N GPU-hours are logged.

## Run the experiments (GPU)

```bash
scripts/run_main.sh            # 8 models × {DIM, DIM+offset, CA} × 6 draws, diagnostics, all judges
scripts/run_strength.sh        # strength vs locality on Gemma-2-9B
scripts/run_baselines.sh       # control directions on the three study models
scripts/run_amplification.sh   # negative strength against jailbreaks
scripts/run_emotion.sh         # the second concept (emotional tone)
```

## Documentation

- [`docs/usage.md`](docs/usage.md): running points, stages, grids, sweeps and analyses.
- [`docs/configuration.md`](docs/configuration.md): every config key, overrides, and what each
  change re-runs.
- [`docs/architecture.md`](docs/architecture.md): layout, stages, cache, config and outputs.
- [`docs/extending.md`](docs/extending.md): adding a model, direction, evaluator, evaluation set,
  analysis or sweep.
