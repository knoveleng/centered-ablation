"""Last-layer origin-corrected ablation: extract → direction → diagnose → steer → evaluate."""

import os

# The steering hook reaches vLLM's engine subprocess through apply_model(), which needs this.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
# FlashInfer's sampler fails at runtime on some GPUs; greedy decoding does not need it.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# Forking after torch CPU work deadlocks the engine process; spawn it instead.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
