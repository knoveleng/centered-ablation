"""Paired statistics over prompts (the resampling unit), as used for every comparison."""

from math import comb

import torch


def mcnemar_p(a: torch.Tensor, b: torch.Tensor) -> float:
    """Exact two-sided McNemar test on paired binary vectors."""
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap(X: torch.Tensor, Y: torch.Tensor, n_boot: int, seed: int) -> tuple[float, float, float]:
    """mean(Y) − mean(X) with a 95% interval. X, Y: (n_draws, n_prompts) binary outcomes; draws are
    averaged inside each replicate, so the prompt is the resampling unit. n_boot and seed: config
    study.bootstrap."""
    g = torch.Generator().manual_seed(seed)
    n = X.shape[1]
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    diffs = Y.float().mean(0)[idx].mean(1) - X.float().mean(0)[idx].mean(1)
    return (Y.float().mean() - X.float().mean()).item(), diffs.quantile(0.025).item(), diffs.quantile(0.975).item()
