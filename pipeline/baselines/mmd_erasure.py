"""Control: rank-1 kernel concept erasure.

The unit u minimizing MMD² between the classes *after* ablating u and applying the model's RMS
normalization, by gradient descent (Adam, `lr`) from DIM with held-out early stopping (the
first `n_stop` held-out rows of each pool). Variants isolate each ingredient:
  kernel=rbf|linear, normalize=true|false (false: per-dim standardization instead of RMS),
  gamma (bandwidth as a multiple of the median pairwise distance), pca_k (0 = unrestricted).
By Propositions B.4-B.5 the linear variants reduce to (reweighted) differences in means.
"""

import torch

from pipeline.directions.base import Direction
from pipeline.operator import Steer, rms_normalize


def _mmd2_rbf(a, b, sigma):
    k = lambda x, y: torch.exp(-torch.cdist(x, y).pow(2) / (2 * sigma ** 2))
    return k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean()


def _mmd2_linear(a, b, sigma=None):
    return (a.mean(0) - b.mean(0)).pow(2).sum()


class MMDErasure(Direction):
    PARAM_SCHEMA = {"kernel": str, "normalize": bool, "gamma": float, "pca_k": int, "steps": int,
                    "lr": float, "n_stop": int, "init_seed": int}

    def fit(self, pos, neg, ref, params, heldout=None):
        torch.manual_seed(params["init_seed"])
        pos, neg = pos.float(), neg.float()
        dim = pos.mean(0) - neg.mean(0)
        if params["normalize"]:
            post = rms_normalize
        else:
            X = torch.cat([pos, neg])
            mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
            post = lambda x: (x - mu) / sd
        allz = post(torch.cat([pos, neg]))
        sigma = params["gamma"] * torch.cdist(allz, allz).flatten().median()
        dist = _mmd2_rbf if params["kernel"] == "rbf" else _mmd2_linear
        B = None
        if params["pca_k"]:
            X = torch.cat([pos, neg])
            V = torch.linalg.svd(X - X.mean(0), full_matrices=False).Vh[:params["pca_k"]]
            B = torch.linalg.qr(torch.cat([dim[None] / dim.norm(), V]).T).Q
        init = dim / dim.norm()
        c = torch.nn.Parameter(B.T @ init if B is not None else init.clone())
        opt = torch.optim.Adam([c], lr=params["lr"])
        unit = lambda: (B @ c if B is not None else c) / (B @ c if B is not None else c).norm()
        loss_on = lambda p, n, u: dist(post(p - (p @ u)[:, None] * u), post(n - (n @ u)[:, None] * u), sigma)
        k = params["n_stop"]
        val = (heldout[0][:k].float(), heldout[1][:k].float()) if heldout else None
        best = (float("inf"), init)
        for step in range(params["steps"] + 1):
            u = unit()
            loss = loss_on(pos, neg, u)
            if step % 10 == 0:
                with torch.no_grad():
                    v = loss_on(*val, u).item() if val else loss.item()
                if v < best[0]:
                    best = (v, u.detach().clone())
            if step == params["steps"]:
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
        return Steer(best[1], 0.0)


DIRECTION = MMDErasure
