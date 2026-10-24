"""Generation-free diagnostics, computed from cached states and the head (W_U, G) alone.

All follow from Proposition 3.1: the steered greedy token is argmax[ℓ(h) − s√d·γ(h)·u_V],
with γ(h) = (h·u − c)/‖h‖ the gate and u_V = W_U G u the footprint.

  omega      |u·μ̂₋| for the fitted direction, and for DIM (the paper's ω)
  per set    mean and sd of the gate; fraction of greedy tokens the operator changes
             (η on the behaviour set; benign change on the benign set); collapse index β
  footprint  the tokens the direction suppresses and promotes most
  first      unsteered and steered greedy first-token ids on the behaviour set (for the
             reduction check against served completions, pipeline/analysis)
  mmlu       accuracy over the four letter tokens at the answer position, unsteered and steered
"""

import torch

from pipeline.operator import Steer


@torch.no_grad()
def token_change(head, X, steer: Steer, s: float) -> tuple[float, torch.Tensor, torch.Tensor]:
    base, new = head.argmax(X), head.argmax(steer.apply(X.to(head.device), s))
    return (base != new).float().mean().item(), base.cpu(), new.cpu()


@torch.no_grad()
def collapse_index(head, X, steer: Steer, s: float) -> torch.Tensor:
    """β(h) = s√d·|γ(h)|·sd(u_V)/sd(ℓ(h)); a field direction uses its per-state footprint."""
    X = X.to(head.device)
    gamma = steer.coefficient(X).abs() / X.norm(dim=-1)
    if hasattr(steer, "u"):
        su = head.footprint(steer.u).std()
    else:
        su = torch.stack([head.footprint(d).std() for d in steer.direction(X)])
    return abs(s) * X.shape[1] ** 0.5 * gamma * su / head.logit_sd(X)


@torch.no_grad()
def diagnose(head, tok, steer: Steer, s: float, sets: dict[str, torch.Tensor], u_dim, mu,
             behaviour: str, mmlu: tuple | None = None, footprint_k: int = 12) -> dict:
    steer = steer.to(head.device)
    out = {"strength": s, "omega": abs(float(steer.u.cpu() @ mu)) if hasattr(steer, "u") else None,
           "omega_dim": abs(float(u_dim @ mu)), "sets": {}}
    for name, X in sets.items():
        X = X.float().to(head.device)
        g = steer.coefficient(X) / X.norm(dim=-1)
        change, base, new = token_change(head, X, steer, s)
        beta = collapse_index(head, X, steer, s)
        out["sets"][name] = {"n": len(X), "gate_mean": g.mean().item(), "gate_sd": g.std().item(),
                             "token_change": change, "beta_mean": beta.mean().item()}
        if name == behaviour:
            out["first"] = {"unsteered": base.tolist(), "steered": new.tolist()}
    if hasattr(steer, "u"):
        fp = head.footprint(steer.u)
        dec = lambda ids: [tok.decode([i]) for i in ids]
        out["footprint"] = {"suppressed": dec(fp.topk(footprint_k).indices.tolist()),
                            "promoted": dec((-fp).topk(footprint_k).indices.tolist())}
    if mmlu is not None:
        X, answers, letters = mmlu
        X = X.float().to(head.device)
        acc = lambda Z: (head.norm(Z) @ head.W[letters].T).argmax(-1).cpu().eq(answers).float().mean().item()
        out["mmlu"] = {"unsteered": acc(X), "steered": acc(steer.apply(X, s))}
    return out
