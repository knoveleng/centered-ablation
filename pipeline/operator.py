"""The operator and its closed form (Proposition 3.1).

Origin-corrected ablation at the output of the last decoder block, every position:

    h  <-  h - s (h·u - c) u                 (u unit; c = 0 for DIM and CA, c = μ₋·u for ACE)

A `Steer` is what a direction's `fit` returns: either a fixed (u, c) or a state-dependent
field u(h) (kernel controls). Both are torch-serializable so vLLM can receive them by value.
"""

import torch


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def rms_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.pow(2).mean(-1, keepdim=True).sqrt()


class Steer:
    """A fixed direction u and offset c."""

    def __init__(self, u: torch.Tensor, c: float = 0.0):
        self.u = unit(u.float())
        self.c = float(c)

    def to(self, device):
        self.u = self.u.to(device)
        return self

    def direction(self, h: torch.Tensor) -> torch.Tensor:
        return self.u.expand_as(h)

    def coefficient(self, h: torch.Tensor) -> torch.Tensor:
        """h·u − c, the scalar that multiplies u (and, after the head, the footprint)."""
        return (h * self.direction(h)).sum(-1) - self.c

    def apply(self, h: torch.Tensor, s: float) -> torch.Tensor:
        return h - s * self.coefficient(h).unsqueeze(-1) * self.direction(h)

    def state(self) -> dict:
        return {"kind": "fixed", "u": self.u.cpu(), "c": self.c}


class FieldSteer(Steer):
    """A state-dependent unit direction u(h) with c = 0 (kernel control directions)."""

    def __init__(self, field):
        self.field, self.c = field, 0.0

    def to(self, device):
        self.field = self.field.to(device)
        return self

    def direction(self, h):
        return self.field(h)

    def state(self) -> dict:
        return {"kind": "field", "field": self.field}


def load_steer(state: dict) -> Steer:
    return Steer(state["u"], state["c"]) if state["kind"] == "fixed" else FieldSteer(state["field"])


def gate(h: torch.Tensor, steer: Steer) -> torch.Tensor:
    """γ(h)/‖h‖: the cosine gate, shifted by the offset for ACE."""
    return steer.coefficient(h) / h.norm(dim=-1)
