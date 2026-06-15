"""Telemetry helpers for OpenMythos observability.

This module is intentionally side-effect free. It does not mutate model behavior;
it provides typed containers and pure helper functions that training, tests, and
future model instrumentation can opt into explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class TelemetryState:
    """Decoupled observability container populated during instrumented runs.

    Modules may write their own operational signals into this object without
    changing their default return signatures. Keep tensors detached unless the
    caller intentionally needs graph-connected diagnostics.
    """

    expert_counts: Optional[torch.Tensor] = None
    act_depths: Optional[torch.Tensor] = None
    hidden_norms: Dict[int, float] = field(default_factory=dict)
    spectral_radius: Optional[float] = None

    def clear(self) -> None:
        """Reset all telemetry fields for reuse across steps."""

        self.expert_counts = None
        self.act_depths = None
        self.hidden_norms.clear()
        self.spectral_radius = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary suitable for metric loggers/checkpoints."""

        return {
            "expert_counts": self.expert_counts,
            "act_depths": self.act_depths,
            "hidden_norms": dict(self.hidden_norms),
            "spectral_radius": self.spectral_radius,
        }


def compute_expert_telemetry(expert_counts: torch.Tensor) -> Dict[str, float | int]:
    """Compute routing health metrics from per-expert token counts.

    Args:
        expert_counts: 1-D tensor of shape ``(n_experts,)`` containing token
            allocation counts for the routed experts.

    Returns:
        Dictionary containing entropy, normalized entropy, Gini coefficient,
        max/min ratio, dead expert count, and total routed assignments.
    """

    if expert_counts.ndim != 1:
        raise ValueError("expert_counts must be a 1-D tensor")

    counts = expert_counts.detach().float().cpu()
    n_experts = int(counts.numel())
    total = counts.sum()

    if n_experts == 0:
        raise ValueError("expert_counts must contain at least one expert")

    if total.item() <= 0:
        return {
            "expert_entropy": 0.0,
            "expert_entropy_normalized": 0.0,
            "expert_gini": 0.0,
            "expert_max_min_ratio": 0.0,
            "dead_experts": n_experts,
            "total_routings": 0,
        }

    probs = counts / total
    probs_nonzero = probs[probs > 0]
    entropy = -(probs_nonzero * torch.log(probs_nonzero)).sum()
    max_entropy = torch.log(torch.tensor(float(n_experts)))
    entropy_normalized = entropy / max_entropy if n_experts > 1 else torch.tensor(1.0)

    diffs = torch.abs(counts.view(-1, 1) - counts.view(1, -1)).sum()
    gini = diffs / (2.0 * n_experts * total)

    min_count = counts.min().item()
    max_count = counts.max().item()
    dead_experts = int((counts == 0).sum().item())
    max_min_ratio = float("inf") if min_count == 0 else max_count / min_count

    return {
        "expert_entropy": float(entropy.item()),
        "expert_entropy_normalized": float(entropy_normalized.item()),
        "expert_gini": float(gini.item()),
        "expert_max_min_ratio": float(max_min_ratio),
        "dead_experts": dead_experts,
        "total_routings": int(total.item()),
    }


def spectral_radius_from_A(A: torch.Tensor) -> float:
    """Return the spectral radius of an LTI transition tensor.

    The current OpenMythos LTI implementation returns a diagonal vector, but this
    helper also supports square matrices or a stack of square matrices for future
    non-diagonal variants.
    """

    A_float = A.detach().float().cpu()

    if A_float.ndim == 1:
        return float(A_float.abs().max().item())

    if A_float.ndim == 2:
        if A_float.shape[0] != A_float.shape[1]:
            raise ValueError("2-D transition matrix must be square")
        return float(torch.linalg.eigvals(A_float).abs().max().item())

    if A_float.ndim == 3:
        if A_float.shape[-1] != A_float.shape[-2]:
            raise ValueError("3-D transition tensor must contain square matrices")
        return float(torch.linalg.eigvals(A_float).abs().amax().item())

    raise ValueError("A must be 1-D diagonal, 2-D square, or 3-D batched square")


def check_spectral_radius(model: torch.nn.Module) -> float:
    """Read ``model.recurrent.injection.get_A()`` and compute spectral radius.

    Supports raw models and common wrappers such as DDP/FSDP that expose the
    underlying module through ``.module``.
    """

    unwrapped = getattr(model, "module", model)
    try:
        A = unwrapped.recurrent.injection.get_A()
    except AttributeError as exc:
        raise AttributeError(
            "model must expose recurrent.injection.get_A() for spectral checks"
        ) from exc
    return spectral_radius_from_A(A)


def active_governance_gate(
    metrics: Dict[str, Any],
    *,
    max_spectral_radius: float = 1.05,
    max_dead_experts: int = 0,
    max_expert_gini: float = 0.85,
    max_grad_norm: Optional[float] = None,
) -> None:
    """Raise when telemetry crosses hard training-abort thresholds.

    This pure gate intentionally performs no checkpoint I/O. Training scripts can
    catch the raised RuntimeError, write an emergency checkpoint with their own
    FSDP/DDP-safe save path, and then halt.
    """

    abort_reasons: list[str] = []

    loss = metrics.get("loss")
    if loss is not None and torch.isnan(torch.tensor(float(loss))):
        abort_reasons.append("loss is NaN")

    spectral_radius = metrics.get("spectral_radius")
    if spectral_radius is not None and float(spectral_radius) > max_spectral_radius:
        abort_reasons.append(
            f"spectral radius {float(spectral_radius):.6f} > {max_spectral_radius:.6f}"
        )

    dead_experts = metrics.get("dead_experts")
    if dead_experts is not None and int(dead_experts) > max_dead_experts:
        abort_reasons.append(
            f"dead experts {int(dead_experts)} > {int(max_dead_experts)}"
        )

    expert_gini = metrics.get("expert_gini", metrics.get("gini"))
    if expert_gini is not None and float(expert_gini) > max_expert_gini:
        abort_reasons.append(
            f"expert gini {float(expert_gini):.6f} > {max_expert_gini:.6f}"
        )

    grad_norm = metrics.get("grad_norm")
    if max_grad_norm is not None and grad_norm is not None:
        if float(grad_norm) > float(max_grad_norm):
            abort_reasons.append(
                f"grad norm {float(grad_norm):.6f} > {float(max_grad_norm):.6f}"
            )

    if abort_reasons:
        joined = "; ".join(abort_reasons)
        raise RuntimeError(f"Training halted by active governance gate: {joined}")
