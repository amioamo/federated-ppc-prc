"""Posterior product composition and projected residual composition.

PPC is q* proportional to q_A q_B / q_0. PRC is the projected residual
q_0 + (q_A - q_0) + (q_B - q_0), with negative mass clipped and renormalized.
The two operators are not the same.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


def _check_factors(base: torch.Tensor, factors: Sequence[torch.Tensor]) -> None:
    if not factors:
        raise ValueError("at least one factor posterior is required")
    for index, factor in enumerate(factors):
        if factor.shape != base.shape:
            raise ValueError(
                f"factor {index} has shape {tuple(factor.shape)}, "
                f"expected {tuple(base.shape)}"
            )


def _weights(count: int, weights: Sequence[float] | None) -> tuple[float, ...]:
    values = tuple(1.0 for _ in range(count)) if weights is None else tuple(float(w) for w in weights)
    if len(values) != count:
        raise ValueError(f"received {len(values)} weights for {count} factors")
    if not all(torch.isfinite(torch.tensor(value)).item() for value in values):
        raise ValueError("composition weights must be finite")
    return values


def posterior_product(
    log_base: torch.Tensor,
    log_factors: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Return normalized log probabilities for weighted PPC.

    q* is proportional to q0 times the product of (qi/q0)**wi.
    Unit weights over M factors therefore give product(qi) / q0**(M-1).
    """

    _check_factors(log_base, log_factors)
    values = _weights(len(log_factors), weights)
    raw = log_base
    for weight, factor in zip(values, log_factors):
        raw = raw + weight * (factor - log_base)
    return raw - torch.logsumexp(raw, dim=dim, keepdim=True)


def posterior_product_two(
    log_base: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    weight_a: float = 1.0,
    weight_b: float = 1.0,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Two-factor PPC compatibility helper."""

    return posterior_product(
        log_base,
        (log_a, log_b),
        (weight_a, weight_b),
        dim=dim,
    )


def first_order_residual_probability(
    base: torch.Tensor,
    factors: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Return the signed first-order residual expansion before projection."""

    _check_factors(base, factors)
    values = _weights(len(factors), weights)
    raw = base
    for weight, factor in zip(values, factors):
        raw = raw + weight * (factor - base)
    return raw


def project_probability(
    probability: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Project a signed categorical vector by clipping and renormalizing.

    Rows with no positive mass fall back to the uniform distribution.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    classes = probability.shape[dim]
    if classes < 2:
        raise ValueError("categorical dimension must have at least two classes")
    positive = probability.clamp_min(0)
    total = positive.sum(dim=dim, keepdim=True)
    uniform = torch.full_like(positive, 1.0 / classes)
    return torch.where(total > eps, positive / total.clamp_min(eps), uniform)


def projected_first_order(
    base: torch.Tensor,
    factors: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    *,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return projected first-order posterior composition."""

    raw = first_order_residual_probability(base, factors, weights)
    return project_probability(raw, dim=dim, eps=eps)


def signed_first_order(
    base_value: torch.Tensor,
    factor_values: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Apply the first-order residual rule to any linear quantity, e.g. flow."""

    return first_order_residual_probability(base_value, factor_values, weights)


@dataclass(frozen=True)
class PRCViews:
    """PPC, the factor average, and the first-order residual for one expert set."""

    ppc_log_probability: torch.Tensor
    posterior_average_probability: torch.Tensor
    first_order_raw_probability: torch.Tensor
    projected_first_order_probability: torch.Tensor


@dataclass(frozen=True)
class ABComposition:
    """The two named A+B posterior compositions for one base/factor triplet.

    ``ppc_probability`` is the normalized posterior product
    ``q_A q_B / q_0``. ``prc_probability`` is the projected residual
    ``project(q_A + q_B - q_0)``.
    """

    ppc_probability: torch.Tensor
    prc_probability: torch.Tensor


def compose_ab(
    log_base: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-8,
) -> ABComposition:
    """Compute PPC and PRC for the controlled two-factor A+B setting."""

    base = log_base.exp()
    factors = (log_a.exp(), log_b.exp())
    return ABComposition(
        ppc_probability=posterior_product_two(
            log_base, log_a, log_b, dim=dim
        ).exp(),
        prc_probability=projected_first_order(
            base, factors, dim=dim, eps=eps
        ),
    )


def compose_views(
    log_base: torch.Tensor,
    log_factors: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    *,
    dim: int = -1,
    eps: float = 1e-8,
) -> PRCViews:
    """Compute PPC, average, and first-order views from the same posteriors."""

    from baselines.controls import posterior_average

    _check_factors(log_base, log_factors)
    base = log_base.exp()
    factors = tuple(value.exp() for value in log_factors)
    raw = first_order_residual_probability(base, factors, weights)
    return PRCViews(
        ppc_log_probability=posterior_product(log_base, log_factors, weights, dim=dim),
        posterior_average_probability=posterior_average(factors, weights),
        first_order_raw_probability=raw,
        projected_first_order_probability=project_probability(raw, dim=dim, eps=eps),
    )
