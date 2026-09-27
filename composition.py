"""Core posterior-space operations for Posterior Residual Composition.

The implementation supports any number of factors. The two-factor helpers
match the frozen experiments exactly:

    PPC: q* proportional to q_A q_B / q_0
    naive product: q* proportional to q_A q_B
    first order: q_0 + (q_A - q_0) + (q_B - q_0)

The projected first-order comparator uses the historical positive-part
normalization from the registered experiments. It is intentionally not
presented as mathematically identical to PPC.
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


def naive_posterior_product(
    log_factors: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Return normalized log probabilities for the uncorrected product.

    q* is proportional to product_i q_i**w_i, with no reference correction.
    Unit weights over two factors therefore give Norm(q_A q_B).
    """

    if not log_factors:
        raise ValueError("at least one factor posterior is required")
    _check_factors(log_factors[0], log_factors)
    values = _weights(len(log_factors), weights)
    raw = values[0] * log_factors[0]
    for weight, factor in zip(values[1:], log_factors[1:]):
        raw = raw + weight * factor
    return raw - torch.logsumexp(raw, dim=dim, keepdim=True)


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

    Rows with no positive mass fall back to the uniform distribution. This is
    the projection rule used by the frozen PRC experiments.
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


def posterior_average(
    factors: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Return a normalized non-negative mixture of factor posteriors."""

    if not factors:
        raise ValueError("at least one factor posterior is required")
    reference = factors[0]
    _check_factors(reference, factors)
    if weights is None:
        values = tuple(1.0 / len(factors) for _ in factors)
    else:
        raw_weights = _weights(len(factors), weights)
        if any(value < 0 for value in raw_weights) or sum(raw_weights) <= 0:
            raise ValueError("average weights must be non-negative with positive sum")
        total = sum(raw_weights)
        values = tuple(value / total for value in raw_weights)
    result = torch.zeros_like(reference)
    for weight, factor in zip(values, factors):
        result = result + weight * factor
    return result


def signed_first_order(
    base_value: torch.Tensor,
    factor_values: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Apply the first-order residual rule to any linear quantity, e.g. flow."""

    return first_order_residual_probability(base_value, factor_values, weights)


@dataclass(frozen=True)
class PRCViews:
    """The registered posterior views for one set of atomic experts."""

    ppc_log_probability: torch.Tensor
    posterior_average_probability: torch.Tensor
    first_order_raw_probability: torch.Tensor
    projected_first_order_probability: torch.Tensor


@dataclass(frozen=True)
class ABComposition:
    """The two named A+B posterior compositions for one base/factor triplet.

    ``ppc_probability`` is the normalized posterior product
    ``q_A q_B / q_0``. ``prc_probability`` is the historical projected
    first-order residual ``project(q_A + q_B - q_0)``. Keeping both fields in
    one value makes the controlled A+B experiment explicit and prevents the
    two methods from being confused by their legacy artifact names.
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
