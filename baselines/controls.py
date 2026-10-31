"""Uncorrected composition controls."""
from __future__ import annotations

from collections.abc import Sequence

import torch


def _check_factors(reference: torch.Tensor, factors: Sequence[torch.Tensor]) -> None:
    for index, factor in enumerate(factors):
        if factor.shape != reference.shape:
            raise ValueError(
                f"factor {index} has shape {tuple(factor.shape)}, "
                f"expected {tuple(reference.shape)}"
            )


def _weights(count: int, weights: Sequence[float] | None) -> tuple[float, ...]:
    values = tuple(1.0 for _ in range(count)) if weights is None else tuple(float(w) for w in weights)
    if len(values) != count:
        raise ValueError(f"received {len(values)} weights for {count} factors")
    if not all(torch.isfinite(torch.tensor(value)).item() for value in values):
        raise ValueError("composition weights must be finite")
    return values


def naive_posterior_product(
    log_factors: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Normalized product of factor posteriors, without reference correction."""
    if not log_factors:
        raise ValueError("at least one factor posterior is required")
    _check_factors(log_factors[0], log_factors)
    values = _weights(len(log_factors), weights)
    raw = values[0] * log_factors[0]
    for weight, factor in zip(values[1:], log_factors[1:]):
        raw = raw + weight * factor
    return raw - torch.logsumexp(raw, dim=dim, keepdim=True)


def posterior_average(
    factors: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Normalized non-negative mixture of factor posteriors."""
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
