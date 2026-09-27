"""Dirichlet-flow operations used after PRC posterior construction."""
from __future__ import annotations

from collections.abc import Sequence

import torch

from composition import posterior_product, signed_first_order


def flow_from_probability(
    state: torch.Tensor,
    endpoint_probability: torch.Tensor,
    c_factor: torch.Tensor,
) -> torch.Tensor:
    """Map endpoint categorical probabilities to the analytic Dirichlet flow."""

    if state.shape != endpoint_probability.shape or state.shape != c_factor.shape:
        raise ValueError("state, endpoint_probability, and c_factor must have equal shapes")
    classes = state.shape[-1]
    eye = torch.eye(classes, device=state.device, dtype=state.dtype)
    return (
        endpoint_probability.unsqueeze(-2)
        * ((eye - state.unsqueeze(-1)) * c_factor.unsqueeze(-2))
    ).sum(-1)


def ppc_velocity(
    state: torch.Tensor,
    log_base: torch.Tensor,
    log_factors: Sequence[torch.Tensor],
    c_factor: torch.Tensor,
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Construct PPC in posterior space and return its Dirichlet velocity."""

    probability = posterior_product(log_base, log_factors, weights).exp()
    return flow_from_probability(state, probability, c_factor)


def first_order_velocity(
    state: torch.Tensor,
    base_probability: torch.Tensor,
    factor_probabilities: Sequence[torch.Tensor],
    c_factor: torch.Tensor,
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Return the signed first-order linear combination of atomic velocities."""

    base_flow = flow_from_probability(state, base_probability, c_factor)
    factor_flows = tuple(
        flow_from_probability(state, factor, c_factor) for factor in factor_probabilities
    )
    return signed_first_order(base_flow, factor_flows, weights)

