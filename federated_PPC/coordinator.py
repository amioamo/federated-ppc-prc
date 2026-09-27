"""One-to-one federated baselines and coordinator helpers for PPC/PRC.

This module deliberately separates three operations that are easy to conflate:

``fedavg_state_dict`` supports the standard global-model comparison across
condition-partitioned sites. ``FederatedCompositionPlan`` expresses the
ordinary coordinator-side PPC/PRC algebra. Privacy mechanisms are out of scope.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from composition import project_probability


def _finite_nonnegative_count(value: int, *, participant: str) -> int:
    count = int(value)
    if count != value or count <= 0:
        raise ValueError(
            f"sample count for {participant!r} must be a positive integer"
        )
    return count


def fedavg_state_dict(
    local_states: Mapping[str, Mapping[str, torch.Tensor]],
    sample_counts: Mapping[str, int],
) -> dict[str, torch.Tensor]:
    """Return the sample-count weighted average of compatible local states.

    Floating tensors are averaged in their native dtype. Non-floating tensors
    (for example counters) must agree exactly and are copied unchanged.
    """

    if not local_states:
        raise ValueError("at least one local state is required")
    participants = tuple(local_states)
    if set(sample_counts) != set(participants):
        raise ValueError("sample counts must name exactly the local participants")
    counts = {
        participant: _finite_nonnegative_count(
            sample_counts[participant], participant=participant
        )
        for participant in participants
    }

    reference_keys = tuple(local_states[participants[0]])
    reference_key_set = set(reference_keys)
    for participant in participants[1:]:
        if set(local_states[participant]) != reference_key_set:
            raise ValueError("all local states must contain the same parameter keys")

    total = sum(counts.values())
    averaged: dict[str, torch.Tensor] = {}
    for key in reference_keys:
        tensors = [local_states[participant][key] for participant in participants]
        reference = tensors[0]
        if any(tensor.shape != reference.shape for tensor in tensors[1:]):
            raise ValueError(f"parameter {key!r} has inconsistent shapes")
        if any(tensor.dtype != reference.dtype for tensor in tensors[1:]):
            raise ValueError(f"parameter {key!r} has inconsistent dtypes")
        if any(tensor.device != reference.device for tensor in tensors[1:]):
            raise ValueError(f"parameter {key!r} has inconsistent devices")

        if reference.is_floating_point() or reference.is_complex():
            value = torch.zeros_like(reference)
            for participant, tensor in zip(participants, tensors):
                value = value + tensor * (counts[participant] / total)
            averaged[key] = value
        else:
            if any(not torch.equal(reference, tensor) for tensor in tensors[1:]):
                raise ValueError(
                    f"non-floating parameter {key!r} must agree across participants"
                )
            averaged[key] = reference.clone()
    return averaged


@dataclass(frozen=True)
class FederatedCompositionPlan:
    """Participant coefficients for weighted reference-relative composition.

    For factor weights ``w_i``, PPC is

        softmax((1 - sum_i w_i) z_bg + sum_i w_i z_i).

    The same coefficients applied to probabilities give the signed PRC vector
    before the registered positive-part normalization.
    """

    reference_id: str
    factor_ids: tuple[str, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.reference_id:
            raise ValueError("reference_id must be non-empty")
        if not self.factor_ids:
            raise ValueError("at least one factor participant is required")
        if len(self.factor_ids) != len(self.weights):
            raise ValueError("factor_ids and weights must have equal length")
        participant_ids = (self.reference_id, *self.factor_ids)
        if len(set(participant_ids)) != len(participant_ids):
            raise ValueError("participant identifiers must be unique")
        weight_tensor = torch.tensor(self.weights, dtype=torch.float64)
        if not bool(torch.isfinite(weight_tensor).all()):
            raise ValueError("composition weights must be finite")

    @classmethod
    def unit_weighted(
        cls, reference_id: str, factor_ids: Sequence[str]
    ) -> "FederatedCompositionPlan":
        factors = tuple(factor_ids)
        return cls(reference_id, factors, tuple(1.0 for _ in factors))

    @property
    def participant_ids(self) -> tuple[str, ...]:
        return (self.reference_id, *self.factor_ids)

    @property
    def reference_coefficient(self) -> float:
        return 1.0 - sum(self.weights)

    def coefficient(self, participant_id: str) -> float:
        if participant_id == self.reference_id:
            return self.reference_coefficient
        try:
            index = self.factor_ids.index(participant_id)
        except ValueError as error:
            raise KeyError(f"unknown participant {participant_id!r}") from error
        return self.weights[index]

    def contribution(
        self, participant_id: str, value: torch.Tensor
    ) -> torch.Tensor:
        """Return one site's coefficient-weighted coordinator payload."""

        if not bool(torch.isfinite(value).all()):
            raise ValueError("site contribution input must be finite")
        return value * self.coefficient(participant_id)

    def contributions(
        self, values: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Build all payloads for tests or a trusted local deployment."""

        if set(values) != set(self.participant_ids):
            raise ValueError("values must name exactly the planned participants")
        shape = values[self.reference_id].shape
        if any(value.shape != shape for value in values.values()):
            raise ValueError("all participant outputs must have the same shape")
        return {
            participant: self.contribution(participant, values[participant])
            for participant in self.participant_ids
        }

    @staticmethod
    def finalize_ppc(
        aggregate_logit_sum: torch.Tensor, *, dim: int = -1
    ) -> torch.Tensor:
        """Normalize the coordinator's sum of coefficient-weighted logits."""

        if not bool(torch.isfinite(aggregate_logit_sum).all()):
            raise ValueError("aggregate logit sum must be finite")
        return torch.log_softmax(aggregate_logit_sum, dim=dim)

    @staticmethod
    def finalize_prc(
        aggregate_probability_sum: torch.Tensor,
        *,
        dim: int = -1,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Apply the registered PRC normalization to the coordinator sum."""

        if not bool(torch.isfinite(aggregate_probability_sum).all()):
            raise ValueError("aggregate probability sum must be finite")
        return project_probability(aggregate_probability_sum, dim=dim, eps=eps)
