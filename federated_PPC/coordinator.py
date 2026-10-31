"""Coordinator-side PPC and PRC.

``FederatedCompositionPlan`` applies the reference-relative coefficients.
Sample-count FedAvg of model parameters is in ``baselines.fedavg``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from composition import project_probability


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
