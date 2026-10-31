"""Sample-count FedAvg of local model parameters."""
from __future__ import annotations

from collections.abc import Mapping

import torch


def _finite_nonnegative_count(value: int, *, participant: str) -> int:
    count = int(value)
    if count != value or count <= 0:
        raise ValueError(f"sample count for {participant!r} must be a positive integer")
    return count


def fedavg_state_dict(
    local_states: Mapping[str, Mapping[str, torch.Tensor]],
    sample_counts: Mapping[str, int],
) -> dict[str, torch.Tensor]:
    """Return the sample-count weighted average of compatible local states.

    Floating tensors are averaged in their native dtype. Non-floating tensors
    must agree exactly and are copied unchanged.
    """
    if not local_states:
        raise ValueError("at least one local state is required")
    participants = tuple(local_states)
    if set(sample_counts) != set(participants):
        raise ValueError("sample counts must name exactly the local participants")
    counts = {
        participant: _finite_nonnegative_count(sample_counts[participant], participant=participant)
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
                raise ValueError(f"non-floating parameter {key!r} must agree across participants")
            averaged[key] = reference.clone()
    return averaged
