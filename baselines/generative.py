"""Comparison generators that do not use reference-corrected composition."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from baselines.controls import naive_posterior_product, posterior_average
from method.model import Denoiser, ModelConfig
from method.sample import CFactor, integrate_dirichlet


def sample_single(
    model: Denoiser,
    *,
    length: int,
    count: int,
    nfe: int,
    seed: int,
    condition: torch.Tensor | None = None,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    c_factor: CFactor | None = None,
) -> np.ndarray:
    """Sample one Dirichlet model. ``condition`` is optional and fixed across steps."""

    def probability_at(state: torch.Tensor, alpha: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        expanded = _expand(state, alpha, prior, model.config)
        cond = None if condition is None else condition.to(state.device).expand(state.shape[0], -1)
        return F.softmax(model(expanded, alpha, cond), dim=-1)

    return integrate_dirichlet(
        probability_at,
        length=length,
        count=count,
        nfe=nfe,
        seed=seed,
        config=model.config,
        batch_size=batch_size,
        device=device,
        c_factor=c_factor,
    )


def sample_naive_product(
    factors: Sequence[Denoiser],
    *,
    length: int,
    count: int,
    nfe: int,
    seed: int,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    return _sample_factors(
        factors,
        combine=lambda logs: naive_posterior_product(logs).exp(),
        length=length,
        count=count,
        nfe=nfe,
        seed=seed,
        batch_size=batch_size,
        device=device,
    )


def sample_posterior_average(
    factors: Sequence[Denoiser],
    *,
    length: int,
    count: int,
    nfe: int,
    seed: int,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    return _sample_factors(
        factors,
        combine=lambda logs: posterior_average(tuple(log.exp() for log in logs)),
        length=length,
        count=count,
        nfe=nfe,
        seed=seed,
        batch_size=batch_size,
        device=device,
    )


def _expand(state, alpha, prior, config: ModelConfig):
    from method.model import expand_state

    return expand_state(state, alpha, prior, config.pseudocount)


def _sample_factors(
    factors: Sequence[Denoiser],
    *,
    combine: Callable[[tuple[torch.Tensor, ...]], torch.Tensor],
    length: int,
    count: int,
    nfe: int,
    seed: int,
    batch_size: int,
    device: str | torch.device,
    c_factor: CFactor | None = None,
) -> np.ndarray:
    if len(factors) < 2:
        raise ValueError("a factor baseline needs at least two models")
    configs = {model.config for model in factors}
    if len(configs) != 1:
        raise ValueError("baseline models must share one ModelConfig")

    def probability_at(state: torch.Tensor, alpha: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        logs = []
        for model in factors:
            expanded = _expand(state, alpha, prior, model.config)
            logs.append(F.log_softmax(model(expanded, alpha), dim=-1))
        return combine(tuple(logs))

    return integrate_dirichlet(
        probability_at,
        length=length,
        count=count,
        nfe=nfe,
        seed=seed,
        config=factors[0].config,
        batch_size=batch_size,
        device=device,
        c_factor=c_factor,
    )
