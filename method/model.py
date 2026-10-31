"""Dirichlet denoiser used by PPC and PRC.

The network reads the 12-channel expanded simplex state and predicts token
logits. DNA tokens are A=0, C=1, G=2, T=3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


VOCABULARY = 4


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 32
    stacks: int = 1
    alpha_scale: float = 2.0
    alpha_max: float = 8.0
    pseudocount: float = 2.0
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_dim % 2:
            raise ValueError("hidden_dim must be a positive even integer")
        if self.stacks <= 0:
            raise ValueError("stacks must be positive")
        if self.alpha_scale <= 0 or self.alpha_max <= 1:
            raise ValueError("invalid Dirichlet alpha range")
        if self.pseudocount <= 0 or self.gradient_clip <= 0:
            raise ValueError("pseudocount and gradient_clip must be positive")


class Fourier(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(93017)
        self.register_buffer("freq", torch.randn(hidden // 2, generator=generator))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        phase = values[:, None] * self.freq[None, :] * 2 * math.pi
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class Denoiser(nn.Module):
    """Token posterior model for a standard Dirichlet conditional path."""

    def __init__(self, config: ModelConfig, condition_dim: int = 0) -> None:
        super().__init__()
        if condition_dim < 0:
            raise ValueError("condition_dim cannot be negative")
        self.config = config
        self.condition_dim = condition_dim
        hidden = config.hidden_dim
        self.inp = nn.Conv1d(12, hidden, 7, padding=3)
        self.time = nn.Sequential(Fourier(hidden), nn.Linear(hidden, hidden))
        self.condition = nn.Linear(condition_dim, hidden, bias=False) if condition_dim else None
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.context = nn.ModuleList()
        for _ in range(config.stacks):
            for dilation in (1, 2, 4, 8, 16):
                self.convs.append(nn.Conv1d(hidden, hidden, 5, dilation=dilation, padding=2 * dilation))
                self.norms.append(nn.LayerNorm(hidden))
                self.context.append(nn.Linear(hidden, hidden))
        self.out = nn.Sequential(nn.Conv1d(hidden, hidden, 1), nn.GELU(), nn.Conv1d(hidden, VOCABULARY, 1))

    def forward(
        self,
        expanded_state: torch.Tensor,
        alpha: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if expanded_state.ndim != 3 or expanded_state.shape[-1] != 12:
            raise ValueError("expanded Dirichlet state must have shape [batch, length, 12]")
        if alpha.shape != (expanded_state.shape[0],):
            raise ValueError("alpha must have shape [batch]")
        context = self.time(alpha)
        if self.condition is not None:
            if condition is None:
                raise ValueError("condition is required")
            context = context + self.condition(condition.float())
        elif condition is not None:
            raise ValueError("unconditional model does not accept a condition vector")
        hidden = F.gelu(self.inp(expanded_state.permute(0, 2, 1)))
        for conv, norm, context_layer in zip(self.convs, self.norms, self.context):
            residual = norm((hidden + context_layer(context)[:, :, None]).permute(0, 2, 1)).permute(0, 2, 1)
            hidden = hidden + F.gelu(conv(residual))
        return self.out(hidden).permute(0, 2, 1)


def expand_state(
    state: torch.Tensor,
    alpha: torch.Tensor,
    prior: torch.Tensor,
    pseudocount: float,
) -> torch.Tensor:
    weight = (pseudocount / (alpha + pseudocount - 1.0))[:, None, None]
    return torch.cat((state * (1.0 - weight), state * weight, prior), dim=-1)


def _gamma_sample(parameters: torch.Tensor, *, seed: int) -> torch.Tensor:
    sample_parameters = parameters.detach().cpu() if parameters.device.type == "mps" else parameters
    devices = [sample_parameters.device] if sample_parameters.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        sampled = torch._standard_gamma(sample_parameters)
    return sampled.to(device=parameters.device, dtype=parameters.dtype)


def dirichlet_training_state(
    target: torch.Tensor,
    *,
    alpha_seed: int,
    gamma_seed: int,
    config: ModelConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the expanded Dirichlet path and its alpha."""
    if target.ndim != 2 or target.dtype != torch.long:
        raise ValueError("target must be a [batch, length] torch.long tensor")
    if torch.any((target < 0) | (target >= VOCABULARY)):
        raise ValueError("target must contain DNA ids 0..3")
    generator = torch.Generator(device="cpu").manual_seed(alpha_seed)
    uniform = torch.rand(target.shape[0], generator=generator).clamp_(1e-7, 1.0 - 1e-7)
    alpha = (1.0 - config.alpha_scale * torch.log1p(-uniform)).clamp_max(config.alpha_max)
    alpha = alpha.to(target.device)
    parameters = torch.ones(target.shape[0], target.shape[1], VOCABULARY, device=target.device)
    parameters.scatter_add_(
        2,
        target[..., None],
        (alpha[:, None, None] - 1.0).expand(target.shape[0], target.shape[1], 1),
    )
    gamma = _gamma_sample(parameters, seed=gamma_seed)
    state = gamma / gamma.sum(dim=-1, keepdim=True)
    prior = torch.full_like(state, 1.0 / VOCABULARY)
    return expand_state(state, alpha, prior, config.pseudocount), alpha


def flow_matching_loss(
    model: Denoiser,
    target: torch.Tensor,
    condition: torch.Tensor | None = None,
    *,
    alpha_seed: int,
    gamma_seed: int,
) -> torch.Tensor:
    """Denoising cross-entropy along the Dirichlet path."""
    expanded, alpha = dirichlet_training_state(
        target,
        alpha_seed=alpha_seed,
        gamma_seed=gamma_seed,
        config=model.config,
    )
    logits = model(expanded, alpha, condition)
    result = F.cross_entropy(logits.reshape(-1, VOCABULARY), target.reshape(-1))
    if not bool(torch.isfinite(result)):
        raise FloatingPointError("non-finite Dirichlet flow-matching loss")
    return result
