"""Euler sampling for reference-corrected PPC and PRC."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.special
import torch
from torch.nn import functional as F

from .composition import compose_ab, compose_views
from .flow import flow_from_probability
from .model import VOCABULARY, Denoiser, ModelConfig, _gamma_sample, expand_state


class CFactor:
    """Analytic Dirichlet velocity coefficient on a fixed alpha grid."""

    def __init__(self, alpha_max: float, alpha_step: float = 0.005, x_points: int = 1001) -> None:
        if alpha_max <= 1 or alpha_step <= 0 or x_points < 2:
            raise ValueError("invalid C-factor grid")
        self.alpha_step = alpha_step
        self.alphas = np.arange(1, alpha_max + alpha_step * 0.5, alpha_step)
        self.grid = np.linspace(0, 1, x_points)
        cdf = np.stack([scipy.special.betainc(alpha, 3, self.grid) for alpha in self.alphas])
        self.derivative = np.gradient(cdf, alpha_step, axis=0)

    def __call__(self, state: torch.Tensor, alpha: float) -> torch.Tensor:
        values = state.detach().cpu().numpy()
        beta = scipy.special.beta(alpha, 3)
        with np.errstate(all="ignore"):
            factor = np.where(values < 1, beta / ((1 - values) ** 3), 0)
            factor = np.where(values ** (alpha - 1) > 0, factor / (values ** (alpha - 1)), 0)
        index = int(np.clip(round((alpha - 1) / self.alpha_step), 0, len(self.alphas) - 1))
        derivative = np.interp(values.reshape(-1), self.grid, self.derivative[index]).reshape(values.shape)
        corrected = np.nan_to_num(-derivative * factor)
        return torch.from_numpy(corrected).to(device=state.device, dtype=state.dtype)


def simplex_project(state: torch.Tensor) -> torch.Tensor:
    classes = state.shape[-1]
    flat = state.reshape(-1, classes)
    ordered, _ = torch.sort(flat, dim=-1, descending=True)
    cumulative = torch.cumsum(ordered, dim=-1) - 1
    index = torch.arange(1, classes + 1, dtype=flat.dtype, device=flat.device)
    support = ordered - cumulative / index > 0
    rho = support.sum(-1).clamp_min(1)
    theta = cumulative[torch.arange(len(flat), device=flat.device), rho - 1] / rho
    return torch.clamp(flat - theta[:, None], min=0).reshape(state.shape)


def integrate_dirichlet(
    probability_at: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    length: int,
    count: int,
    nfe: int,
    seed: int,
    config: ModelConfig,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    c_factor: CFactor | None = None,
) -> np.ndarray:
    """Integrate a categorical endpoint field and decode by argmax."""
    if length <= 0 or count <= 0 or nfe < 2 or batch_size <= 0:
        raise ValueError("length, count, batch_size must be positive and nfe must be at least 2")
    device = torch.device(device)
    table = c_factor or CFactor(config.alpha_max)
    points = torch.linspace(1.0, config.alpha_max, nfe + 1, device=device)
    pieces: list[np.ndarray] = []
    for start_index in range(0, count, batch_size):
        batch = min(batch_size, count - start_index)
        parameters = torch.ones(batch, length, VOCABULARY, device=device)
        gamma = _gamma_sample(parameters, seed=seed + start_index * 131 + 17)
        state = gamma / gamma.sum(dim=-1, keepdim=True)
        prior = torch.full_like(state, 1.0 / VOCABULARY)
        final_probability: torch.Tensor | None = None
        for start, stop in zip(points[:-1], points[1:]):
            alpha = start.expand(batch)
            final_probability = probability_at(state, alpha, prior)
            flow = flow_from_probability(state, final_probability, table(state, float(start)))
            state = state + flow * (stop - start)
            invalid = (
                bool((state < 0).any())
                or not bool(torch.isfinite(state).all())
                or not torch.allclose(state.sum(dim=-1), torch.ones_like(state[..., 0]), atol=1e-4)
            )
            if invalid:
                state = simplex_project(state)
        if final_probability is None:
            raise RuntimeError("sampling performed no function evaluation")
        pieces.append(final_probability.argmax(dim=-1).cpu().numpy().astype(np.uint8))
    return np.concatenate(pieces, axis=0)


@torch.no_grad()
def sample_composition(
    method: str,
    reference: Denoiser,
    factors: Sequence[Denoiser],
    *,
    length: int,
    count: int,
    nfe: int,
    seed: int,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    c_factor: CFactor | None = None,
) -> np.ndarray:
    """Sample PPC or PRC from one reference model and one or more factor models."""
    if method not in {"ppc", "prc"}:
        raise ValueError("method must be 'ppc' or 'prc'")
    if not factors:
        raise ValueError("at least one factor model is required")
    models = (reference, *factors)
    if any(model.config != reference.config for model in models[1:]):
        raise ValueError("composition models must share one ModelConfig")

    def probability_at(state: torch.Tensor, alpha: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        logs = []
        for model in models:
            expanded = expand_state(state, alpha, prior, model.config.pseudocount)
            logs.append(F.log_softmax(model(expanded, alpha), dim=-1))
        if len(factors) == 2:
            composed = compose_ab(logs[0], logs[1], logs[2])
            return composed.ppc_probability if method == "ppc" else composed.prc_probability
        views = compose_views(logs[0], logs[1:])
        if method == "ppc":
            return views.ppc_log_probability.exp()
        return views.projected_first_order_probability

    return integrate_dirichlet(
        probability_at,
        length=length,
        count=count,
        nfe=nfe,
        seed=seed,
        config=reference.config,
        batch_size=batch_size,
        device=device,
        c_factor=c_factor,
    )
