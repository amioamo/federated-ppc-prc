"""Checks for the Dirichlet denoiser and separated baselines."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from baselines import fedavg_state_dict, sample_single
from method import CFactor, Denoiser, ModelConfig, expand_state, flow_matching_loss, sample_composition


def _tiny_config() -> ModelConfig:
    return ModelConfig(hidden_dim=8, stacks=1, alpha_max=2.0, pseudocount=2.0)


def _coarse_factor(config: ModelConfig) -> CFactor:
    return CFactor(config.alpha_max, alpha_step=0.25, x_points=21)


class TestMethodCore(unittest.TestCase):
    def test_expanded_state_keeps_the_uniform_prior_block(self) -> None:
        state = torch.full((2, 5, 4), 0.25)
        alpha = torch.tensor([2.0, 3.0])
        prior = torch.full_like(state, 0.25)
        expanded = expand_state(state, alpha, prior, pseudocount=2.0)
        self.assertEqual(tuple(expanded.shape), (2, 5, 12))
        torch.testing.assert_close(expanded[..., 8:], prior)

    def test_flow_matching_loss_is_finite(self) -> None:
        config = _tiny_config()
        model = Denoiser(config)
        target = torch.randint(0, 4, (4, 12))
        loss = flow_matching_loss(model, target, alpha_seed=1, gamma_seed=2)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_ppc_and_single_model_samples_are_dna(self) -> None:
        config = _tiny_config()
        torch.manual_seed(0)
        reference = Denoiser(config).eval()
        factor_a = Denoiser(config).eval()
        factor_b = Denoiser(config).eval()
        table = _coarse_factor(config)
        composed = sample_composition(
            "ppc",
            reference,
            (factor_a, factor_b),
            length=8,
            count=2,
            nfe=2,
            seed=3,
            batch_size=2,
            c_factor=table,
        )
        single = sample_single(
            reference,
            length=8,
            count=2,
            nfe=2,
            seed=3,
            batch_size=2,
            c_factor=table,
        )
        self.assertEqual(composed.shape, (2, 8))
        self.assertEqual(single.shape, (2, 8))
        self.assertTrue(np.isin(composed, (0, 1, 2, 3)).all())

    def test_fedavg_lives_with_baselines(self) -> None:
        states = {
            "a": {"weight": torch.tensor([1.0, 3.0])},
            "b": {"weight": torch.tensor([5.0, 7.0])},
        }
        observed = fedavg_state_dict(states, {"a": 3, "b": 1})
        torch.testing.assert_close(observed["weight"], torch.tensor([2.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
