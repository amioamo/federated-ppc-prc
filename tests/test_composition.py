from __future__ import annotations

import unittest

import torch

from federated_PPC import (
    compose_ab,
    posterior_product,
    posterior_product_two,
)
from PRC import (
    first_order_residual_probability,
    first_order_velocity,
    flow_from_probability,
    project_probability,
)
from baselines import naive_posterior_product
from composition import compose_views


class TestPosteriorResidualComposition(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(20260811)
        self.q0 = torch.softmax(torch.randn(2, 7, 4, generator=generator), dim=-1)
        self.qa = torch.softmax(torch.randn(2, 7, 4, generator=generator), dim=-1)
        self.qb = torch.softmax(torch.randn(2, 7, 4, generator=generator), dim=-1)

    def test_two_factor_ppc_matches_density_ratio(self) -> None:
        observed = posterior_product_two(self.q0.log(), self.qa.log(), self.qb.log()).exp()
        expected = self.qa * self.qb / self.q0
        expected = expected / expected.sum(-1, keepdim=True)
        torch.testing.assert_close(observed, expected)

    def test_naive_product_is_norm_of_component_product(self) -> None:
        observed = naive_posterior_product((self.qa.log(), self.qb.log())).exp()
        expected = self.qa * self.qb
        expected = expected / expected.sum(-1, keepdim=True)
        torch.testing.assert_close(observed, expected)
        ppc = posterior_product_two(self.q0.log(), self.qa.log(), self.qb.log()).exp()
        average = 0.5 * (self.qa + self.qb)
        self.assertFalse(torch.allclose(observed, ppc))
        self.assertFalse(torch.allclose(observed, average))
        uniform = torch.full_like(self.q0, 0.25)
        corrected_uniform = posterior_product_two(uniform.log(), self.qa.log(), self.qb.log()).exp()
        torch.testing.assert_close(observed, corrected_uniform)

    def test_multi_factor_ppc_uses_base_power_m_minus_one(self) -> None:
        qc = torch.roll(self.qb, shifts=1, dims=-1)
        observed = posterior_product(self.q0.log(), (self.qa.log(), self.qb.log(), qc.log())).exp()
        expected = self.qa * self.qb * qc / self.q0.square()
        expected = expected / expected.sum(-1, keepdim=True)
        torch.testing.assert_close(observed, expected)

    def test_first_order_and_registered_projection(self) -> None:
        raw = first_order_residual_probability(self.q0, (self.qa, self.qb))
        torch.testing.assert_close(raw, self.qa + self.qb - self.q0)
        projected = project_probability(raw)
        self.assertTrue(bool((projected >= 0).all()))
        torch.testing.assert_close(projected.sum(-1), torch.ones_like(projected[..., 0]))

    def test_zero_positive_mass_falls_back_to_uniform(self) -> None:
        projected = project_probability(torch.full((3, 4), -1.0))
        torch.testing.assert_close(projected, torch.full((3, 4), 0.25))

    def test_views_keep_ppc_and_first_order_separate(self) -> None:
        views = compose_views(self.q0.log(), (self.qa.log(), self.qb.log()))
        torch.testing.assert_close(views.first_order_raw_probability, self.qa + self.qb - self.q0)
        self.assertFalse(torch.equal(views.ppc_log_probability.exp(), views.projected_first_order_probability))

    def test_explicit_ab_api_names_ppc_and_prc(self) -> None:
        composition = compose_ab(self.q0.log(), self.qa.log(), self.qb.log())
        torch.testing.assert_close(
            composition.ppc_probability,
            posterior_product_two(self.q0.log(), self.qa.log(), self.qb.log()).exp(),
        )
        torch.testing.assert_close(
            composition.prc_probability,
            project_probability(self.qa + self.qb - self.q0),
        )

    def test_views_apply_weights_consistently(self) -> None:
        views = compose_views(
            self.q0.log(),
            (self.qa.log(), self.qb.log()),
            weights=(3.0, 1.0),
        )
        torch.testing.assert_close(
            views.posterior_average_probability,
            0.75 * self.qa + 0.25 * self.qb,
        )
        torch.testing.assert_close(
            views.first_order_raw_probability,
            self.q0 + 3.0 * (self.qa - self.q0) + (self.qb - self.q0),
        )

    def test_signed_velocity_matches_linearity(self) -> None:
        generator = torch.Generator().manual_seed(77)
        state = torch.softmax(torch.randn(2, 7, 4, generator=generator), dim=-1)
        c_factor = torch.rand(2, 7, 4, generator=generator)
        observed = first_order_velocity(state, self.q0, (self.qa, self.qb), c_factor)
        expected = (
            flow_from_probability(state, self.qa, c_factor)
            + flow_from_probability(state, self.qb, c_factor)
            - flow_from_probability(state, self.q0, c_factor)
        )
        torch.testing.assert_close(observed, expected)


if __name__ == "__main__":
    unittest.main()
