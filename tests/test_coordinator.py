from __future__ import annotations

import unittest

import torch

from baselines import fedavg_state_dict
from federated_PPC import FederatedCompositionPlan, posterior_product
from PRC import projected_first_order


class TestFederatedComposition(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(20260901)
        self.logits = {
            participant: torch.randn(3, 11, 4, generator=generator)
            for participant in ("background", "factor_a", "factor_b")
        }
        self.plan = FederatedCompositionPlan.unit_weighted(
            "background", ("factor_a", "factor_b")
        )

    def test_coordinator_ppc_matches_central_composition(self) -> None:
        contributions = self.plan.contributions(self.logits)
        aggregate = sum(contributions.values())
        observed = self.plan.finalize_ppc(aggregate)

        log_probabilities = {
            key: torch.log_softmax(value, dim=-1)
            for key, value in self.logits.items()
        }
        expected = posterior_product(
            log_probabilities["background"],
            (
                log_probabilities["factor_a"],
                log_probabilities["factor_b"],
            ),
        )
        torch.testing.assert_close(observed, expected)

    def test_coordinator_prc_matches_central_composition(self) -> None:
        probabilities = {
            key: torch.softmax(value, dim=-1) for key, value in self.logits.items()
        }
        aggregate = sum(self.plan.contributions(probabilities).values())
        observed = self.plan.finalize_prc(aggregate)
        expected = projected_first_order(
            probabilities["background"],
            (probabilities["factor_a"], probabilities["factor_b"]),
        )
        torch.testing.assert_close(observed, expected)

    def test_weighted_coefficients_match_weighted_ppc(self) -> None:
        plan = FederatedCompositionPlan(
            "background", ("factor_a", "factor_b"), (0.25, 1.75)
        )
        aggregate = sum(plan.contributions(self.logits).values())
        observed = plan.finalize_ppc(aggregate)
        log_probabilities = {
            key: torch.log_softmax(value, dim=-1)
            for key, value in self.logits.items()
        }
        expected = posterior_product(
            log_probabilities["background"],
            (
                log_probabilities["factor_a"],
                log_probabilities["factor_b"],
            ),
            weights=(0.25, 1.75),
        )
        torch.testing.assert_close(observed, expected)

    def test_plan_rejects_missing_or_duplicate_participants(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            FederatedCompositionPlan.unit_weighted("background", ("background",))
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.plan.contributions({"background": self.logits["background"]})


class TestFederatedAveraging(unittest.TestCase):
    def test_sample_weighted_average(self) -> None:
        states = {
            "hospital_a": {
                "weight": torch.tensor([1.0, 3.0]),
                "counter": torch.tensor(4),
            },
            "hospital_b": {
                "weight": torch.tensor([5.0, 7.0]),
                "counter": torch.tensor(4),
            },
        }
        observed = fedavg_state_dict(states, {"hospital_a": 3, "hospital_b": 1})
        torch.testing.assert_close(observed["weight"], torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(observed["counter"], torch.tensor(4))

    def test_average_rejects_incompatible_states(self) -> None:
        states = {
            "a": {"weight": torch.ones(2)},
            "b": {"bias": torch.ones(2)},
        }
        with self.assertRaisesRegex(ValueError, "same parameter keys"):
            fedavg_state_dict(states, {"a": 1, "b": 1})

    def test_average_rejects_nonpositive_counts(self) -> None:
        states = {"a": {"weight": torch.ones(2)}}
        with self.assertRaisesRegex(ValueError, "positive integer"):
            fedavg_state_dict(states, {"a": 0})


if __name__ == "__main__":
    unittest.main()
