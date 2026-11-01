"""Synthetic checks for sequence metrics and DHS scorers."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from evaluation import (
    DeepSTARR,
    SequenceClassifier,
    activity_metrics,
    admit_scorer,
    average_probabilities,
    canonical_unique,
    class_probabilities,
    classifier_features,
    energy_distance,
    exact_nearest_hamming,
    fit_multinomial,
    fit_sequence_classifier,
    jensen_shannon_divergence,
    logistic_probabilities,
    predict_exact_tokens,
    reverse_complement,
    scorer_ratios,
    sequence_metrics,
)


class TestEvaluation(unittest.TestCase):
    def test_exact_nearest_handles_both_orientations(self) -> None:
        reference = np.asarray([[0, 1, 2, 3, 0], [3, 3, 3, 3, 3]], dtype=np.uint8)
        query = np.asarray([[3, 0, 1, 2, 3], [0, 0, 0, 0, 1]], dtype=np.uint8)
        augmented = np.concatenate((reference, reverse_complement(reference)))
        expected = [np.min(np.mean(row[None, :] != augmented, axis=1)) for row in query]
        with tempfile.TemporaryDirectory() as temporary:
            first = exact_nearest_hamming(query, reference, output_dir=Path(temporary))
            second = exact_nearest_hamming(query, reference, output_dir=Path(temporary))
        np.testing.assert_allclose(first, expected)
        np.testing.assert_array_equal(first, second)

    def test_canonical_units_collapse_reverse_complement_pairs(self) -> None:
        tokens = np.asarray([[0, 0, 1, 2, 3], [0, 0, 0, 0, 0]], dtype=np.uint8)
        paired = np.concatenate((tokens, reverse_complement(tokens)), axis=0)
        self.assertEqual(len(canonical_unique(paired)), 2)

    def test_identical_inputs_have_zero_discrepancy(self) -> None:
        values = np.asarray([[0.0, 1.0], [2.0, 3.0]])
        self.assertAlmostEqual(energy_distance(values, values), 0.0, places=12)
        self.assertGreater(energy_distance(values, values + 2.0), 0.0)
        self.assertAlmostEqual(jensen_shannon_divergence([0.5, 0.5], [0.5, 0.5]), 0.0)
        generated = np.asarray([[0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 0, 1]], dtype=np.uint8)
        metrics = sequence_metrics(generated, generated, generated, include_exact_nearest=True)
        self.assertAlmostEqual(metrics["kmer_jsd_3"], 0.0)
        self.assertAlmostEqual(metrics["gc_absolute_error"], 0.0)
        self.assertAlmostEqual(metrics["exact_rc_copy_rate"], 1.0)
        self.assertAlmostEqual(metrics["nearest_train_hamming_p50"], 0.0)

    def test_activity_metrics_match_identical_scores(self) -> None:
        scores = np.asarray([[0.1, 0.2], [0.95, 0.4], [0.2, 0.1]], dtype=np.float64)
        metrics = activity_metrics(scores, scores)
        self.assertAlmostEqual(metrics["functional_energy_distance"], 0.0)
        self.assertAlmostEqual(metrics["developmental_mean_gap"], 0.0)
        self.assertAlmostEqual(metrics["functional_target_rate"], 1.0 / 3.0)

    def test_deepstarr_exact_path_is_orientation_invariant(self) -> None:
        generator = np.random.default_rng(7)
        tokens = generator.integers(0, 4, size=(2, 249), dtype=np.uint8)
        model = DeepSTARR().eval()
        left = predict_exact_tokens(model, tokens, batch_size=2)
        right = predict_exact_tokens(model, reverse_complement(tokens), batch_size=2)
        np.testing.assert_allclose(left, right, atol=1e-6, rtol=1e-6)
        with self.assertRaises(ValueError):
            predict_exact_tokens(model, tokens[:, :-1])

    def test_dhs_scorers_and_ratios(self) -> None:
        generator = np.random.default_rng(11)
        tokens = generator.integers(0, 4, size=(8, 200), dtype=np.uint8)
        labels = np.tile(np.arange(4), 2)
        features = classifier_features(tokens)
        self.assertEqual(features.shape, (8, 4**3 + 4**4))
        fitted = fit_multinomial(features, labels, seed=0)
        probabilities = logistic_probabilities(features, fitted["coefficients"], fitted["intercepts"])
        self.assertEqual(probabilities.shape, (8, 4))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        state, losses = fit_sequence_classifier(tokens, labels, epochs=1, seed=0, batch_size=4)
        self.assertEqual(len(losses), 1)
        network = SequenceClassifier()
        network.load_state_dict(state)
        scores = class_probabilities(network, tokens, batch_size=4)
        self.assertEqual(scores.shape, (8, 4))
        averaged = average_probabilities(probabilities[:, 3], scores[:, 3])
        self.assertEqual(averaged.shape, (8,))
        ratios = scorer_ratios(
            {"logistic": 0.2, "convolution": 0.4, "lstm": 0.6},
            {"logistic": 0.4, "convolution": 0.4, "lstm": 0.3},
        )
        self.assertAlmostEqual(ratios["average"], (0.5 + 1.0 + 2.0) / 3.0)
        self.assertTrue(
            admit_scorer(
                {"0000": 0.1, "0100": 0.1, "0010": 0.1, "0110": 0.7, "uniform": 0.05},
                [0.1, 0.1, 0.1, 0.2, 0.5],
            )
        )
        torch.manual_seed(0)


if __name__ == "__main__":
    unittest.main()
