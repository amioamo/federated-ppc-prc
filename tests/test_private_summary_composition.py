from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.distance import pdist
from scipy.special import softmax

from private_summary_composition import (
    PublicFeatureMap,
    correct_gc,
    conditional_weights,
    fit_conditional,
    fit_energy,
    independent_mh,
    orbit_frequencies,
    privacy_rho,
    release_summary,
    temper_within_strata,
)


class TestPrivateComposition(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(937061)
        self.x = rng.integers(0, 4, (256, 249), dtype=np.uint8)
        self.features = PublicFeatureMap.fit_public(self.x)

    def test_features_preserve_rc_and_privacy_geometry(self) -> None:
        x = np.concatenate(
            [self.x[:48], np.stack([np.full(249, a, dtype=np.uint8) for a in range(4)])]
        )
        features = self.features
        self.assertTrue(np.allclose(features.projection5.T @ features.projection5, np.eye(32), atol=1e-12))
        for name in ["k3", "multiscale"]:
            f = features.transform(x, name)
            rc = features.transform(3 - x[:, ::-1], name)
            self.assertTrue(np.allclose(f, rc, atol=1e-13))
            self.assertLessEqual(pdist(f).max(), 1 + 1e-12)
            self.assertLessEqual(np.linalg.norm(f, axis=1).max(), 1 / np.sqrt(2) + 1e-12)
            joint = np.zeros((len(x), 4, f.shape[1]))
            joint[np.arange(len(x)), features.strata(x)] = f
            self.assertLessEqual(pdist(joint.reshape(len(x), -1)).max(), 1 + 1e-12)
            gc = sum(((x[:, j : j + 247] == 1) | (x[:, j : j + 247] == 2)).mean(1) for j in range(3)) / 3
            self.assertTrue(np.allclose(f @ features.gc_direction(name), gc, atol=1e-13))
        h = orbit_frequencies(x, 3)
        self.assertTrue(np.allclose(h.sum(1), 1))
        self.assertTrue(np.isclose(np.linalg.norm(h[-4] / np.sqrt(2) - h[-3] / np.sqrt(2)), 1))
        full = np.repeat(h / 2, 2, axis=1)
        compact = h / np.sqrt(2)
        self.assertTrue(np.allclose(full @ full.T, compact @ compact.T, atol=1e-13))

    def test_joint_release_budget_and_no_noise(self) -> None:
        x, features = self.x, self.features
        f = features.transform(x, "multiscale")
        g = features.strata(x)
        for eps in [1.0, 3.0, 8.0]:
            r = release_summary(f, g, conditional=True, epsilon=eps, delta=1e-6, rng=np.random.default_rng(813))
            feature_rho = (1 / len(x)) ** 2 / (2 * r["feature_sd"] ** 2)
            count_rho = (np.sqrt(2) / len(x)) ** 2 / (2 * r["count_sd"] ** 2)
            rho = privacy_rho(eps, 1e-6)
            self.assertTrue(np.isclose(feature_rho + count_rho, rho))
            self.assertTrue(np.isclose(rho + 2 * np.sqrt(rho * np.log(1e6)), eps))
        r = release_summary(f, g, conditional=True, epsilon=None, delta=1e-6, rng=np.random.default_rng(81))
        self.assertTrue(np.allclose(r["mean"].sum(0), f.mean(0)))
        self.assertTrue(np.isclose(r["proportions"].sum(), 1))

    def test_conditional_normalization_and_diversity(self) -> None:
        g = np.repeat(np.arange(4), 20)
        score = np.linspace(-30, 30, len(g))
        mass = np.array([0.4, 0.3, 0.2, 0.1])
        w = conditional_weights(score, g, mass)
        self.assertTrue(np.allclose(np.bincount(g, weights=w), mass))
        q, b, ok = temper_within_strata(score, g, mass, 0.5)
        self.assertTrue(ok and 0 <= b <= 1 and 1 / (q @ q) >= 0.5 * len(g) - 1e-7)
        _, _, ok = temper_within_strata(score, g, np.array([0.97, 0.01, 0.01, 0.01]), 0.9)
        self.assertFalse(ok)

    def test_realistic_conditional_fit_and_gc_constraint(self) -> None:
        x, features = self.x, self.features
        f = features.transform(x, "k3")
        g = features.strata(x)
        r = release_summary(f, g, conditional=True, epsilon=None, delta=1e-6, rng=np.random.default_rng(9))
        fit = fit_conditional(f, g, [r, r])
        self.assertTrue(np.allclose(fit["weights"], np.full(len(x), 1 / len(x)), atol=1e-7))
        direction = features.gc_direction("k3")
        score = -500 * (f @ direction)
        w, gamma, target = correct_gc(score, f, [f.mean(0), f.mean(0)], direction)
        self.assertTrue(gamma > 0 and abs(w @ (f @ direction) - target) < 1e-9)
        r["proportions"] = np.array([-0.01, 0.25, 0.3, 0.46])
        fit = fit_conditional(f, g, [r, r], reg=1e-3)
        self.assertTrue(np.isfinite(fit["weights"]).all() and np.isclose(fit["weights"].sum(), 1))

    def test_mh_stationarity_and_repeated_states(self) -> None:
        rng = np.random.default_rng(5121)
        result = independent_mh(np.zeros(22), chains=2, steps=10, burn=5, draws=10, rng=rng)
        self.assertEqual(result["acceptance"], 1)
        self.assertTrue(np.array_equal(result["indices"], np.arange(12, 22)))
        result = independent_mh(np.r_[np.full(2, 1e6), np.zeros(20)], chains=2, steps=10, burn=5, draws=10, rng=rng)
        self.assertEqual(result["acceptance"], 0)
        self.assertEqual(len(np.unique(result["indices"])), 2)

    def test_uncertainty_aware_regularization(self) -> None:
        x, features = self.x, self.features
        f = features.transform(x, "k3")
        g = features.strata(x)
        release = release_summary(f, g, conditional=True, epsilon=3.0, delta=1e-6, rng=np.random.default_rng(73))
        fit = fit_conditional(f, g, [release, release], noise_aware=True)
        self.assertTrue(np.allclose(np.bincount(g, weights=fit["weights"]), fit["mass"]))
        center = np.linspace(-1, 1, f.shape[1])
        theta, _ = fit_energy(f, f.mean(0), reg=1e5, center=center)
        self.assertLess(np.linalg.norm(theta - center), 1e-5)

    def test_invalid_public_map_cannot_claim_contraction(self) -> None:
        features = self.features
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            PublicFeatureMap(features.projection5 * 1.1, features.gc_edges)
        with self.assertRaisesRegex(ValueError, "boundaries"):
            PublicFeatureMap(features.projection5, np.array([0.4, 0.4, 0.5]))


if __name__ == "__main__":
    unittest.main()
