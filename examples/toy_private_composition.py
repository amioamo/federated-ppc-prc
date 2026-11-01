"""Differentially private summary release and composition."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from method import (
    PublicFeatureMap,
    fit_conditional,
    privacy_rho,
    release_summary,
)


def main() -> None:
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 4, size=(256, 249), dtype=np.uint8)
    features = PublicFeatureMap.fit_public(tokens)
    mapped = features.transform(tokens, "k3")
    strata = features.strata(tokens)
    epsilon, delta = 3.0, 1e-6
    released = release_summary(
        mapped, strata, conditional=True, epsilon=epsilon, delta=delta, rng=rng
    )
    fit = fit_conditional(mapped, strata, [released, released])
    print("epsilon", epsilon, "delta", delta, "rho", privacy_rho(epsilon, delta))
    print("released n", released["n"])
    print("composition effective count", float(1.0 / (fit["weights"] @ fit["weights"])))


if __name__ == "__main__":
    main()
