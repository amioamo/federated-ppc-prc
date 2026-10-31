"""Score random sequences with the published metrics."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation import activity_metrics, sequence_metrics


def main() -> None:
    generator = np.random.default_rng(0)
    generated = generator.integers(0, 4, size=(32, 64), dtype=np.uint8)
    target = generator.integers(0, 4, size=(32, 64), dtype=np.uint8)
    training = generator.integers(0, 4, size=(16, 64), dtype=np.uint8)
    print(sequence_metrics(generated, target, training, include_exact_nearest=True))
    print(activity_metrics(generator.random((32, 2)), generator.random((32, 2))))


if __name__ == "__main__":
    main()
