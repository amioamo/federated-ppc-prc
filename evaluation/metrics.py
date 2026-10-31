"""Sequence and activity metrics.

DNA tokens use A=0, C=1, G=2, T=3. Activity scores are two columns,
developmental then housekeeping.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


DEVELOPMENTAL_THRESHOLD = 0.896042879723438
HOUSEKEEPING_THRESHOLD = 0.273656882699098
_BYTE_MISMATCH = np.asarray(
    [sum(((value >> shift) & 3) != 0 for shift in (0, 2, 4, 6)) for value in range(256)],
    dtype=np.uint8,
)


def reverse_complement(tokens: np.ndarray) -> np.ndarray:
    values = np.asarray(tokens, dtype=np.uint8)
    if values.ndim != 2 or np.any(values > 3):
        raise ValueError("tokens must be a [rows, length] array in {0, 1, 2, 3}")
    return 3 - values[:, ::-1]


def canonicalize(tokens: np.ndarray) -> np.ndarray:
    values = np.asarray(tokens, dtype=np.uint8)
    rc = reverse_complement(values)
    choose_rc = np.zeros(len(values), dtype=bool)
    undecided = np.ones(len(values), dtype=bool)
    for column in range(values.shape[1]):
        lower = rc[:, column] < values[:, column]
        higher = rc[:, column] > values[:, column]
        choose_rc |= undecided & lower
        undecided &= ~(lower | higher)
    return np.where(choose_rc[:, None], rc, values)


def canonical_unique(tokens: np.ndarray) -> np.ndarray:
    return np.unique(canonicalize(tokens), axis=0)


def kmer_frequency(tokens: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(tokens, dtype=np.uint8)
    if values.ndim != 2 or not 1 <= k <= values.shape[1]:
        raise ValueError("invalid tokens or k")
    counts = np.zeros(4**k, dtype=np.int64)
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    for oriented in (values, reverse_complement(values)):
        for start in range(values.shape[1] - k + 1):
            codes = oriented[:, start : start + k].astype(np.int64) @ powers
            counts += np.bincount(codes, minlength=4**k)
    return counts.astype(np.float64) / counts.sum()


def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    if p.shape != q.shape or np.any(p < 0) or np.any(q < 0):
        raise ValueError("JSD inputs must be compatible non-negative vectors")
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    left_terms = np.zeros_like(p)
    right_terms = np.zeros_like(q)
    left_mask = p > 0
    right_mask = q > 0
    left_terms[left_mask] = p[left_mask] * np.log(p[left_mask] / midpoint[left_mask])
    right_terms[right_mask] = q[right_mask] * np.log(q[right_mask] / midpoint[right_mask])
    return float(0.5 * (left_terms.sum() + right_terms.sum()))


def energy_distance(reference: np.ndarray, generated: np.ndarray, *, chunk_size: int = 1024) -> float:
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(generated, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or min(len(x), len(y)) < 2:
        raise ValueError("energy distance requires compatible two-dimensional arrays")

    def mean_distance(left: np.ndarray, right: np.ndarray) -> float:
        total = 0.0
        for start in range(0, len(left), chunk_size):
            total += float(cdist(left[start : start + chunk_size], right).sum())
        return total / (len(left) * len(right))

    result = 2.0 * mean_distance(x, y) - mean_distance(x, x) - mean_distance(y, y)
    return float(max(0.0, result))


def _pack_four(tokens: np.ndarray) -> np.ndarray:
    values = np.asarray(tokens, dtype=np.uint8)
    if values.ndim != 2 or np.any(values > 3):
        raise ValueError("invalid DNA tokens")
    padding = (-values.shape[1]) % 4
    if padding:
        values = np.pad(values, ((0, 0), (0, padding)), constant_values=0)
    blocks = values.reshape(len(values), -1, 4)
    return (
        (blocks[..., 0] << 6)
        | (blocks[..., 1] << 4)
        | (blocks[..., 2] << 2)
        | blocks[..., 3]
    ).astype(np.uint8)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_numpy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def exact_nearest_hamming(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    output_dir: Path | None = None,
    query_chunk: int = 32,
    reference_chunk: int = 512,
) -> np.ndarray:
    """Exact orientation-aware nearest Hamming distance, divided by length."""
    q_values = np.asarray(query, dtype=np.uint8)
    r_values = np.asarray(reference, dtype=np.uint8)
    if q_values.ndim != 2 or r_values.ndim != 2 or q_values.shape[1] != r_values.shape[1]:
        raise ValueError("query/reference sequence arrays are incompatible")
    length = q_values.shape[1]
    q = _pack_four(q_values)
    r = _pack_four(np.concatenate((r_values, reverse_complement(r_values)), axis=0))
    completed = 0
    state_path: Path | None = None
    partial_path: Path | None = None
    if output_dir is None:
        nearest: np.ndarray = np.full(len(q), np.nan, dtype=np.float64)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / "nearest_training_hamming.npy"
        if final_path.is_file():
            result = np.load(final_path, allow_pickle=False)
            if result.shape != (len(q),):
                raise ValueError("cached nearest-training array shape mismatch")
            return result
        state_path = output_dir / "nearest_training_hamming.progress.json"
        partial_path = output_dir / "nearest_training_hamming.partial.npy"
        if state_path.is_file() and partial_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            expected = {
                "query_rows": len(q),
                "reference_rows": len(r_values),
                "sequence_length": length,
            }
            if any(int(state.get(key, -1)) != value for key, value in expected.items()):
                raise ValueError("nearest-training resume cache identity mismatch")
            completed = int(state["completed_query_rows"])
            nearest = np.lib.format.open_memmap(partial_path, mode="r+")
        else:
            nearest = np.lib.format.open_memmap(
                partial_path, mode="w+", dtype=np.float64, shape=(len(q),)
            )
            nearest[:] = np.nan

    for q_start in range(completed, len(q), query_chunk):
        q_block = q[q_start : q_start + query_chunk]
        best = np.full(len(q_block), length + 1, dtype=np.int16)
        for r_start in range(0, len(r), reference_chunk):
            xor = np.bitwise_xor(q_block[:, None, :], r[None, r_start : r_start + reference_chunk, :])
            distance = _BYTE_MISMATCH[xor].sum(axis=2, dtype=np.int16)
            best = np.minimum(best, distance.min(axis=1))
        nearest[q_start : q_start + len(q_block)] = best.astype(np.float64) / length
        if output_dir is not None and state_path is not None:
            nearest.flush()
            _atomic_json(
                state_path,
                {
                    "query_rows": len(q),
                    "reference_rows": len(r_values),
                    "sequence_length": length,
                    "completed_query_rows": q_start + len(q_block),
                },
            )
    result = np.asarray(nearest).copy()
    if output_dir is not None and partial_path is not None and state_path is not None:
        _atomic_numpy(output_dir / "nearest_training_hamming.npy", result)
        partial_path.unlink()
        state_path.unlink()
    return result


def sequence_metrics(
    generated: np.ndarray,
    target: np.ndarray,
    training: np.ndarray | None = None,
    *,
    include_exact_nearest: bool = False,
) -> dict[str, float]:
    """3-mer and 6-mer JSD, GC error, uniqueness, and optional training overlap."""
    generated = np.asarray(generated, dtype=np.uint8)
    target = np.asarray(target, dtype=np.uint8)
    canonical_generated = canonicalize(generated)
    metrics = {
        "kmer_jsd_3": jensen_shannon_divergence(kmer_frequency(generated, 3), kmer_frequency(target, 3)),
        "kmer_jsd_6": jensen_shannon_divergence(kmer_frequency(generated, 6), kmer_frequency(target, 6)),
        "gc_absolute_error": float(
            abs(np.mean((generated == 1) | (generated == 2)) - np.mean((target == 1) | (target == 2)))
        ),
        "unique_rate": float(len(np.unique(canonical_generated, axis=0)) / len(generated)),
    }
    if training is None:
        return metrics
    observed = canonical_unique(training)
    observed_rows = {row.tobytes() for row in observed}
    metrics["exact_rc_copy_rate"] = float(
        np.mean([row.tobytes() in observed_rows for row in canonical_generated])
    )
    if include_exact_nearest:
        nearest = exact_nearest_hamming(generated, observed)
        metrics["nearest_train_hamming_p10"] = float(np.quantile(nearest, 0.1))
        metrics["nearest_train_hamming_p50"] = float(np.quantile(nearest, 0.5))
    return metrics


def activity_metrics(
    target_scores: np.ndarray,
    generated_scores: np.ndarray,
    *,
    developmental_threshold: float = DEVELOPMENTAL_THRESHOLD,
    housekeeping_threshold: float = HOUSEKEEPING_THRESHOLD,
) -> dict[str, Any]:
    """Compare two-column activity predictions."""
    target = np.asarray(target_scores, dtype=np.float64)
    generated = np.asarray(generated_scores, dtype=np.float64)
    if target.ndim != 2 or generated.ndim != 2 or target.shape[1] != 2 or generated.shape[1] != 2:
        raise ValueError("activity scores must have shape [rows, 2]")
    quantiles = (0.1, 0.5, 0.9)
    return {
        "functional_energy_distance": energy_distance(target, generated),
        "functional_target_rate": float(
            np.mean(
                (generated[:, 0] >= developmental_threshold)
                & (generated[:, 1] >= housekeeping_threshold)
            )
        ),
        "developmental_wasserstein": float(wasserstein_distance(target[:, 0], generated[:, 0])),
        "housekeeping_wasserstein": float(wasserstein_distance(target[:, 1], generated[:, 1])),
        "developmental_mean_gap": float(generated[:, 0].mean() - target[:, 0].mean()),
        "housekeeping_mean_gap": float(generated[:, 1].mean() - target[:, 1].mean()),
        "developmental_quantile_gaps": {
            str(q): float(np.quantile(generated[:, 0], q) - np.quantile(target[:, 0], q))
            for q in quantiles
        },
        "housekeeping_quantile_gaps": {
            str(q): float(np.quantile(generated[:, 1], q) - np.quantile(target[:, 1], q))
            for q in quantiles
        },
    }
