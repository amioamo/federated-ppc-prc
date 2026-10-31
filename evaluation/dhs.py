"""DHS activity scorers for 200-bp sequences.

Classes are ``0000``, ``0100``, ``0010``, and ``0110`` in that order.
The convolutional and LSTM networks are fit on caller-supplied rows.
Logistic regression uses both-orientation 3-mer and 4-mer frequencies.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch import nn


CLASS_ORDER = ("0000", "0100", "0010", "0110")
TARGET_CLASS = "0110"
TARGET_INDEX = CLASS_ORDER.index(TARGET_CLASS)
CLASSIFIER_KS = (3, 4)
CLASSIFIER_C = 1.0
CLASSIFIER_MAX_ITER = 2000
CHANNELS = 64
KERNELS = (11, 7)
LSTM_HIDDEN = 64
SEQUENCE_LENGTH = 200
AUROC_MIN = 0.65
BRIER_MAX = 0.25
UNIFORM_CLASS_NAMES = ("0000", "0100", "0010", "0110", "uniform")
UNIFORM_INDEX = 4


def _as_tokens(tokens: np.ndarray) -> np.ndarray:
    values = np.asarray(tokens)
    if values.ndim != 2 or values.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(f"expected [rows, {SEQUENCE_LENGTH}] tokens, got {values.shape}")
    if not np.issubdtype(values.dtype, np.integer) or int(values.min()) < 0 or int(values.max()) > 3:
        raise ValueError("tokens must be integer DNA codes in {0, 1, 2, 3}")
    return values.astype(np.uint8, copy=False)


class SequenceClassifier(nn.Module):
    """Two same-length convolutions, global average pooling, four logits."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(4, CHANNELS, kernel_size=KERNELS[0], padding=KERNELS[0] // 2)
        self.conv2 = nn.Conv1d(CHANNELS, CHANNELS, kernel_size=KERNELS[1], padding=KERNELS[1] // 2)
        self.output = nn.Linear(CHANNELS, len(CLASS_ORDER))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != SEQUENCE_LENGTH:
            raise ValueError(f"expected [batch, {SEQUENCE_LENGTH}] tokens, got {tuple(tokens.shape)}")
        one_hot = torch.nn.functional.one_hot(tokens.long(), num_classes=4).permute(0, 2, 1).float()
        hidden = torch.relu(self.conv1(one_hot))
        hidden = torch.relu(self.conv2(hidden))
        return self.output(hidden.mean(dim=-1))


class SequenceLSTM(nn.Module):
    """One bidirectional LSTM over one-hot DNA, then mean pooling."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=LSTM_HIDDEN,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Linear(2 * LSTM_HIDDEN, len(CLASS_ORDER))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != SEQUENCE_LENGTH:
            raise ValueError(f"expected [batch, {SEQUENCE_LENGTH}] tokens, got {tuple(tokens.shape)}")
        one_hot = torch.nn.functional.one_hot(tokens.long(), num_classes=4).float()
        sequence, _ = self.lstm(one_hot)
        return self.output(sequence.mean(dim=1))


def _fit_network(
    model: nn.Module,
    tokens: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    seed: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[dict[str, torch.Tensor], list[float]]:
    values = _as_tokens(tokens)
    targets = np.asarray(labels, dtype=np.int64)
    if len(values) != len(targets) or epochs <= 0 or batch_size <= 0:
        raise ValueError("invalid network training inputs")
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator()
    generator.manual_seed(seed)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(values), generator=generator).numpy()
        total = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            batch_tokens = torch.from_numpy(values[index].astype(np.int64, copy=False))
            batch_labels = torch.from_numpy(targets[index])
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(batch_tokens), batch_labels)
            if not torch.isfinite(loss):
                raise RuntimeError("training loss is non-finite")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
            seen += len(index)
        losses.append(total / seen)
    model.eval()
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, losses


def fit_sequence_classifier(
    tokens: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    seed: int,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[dict[str, torch.Tensor], list[float]]:
    return _fit_network(
        SequenceClassifier(),
        tokens,
        labels,
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


def fit_sequence_lstm(
    tokens: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    seed: int,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[dict[str, torch.Tensor], list[float]]:
    return _fit_network(
        SequenceLSTM(),
        tokens,
        labels,
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


def load_sequence_classifier(state: dict[str, torch.Tensor]) -> SequenceClassifier:
    model = SequenceClassifier()
    model.load_state_dict(state)
    model.eval()
    return model


def load_sequence_lstm(state: dict[str, torch.Tensor]) -> SequenceLSTM:
    model = SequenceLSTM()
    model.load_state_dict(state)
    model.eval()
    return model


def class_probabilities(model: nn.Module, tokens: np.ndarray, *, batch_size: int = 256) -> np.ndarray:
    values = _as_tokens(tokens)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size].astype(np.int64, copy=False))
            outputs.append(torch.softmax(model(batch), dim=1).numpy())
    return np.concatenate(outputs, axis=0)


def reverse_complement_kmer_features(
    tokens: np.ndarray, *, ks: tuple[int, ...] = CLASSIFIER_KS
) -> np.ndarray:
    """Per-sequence k-mer frequencies, averaged over both orientations."""
    values = np.asarray(tokens, dtype=np.uint8)
    if values.ndim != 2 or np.any(values > 3):
        raise ValueError("tokens must be [rows, length] values in {0, 1, 2, 3}")
    if not ks or any(k <= 0 or k > values.shape[1] for k in ks):
        raise ValueError("invalid k-mer sizes")
    rows = np.arange(len(values), dtype=np.int64)
    outputs: list[np.ndarray] = []
    reverse = 3 - values[:, ::-1]
    for k in ks:
        counts = np.zeros((len(values), 4**k), dtype=np.float32)
        powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
        for oriented in (values, reverse):
            for start in range(values.shape[1] - k + 1):
                codes = oriented[:, start : start + k].astype(np.int64) @ powers
                np.add.at(counts, (rows, codes), 1.0)
        counts /= float(2 * (values.shape[1] - k + 1))
        outputs.append(counts)
    return np.concatenate(outputs, axis=1)


def classifier_features(tokens: np.ndarray, *, chunk_rows: int = 4096) -> np.ndarray:
    dimension = sum(4**k for k in CLASSIFIER_KS)
    output = np.empty((len(tokens), dimension), dtype=np.float32)
    for start in range(0, len(tokens), chunk_rows):
        output[start : start + chunk_rows] = reverse_complement_kmer_features(tokens[start : start + chunk_rows])
    return output


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64)
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    exponents = np.exp(shifted)
    return exponents / exponents.sum(axis=1, keepdims=True)


def logistic_probabilities(
    features: np.ndarray,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    weights = np.asarray(coefficients, dtype=np.float64)
    bias = np.asarray(intercepts, dtype=np.float64)
    if weights.shape[0] != bias.shape[0] or weights.shape[1] != values.shape[1]:
        raise ValueError("logistic coefficient shape mismatch")
    return _softmax(values @ weights.T + bias[None, :])


def target_class_probability(
    features: np.ndarray,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    probabilities = logistic_probabilities(features, coefficients, intercepts)
    if probabilities.shape[1] <= TARGET_INDEX:
        raise ValueError("logistic model does not include the 0110 class")
    return probabilities[:, TARGET_INDEX]


def fit_multinomial(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if set(np.unique(targets).tolist()) != set(range(len(CLASS_ORDER))):
        raise ValueError("training labels must contain every class")
    classifier = LogisticRegression(
        C=CLASSIFIER_C,
        solver="lbfgs",
        max_iter=CLASSIFIER_MAX_ITER,
        tol=1e-7,
        random_state=seed,
    )
    classifier.fit(values, targets)
    if list(classifier.classes_) != list(range(len(CLASS_ORDER))):
        raise ValueError(f"unexpected class order: {classifier.classes_.tolist()}")
    iterations = int(np.max(classifier.n_iter_))
    if iterations >= CLASSIFIER_MAX_ITER:
        raise RuntimeError("multinomial logistic regression did not converge")
    return {
        "coefficients": np.asarray(classifier.coef_, dtype=np.float64),
        "intercepts": np.asarray(classifier.intercept_, dtype=np.float64),
        "iterations": np.asarray([iterations], dtype=np.int64),
    }


def average_probabilities(*members: np.ndarray) -> np.ndarray:
    rows = [np.asarray(member, dtype=np.float64) for member in members]
    if len(rows) < 2 or any(row.shape != rows[0].shape for row in rows):
        raise ValueError("ensemble members must be at least two equal-shaped scores")
    if np.any(rows[0] < 0) or np.any(rows[0] > 1):
        raise ValueError("ensemble scores must lie in [0, 1]")
    return np.mean(rows, axis=0)


def discrimination(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """0110-versus-rest AUROC and Brier score."""
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    binary = targets == TARGET_INDEX
    if binary.sum() == 0 or binary.sum() == len(binary):
        raise ValueError("0110-versus-rest needs both classes")
    per_class = {}
    for index, name in enumerate(CLASS_ORDER):
        mask = targets == index
        per_class[name] = {"rows": int(mask.sum()), "mean_target_probability": float(values[mask].mean())}
    return {
        "auroc_0110_vs_rest": float(roc_auc_score(binary.astype(np.int64), values)),
        "brier_0110_vs_rest": float(brier_score_loss(binary.astype(np.int64), values)),
        "per_class": per_class,
    }


def lowered_gate(member_aurocs: dict[str, float], ensemble_auroc: float, ensemble_brier: float) -> str:
    passed = (
        all(value >= AUROC_MIN for value in member_aurocs.values())
        and ensemble_auroc >= AUROC_MIN
        and ensemble_brier < BRIER_MAX
    )
    return "PASS" if passed else "HOLD_UNINFORMATIVE_CLASSIFIER"


def scorer_ratios(generated: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    """Equal-weight average of generated-to-real score ratios."""
    ratios = {}
    for name in ("logistic", "convolution", "lstm"):
        denominator = reference[name]
        if denominator <= 0:
            raise ValueError(f"{name} reference score is not positive")
        ratios[name] = float(generated[name]) / float(denominator)
    ratios["average"] = float(np.mean([ratios[name] for name in ("logistic", "convolution", "lstm")]))
    return ratios


def admit_scorer(mean_target_probability: dict[str, float], uniform_class_means: list[float]) -> bool:
    """Keep a scorer only when real 0110 outranks every alternative on P(0110)."""
    target = mean_target_probability["0110"]
    if any(target <= mean_target_probability[name] for name in UNIFORM_CLASS_NAMES if name != "0110"):
        return False
    if len(uniform_class_means) != len(UNIFORM_CLASS_NAMES):
        raise ValueError("uniform class means must cover all five classes")
    return int(np.argmax(uniform_class_means)) == UNIFORM_INDEX
