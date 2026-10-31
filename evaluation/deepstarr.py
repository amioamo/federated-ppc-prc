"""Inference mirror of the public two-output DeepSTARR model.

The original model is Keras 2.2.4 / TensorFlow. This module reads those HDF5
weights and does not retrain them. Place ``DeepSTARR.model.json`` and
``DeepSTARR.model.h5`` in the directory passed to ``load_model``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MODEL_INPUT_LENGTH = 249
TARGET_INPUT_LENGTH = 500
TILE_STARTS = (0, 125, 251)
OUTPUT_NAMES = ("developmental_activity", "housekeeping_activity")
EXPECTED_MD5 = {
    "DeepSTARR.model.json": "9b796f79441e53dc75dd79b911fff872",
    "DeepSTARR.model.h5": "7e53a9351b2520a4713a5ffdb5f1566c",
}
EXPECTED_SHA256 = {
    "DeepSTARR.model.json": "679a6998e23adcdbb489d11d998eb0132f9e3076fb6cac54e515e26381016291",
    "DeepSTARR.model.h5": "b1077afeb01570b4e5b79936d1f5ad6725a3499919957492d6ef0ecf8b188b46",
}


def _digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_assets(asset_dir: Path) -> dict[str, dict[str, str | int]]:
    asset_dir = Path(asset_dir)
    audit: dict[str, dict[str, str | int]] = {}
    for name in EXPECTED_SHA256:
        path = asset_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing DeepSTARR asset: {name}")
        md5 = _digest(path, "md5")
        sha256 = _digest(path, "sha256")
        if md5 != EXPECTED_MD5[name] or sha256 != EXPECTED_SHA256[name]:
            raise ValueError(f"DeepSTARR asset checksum mismatch: {name}")
        audit[name] = {
            "bytes": path.stat().st_size,
            "md5": md5,
            "sha256": sha256,
        }
    model_json = json.loads((asset_dir / "DeepSTARR.model.json").read_text())
    if model_json.get("keras_version") != "2.2.4" or model_json.get("backend") != "tensorflow":
        raise ValueError("DeepSTARR architecture metadata drifted")
    return audit


class DeepSTARR(nn.Module):
    """Inference-only mirror of the public two-output DeepSTARR model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(4, 256, 7, padding="same")
        self.bn1 = nn.BatchNorm1d(256, eps=1e-3)
        self.conv2 = nn.Conv1d(256, 60, 3, padding="same")
        self.bn2 = nn.BatchNorm1d(60, eps=1e-3)
        self.conv3 = nn.Conv1d(60, 60, 5, padding="same")
        self.bn3 = nn.BatchNorm1d(60, eps=1e-3)
        self.conv4 = nn.Conv1d(60, 120, 3, padding="same")
        self.bn4 = nn.BatchNorm1d(120, eps=1e-3)
        self.dense1 = nn.Linear(1800, 256)
        self.bn5 = nn.BatchNorm1d(256, eps=1e-3)
        self.dense2 = nn.Linear(256, 256)
        self.bn6 = nn.BatchNorm1d(256, eps=1e-3)
        self.dev = nn.Linear(256, 1)
        self.hk = nn.Linear(256, 1)

    def forward(self, one_hot: torch.Tensor) -> torch.Tensor:
        if one_hot.ndim != 3 or tuple(one_hot.shape[1:]) != (MODEL_INPUT_LENGTH, 4):
            raise ValueError(f"DeepSTARR expects [N, {MODEL_INPUT_LENGTH}, 4]")
        x = one_hot.transpose(1, 2)
        x = F.max_pool1d(F.relu(self.bn1(self.conv1(x))), 2, 2)
        x = F.max_pool1d(F.relu(self.bn2(self.conv2(x))), 2, 2)
        x = F.max_pool1d(F.relu(self.bn3(self.conv3(x))), 2, 2)
        x = F.max_pool1d(F.relu(self.bn4(self.conv4(x))), 2, 2)
        x = x.transpose(1, 2).reshape(x.shape[0], -1)
        x = F.relu(self.bn5(self.dense1(x)))
        x = F.relu(self.bn6(self.dense2(x)))
        return torch.cat((self.dev(x), self.hk(x)), dim=1)


def _dataset(group: h5py.File, layer: str, suffix: str) -> np.ndarray:
    layer_group = group[layer]
    nested = next(value for value in layer_group.values() if isinstance(value, h5py.Group))
    key = next(key for key in nested.keys() if key.endswith(suffix))
    return np.asarray(nested[key])


def _load_conv(group: h5py.File, layer: str, module: nn.Conv1d) -> None:
    kernel = _dataset(group, layer, "kernel:0")
    bias = _dataset(group, layer, "bias:0")
    module.weight.data.copy_(torch.from_numpy(kernel.transpose(2, 1, 0)))
    module.bias.data.copy_(torch.from_numpy(bias))


def _load_dense(group: h5py.File, layer: str, module: nn.Linear) -> None:
    kernel = _dataset(group, layer, "kernel:0")
    bias = _dataset(group, layer, "bias:0")
    module.weight.data.copy_(torch.from_numpy(kernel.T))
    module.bias.data.copy_(torch.from_numpy(bias))


def _load_batch_norm(group: h5py.File, layer: str, module: nn.BatchNorm1d) -> None:
    module.weight.data.copy_(torch.from_numpy(_dataset(group, layer, "gamma:0")))
    module.bias.data.copy_(torch.from_numpy(_dataset(group, layer, "beta:0")))
    module.running_mean.data.copy_(torch.from_numpy(_dataset(group, layer, "moving_mean:0")))
    module.running_var.data.copy_(torch.from_numpy(_dataset(group, layer, "moving_variance:0")))


def load_model(asset_dir: Path, device: str | torch.device = "cpu") -> DeepSTARR:
    asset_dir = Path(asset_dir)
    verify_assets(asset_dir)
    model = DeepSTARR()
    with h5py.File(asset_dir / "DeepSTARR.model.h5", "r") as group:
        for layer, module in zip(
            ("Conv1D_1st", "Conv1D_2", "Conv1D_3", "Conv1D_4"),
            (model.conv1, model.conv2, model.conv3, model.conv4),
        ):
            _load_conv(group, layer, module)
        for layer, module in zip(
            (
                "batch_normalization_60",
                "batch_normalization_61",
                "batch_normalization_62",
                "batch_normalization_63",
                "batch_normalization_64",
                "batch_normalization_65",
            ),
            (model.bn1, model.bn2, model.bn3, model.bn4, model.bn5, model.bn6),
        ):
            _load_batch_norm(group, layer, module)
        for layer, module in zip(
            ("Dense_1", "Dense_2", "Dense_Dev", "Dense_Hk"),
            (model.dense1, model.dense2, model.dev, model.hk),
        ):
            _load_dense(group, layer, module)
    return model.to(device).eval()


def _as_tokens(tokens: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(tokens)
    if values.ndim != 2 or values.shape[1] != length:
        raise ValueError(f"DeepSTARR input must be [N, {length}]")
    if values.dtype.kind not in "iu" or np.any((values < 0) | (values > 3)):
        raise ValueError("DNA tokens must be integers in {0, 1, 2, 3}")
    return values.astype(np.int64, copy=False)


def _views(tokens: np.ndarray) -> Iterable[np.ndarray]:
    for values in (tokens, 3 - tokens[:, ::-1]):
        for start in TILE_STARTS:
            yield values[:, start : start + MODEL_INPUT_LENGTH]


@torch.inference_mode()
def predict_tokens(
    model: DeepSTARR,
    tokens: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Score 500-bp sequences by averaging three tiles and both orientations."""
    values = _as_tokens(tokens, TARGET_INPUT_LENGTH)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output = np.zeros((len(values), len(OUTPUT_NAMES)), dtype=np.float64)
    for view in _views(values):
        for start in range(0, len(view), batch_size):
            batch = torch.as_tensor(view[start : start + batch_size], device=device)
            one_hot = F.one_hot(batch, num_classes=4).to(torch.float32)
            output[start : start + len(batch)] += model(one_hot).cpu().numpy()
    return (output / 6.0).astype(np.float32)


@torch.inference_mode()
def predict_exact_tokens(
    model: DeepSTARR,
    tokens: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Score exact 249-bp sequences and average the two orientations."""
    values = _as_tokens(tokens, MODEL_INPUT_LENGTH)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output = np.zeros((len(values), len(OUTPUT_NAMES)), dtype=np.float64)
    for view in (values, 3 - values[:, ::-1]):
        for start in range(0, len(view), batch_size):
            batch = torch.as_tensor(view[start : start + batch_size], device=device)
            one_hot = F.one_hot(batch, num_classes=4).to(torch.float32)
            output[start : start + len(batch)] += model(one_hot).cpu().numpy()
    return (output / 2.0).astype(np.float32)
