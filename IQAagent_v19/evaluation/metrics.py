"""Offline-only evaluation metrics and score-scale conversion."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr


def label_to_100(value: float, label_min: float, label_max: float) -> float:
    if label_max <= label_min:
        raise ValueError("label_max must be greater than label_min")
    return float((value - label_min) / (label_max - label_min) * 100.0)


def score_100_to_native(value: float, label_min: float, label_max: float) -> float:
    if label_max <= label_min:
        raise ValueError("label_max must be greater than label_min")
    return float(label_min + value / 100.0 * (label_max - label_min))


def evaluate(
    predicted_100: list[float],
    labels_native: list[float],
    label_min: float,
    label_max: float,
) -> dict[str, float | int | None]:
    if len(predicted_100) != len(labels_native):
        raise ValueError("prediction and label lengths differ")

    valid = [
        (float(pred), float(label))
        for pred, label in zip(predicted_100, labels_native)
        if math.isfinite(float(pred)) and math.isfinite(float(label))
    ]
    if not valid:
        return {"N": 0, "SRCC": None, "MAE_100": None, "MAE_native": None}

    predictions = np.array([item[0] for item in valid], dtype=np.float64)
    labels = np.array([item[1] for item in valid], dtype=np.float64)
    labels_100 = np.array(
        [label_to_100(value, label_min, label_max) for value in labels],
        dtype=np.float64,
    )
    predictions_native = np.array(
        [score_100_to_native(value, label_min, label_max) for value in predictions],
        dtype=np.float64,
    )

    srcc = None
    if len(valid) >= 2:
        value, _ = spearmanr(predictions, labels)
        if np.isfinite(value):
            srcc = float(value)

    return {
        "N": len(valid),
        "SRCC": srcc,
        "MAE_100": float(np.mean(np.abs(predictions - labels_100))),
        "MAE_native": float(np.mean(np.abs(predictions_native - labels))),
    }
