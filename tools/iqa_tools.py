"""Dataset-independent objective evidence for image quality assessment.

The values produced here are image-derived engineering indices, not MOS
predictions and not mappings fitted on KonIQ-10k or SPAQ.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(np.clip(value, low, high))


def brisque_score(img_path: str) -> float | None:
    """Return raw BRISQUE score, or ``None`` when the optional tool fails.

    A failed BRISQUE call must never be converted into a high-quality score.
    """

    try:
        import cv2
        import piq
        import torch

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
        with torch.inference_mode():
            score = piq.brisque(tensor, data_range=1.0, reduction="mean")
        value = float(score.item())
        return value if math.isfinite(value) else None
    except Exception:  # noqa: BLE001 - optional BRISQUE dependency may fail broadly
        return None


def sharpness_index(laplacian_variance: float) -> float:
    """Map Laplacian variance to a fixed 0-100 evidence index.

    Log scaling reduces content/resolution sensitivity. The endpoints are fixed
    engineering thresholds (8 and 3000), not dataset statistics.
    """

    value = max(float(laplacian_variance), 0.0)
    low = math.log1p(8.0)
    high = math.log1p(3000.0)
    return _clip(100.0 * (math.log1p(value) - low) / (high - low))


def brisque_quality_index(raw_brisque: float | None) -> float | None:
    """Convert the conventional BRISQUE direction to a 0-100 quality index.

    This is the fixed transform ``100 - clip(raw, 0, 100)`` and is not fitted
    to subjective labels.
    """

    if raw_brisque is None or not math.isfinite(raw_brisque):
        return None
    return 100.0 - _clip(raw_brisque)


def noise_severity_index(noise_sigma: float) -> float:
    """Fixed 0-100 degradation index from residual standard deviation."""

    return _clip(float(noise_sigma) / 20.0 * 100.0)


def exposure_quality_index(
    mean_luma: float,
    clipped_dark_ratio: float,
    clipped_bright_ratio: float,
) -> float:
    """Fixed exposure evidence using mean luminance and clipped-pixel ratios."""

    center_penalty = abs(float(mean_luma) - 127.5) / 127.5 * 45.0
    clipping_penalty = (
        max(float(clipped_dark_ratio), 0.0) + max(float(clipped_bright_ratio), 0.0)
    ) * 220.0
    return _clip(100.0 - center_penalty - clipping_penalty)


def blockiness_quality_index(blockiness: float) -> float:
    """Fixed JPEG-blockiness evidence; higher output means better quality."""

    penalty = max(float(blockiness), 0.0) / 12.0 * 100.0
    return _clip(100.0 - penalty)


def build_evidence_indices(raw: dict[str, Any]) -> dict[str, float | None]:
    """Create online-safe evidence indices from raw image measurements."""

    return {
        "global_sharpness": sharpness_index(raw["sharpness_raw"]),
        "local_sharpness": sharpness_index(raw["worst_patch_sharpness_raw"]),
        "brisque_quality": brisque_quality_index(raw.get("brisque_raw")),
        "noise_severity": noise_severity_index(raw["noise_raw"]),
        "exposure_quality": exposure_quality_index(
            raw["mean_luma"],
            raw["clipped_dark_ratio"],
            raw["clipped_bright_ratio"],
        ),
        "blockiness_quality": blockiness_quality_index(raw["blockiness_raw"]),
    }
