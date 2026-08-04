"""Restricted Router/Decision logic.

The router only:
1. selects image-derived evidence rules;
2. resolves conflicts between VLM and deterministic evidence;
3. exposes an explanation of that decision.

It never receives dataset names, image identifiers, file names, splits or MOS.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

DISTORTION_TYPES = {
    "Blurs",
    "Noise",
    "Compression",
    "Brightness change",
    "Color distortions",
    "Sharpness",
    "Contrast",
}

_DEFAULT_WEIGHTS = {
    "global_sharpness": 0.30,
    "local_sharpness": 0.15,
    "brisque_quality": 0.25,
    "noise_quality": 0.12,
    "exposure_quality": 0.10,
    "blockiness_quality": 0.08,
}

_PROFILES = {
    "blur": {
        "global_sharpness": 0.40,
        "local_sharpness": 0.25,
        "brisque_quality": 0.18,
        "noise_quality": 0.05,
        "exposure_quality": 0.07,
        "blockiness_quality": 0.05,
    },
    "noise": {
        "global_sharpness": 0.16,
        "local_sharpness": 0.10,
        "brisque_quality": 0.27,
        "noise_quality": 0.32,
        "exposure_quality": 0.08,
        "blockiness_quality": 0.07,
    },
    "compression": {
        "global_sharpness": 0.17,
        "local_sharpness": 0.10,
        "brisque_quality": 0.28,
        "noise_quality": 0.08,
        "exposure_quality": 0.07,
        "blockiness_quality": 0.30,
    },
    "exposure": {
        "global_sharpness": 0.17,
        "local_sharpness": 0.08,
        "brisque_quality": 0.20,
        "noise_quality": 0.10,
        "exposure_quality": 0.38,
        "blockiness_quality": 0.07,
    },
}


def sanitize_distortions(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        if isinstance(value, str) and value in DISTORTION_TYPES and value not in result:
            result.append(value)
    return result


def select_rule_profile(distortions: list[str]) -> str:
    distortion_set = set(distortions)
    if distortion_set & {"Blurs", "Sharpness"}:
        return "blur"
    if "Compression" in distortion_set:
        return "compression"
    if "Noise" in distortion_set:
        return "noise"
    if distortion_set & {"Brightness change", "Contrast", "Color distortions"}:
        return "exposure"
    return "general"


def select_evidence_types(distortions: list[str]) -> list[str]:
    """Return the visual evidence maps selected for this image."""

    selected = ["detail"]
    distortion_set = set(distortions)
    if distortion_set & {"Blurs", "Sharpness", "Compression"}:
        selected.append("gradient")
    if distortion_set & {"Noise", "Compression"}:
        selected.append("noise")
    return selected


def _available_quality_values(
    evidence: Mapping[str, float | None],
) -> dict[str, float]:
    values: dict[str, float] = {}
    direct_keys = {
        "global_sharpness",
        "local_sharpness",
        "brisque_quality",
        "exposure_quality",
        "blockiness_quality",
    }
    for key in direct_keys:
        value = evidence.get(key)
        if value is not None and math.isfinite(float(value)):
            values[key] = float(np.clip(value, 0.0, 100.0))

    noise = evidence.get("noise_severity")
    if noise is not None and math.isfinite(float(noise)):
        values["noise_quality"] = 100.0 - float(np.clip(noise, 0.0, 100.0))
    return values


def compute_evidence_anchor(
    evidence: Mapping[str, float | None],
    distortions: list[str],
) -> tuple[float, dict[str, float], str]:
    """Compute a fixed, dataset-independent evidence anchor."""

    profile = select_rule_profile(distortions)
    requested_weights = _PROFILES.get(profile, _DEFAULT_WEIGHTS)
    values = _available_quality_values(evidence)
    if not values:
        return 50.0, {}, profile

    active = {
        key: weight
        for key, weight in requested_weights.items()
        if key in values and weight > 0
    }
    total = sum(active.values())
    normalized = {key: weight / total for key, weight in active.items()}
    anchor = sum(values[key] * normalized[key] for key in normalized)
    return float(np.clip(anchor, 0.0, 100.0)), normalized, profile


_DIMENSION_NAMES = (
    "sharpness",
    "noise_cleanliness",
    "exposure",
    "color_fidelity",
    "artifact_free",
)


def dimension_score(
    dimensions: Mapping[str, object],
) -> tuple[float | None, dict[str, float]]:
    """Average available VLM technical dimensions with fixed equal weights."""

    valid: dict[str, float] = {}
    for name in _DIMENSION_NAMES:
        try:
            value = float(dimensions.get(name))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and 0.0 <= value <= 100.0:
            valid[name] = value
    if not valid:
        return None, {}
    weights = {name: 1.0 / len(valid) for name in valid}
    score = sum(valid[name] * weights[name] for name in valid)
    return float(score), weights


def fuse_scores(
    evidence_anchor: float,
    vlm_direct: float | None,
    vlm_dimensions: float | None,
    _unused_confidence: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """Use deterministic evidence only as a guardrail around VLM judgments."""

    components: dict[str, float] = {"evidence": float(evidence_anchor)}
    if vlm_direct is not None and math.isfinite(float(vlm_direct)):
        components["vlm_direct"] = float(np.clip(vlm_direct, 0.0, 100.0))
    if vlm_dimensions is not None and math.isfinite(float(vlm_dimensions)):
        components["vlm_dimensions"] = float(np.clip(vlm_dimensions, 0.0, 100.0))

    if len(components) == 1:
        return components["evidence"], {"evidence": 1.0}

    if "vlm_dimensions" in components and "vlm_direct" in components:
        requested = {
            "evidence": 0.10,
            "vlm_direct": 0.50,
            "vlm_dimensions": 0.40,
        }
    elif "vlm_dimensions" in components:
        requested = {"evidence": 0.15, "vlm_dimensions": 0.85}
    else:
        requested = {"evidence": 0.15, "vlm_direct": 0.85}

    active = {key: value for key, value in requested.items() if key in components}
    total = sum(active.values())
    weights = {key: value / total for key, value in active.items()}
    fused = sum(components[key] * weights[key] for key in weights)
    return float(np.clip(fused, 0.0, 100.0)), weights
