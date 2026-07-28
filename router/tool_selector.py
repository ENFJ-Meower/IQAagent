"""
Router layer: dynamically weight sharpness_mos vs brisque_mos based on detected distortions.

BRISQUE is an NSS (natural scene statistics) model — it gives unreliable scores for:
- Images with many detected distortions (it saturates near 5.0 on already-degraded textures)
- Compression / Noise dominated images (BRISQUE captures global statistics, misses local artifacts)

Sharpness (Laplacian variance) is more reliable for blur/sharpness distortions.
When BRISQUE MOS is very high (≥ 4.5) but distortions are detected, trust sharpness more.
"""

# Distortion type → (sharpness_weight, brisque_weight)
_DISTORTION_WEIGHTS = {
    'Blurs':             (0.70, 0.30),
    'Sharpness':         (0.70, 0.30),
    'Noise':             (0.35, 0.65),
    'Compression':       (0.30, 0.70),
    'Brightness change': (0.40, 0.60),
    'Color distortions': (0.45, 0.55),
    'Contrast':          (0.40, 0.60),
}

_DEFAULT_WEIGHTS = (0.45, 0.55)


def get_tool_weights(distortions: list[str], brisque_mos: float) -> tuple[float, float]:
    """Return (sharpness_weight, brisque_weight) for the given distortion profile.

    When BRISQUE MOS saturates near 5.0 with distortions present, reduce brisque trust.
    """
    if not distortions:
        # No distortions detected — if brisque_mos is suspiciously high, down-weight it
        if brisque_mos >= 4.5:
            return (0.55, 0.45)
        return _DEFAULT_WEIGHTS

    # Average weights across all detected distortion types
    total_s, total_b = 0.0, 0.0
    count = 0
    for d in distortions:
        s, b = _DISTORTION_WEIGHTS.get(d, _DEFAULT_WEIGHTS)
        total_s += s
        total_b += b
        count += 1

    sw = total_s / count
    bw = total_b / count

    # Saturation penalty: if brisque_mos ≥ 4.5 but distortions detected,
    # BRISQUE is likely unreliable — shift 0.15 weight toward sharpness
    if brisque_mos >= 4.5 and distortions:
        sw = min(sw + 0.15, 0.85)
        bw = 1.0 - sw

    return (sw, bw)


def compute_tool_composite(
    sharpness_mos: float,
    brisque_mos: float,
    distortions: list[str],
) -> tuple[float, float, float]:
    """Return (composite_score, sharpness_weight, brisque_weight)."""
    sw, bw = get_tool_weights(distortions, brisque_mos)
    composite = sw * sharpness_mos + bw * brisque_mos
    return composite, sw, bw
