import numpy as np
from scipy.optimize import curve_fit


def _f(x, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def fit(pred_scores: list, mos_scores: list) -> dict:
    """Fit 5-parameter logistic mapping from VLM scores to MOS scale.

    Returns parameter dict. Raises RuntimeError if fitting fails.
    """
    x = np.array(pred_scores, dtype=np.float64)
    y = np.array(mos_scores, dtype=np.float64)

    # Initial guess: identity-like mapping
    p0 = [1.0, 1.0, np.mean(x), 1.0, 0.0]
    bounds = (
        [-10, 0.01, 1.0, 0.0, -10],
        [10,  10.0, 5.0, 5.0,  10],
    )

    try:
        params, _ = curve_fit(_f, x, y, p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError as e:
        raise RuntimeError(f"Calibration fitting failed: {e}")

    return {
        'b1': float(params[0]),
        'b2': float(params[1]),
        'b3': float(params[2]),
        'b4': float(params[3]),
        'b5': float(params[4]),
    }


def apply_calibration(score: float, params: dict) -> float:
    """Apply fitted calibration to a single score. Clamps output to [1, 5]."""
    b1, b2, b3, b4, b5 = params['b1'], params['b2'], params['b3'], params['b4'], params['b5']
    calibrated = _f(np.array([score]), b1, b2, b3, b4, b5)[0]
    return float(np.clip(calibrated, 1.0, 5.0))
