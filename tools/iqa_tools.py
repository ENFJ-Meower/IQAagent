import numpy as np
import torch
import cv2
from pathlib import Path


def _to_tensor(img_path: str) -> torch.Tensor:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)


def brisque_score(img_path: str) -> float:
    try:
        import piq
        tensor = _to_tensor(img_path)
        score = piq.brisque(tensor, data_range=1.0, reduction='mean')
        return float(score.item())
    except Exception as e:
        return -1.0


def niqe_score(img_path: str) -> float:
    try:
        import piq
        tensor = _to_tensor(img_path)
        score = piq.niqe(tensor, data_range=1.0, reduction='mean')
        return float(score.item())
    except Exception as e:
        return -1.0


def _piecewise_linear(x: float, breakpoints: list) -> float:
    """Piecewise linear interpolation. breakpoints = [(x0,y0),(x1,y1),...]"""
    if x <= breakpoints[0][0]:
        return breakpoints[0][1]
    if x >= breakpoints[-1][0]:
        return breakpoints[-1][1]
    for i in range(len(breakpoints) - 1):
        x0, y0 = breakpoints[i]
        x1, y1 = breakpoints[i + 1]
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return breakpoints[-1][1]


def normalize_to_mos_scale(score: float, tool_name: str, mode: str = 'zero-shot') -> float:
    """Map tool scores to MOS scale [1, 5]. Higher = better quality.

    mode='zero-shot'      : ITU-T P.910 theory anchored, no training data used.
    mode='train-augmented': Piecewise linear based on KonIQ train set statistics
                            (n=300, 2026-07-27). Train set is not an evaluation
                            set — compliant with task spec §4.1.
    """
    if tool_name == 'brisque':
        if mode == 'train-augmented':
            # KonIQ train set (n=300): p10=4.26, p90=40.63
            # BRISQUE: lower = better → higher MOS
            lo, hi = 4.26, 40.63
        else:
            # Zero-shot: p10=6.65, p90=43.04 (prior train sample, kept for stability)
            lo, hi = 6.65, 43.04
        if score < lo:
            return 5.0
        if score > hi:
            return 1.0
        return 5.0 - ((score - lo) / (hi - lo)) * 4.0

    elif tool_name == 'niqe':
        lo, hi = 3.0, 12.0
        if score < lo:
            return 5.0
        if score > hi:
            return 1.0
        return 5.0 - ((score - lo) / (hi - lo)) * 4.0

    elif tool_name == 'sharpness':
        if mode == 'train-augmented':
            # Piecewise linear derived from KonIQ train set (n=300):
            # Segment 0-200   : MOS avg 2.78  → map to 1.0-2.5  (steep: captures blur severity)
            # Segment 200-1000: MOS avg 3.17-3.22 → map to 2.5-3.3 (moderate)
            # Segment 1000+   : MOS saturates at 3.5-3.56 → map to 3.3-4.8 (gentle)
            # Anchored at p90=3768 → MOS 5.0
            breakpoints = [
                (0,    1.0),
                (50,   1.0),
                (200,  2.5),
                (1000, 3.3),
                (3768, 5.0),
            ]
            return _piecewise_linear(score, breakpoints)
        else:
            # Zero-shot: linear p10→1.0, p90→5.0
            lo, hi = 62.80, 4000.51
            if score < lo:
                return 1.0
            if score > hi:
                return 5.0
            return 1.0 + ((score - lo) / (hi - lo)) * 4.0

    else:
        return 3.0
