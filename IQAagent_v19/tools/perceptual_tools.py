"""Deterministic image-derived evidence tools.

All visualization maps use fixed scaling. Per-image min/max normalization is
intentionally avoided because it destroys cross-image magnitude information.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _load_bgr(img_path: str) -> np.ndarray:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    return img


def _load_gray(img_path: str) -> np.ndarray:
    return cv2.cvtColor(_load_bgr(img_path), cv2.COLOR_BGR2GRAY)


def _save_checked(path: str | None, image: np.ndarray) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image: {path}")


def noise_residual_map(img_path: str, save_path: str | None = None) -> np.ndarray:
    """Return a fixed-scale residual map.

    Residual 0 maps to black; residual >= 24 gray levels maps to white.
    """

    gray = _load_gray(img_path).astype(np.float32)
    denoised = cv2.medianBlur(gray.astype(np.uint8), 5).astype(np.float32)
    residual = np.abs(gray - denoised)
    residual_u8 = np.clip(residual / 24.0 * 255.0, 0, 255).astype(np.uint8)
    _save_checked(save_path, residual_u8)
    return residual_u8


def gradient_magnitude_map(img_path: str, save_path: str | None = None) -> np.ndarray:
    """Return a fixed-scale Sobel gradient map.

    Gradient magnitude >= 512 maps to white.
    """

    gray = _load_gray(img_path).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    magnitude_u8 = np.clip(magnitude / 512.0 * 255.0, 0, 255).astype(np.uint8)
    _save_checked(save_path, magnitude_u8)
    return magnitude_u8


def detail_contact_sheet(
    img_path: str,
    save_path: str | None = None,
    tile_size: int = 224,
) -> np.ndarray:
    """Create a 2x3 sheet: full image plus five native-detail regions."""

    img = _load_bgr(img_path)
    height, width = img.shape[:2]
    crop_side = max(32, min(height, width, 448))
    crop_side = min(crop_side, height, width)

    positions = [
        (0, 0),
        (0, width - crop_side),
        (height - crop_side, 0),
        (height - crop_side, width - crop_side),
        ((height - crop_side) // 2, (width - crop_side) // 2),
    ]

    interpolation = (
        cv2.INTER_AREA if max(height, width) > tile_size else cv2.INTER_CUBIC
    )
    full = cv2.resize(img, (tile_size, tile_size), interpolation=interpolation)
    tiles = [full]
    for top, left in positions:
        crop = img[top : top + crop_side, left : left + crop_side]
        tiles.append(
            cv2.resize(crop, (tile_size, tile_size), interpolation=interpolation)
        )

    rows = [
        np.hstack(tiles[:3]),
        np.hstack(tiles[3:6]),
    ]
    sheet = np.vstack(rows)
    _save_checked(save_path, sheet)
    return sheet


def gradient_sharpness_score(img_path: str) -> float:
    gray = _load_gray(img_path).astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    return float(laplacian.var())


def noise_level_score(img_path: str) -> float:
    gray = _load_gray(img_path).astype(np.float32)
    denoised = cv2.medianBlur(gray.astype(np.uint8), 5).astype(np.float32)
    residual = gray - denoised
    return float(np.std(residual))


def exposure_statistics(img_path: str) -> dict[str, float]:
    gray = _load_gray(img_path).astype(np.float32)
    return {
        "mean_luma": float(gray.mean()),
        "luma_std": float(gray.std()),
        "clipped_dark_ratio": float(np.mean(gray <= 4.0)),
        "clipped_bright_ratio": float(np.mean(gray >= 251.0)),
    }


def blockiness_score(img_path: str, block_size: int = 8) -> float:
    """Estimate JPEG block boundaries without a learned model."""

    gray = _load_gray(img_path).astype(np.float32)
    if gray.shape[0] <= block_size * 2 or gray.shape[1] <= block_size * 2:
        return 0.0

    vertical_diff = np.abs(np.diff(gray, axis=1))
    horizontal_diff = np.abs(np.diff(gray, axis=0))

    v_boundary = vertical_diff[:, block_size - 1 :: block_size].mean()
    h_boundary = horizontal_diff[block_size - 1 :: block_size, :].mean()

    v_inside_mask = np.ones(vertical_diff.shape[1], dtype=bool)
    h_inside_mask = np.ones(horizontal_diff.shape[0], dtype=bool)
    v_inside_mask[block_size - 1 :: block_size] = False
    h_inside_mask[block_size - 1 :: block_size] = False

    v_inside = vertical_diff[:, v_inside_mask].mean()
    h_inside = horizontal_diff[h_inside_mask, :].mean()
    return float(max(0.0, (v_boundary + h_boundary - v_inside - h_inside) / 2.0))


def patch_quality_scores(img_path: str, patch_size: int = 256) -> dict:
    """Return raw sharpness/noise measurements for five spatial regions."""

    gray = _load_gray(img_path).astype(np.float32)
    height, width = gray.shape
    side = max(1, min(patch_size, max(1, height // 2), max(1, width // 2)))

    positions = {
        "top_left": (0, 0),
        "top_right": (0, width - side),
        "bottom_left": (height - side, 0),
        "bottom_right": (height - side, width - side),
        "center": ((height - side) // 2, (width - side) // 2),
    }

    regions: dict[str, dict[str, float]] = {}
    for name, (top, left) in positions.items():
        patch = gray[top : top + side, left : left + side]
        laplacian = cv2.Laplacian(patch, cv2.CV_32F)
        denoised = cv2.medianBlur(patch.astype(np.uint8), 5).astype(np.float32)
        regions[name] = {
            "sharpness_raw": float(laplacian.var()),
            "noise_raw": float(np.std(patch - denoised)),
        }

    return {
        "regions": regions,
        "worst_patch_sharpness_raw": min(
            item["sharpness_raw"] for item in regions.values()
        ),
        "worst_patch_noise_raw": max(item["noise_raw"] for item in regions.values()),
    }


def compute_raw_evidence(img_path: str) -> dict[str, float | None]:
    """Compute every scalar evidence item for one image."""

    from tools.iqa_tools import brisque_score

    patch = patch_quality_scores(img_path)
    exposure = exposure_statistics(img_path)
    return {
        "sharpness_raw": gradient_sharpness_score(img_path),
        "noise_raw": noise_level_score(img_path),
        "brisque_raw": brisque_score(img_path),
        "worst_patch_sharpness_raw": patch["worst_patch_sharpness_raw"],
        "worst_patch_noise_raw": patch["worst_patch_noise_raw"],
        "blockiness_raw": blockiness_score(img_path),
        **exposure,
    }
