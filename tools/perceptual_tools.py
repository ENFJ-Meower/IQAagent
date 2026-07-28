import cv2
import numpy as np
from pathlib import Path


def _load_gray(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _load_bgr(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    return img


def noise_residual_map(img_path: str, save_path: str = None) -> np.ndarray:
    gray = _load_gray(img_path).astype(np.float32)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = np.abs(gray - denoised)
    residual_u8 = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if save_path:
        cv2.imwrite(save_path, residual_u8)
    return residual_u8


def gradient_magnitude_map(img_path: str, save_path: str = None) -> np.ndarray:
    gray = _load_gray(img_path).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    mag_u8 = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if save_path:
        cv2.imwrite(save_path, mag_u8)
    return mag_u8


def fourier_magnitude_spectrum(img_path: str, save_path: str = None) -> np.ndarray:
    gray = _load_gray(img_path).astype(np.float32)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20 * np.log(np.abs(dft_shift) + 1)
    spec_u8 = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if save_path:
        cv2.imwrite(save_path, spec_u8)
    return spec_u8


def luminance_histogram(img_path: str) -> np.ndarray:
    gray = _load_gray(img_path)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist.flatten().astype(np.float32) / (gray.shape[0] * gray.shape[1])


def gradient_sharpness_score(img_path: str) -> float:
    gray = _load_gray(img_path).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap.var())


def noise_level_score(img_path: str) -> float:
    gray = _load_gray(img_path).astype(np.float32)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray - denoised
    return float(np.std(residual))


def colorfulness_score(img_path: str) -> float:
    img = _load_bgr(img_path).astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))


def patch_quality_scores(img_path: str, patch_size: int = 128) -> dict:
    """Compute sharpness and noise scores for 4 corner patches.

    Returns worst-patch sharpness MOS and a summary dict.
    Useful as a local quality lower-bound signal.
    """
    from tools.iqa_tools import normalize_to_mos_scale

    img = _load_gray(img_path).astype(np.float32)
    h, w = img.shape

    ps = min(patch_size, h // 3, w // 3)

    patches = {
        'top_left':     img[:ps, :ps],
        'top_right':    img[:ps, w - ps:],
        'bottom_left':  img[h - ps:, :ps],
        'bottom_right': img[h - ps:, w - ps:],
    }

    results = {}
    for name, patch in patches.items():
        lap = cv2.Laplacian(patch, cv2.CV_32F)
        sharpness = float(lap.var())
        denoised = cv2.GaussianBlur(patch, (5, 5), 0)
        noise = float(np.std(patch - denoised))
        results[name] = {'sharpness': sharpness, 'noise': noise}

    worst_sharpness = min(v['sharpness'] for v in results.values())
    worst_sharpness_mos = normalize_to_mos_scale(worst_sharpness, 'sharpness')
    worst_noise = max(v['noise'] for v in results.values())

    return {
        'patches': results,
        'worst_sharpness': worst_sharpness,
        'worst_sharpness_mos': worst_sharpness_mos,
        'worst_noise': worst_noise,
    }

    gray = _load_gray(img_path).astype(np.float32)
    return float(np.mean(gray) / 255.0)


def contrast_score(img_path: str) -> float:
    gray = _load_gray(img_path).astype(np.float32)
    return float(np.std(gray) / 128.0)
