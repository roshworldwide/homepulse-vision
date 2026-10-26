"""improve_view operations: the agent's tools for making a bad frame readable."""

from __future__ import annotations

import cv2 as cv
import numpy as np

OPS = ("clahe", "gamma", "sharpen", "crop_upscale")


def clahe(img: np.ndarray, clip: float = 3.0) -> np.ndarray:
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    l = cv.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
    return cv.cvtColor(cv.merge((l, a, b)), cv.COLOR_LAB2BGR)


def gamma(img: np.ndarray, target_median: float = 115.0) -> tuple[np.ndarray, float]:
    """Brighten with a gamma chosen so the median luma lands near target_median."""
    med = float(np.median(cv.cvtColor(img, cv.COLOR_BGR2GRAY)))
    med = min(max(med, 2.0), 250.0)
    g = float(np.clip(np.log(target_median / 255.0) / np.log(med / 255.0), 0.2, 1.0))
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
    out = cv.LUT(img, lut)
    if g < 0.6:  # lifting deep shadows amplifies sensor noise
        out = cv.bilateralFilter(out, 5, 30, 5)
    return out, round(g, 3)


def sharpen(img: np.ndarray, amount: float = 1.2, sigma: float = 2.0) -> np.ndarray:
    blur = cv.GaussianBlur(img, (0, 0), sigma)
    return cv.addWeighted(img, 1 + amount, blur, -amount, 0)


def crop_upscale(img: np.ndarray, box: list[int], target: int = 720) -> tuple[np.ndarray, list[int], float]:
    """Crop the region of interest (with margin) and upscale it so small QR modules span more pixels."""
    h, w = img.shape[:2]
    x, y, bw, bh = box
    m = int(0.18 * max(bw, bh))
    x0, y0 = max(x - m, 0), max(y - m, 0)
    x1, y1 = min(x + bw + m, w), min(y + bh + m, h)
    crop = img[y0:y1, x0:x1]
    s = float(np.clip(target / max(crop.shape[:2]), 1.0, 4.0))
    up = cv.resize(crop, None, fx=s, fy=s, interpolation=cv.INTER_CUBIC)
    return sharpen(up, 0.8, 1.5), [x0, y0, x1 - x0, y1 - y0], round(s, 2)
