"""Frame quality gate and camera-overlay masking.

Measures focus (variance of the Laplacian, plus a re-blur ratio that does not
depend on how much detail the scene has), exposure (luma histogram) and
backlight, and finds the doorbell app's overlays (top-left logo, top-right
text, bottom caption band) so they can be masked before analysis.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

# Thresholds set on calibration frames (seeds 3 and 7, not in the eval set); see docs/report.md
BLUR_EFFECT_MAX = 0.195     # re-blur ratio above this = blurry (QR decoding fails from ~0.198)
DARK_MEAN_MAX = 70.0        # mean luma below this = dark
DARK_P90_MAX = 110.0        # ...or 90th percentile below this
BACKLIGHT_MIN = 0.45        # backlight score above this = backlit


def find_overlays(img: np.ndarray) -> list[list[int]]:
    """Boxes [x, y, w, h] of burned-in camera overlays: logo top-left, text top-right, caption band."""
    h, w = img.shape[:2]
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    boxes: list[list[int]] = []
    for x0, y0, x1, y1 in ((0, 0, int(0.30 * w), int(0.13 * h)), (int(0.50 * w), 0, w, int(0.10 * h))):
        roi = hsv[y0:y1, x0:x1]
        white = cv.inRange(roi, (0, 0, 200), (180, 60, 255))
        white = cv.morphologyEx(white, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (25, 9)))
        n, _, stats, _ = cv.connectedComponentsWithStats(white)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area >= 60 and bh <= 0.12 * h:
                boxes.append([int(x0 + x - 4), int(y0 + y - 4), int(bw + 8), int(bh + 8)])
    # caption band: near-black full-width rows at the bottom, clearly darker than the scene above
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dark_rows = (gray < 30).mean(axis=1) > 0.6
    band, gap = 0, 0
    for r in range(h - 1, int(0.85 * h), -1):
        if dark_rows[r]:
            band, gap = h - r, 0
        else:
            gap += 1
            if gap > 14:
                break
    if band >= 12:
        above = float(gray[max(h - band - 60, 0):h - band - 4].mean())
        inside = float(np.median(gray[h - band:, :]))
        if above - inside > 25:
            boxes.append([0, h - band - 2, w, band + 2])
    return boxes


def mask_overlays(img: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    """Remove overlays with OpenCV inpainting so downstream detectors never read them."""
    if not boxes:
        return img
    m = np.zeros(img.shape[:2], np.uint8)
    for x, y, bw, bh in boxes:
        cv.rectangle(m, (max(x, 0), max(y, 0)), (x + bw, y + bh), 255, -1)
    small = 0.5
    sm = cv.resize(img, None, fx=small, fy=small, interpolation=cv.INTER_AREA)
    mm = cv.resize(m, (sm.shape[1], sm.shape[0]), interpolation=cv.INTER_NEAREST)
    filled = cv.inpaint(sm, mm, 3, cv.INPAINT_TELEA)
    filled = cv.resize(filled, (img.shape[1], img.shape[0]), interpolation=cv.INTER_LINEAR)
    out = img.copy()
    out[m > 0] = filled[m > 0]
    return out


def blur_effect(gray: np.ndarray, valid: np.ndarray | None = None, k: int = 11) -> float:
    """Crete et al. (2007) no-reference blur: how little extra blurring changes the image. 0 sharp, 1 blurred."""
    g = gray.astype(np.float32)
    worst = 0.0
    for ksize, axis in (((1, k), 0), ((k, 1), 1)):
        d_org = np.abs(np.diff(g, axis=axis))
        d_blur = np.abs(np.diff(cv.blur(g, ksize), axis=axis))
        gain = np.maximum(0, d_org - d_blur)
        if valid is not None:
            v = valid[1:, :] if axis == 0 else valid[:, 1:]
            d_org, gain = d_org[v], gain[v]
        m1 = float(d_org.sum())
        worst = max(worst, abs(m1 - float(gain.sum())) / max(m1, 1e-6))
    return worst


def measure(img: np.ndarray, overlay_boxes: list[list[int]] | None = None) -> dict:
    h, w = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    valid = np.ones_like(gray, bool)
    for x, y, bw, bh in overlay_boxes or []:
        valid[max(y, 0):y + bh, max(x, 0):x + bw] = False
    scale = 640.0 / w
    g = cv.resize(gray, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    v = cv.resize(valid.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv.INTER_NEAREST) > 0
    lap = cv.Laplacian(g, cv.CV_64F, ksize=3)
    v_in = cv.erode(v.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    # sharpness measured on the edge-carrying pixels: flat walls say nothing about focus
    edges = cv.dilate(cv.Canny(g, 30, 90), np.ones((3, 3), np.uint8)) > 0
    sel = v_in & edges
    lap_var = float(lap[v_in].var()) if v_in.any() else 0.0
    edge_lap_var = float(lap[sel].var()) if sel.sum() > 200 else lap_var
    # normalise by scene contrast so darkness alone does not read as blur
    sharp = edge_lap_var / max(float(g[v_in].var()), 25.0)
    # saturated glow (backlight) is smooth by nature and would read as blur
    unsat = valid & (cv.dilate((gray > 235).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0)
    be = blur_effect(gray, unsat if unsat.mean() > 0.3 else valid)
    px = gray[valid]
    hist = cv.calcHist([gray], [0], valid.astype(np.uint8) * 255, [256], [0, 256]).ravel()
    hist /= max(hist.sum(), 1)
    cdf = np.cumsum(hist)
    p10, p50, p90 = (int(np.searchsorted(cdf, q)) for q in (0.10, 0.50, 0.90))
    mean = float(px.mean())
    bright_frac = float(hist[230:].sum())
    dark_frac = float(hist[:50].sum())
    # backlight: bright border/top versus a much darker centre where the visitor stands
    cy0, cy1, cx0, cx1 = int(0.30 * h), int(0.95 * h), int(0.15 * w), int(0.70 * w)
    centre = float(gray[cy0:cy1, cx0:cx1].mean())
    top = float(gray[int(0.12 * h):int(0.30 * h), :].mean())
    contrast_ratio = (top + 1) / (centre + 1)
    backlight = float(np.clip((contrast_ratio - 1.3) / 1.7, 0, 1) * 0.6 + np.clip(bright_frac / 0.25, 0, 1) * 0.4)
    if bright_frac < 0.05:
        backlight *= 0.5
    issues = []
    if be > BLUR_EFFECT_MAX:
        issues.append("blurry")
    if mean < DARK_MEAN_MAX and p90 < DARK_P90_MAX:
        issues.append("dark")
    if backlight > BACKLIGHT_MIN:
        issues.append("backlit")
    overlay_area = 0
    if overlay_boxes:
        om = np.zeros_like(gray, np.uint8)
        for x, y, bw, bh in overlay_boxes:
            om[max(y, 0):y + bh, max(x, 0):x + bw] = 1
        overlay_area = int(om.sum())
    return {
        "blur_effect": round(be, 3),
        "sharpness": round(sharp, 2),
        "laplacian_var_edges": round(edge_lap_var, 1),
        "lapvar_full": round(lap_var, 1),
        "mean_luma": round(mean, 1),
        "p10": p10, "p50": p50, "p90": p90,
        "dark_fraction": round(dark_frac, 3),
        "bright_fraction": round(bright_frac, 3),
        "backlight_score": round(backlight, 3),
        "watermark_fraction": round(overlay_area / (h * w), 4),
        "issues": issues,
        "trustworthy": not issues,
    }
