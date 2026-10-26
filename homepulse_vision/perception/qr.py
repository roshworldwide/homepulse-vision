"""QR decoding with OpenCV's two detectors, plus candidate regions for active perception."""

from __future__ import annotations

import cv2 as cv
import numpy as np

_aruco = cv.QRCodeDetectorAruco()
_std = cv.QRCodeDetector()


def _box(pts) -> list[int]:
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    x, y = p.min(0)
    x1, y1 = p.max(0)
    return [int(x), int(y), int(x1 - x), int(y1 - y)]


def decode_roi(img: np.ndarray) -> dict:
    """Decode on an improve_view crop: a bounded set of re-sharpen / binarise variants (at most 6 decodes)."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img
    res = decode(gray)
    if res["codes"]:
        res["variant"] = "as_is"
        return res
    blur = cv.GaussianBlur(gray, (0, 0), 2.0)
    variants = [
        ("sharpen_x2", cv.addWeighted(gray, 3.0, blur, -2.0, 0)),
        ("sharpen_x3", cv.addWeighted(gray, 4.0, blur, -3.0, 0)),
        ("adaptive_bin", cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 2)),
        ("downscale_0.6", cv.resize(gray, None, fx=0.6, fy=0.6, interpolation=cv.INTER_AREA)),
        ("otsu", cv.threshold(cv.GaussianBlur(gray, (3, 3), 0), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]),
    ]
    for name, v in variants:
        r = decode(v)
        if r["codes"]:
            if name.startswith("downscale"):
                for c in r["codes"]:
                    c["points"] = (np.asarray(c["points"]) / 0.6).round().astype(int).tolist()
            r["variant"] = name
            return r
    res["variant"] = None
    return res


def decode(img: np.ndarray) -> dict:
    """Return decoded codes and boxes where a QR was located but not decoded."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img
    codes: list[dict] = []
    located: list[list[int]] = []
    try:
        ok, texts, pts, _ = _aruco.detectAndDecodeMulti(gray)
    except cv.error:
        ok, texts, pts = False, (), None
    if ok and pts is not None:
        for t, p in zip(texts, pts):
            if t:
                codes.append({"text": t, "points": np.asarray(p).reshape(-1, 2).round().astype(int).tolist(),
                              "detector": "QRCodeDetectorAruco"})
            else:
                located.append(_box(p))
    if not codes:
        try:
            t, p, _ = _std.detectAndDecode(gray)
        except cv.error:
            t, p = "", None
        if t and p is not None:
            codes.append({"text": t, "points": np.asarray(p).reshape(-1, 2).round().astype(int).tolist(), "detector": "QRCodeDetector"})
        elif p is not None:
            located.append(_box(p))
    return {"codes": codes, "located_undecoded": located}


def bright_panels(img: np.ndarray, max_n: int = 3) -> list[list[int]]:
    """Phone screens and paper are bright, rectangular and small: likely places for a pass QR."""
    h, w = img.shape[:2]
    gray = cv.GaussianBlur(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (5, 5), 0)
    thr = max(float(np.percentile(gray, 90)), float(gray.mean() + 1.2 * gray.std()))
    _, bw = cv.threshold(gray, min(thr, 250), 255, cv.THRESH_BINARY)
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (15, 15)))
    cnts, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in cnts:
        a = cv.contourArea(c)
        if not (0.006 * w * h <= a <= 0.25 * w * h):
            continue
        (cx, cy), (rw, rh), _ = cv.minAreaRect(c)
        if rw * rh <= 0 or a / (rw * rh) < 0.7:
            continue
        x, y, bw_, bh_ = cv.boundingRect(c)
        if y < 0.12 * h and x > 0.5 * w:  # overlay corner
            continue
        ar = max(bw_, bh_) / max(min(bw_, bh_), 1)
        if ar > 3.5:
            continue
        cands.append((a, [int(x), int(y), int(bw_), int(bh_)]))
    cands.sort(key=lambda t: -t[0])
    return [b for _, b in cands[:max_n]]
