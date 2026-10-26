"""analyze_frame: the OpenCV 5 perception pipeline the agent calls."""

from __future__ import annotations

import time

import cv2 as cv
import numpy as np

from . import presence, qr, quality, text


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def analyze(img: np.ndarray, *, mask: bool = True, roi_crop: bool = False) -> dict:
    """Run every perception stage on one frame. Returns plain JSON-able data plus timings.
    roi_crop: the frame is an improve_view crop, so overlays are gone and QR decoding may try variants."""
    timing = {}
    t0 = time.perf_counter()
    t = time.perf_counter()
    overlays = quality.find_overlays(img) if mask else []
    work = quality.mask_overlays(img, overlays)
    timing["watermark_mask"] = _ms(t)

    t = time.perf_counter()
    q = quality.measure(img, overlays)
    timing["quality"] = _ms(t)

    t = time.perf_counter()
    qr_res = qr.decode_roi(work) if roi_crop else qr.decode(work)
    panels = qr.bright_panels(work)
    timing["qr"] = _ms(t)

    t = time.perf_counter()
    regions = text.read_text(work)
    txt = text.assess(regions)
    timing["text"] = _ms(t)

    t = time.perf_counter()
    faces = presence.detect_faces(work)
    timing["presence"] = _ms(t)
    timing["total"] = _ms(t0)

    roi = []
    for b in qr_res["located_undecoded"]:
        roi.append({"kind": "qr_located_not_decoded", "box": b})
    for b in panels:
        roi.append({"kind": "bright_panel", "box": b})
    if regions:
        xs = [r["box"] for r in regions]
        x0, y0 = min(b[0] for b in xs), min(b[1] for b in xs)
        x1, y1 = max(b[0] + b[2] for b in xs), max(b[1] + b[3] for b in xs)
        roi.append({"kind": "text_block", "box": [x0, y0, x1 - x0, y1 - y0]})
    return {
        "frame_size": [int(img.shape[1]), int(img.shape[0])],
        "quality": q,
        "overlays_masked": overlays,
        "qr": qr_res,
        "text_regions": regions,
        "text": txt,
        "faces": faces,
        "roi_candidates": roi,
        "timing_ms": timing,
        "_work": work,
    }


def draw(img: np.ndarray, a: dict) -> np.ndarray:
    """Overlay for the UI: faces (presence only), text boxes, QR outline, masked overlays, quality."""
    out = img.copy()
    for x, y, w, h in a["overlays_masked"]:
        cv.rectangle(out, (x, y), (x + w, y + h), (128, 128, 128), 1)
        cv.putText(out, "masked", (x + 2, y + h - 3), cv.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv.LINE_AA)
    for f in a["faces"]:
        x, y, w, h = f["box"]
        cv.rectangle(out, (x, y), (x + w, y + h), (80, 220, 80), 2)
        cv.putText(out, f"person {f['score']:.2f}", (x, max(y - 6, 12)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 1, cv.LINE_AA)
    colour = (40, 40, 230) if a["text"]["instruction_like"] else (0, 200, 255)
    for r in a["text_regions"]:
        x, y, w, h = r["box"]
        cv.rectangle(out, (x, y), (x + w, y + h), colour, 2)
        cv.putText(out, r["text"], (x, y + h + 14), cv.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv.LINE_AA)
    for c in a["qr"]["codes"]:
        p = np.array(c["points"], np.int32)
        cv.polylines(out, [p], True, (255, 140, 0), 3)
        cv.putText(out, "QR " + c["text"][:10] + "...", tuple(p[0] + [0, -8]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1, cv.LINE_AA)
    for r in a["roi_candidates"]:
        if r["kind"] == "text_block":
            continue
        x, y, w, h = r["box"]
        cv.rectangle(out, (x, y), (x + w, y + h), (255, 0, 255), 1, cv.LINE_4)
    q = a["quality"]
    label = f"blur {q['blur_effect']:.3f}  luma {q['mean_luma']:.0f}  backlight {q['backlight_score']:.2f}"
    label += "  " + (",".join(q["issues"]) if q["issues"] else "OK")
    cv.rectangle(out, (0, out.shape[0] - 26), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    cv.putText(out, label, (8, out.shape[0] - 8), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)
    return out
