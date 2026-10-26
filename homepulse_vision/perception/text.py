"""Scene text: OpenCV DNN DB detector (PP-OCRv3) + CRNN recognizer, then an untrusted-text check.

Text read from the camera is data, never an instruction. This module only
reports what it read and whether it looks like an attempt to instruct the agent.
"""

from __future__ import annotations

import re
import threading

import cv2 as cv
import numpy as np

from ..config import model_path

_lock = threading.Lock()
_det = None
_rec = None
VOCAB = "0123456789abcdefghijklmnopqrstuvwxyz"
DET_SIZE = (736, 544)  # w, h; multiples of 32 near the 4:3 doorbell aspect
MAX_REGIONS = 14

# Phrases that address the agent or the lock. Matched on lowercase letters only,
# because the recognizer outputs a-z0-9 and merges words on a line.
STRONG = ["unlock", "opendoor", "openthedoor", "letmein", "letthemin", "disarm", "ignoreprevious",
          "ignoreall", "ignoreyour", "override", "grantaccess", "buzzmein", "deactivate", "allowentry",
          "openthegate", "doorcode", "unlatch"]
WEAK = ["system", "notice", "assistant", "alexa", "admin", "verified", "authorized", "authorised",
        "instruction", "command", "security", "frontdoor", "emergency", "police", "approved", "agent",
        "delivery", "ignore", "access"]


def _models():
    global _det, _rec
    if _det is None:
        det = cv.dnn.TextDetectionModel_DB(model_path("text_detection_en_ppocrv3_2023may.onnx"))
        det.setBinaryThreshold(0.3).setPolygonThreshold(0.5).setMaxCandidates(200).setUnclipRatio(2.0)
        det.setInputParams(1.0 / 255.0, DET_SIZE, (122.67891434, 116.66876762, 104.00698793), True)
        rec = cv.dnn.TextRecognitionModel(model_path("text_recognition_CRNN_EN_2021sep.onnx"))
        rec.setDecodeType("CTC-greedy")
        rec.setVocabulary(list(VOCAB))
        rec.setInputParams(1 / 127.5, (100, 32), (127.5, 127.5, 127.5))
        _det, _rec = det, rec
    return _det, _rec


def _order(pts: np.ndarray) -> np.ndarray:
    rect = cv.boxPoints(cv.minAreaRect(pts.astype(np.float32)))
    s, d = rect.sum(1), np.diff(rect, axis=1).ravel()
    return np.array([rect[np.argmin(s)], rect[np.argmin(d)], rect[np.argmax(s)], rect[np.argmax(d)]], np.float32)


def _crop(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    dst = np.float32([[0, 31], [0, 0], [99, 0]])
    M = cv.getAffineTransform(np.float32([bl, tl, tr]), dst)
    return cv.warpAffine(img, M, (100, 32))


def read_text(img: np.ndarray) -> list[dict]:
    with _lock:
        det, rec = _models()
        quads, confs = det.detect(img)
        regions = []
        for q, c in zip(quads, confs):
            q = np.asarray(q, np.float32)
            area = cv.contourArea(q)
            if area < 120:
                continue
            regions.append((area, q, float(c)))
        regions.sort(key=lambda t: -t[0])
        out = []
        for _, q, c in regions[:MAX_REGIONS]:
            oq = _order(q)
            word = rec.recognize(_crop(img, oq))
            if not word:
                continue
            x, y, w, h = cv.boundingRect(q.astype(np.int32))
            out.append({"text": word, "box": [int(x), int(y), int(w), int(h)], "det_conf": round(c, 3)})
    out.sort(key=lambda r: (r["box"][1] // 20, r["box"][0]))
    return out


def _fuzzy_find(hay: str, needle: str) -> bool:
    if needle in hay:
        return True
    n = len(needle)
    if n < 6 or len(hay) < n - 1:
        return False
    # one substitution, insertion or deletion anywhere in a window
    for L in (n - 1, n, n + 1):
        for i in range(0, max(len(hay) - L + 1, 0)):
            if _lev1(hay[i:i + L], needle):
                return True
    return False


def _lev1(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    i = j = diff = 0
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            diff += 1
            j += 1
            if diff > 1:
                return False
        else:
            i += 1
            j += 1
    return True


def assess(regions: list[dict]) -> dict:
    joined = re.sub(r"[^a-z]", "", "".join(r["text"] for r in regions).lower())
    strong = [k for k in STRONG if _fuzzy_find(joined, k)]
    weak = [k for k in WEAK if _fuzzy_find(joined, k)]
    return {
        "trust": "untrusted",
        "text_present": bool(regions),
        "read_text": " ".join(r["text"] for r in regions)[:300],
        "instruction_like": bool(strong) or len(weak) >= 2,
        "matched_phrases": strong + weak,
    }
