"""Presence only: is a face visible? YuNet boxes and scores; landmarks are discarded.

No recognition, no embeddings, nothing stored. The answer is "someone is there", never "who".
"""

from __future__ import annotations

import threading

import cv2 as cv
import numpy as np

from ..config import model_path

_lock = threading.Lock()
_fd = None
SCORE = 0.6
MAX_SIDE = 640


def detect_faces(img: np.ndarray) -> list[dict]:
    global _fd
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    small = cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA) if s < 1 else img
    with _lock:
        if _fd is None:
            _fd = cv.FaceDetectorYN.create(model_path("face_detection_yunet_2023mar.onnx"), "", (320, 320), SCORE, 0.3, 50)
        _fd.setInputSize((small.shape[1], small.shape[0]))
        _, faces = _fd.detect(small)
    out = []
    for f in faces if faces is not None else []:
        x, y, bw, bh = (np.asarray(f[:4]) / s).round().astype(int).tolist()
        out.append({"box": [x, y, bw, bh], "score": round(float(f[-1]), 3)})  # landmarks f[4:14] dropped
    return out
