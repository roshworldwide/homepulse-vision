"""FastAPI app: web UI + JSON API. Runs locally with uvicorn and on AWS Lambda via Mangum."""

from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(4096 * 4096))  # decompression-bomb cap, read at first decode

import base64
import binascii
import hashlib
import hmac
import json
import struct
import time
from pathlib import Path

import cv2 as cv
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from .agent import loop
from .agent.tools import Session
from .config import MAX_IMAGE_SIDE, MAX_UPLOAD_BYTES, MODEL_ID, WORK_MAX_SIDE
from .passes import Issuer, UsedPasses
from .ratelimit import RateLimiter
from .scenarios import BY_ID, SCENARIOS, Camera, build_burst, thumbnail

STATIC = Path(__file__).parent / "static"
ISSUER = Issuer()
USED = UsedPasses()
LIMITER = RateLimiter()
_APPROVAL_KEY = hashlib.sha256(b"hpv-approval|" + ISSUER.sk.private_bytes_raw()).digest()
APPROVAL_TTL_S = 15 * 60
_thumbs: dict[str, bytes] = {}
_planner = None

app = FastAPI(title="HomePulse Vision", docs_url=None, redoc_url=None, openapi_url=None)


def planner():
    global _planner
    if _planner is None:
        _planner = loop.ScriptedPlanner() if os.environ.get("HPV_PLANNER") == "scripted" else loop.BedrockPlanner()
    return _planner


class RunRequest(BaseModel):
    scenario: str | None = None
    image_b64: str | None = None


class FamilyReply(BaseModel):
    approval_token: str
    approve: bool


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _limited(request: Request) -> JSONResponse | None:
    ok, wait, scope = LIMITER.check(_client_ip(request))
    if ok:
        return None
    return JSONResponse({"error": "rate_limited", "scope": scope, "retry_after_s": wait}, status_code=429,
                        headers={"Retry-After": str(wait)})


@app.middleware("http")
async def security_headers(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES * 4 // 3 + 4096:
        return JSONResponse({"error": "request_too_large", "max_image_bytes": MAX_UPLOAD_BYTES}, status_code=413)
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = ("default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                                               "script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
    return resp


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/static/{name}")
def static(name: str):
    p = (STATIC / name).resolve()
    if p.parent != STATIC.resolve() or not p.exists():
        return JSONResponse({"error": "not_found"}, status_code=404)
    return FileResponse(p)


@app.get("/api/health")
def health():
    return {"ok": True, "opencv": cv.__version__, "model": MODEL_ID, "planner": getattr(planner(), "name", "?"),
            "issuer_key": "ephemeral" if ISSUER.ephemeral else "configured", "camera": "simulated", "lock": "simulated"}


@app.get("/api/scenarios")
def scenarios():
    return [{"id": s.id, "title": s.title, "blurb": s.blurb, "expect": s.expect} for s in SCENARIOS]


@app.get("/api/scenarios/{sid}/thumb.jpg")
def scenario_thumb(sid: str):
    if sid not in BY_ID:
        return JSONResponse({"error": "unknown scenario"}, status_code=404)
    if sid not in _thumbs:
        _thumbs[sid] = thumbnail(sid, ISSUER)
    return Response(_thumbs[sid], media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})


def _dims(raw: bytes) -> tuple[int, int] | None:
    """Width/height from the file header, before any pixel is decoded."""
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24:
        return struct.unpack(">II", raw[16:24])
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP" and len(raw) >= 30:
        kind = raw[12:16]
        if kind == b"VP8X":
            return 1 + int.from_bytes(raw[24:27], "little"), 1 + int.from_bytes(raw[27:30], "little")
        if kind == b"VP8 ":
            w, h = struct.unpack("<HH", raw[26:30])
            return w & 0x3FFF, h & 0x3FFF
        if kind == b"VP8L":
            b = int.from_bytes(raw[21:25], "little")
            return (b & 0x3FFF) + 1, ((b >> 14) & 0x3FFF) + 1
        return None
    if raw[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(raw):
            if raw[i] != 0xFF:
                i += 1
                continue
            m = raw[i + 1]
            if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
                i += 2
                continue
            seg = struct.unpack(">H", raw[i + 2:i + 4])[0]
            if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", raw[i + 5:i + 9])
                return w, h
            i += 2 + seg
    return None


def _decode_upload(data: str) -> tuple[np.ndarray | None, str | None]:
    if "," in data[:100] and data.startswith("data:"):
        data = data.split(",", 1)[1]
    if len(data) > MAX_UPLOAD_BYTES * 4 // 3 + 8:
        return None, f"image larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None, "image is not valid base64"
    if len(raw) > MAX_UPLOAD_BYTES:
        return None, f"image larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
    if not (raw[:3] == b"\xff\xd8\xff" or raw[:8] == b"\x89PNG\r\n\x1a\n" or raw[8:12] == b"WEBP"):
        return None, "only JPEG, PNG or WebP images are accepted"
    dims = _dims(raw)
    if dims is None:
        return None, "could not read image dimensions"
    if max(dims) > MAX_IMAGE_SIDE:
        return None, f"image larger than {MAX_IMAGE_SIDE} px on a side"
    arr = np.frombuffer(raw, np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    del raw, arr
    if img is None or min(img.shape[:2]) < 64:
        return None, "image too small"
    s = WORK_MAX_SIDE / max(img.shape[:2])
    if s < 1:
        img = cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
    return img, None


def _approval_token(session_id: str, request_id: str) -> str:
    body = json.dumps({"s": session_id, "r": request_id, "exp": int(time.time()) + APPROVAL_TTL_S}, separators=(",", ":")).encode()
    sig = hmac.new(_APPROVAL_KEY, body, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(body).decode().rstrip("=") + "." + base64.urlsafe_b64encode(sig).decode().rstrip("=")


def _check_approval(token: str) -> dict | None:
    try:
        b, s = token.split(".")
        body = base64.urlsafe_b64decode(b + "=" * (-len(b) % 4))
        sig = base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
    except Exception:
        return None
    if not hmac.compare_digest(sig, hmac.new(_APPROVAL_KEY, body, hashlib.sha256).digest()):
        return None
    d = json.loads(body)
    return d if d.get("exp", 0) >= time.time() else None


@app.post("/api/run")
def run(req: RunRequest, request: Request):
    if (lim := _limited(request)) is not None:
        return lim
    if req.image_b64:
        img, err = _decode_upload(req.image_b64)
        if err:
            return JSONResponse({"error": err}, status_code=400)
        camera, source = Camera([img], "uploaded image (not stored)"), "upload"
    elif req.scenario in BY_ID:
        camera, source = Camera(build_burst(req.scenario, ISSUER, USED), "simulated doorbell camera"), "scenario:" + req.scenario
    else:
        return JSONResponse({"error": "pick a scenario or upload an image"}, status_code=400)
    t0 = time.perf_counter()
    s = Session(camera, ISSUER.pk, USED, source=source)
    result = loop.run(s, planner())
    approval = None
    if result["decision"] == "ASKED_FAMILY" and s.family_requests:
        approval = {"token": _approval_token(s.id, s.family_requests[-1]["request_id"]),
                    "request": {k: s.family_requests[-1][k] for k in ("request_id", "reason", "message")}}
    return {
        **result,
        "expected": BY_ID[req.scenario].expect if req.scenario in BY_ID else None,
        "trace": s.trace.events,
        "views": loop.views_json(s),
        "counts": s.counts,
        "stage_ms": s.stage_ms,
        "wall_ms": round((time.perf_counter() - t0) * 1000, 1),
        "family_approval": approval,
        "simulated": {"camera": source != "upload", "lock": True, "family_phone": True},
    }


@app.post("/api/family")
def family(reply: FamilyReply, request: Request):
    if (lim := _limited(request)) is not None:
        return lim
    d = _check_approval(reply.approval_token)
    if not d:
        return JSONResponse({"error": "approval token invalid or expired"}, status_code=403)
    s = Session(Camera([np.zeros((8, 8, 3), np.uint8)], "none"), ISSUER.pk, USED, source="family_reply")
    s.id = d["s"]
    s.trace.session_id = d["s"]
    s.trace.add("family_reply", request_id=d["r"], approve=reply.approve, channel="simulated family phone")
    if reply.approve:
        s.family_approval = {"approved": True, "session": d["s"], "request_id": d["r"]}
        from .agent.tools import unlock_door

        out = unlock_door(s, {"reason": "family approved"})
        decision = "ALLOWED" if out.get("unlocked") else "BLOCKED"
    else:
        decision = "BLOCKED"
    s.trace.add("decision", decision=decision, reason="family " + ("approved" if reply.approve else "declined"),
                unlocked=s.unlocked, basis=s.unlock_basis)
    return {"decision": decision, "trace": s.trace.events}
