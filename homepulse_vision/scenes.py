"""Procedural doorbell frames (synthetic, drawn with OpenCV).

Every frame here is generated, not photographed, and carries "SIMULATED" in its
camera overlay. Faces are drawn shapes: there is no real person in any frame.
The same renderer feeds the evaluation set and the live demo scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

W, H = 960, 720

SKIN = [(150, 175, 215), (120, 150, 195), (90, 120, 165), (70, 95, 135), (175, 195, 225)]
SCRUBS = [(140, 120, 40), (150, 140, 60), (120, 90, 140), (90, 130, 60), (160, 110, 70)]


@dataclass
class SceneSpec:
    visitor: bool = True
    qr_text: str | None = None          # token shown on a phone screen
    qr_on_paper: bool = False           # printed copy instead of phone
    qr_tilt_deg: float = 0.0            # out-of-plane tilt of the phone
    qr_scale: float = 1.0               # phone size multiplier
    sign_lines: list[str] = field(default_factory=list)
    badge: str | None = None
    motion_blur: int = 0                # kernel length in px (0 = none)
    gaussian_blur: float = 0.0          # sigma
    darkness: float = 0.0               # 0..1 (1 = almost black)
    backlight: float = 0.0              # 0..1
    watermark: bool = True
    noise: float = 3.0
    seed: int = 0
    clock: str = "2:14 PM"
    skin: int | None = None             # index into SKIN; None = random from seed


def _qr_image(text: str, border: int = 2) -> np.ndarray:
    p = cv.QRCodeEncoder_Params()
    p.correction_level = cv.QRCodeEncoder_CORRECT_LEVEL_M
    enc = cv.QRCodeEncoder.create(p)
    qr = enc.encode(text)
    return cv.copyMakeBorder(qr, border, border, border, border, cv.BORDER_CONSTANT, value=255)


def draw_face(img: np.ndarray, cx: int, cy: int, r: int, skin, hair=(30, 35, 45)) -> None:
    y0, y1 = max(cy - r - 2, 0), min(cy + r + 2, img.shape[0])
    x0, x1 = max(int(cx - r * 0.8) - 2, 0), min(int(cx + r * 0.8) + 2, img.shape[1])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    ax, ay = r * 0.78, r
    m = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2
    shade = np.clip(1.15 - 0.45 * m - 0.15 * (xx - cx) / ax, 0.5, 1.1)
    roi = img[y0:y1, x0:x1]
    mask = m <= 1
    for c in range(3):
        ch = roi[..., c]
        ch[mask] = np.clip(skin[c] * shade[mask], 0, 255)
    cv.ellipse(img, (cx, int(cy - r * 0.55)), (int(ax * 1.02), int(r * 0.5)), 0, 180, 360, hair, -1)
    for s in (-1, 1):
        ex, ey = int(cx + s * r * 0.33), int(cy - r * 0.1)
        cv.ellipse(img, (ex, int(ey - r * 0.16)), (int(r * 0.17), max(int(r * 0.04), 1)), 0, 180, 360, (40, 40, 50), -1)
        cv.ellipse(img, (ex, ey), (int(r * 0.14), max(int(r * 0.07), 1)), 0, 0, 360, (245, 245, 245), -1)
        cv.circle(img, (ex, ey), max(int(r * 0.065), 1), (60, 45, 30), -1)
        cv.circle(img, (ex, ey), max(int(r * 0.03), 1), (10, 10, 10), -1)
    nose = np.array([[cx, int(cy - r * 0.05)], [int(cx - r * 0.09), int(cy + r * 0.25)],
                     [int(cx + r * 0.09), int(cy + r * 0.25)]], np.int32)
    cv.fillPoly(img, [nose], tuple(int(c * 0.8) for c in skin))
    cv.ellipse(img, (cx, int(cy + r * 0.48)), (int(r * 0.25), max(int(r * 0.08), 1)), 0, 0, 180, (70, 70, 150), -1)


def _paste_quad(dst: np.ndarray, src: np.ndarray, quad: np.ndarray) -> None:
    h, w = src.shape[:2]
    M = cv.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), quad.astype(np.float32))
    warped = cv.warpPerspective(src, M, (dst.shape[1], dst.shape[0]), flags=cv.INTER_AREA)
    mask = cv.warpPerspective(np.full((h, w), 255, np.uint8), M, (dst.shape[1], dst.shape[0]))
    dst[mask > 127] = warped[mask > 127]


def _tilted_quad(cx: float, cy: float, w: float, h: float, tilt_deg: float) -> np.ndarray:
    """Rectangle rotated about its vertical axis by tilt_deg (perspective foreshortening)."""
    t = np.deg2rad(tilt_deg)
    half = w / 2 * np.cos(t)
    depth = 0.18 * np.sin(t)  # far edge shrinks
    return np.array([
        [cx - half, cy - h / 2 * (1 + depth)], [cx + half, cy - h / 2 * (1 - depth)],
        [cx + half, cy + h / 2 * (1 - depth)], [cx - half, cy + h / 2 * (1 + depth)],
    ])


def overlay_watermark(img: np.ndarray, clock: str) -> None:
    """Generic doorbell-app overlay in the positions a Ring frame uses (logo top-left, text top-right).
    Not Ring's logo: a plain white glyph and wordmark drawn for the simulation."""
    over = img.copy()
    cv.circle(over, (44, 40), 18, (255, 255, 255), 3)
    cv.circle(over, (44, 40), 6, (255, 255, 255), -1)
    cv.putText(over, "doorcam", (70, 52), cv.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv.LINE_AA)
    label = f"SIMULATED  Front Door  {clock}"
    (tw, _), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv.putText(over, label, (W - tw - 18, 36), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA)
    cv.addWeighted(over, 0.85, img, 0.15, 0, dst=img)


def render(spec: SceneSpec, meta: dict | None = None) -> np.ndarray:
    """Draw the frame. If `meta` is a dict it is filled with ground truth (face box, sign line boxes, skin index)."""
    meta = {} if meta is None else meta
    rng = np.random.default_rng(spec.seed)
    wall = np.array([130, 150, 170]) + rng.integers(-20, 20, 3)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = np.clip(wall, 0, 255)
    # siding texture
    for y in range(0, 600, 24):
        cv.line(img, (0, y), (W, y), tuple(int(c * 0.9) for c in np.clip(wall, 0, 255)), 1)
    cv.rectangle(img, (0, 600), (W, H), (110, 115, 120), -1)  # porch floor
    dx = int(rng.integers(560, 640))
    cv.rectangle(img, (dx, 90), (dx + 260, 600), (40, 60, 90), -1)
    cv.rectangle(img, (dx + 20, 110), (dx + 240, 600), (55, 85, 120), -1)
    cv.circle(img, (dx + 210, 370), 10, (90, 160, 170), -1)
    cv.rectangle(img, (dx - 70, 200), (dx - 40, 260), (60, 200, 240), -1)  # porch lamp
    bg = img.copy()

    if spec.visitor:
        vx = int(rng.integers(260, 400))
        si = int(rng.integers(len(SKIN)))
        si = si if spec.skin is None else spec.skin
        skin = SKIN[si]
        scrubs = SCRUBS[int(rng.integers(len(SCRUBS)))]
        cv.rectangle(img, (vx - 95, 250), (vx + 95, 560), scrubs, -1)
        cv.rectangle(img, (vx - 70, 560), (vx + 70, 690), (60, 50, 50), -1)
        fr = int(rng.integers(58, 72))
        draw_face(img, vx, 175, fr, skin)
        meta["skin"] = si
        meta["face_box"] = [int(vx - fr * 0.8), 175 - fr, int(fr * 1.6), 2 * fr]
        if spec.badge:
            cv.rectangle(img, (vx + 20, 300), (vx + 90, 335), (250, 250, 250), -1)
            cv.putText(img, spec.badge, (vx + 24, 323), cv.FONT_HERSHEY_SIMPLEX, 0.4, (120, 60, 20), 1, cv.LINE_AA)
    else:
        vx = 330

    if spec.sign_lines:
        sx0, sy0 = vx - 220, 260
        lh = 52
        sw = 470
        sh = 30 + lh * len(spec.sign_lines)
        cv.rectangle(img, (sx0, sy0), (sx0 + sw, sy0 + sh), (245, 248, 248), -1)
        cv.rectangle(img, (sx0, sy0), (sx0 + sw, sy0 + sh), (30, 30, 30), 3)
        meta["sign_lines"] = []
        for i, line in enumerate(spec.sign_lines):
            org = (sx0 + 22, sy0 + 55 + i * lh)
            cv.putText(img, line, org, cv.FONT_HERSHEY_DUPLEX, 1.05, (20, 20, 20), 2, cv.LINE_AA)
            (tw, th), base = cv.getTextSize(line, cv.FONT_HERSHEY_DUPLEX, 1.05, 2)
            meta["sign_lines"].append({"text": line, "box": [org[0], org[1] - th, tw, th + base]})

    if spec.qr_text:
        qr = cv.cvtColor(_qr_image(spec.qr_text), cv.COLOR_GRAY2BGR)
        s = spec.qr_scale
        if spec.qr_on_paper:
            pw, ph = 170 * s, 220 * s
            paper = np.full((440, 340, 3), 238, np.uint8)
            q = cv.resize(qr, (300, 300), interpolation=cv.INTER_NEAREST)
            paper[30:330, 20:320] = q
            cv.putText(paper, "VISITOR PASS", (60, 395), cv.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv.LINE_AA)
            _paste_quad(img, paper, _tilted_quad(vx + 150, 420, pw, ph, spec.qr_tilt_deg))
        else:
            pw, ph = 150 * s, 270 * s
            phone = np.full((540, 300, 3), (25, 25, 25), np.uint8)
            screen = np.full((470, 270, 3), 252, np.uint8)
            q = cv.resize(qr, (250, 250), interpolation=cv.INTER_NEAREST)
            screen[70:320, 10:260] = q
            cv.putText(screen, "HomePulse pass", (30, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (90, 60, 20), 2, cv.LINE_AA)
            phone[35:505, 15:285] = screen
            _paste_quad(img, phone, _tilted_quad(vx + 150, 400, pw, ph, spec.qr_tilt_deg))
            # a hand holding it
            cv.ellipse(img, (int(vx + 150), int(400 + ph / 2)), (int(pw * 0.42), 24), 0, 0, 360,
                       SKIN[int(rng.integers(len(SKIN)))], -1)

    fg_mask = (np.abs(img.astype(np.int16) - bg.astype(np.int16)).sum(axis=2) > 0).astype(np.float32)
    fg_mask = cv.dilate(fg_mask, np.ones((3, 3), np.uint8))
    img = _degrade(img, bg, fg_mask, spec, rng)
    if spec.watermark:
        overlay_watermark(img, spec.clock)
    return img


def _degrade(img: np.ndarray, bg: np.ndarray, fg_mask: np.ndarray, spec: SceneSpec,
             rng: np.random.Generator) -> np.ndarray:
    f = img.astype(np.float32)
    if spec.backlight > 0:
        # bright sky/glass behind the visitor; auto-exposure meters on it and the subject turns into a silhouette
        glow = np.zeros((H, W), np.float32)
        cv.rectangle(glow, (0, 0), (W, 240), 1.0, -1)
        cv.rectangle(glow, (W - 300, 0), (W, 600), 1.0, -1)
        glow = cv.GaussianBlur(glow, (0, 0), 25)
        gain = 1.0 - 0.8 * spec.backlight
        lit_bg = bg.astype(np.float32) * gain + glow[..., None] * 255 * spec.backlight * 1.2
        m = fg_mask[..., None]
        f = m * (f * gain) + (1 - m) * lit_bg
    if spec.darkness > 0:
        gain = 1.0 - 0.93 * spec.darkness
        f = f * gain
    if spec.motion_blur > 1:
        k = np.zeros((spec.motion_blur, spec.motion_blur), np.float32)
        k[spec.motion_blur // 2, :] = 1.0 / spec.motion_blur
        f = cv.filter2D(f, -1, k)
    if spec.gaussian_blur > 0:
        f = cv.GaussianBlur(f, (0, 0), spec.gaussian_blur)
    noise = spec.noise * (1 + 2 * spec.darkness)
    if noise > 0:
        f = f + rng.normal(0, noise, f.shape).astype(np.float32)
    out = np.clip(f, 0, 255).astype(np.uint8)
    # JPEG round-trip like a real camera stream
    ok, buf = cv.imencode(".jpg", out, [cv.IMWRITE_JPEG_QUALITY, 85])
    return cv.imdecode(buf, cv.IMREAD_COLOR)
