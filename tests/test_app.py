import base64

import cv2 as cv
import numpy as np
from fastapi.testclient import TestClient

from homepulse_vision import app as app_mod
from homepulse_vision.ratelimit import RateLimiter

client = TestClient(app_mod.app)


def test_health_and_scenarios():
    h = client.get("/api/health").json()
    assert h["opencv"].startswith("5.") and h["camera"] == "simulated"
    assert len(client.get("/api/scenarios").json()) >= 10
    assert client.get("/api/scenarios/nurse_valid/thumb.jpg").headers["content-type"] == "image/jpeg"


def test_run_scenario_and_headers():
    app_mod.LIMITER = RateLimiter()
    r = client.post("/api/run", json={"scenario": "forged_pass"})
    assert r.status_code == 200 and r.json()["decision"] == "BLOCKED"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]


def test_upload_validation():
    app_mod.LIMITER = RateLimiter()
    assert client.post("/api/run", json={"image_b64": "@@@"}).status_code == 400
    assert client.post("/api/run", json={"image_b64": base64.b64encode(b"GIF89a....").decode()}).status_code == 400
    big = cv.imencode(".png", np.zeros((5000, 64, 3), np.uint8))[1].tobytes()
    r = client.post("/api/run", json={"image_b64": base64.b64encode(big).decode()})
    assert r.status_code == 400 and "4096" in r.json()["error"]
    ok = cv.imencode(".jpg", np.full((240, 320, 3), 120, np.uint8))[1].tobytes()
    r = client.post("/api/run", json={"image_b64": "data:image/jpeg;base64," + base64.b64encode(ok).decode()})
    assert r.status_code == 200 and r.json()["simulated"]["camera"] is False


def test_rate_limit_per_ip():
    lim = RateLimiter()
    lim.rules_ip = [(60, 2)]
    app_mod.LIMITER = lim
    codes = [client.post("/api/run", json={"scenario": "nope"}).status_code for _ in range(3)]
    assert codes[-1] == 429


def test_global_cap():
    lim = RateLimiter()
    lim.rules_global = [(60, 2)]
    assert lim.check("a")[0] and lim.check("b")[0]
    ok, wait, scope = lim.check("c")
    assert not ok and scope == "global" and wait > 0


def test_family_token_tamper_rejected():
    app_mod.LIMITER = RateLimiter()
    r = client.post("/api/family", json={"approval_token": "abc.def", "approve": True})
    assert r.status_code == 403
