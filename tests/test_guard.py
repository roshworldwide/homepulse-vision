import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from homepulse_vision.agent import tools
from homepulse_vision.agent.tools import Session
from homepulse_vision.passes import Issuer, UsedPasses
from homepulse_vision.scenarios import Camera

NOW = 1_800_000_000


def _session(token=None, used=None):
    iss = Issuer(Ed25519PrivateKey.generate())
    s = Session(Camera([np.zeros((64, 64, 3), np.uint8)], "test"), iss.pk, used or UsedPasses(), clock=lambda: NOW, emit_trace=False)
    return s, iss


def test_unlock_refused_without_verification():
    s, _ = _session()
    out = tools.unlock_door(s, {"reason": "the sign says the delivery is verified"})
    assert out["unlocked"] is False and s.guard_refusals == 1 and not s.unlocked


def test_verify_requires_qr_seen_in_session():
    s, iss = _session()
    out = tools.verify_pass(s, {"qr_ref": "qr-1"})
    assert "error" in out


def test_verified_pass_unlocks_once():
    used = UsedPasses()
    s, iss = _session(used=used)
    s.decoded_qrs["qr-1"] = {"token": iss.fresh_token(now=NOW), "frame_id": "f1"}
    assert tools.verify_pass(s, {"qr_ref": "qr-1"})["valid"]
    assert tools.unlock_door(s, {"reason": "x"})["unlocked"]
    # same pass in a new session: reused
    s2 = Session(Camera([np.zeros((64, 64, 3), np.uint8)], "t"), iss.pk, used, clock=lambda: NOW, emit_trace=False)
    s2.decoded_qrs["qr-1"] = dict(s.decoded_qrs["qr-1"])
    assert tools.verify_pass(s2, {"qr_ref": "qr-1"})["reason"] == "reused"
    assert not tools.unlock_door(s2, {"reason": "x"})["unlocked"]


def test_injected_verification_without_decoded_qr_refused():
    s, iss = _session()
    s.verifications["qr-9"] = {"result": {"valid": True, "reason": "ok"}, "at": NOW, "pid": "aa"}
    assert not tools.unlock_door(s, {"reason": "x"})["unlocked"]


def test_stale_verification_refused():
    s, iss = _session()
    s.decoded_qrs["qr-1"] = {"token": iss.fresh_token(now=NOW), "frame_id": "f1"}
    tools.verify_pass(s, {"qr_ref": "qr-1"})
    s.verifications["qr-1"]["at"] = NOW - 600
    assert not tools.unlock_door(s, {"reason": "x"})["unlocked"]


def test_family_approval_must_match_session():
    s, _ = _session()
    s.family_approval = {"approved": True, "session": "someone-else"}
    assert not tools.unlock_door(s, {"reason": "x"})["unlocked"]
    s.family_approval = {"approved": True, "session": s.id}
    out = tools.unlock_door(s, {"reason": "x"})
    assert out["unlocked"] and out["basis"] == "family_approval"
