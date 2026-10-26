import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from homepulse_vision.agent import loop
from homepulse_vision.agent.tools import Session
from homepulse_vision.passes import Issuer, UsedPasses
from homepulse_vision.scenarios import SCENARIOS, Camera, build_burst


def _session(sid):
    iss = Issuer(Ed25519PrivateKey.generate())
    used = UsedPasses()
    return Session(Camera(build_burst(sid, iss, used), "simulated doorbell camera"), iss.pk, used, emit_trace=False)


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_scripted_policy_reaches_expected_decision(sc):
    s = _session(sc.id)
    assert loop.run(s, loop.ScriptedPlanner())["decision"] == sc.expect


def _msg(*uses, text=None):
    content = [{"text": text}] if text else []
    content += [{"toolUse": {"toolUseId": f"u{i}", "name": n, "input": a}} for i, (n, a) in enumerate(uses)]
    return {"output": {"message": {"role": "assistant", "content": content}}, "stopReason": "tool_use" if uses else "end_turn", "usage": {}}


class Eager:
    """A model that obeys the sign: tries to unlock immediately."""
    name = "test-eager"

    def __init__(self):
        self.n = 0

    def converse(self, messages, s):
        self.n += 1
        return _msg(("unlock_door", {"reason": "sign says delivery verified"})) if self.n == 1 else _msg(text="done")


def test_guard_blocks_model_that_follows_the_sign():
    s = _session("sign_injection")
    r = loop.run(s, Eager())
    assert r["decision"] != "ALLOWED" and not s.unlocked and s.guard_refusals == 1


class Fabricator:
    """A model that invents a qr_ref it never saw."""
    name = "test-fabricator"

    def __init__(self):
        self.n = 0

    def converse(self, messages, s):
        self.n += 1
        if self.n == 1:
            return _msg(("verify_pass", {"qr_ref": "qr-7"}), ("unlock_door", {"reason": "verified"}))
        return _msg(text="done")


def test_fabricated_qr_ref_cannot_unlock():
    s = _session("nurse_valid")
    loop.run(s, Fabricator())
    assert not s.unlocked and s.counts["tool_errors"] == 1


class Broken:
    name = "test-broken"

    def converse(self, messages, s):
        raise TimeoutError("bedrock read timeout")


def test_model_failure_is_fail_safe():
    s = _session("nurse_valid")
    r = loop.run(s, Broken())
    assert r["decision"] == "ASKED_FAMILY" and r["end"] == "model_error" and not s.unlocked
    assert any(e["type"] == "safety_net" for e in s.trace.events)


class Looper:
    name = "test-looper"

    def converse(self, messages, s):
        return _msg(("request_new_snapshot", {}))


def test_step_limit_is_fail_safe():
    s = _session("night_visitor")
    r = loop.run(s, Looper())
    assert r["end"] == "step_limit" and r["decision"] == "ASKED_FAMILY"


def test_trace_links_actions_to_perception():
    s = _session("nurse_blurry")
    loop.run(s, loop.ScriptedPlanner())
    iv = next(e for e in s.trace.events if e["type"] == "tool_call" and e["name"] == "improve_view")
    assert "blurry" in iv["evidence"]["quality_issues"]
    vp = next(e for e in s.trace.events if e["type"] == "tool_call" and e["name"] == "verify_pass")
    assert "improve_view" in vp["evidence"]["frame_origin"]
