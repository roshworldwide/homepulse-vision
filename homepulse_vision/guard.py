"""The unlock guard: a deterministic check, never the model's opinion.

unlock_door succeeds only if, in this session,
  (a) a QR was decoded by analyze_frame, verify_pass returned valid for that QR
      within the last VERIFY_FRESH_S seconds, and its pass id is consumed now
      (one-time use, atomic), or
  (b) a family member approved this session through a signed approval.
The model's arguments to unlock_door are recorded but never consulted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VERIFY_FRESH_S = 120


@dataclass
class GuardResult:
    allowed: bool
    basis: str | None  # "verified_pass" | "family_approval" | None
    checks: dict = field(default_factory=dict)
    refused_reason: str | None = None


def check_unlock(session, now: int) -> GuardResult:
    checks: dict = {}
    # (b) family approval
    appr = session.family_approval
    checks["family_approval"] = bool(appr and appr.get("approved") and appr.get("session") == session.id)
    if checks["family_approval"]:
        return GuardResult(True, "family_approval", checks)
    # (a) verified pass seen in this session
    candidates = []
    for ref, v in session.verifications.items():
        seen = ref in session.decoded_qrs
        fresh = now - v["at"] <= VERIFY_FRESH_S
        candidates.append({"qr_ref": ref, "seen_in_session": seen, "verified_valid": v["result"]["valid"], "fresh": fresh})
        if seen and v["result"]["valid"] and fresh:
            pid = v["pid"]
            if session.used.consume(pid, now):
                checks["pass"] = candidates
                return GuardResult(True, "verified_pass", checks)
            candidates[-1]["consumed_already"] = True
    checks["pass"] = candidates
    reason = "no valid pass verified in this session and no family approval"
    return GuardResult(False, None, checks, reason)
