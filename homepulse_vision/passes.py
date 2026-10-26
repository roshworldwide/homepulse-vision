"""Signed one-time visitor passes (the "HP1" QR format).

A pass is issued by the family (HomePulse) to a visitor's phone. The phone app
re-fetches a freshly signed display token every 60 s, so the QR on screen
always carries a recent `iat`. A printed copy or screenshot goes stale after
`DISPLAY_TTL_S` seconds, an expired visit window fails, a forged signature
fails, and a pass id can open the door once.

Wire format (all integers big-endian):

    payload = version:u8 | pid:8 bytes | door:u8 | nbf:u32 | exp:u32 | iat:u32   (22 bytes)
    token   = "HP1:" + base32(payload | ed25519_signature)                         (142 chars)

Base32 keeps the QR in alphanumeric mode, which needs fewer modules than byte mode.
"""

from __future__ import annotations

import base64
import os
import secrets
import struct
import threading
import time
from dataclasses import asdict, dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

PREFIX = "HP1:"
VERSION = 1
_FMT = ">B8sBIII"
_PAYLOAD_LEN = struct.calcsize(_FMT)
_SIG_LEN = 64
DISPLAY_TTL_S = 90  # the phone re-signs every 60 s; allow 30 s of clock skew
FRONT_DOOR = 1


@dataclass
class PassClaims:
    pid: str  # hex
    door: int
    nbf: int
    exp: int
    iat: int


@dataclass
class Verification:
    valid: bool
    reason: str  # ok | malformed | forged | wrong_door | not_yet_valid | expired | stale_display | reused
    pid_short: str | None = None
    visit_window: list[int] | None = None
    display_age_s: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _b32e(b: bytes) -> str:
    return base64.b32encode(b).decode().rstrip("=")


def _b32d(s: str) -> bytes:
    return base64.b32decode(s + "=" * (-len(s) % 8))


def public_key_bytes(pk: Ed25519PublicKey) -> bytes:
    return pk.public_bytes(Encoding.Raw, PublicFormat.Raw)


def load_private_key(raw_b64: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(base64.b64decode(raw_b64))


def issue_token(sk: Ed25519PrivateKey, *, nbf: int, exp: int, iat: int, door: int = FRONT_DOOR,
                pid: bytes | None = None) -> str:
    pid = pid or secrets.token_bytes(8)
    payload = struct.pack(_FMT, VERSION, pid, door, nbf, exp, iat)
    return PREFIX + _b32e(payload + sk.sign(payload))


def parse_token(token: str) -> tuple[bytes, bytes, PassClaims] | None:
    if not isinstance(token, str) or not token.startswith(PREFIX):
        return None
    try:
        raw = _b32d(token[len(PREFIX):])
    except Exception:
        return None
    if len(raw) != _PAYLOAD_LEN + _SIG_LEN:
        return None
    payload, sig = raw[:_PAYLOAD_LEN], raw[_PAYLOAD_LEN:]
    version, pid, door, nbf, exp, iat = struct.unpack(_FMT, payload)
    if version != VERSION:
        return None
    return payload, sig, PassClaims(pid.hex(), door, nbf, exp, iat)


class UsedPasses:
    """One-time-use ledger. In-memory per process; production would use a conditional write in a table."""

    def __init__(self) -> None:
        self._used: dict[str, int] = {}
        self._lock = threading.Lock()

    def is_used(self, pid: str) -> bool:
        with self._lock:
            return pid in self._used

    def consume(self, pid: str, now: int) -> bool:
        """Atomically mark pid used. Returns False if it was already used."""
        with self._lock:
            if pid in self._used:
                return False
            self._used[pid] = now
            return True


def verify_token(token: str, pk: Ed25519PublicKey, used: UsedPasses, *, now: int | None = None,
                 door: int = FRONT_DOOR) -> Verification:
    """Deterministic pass check. Order matters: signature before any claim is trusted."""
    now = int(time.time()) if now is None else now
    parsed = parse_token(token)
    if parsed is None:
        return Verification(False, "malformed")
    payload, sig, c = parsed
    try:
        pk.verify(sig, payload)
    except InvalidSignature:
        return Verification(False, "forged")
    v = Verification(False, "ok", pid_short=c.pid[:6], visit_window=[c.nbf, c.exp],
                     display_age_s=now - c.iat)
    if c.door != door:
        v.reason = "wrong_door"
    elif now < c.nbf:
        v.reason = "not_yet_valid"
    elif now > c.exp:
        v.reason = "expired"
    elif not (-30 <= now - c.iat <= DISPLAY_TTL_S):
        v.reason = "stale_display"
    elif used.is_used(c.pid):
        v.reason = "reused"
    else:
        v.valid = True
    return v


class Issuer:
    """Demo issuer standing in for the HomePulse family app.

    The signing key comes from HPV_ISSUER_SK (base64 raw 32 bytes, set as a Lambda
    environment variable by the deploy script, never committed). Without it an
    ephemeral key is generated per process.
    """

    def __init__(self, sk: Ed25519PrivateKey | None = None) -> None:
        env = os.environ.get("HPV_ISSUER_SK")
        self.ephemeral = sk is None and not env
        self.sk = sk or (load_private_key(env) if env else Ed25519PrivateKey.generate())
        self.pk = self.sk.public_key()

    def fresh_token(self, *, now: int | None = None, visit_hours: float = 2.0, **kw) -> str:
        now = int(time.time()) if now is None else now
        nbf = kw.pop("nbf", now - 600)
        exp = kw.pop("exp", now + int(visit_hours * 3600))
        iat = kw.pop("iat", now)
        return issue_token(self.sk, nbf=nbf, exp=exp, iat=iat, **kw)
