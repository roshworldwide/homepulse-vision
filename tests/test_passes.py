from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from homepulse_vision.passes import DISPLAY_TTL_S, PREFIX, Issuer, UsedPasses, issue_token, parse_token, verify_token

NOW = 1_800_000_000


def _iss():
    return Issuer(Ed25519PrivateKey.generate())


def test_valid_pass():
    iss = _iss()
    v = verify_token(iss.fresh_token(now=NOW), iss.pk, UsedPasses(), now=NOW)
    assert v.valid and v.reason == "ok"


def test_token_is_qr_alphanumeric():
    tok = _iss().fresh_token(now=NOW)
    assert tok.startswith(PREFIX)
    assert set(tok) <= set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:")


def test_forged_signature():
    iss = _iss()
    tok = issue_token(Ed25519PrivateKey.generate(), nbf=NOW - 60, exp=NOW + 60, iat=NOW)
    assert verify_token(tok, iss.pk, UsedPasses(), now=NOW).reason == "forged"


def test_tampered_payload_is_forged():
    iss = _iss()
    tok = iss.fresh_token(now=NOW)
    flipped = tok[:10] + ("A" if tok[10] != "A" else "B") + tok[11:]
    assert verify_token(flipped, iss.pk, UsedPasses(), now=NOW).reason in ("forged", "malformed")


def test_expired_and_not_yet_valid():
    iss = _iss()
    used = UsedPasses()
    assert verify_token(iss.fresh_token(now=NOW, nbf=NOW - 7200, exp=NOW - 1), iss.pk, used, now=NOW).reason == "expired"
    assert verify_token(iss.fresh_token(now=NOW, nbf=NOW + 600, exp=NOW + 7200), iss.pk, used, now=NOW).reason == "not_yet_valid"


def test_printed_copy_is_stale():
    iss = _iss()
    tok = iss.fresh_token(now=NOW, iat=NOW - DISPLAY_TTL_S - 1, nbf=NOW - 86400, exp=NOW + 86400)
    assert verify_token(tok, iss.pk, UsedPasses(), now=NOW).reason == "stale_display"


def test_reused():
    iss = _iss()
    used = UsedPasses()
    tok = iss.fresh_token(now=NOW)
    assert used.consume(parse_token(tok)[2].pid, NOW)
    assert verify_token(tok, iss.pk, used, now=NOW).reason == "reused"
    assert not used.consume(parse_token(tok)[2].pid, NOW)


def test_wrong_door_and_malformed():
    iss = _iss()
    assert verify_token(iss.fresh_token(now=NOW, door=2), iss.pk, UsedPasses(), now=NOW).reason == "wrong_door"
    for bad in ("", "hello", "HP1:", "HP1:!!!!", "HP1:" + "A" * 50):
        assert verify_token(bad, iss.pk, UsedPasses(), now=NOW).reason == "malformed"
