"""Evaluate HomePulse Vision on docs/eval/set. Reports only measured numbers.

  python scripts/run_eval.py perception            # OpenCV-only metrics (no AWS)
  python scripts/run_eval.py agent --planner nova  # end-to-end decisions with Nova 2 Lite on Bedrock
  python scripts/run_eval.py agent --planner scripted
  python scripts/run_eval.py lambda                # per-stage latency on the deployed Lambda
  python scripts/run_eval.py report                # docs/eval/metrics.md + failure images
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HPV_TRACE_STDOUT", "0")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2 as cv  # noqa: E402
import numpy as np  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey  # noqa: E402

from homepulse_vision.agent import loop  # noqa: E402
from homepulse_vision.agent.tools import Session  # noqa: E402
from homepulse_vision.passes import UsedPasses, verify_token  # noqa: E402
from homepulse_vision.perception import enhance, pipeline, presence, qr, quality  # noqa: E402
from homepulse_vision.scenarios import Camera  # noqa: E402

EVAL = ROOT / "docs" / "eval"
SET = EVAL / "set"


def load():
    m = json.loads((SET / "manifest.json").read_text())
    pk = Ed25519PublicKey.from_public_bytes(base64.b64decode(m["issuer_public_key_b64"]))
    return m, pk


def pct(a, b):
    return None if not b else round(100.0 * a / b, 1)


def p(vals, q):
    if not vals:
        return None
    v = sorted(vals)
    return round(v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))], 1)


def env_info():
    import platform

    return {"opencv": cv.__version__, "python": platform.python_version(), "machine": platform.machine(),
            "system": platform.system(), "date": time.strftime("%Y-%m-%d %H:%M %Z")}


# ------------------------------------------------------------------ perception

def _valid_decoded(codes, pk, now):
    return any(verify_token(c["text"], pk, UsedPasses(), now=now).reason in ("ok",) for c in codes)


def improve_chain(img, a, pk, now):
    """The same operations the agent can request through improve_view (at most 4), applied deterministically."""
    work = a["_work"]
    q = a["quality"]
    rois = [r["box"] for r in a["roi_candidates"] if r["kind"] != "text_block"]
    tried = []

    def crop_try(src, boxes):
        for b in boxes[:2]:
            if len(tried) >= 4:
                return False
            c, _, _ = enhance.crop_upscale(src, b)
            tried.append("crop_upscale")
            if _valid_decoded(qr.decode_roi(c)["codes"], pk, now):
                return True
        return False

    if crop_try(work, rois):
        return True, tried
    for issue, op in (("dark", "gamma"), ("backlit", "clahe")):
        if issue in q["issues"] and len(tried) < 4:
            e = enhance.gamma(work)[0] if op == "gamma" else enhance.clahe(work)
            tried.append(op)
            if _valid_decoded(qr.decode(e)["codes"], pk, now):
                return True, tried
            if crop_try(e, rois or qr.bright_panels(e)):
                return True, tried
    return False, tried


def face_hit(faces, box):
    if not box:
        return False
    x, y, w, h = box
    for f in faces:
        fx, fy, fw, fh = f["box"]
        cx, cy = fx + fw / 2, fy + fh / 2
        if x - 10 <= cx <= x + w + 10 and y - 10 <= cy <= y + h + 10:
            return True
    return False


_fd_low = None


def best_face_score(img, box):
    """Highest YuNet score (threshold 0.3) inside the drawn face, or anywhere when box is None (empty porch)."""
    global _fd_low
    if _fd_low is None:
        from homepulse_vision.config import model_path
        _fd_low = cv.FaceDetectorYN.create(model_path("face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.3, 0.3, 50)
    s = min(1.0, 640 / max(img.shape[:2]))
    small = cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA) if s < 1 else img
    _fd_low.setInputSize((small.shape[1], small.shape[0]))
    _, faces = _fd_low.detect(small)
    best = 0.0
    for f in faces if faces is not None else []:
        fb = {"box": (np.asarray(f[:4]) / s).tolist()}
        if box is None or face_hit([fb], box):
            best = max(best, float(f[-1]))
    return round(best, 3)


def perception(_args):
    m, pk = load()
    now = m["eval_now"] + 5
    rows = []
    timings = defaultdict(list)
    for c in m["cases"]:
        img = cv.imread(str(SET / c["frames"][0]))
        a = pipeline.analyze(img)
        for k, v in a["timing_ms"].items():
            timings[k].append(v)
        row = {"id": c["id"], "group": c["group"], "label": c["label"], "quality": {k: a["quality"][k] for k in (
            "issues", "blur_effect", "laplacian_var_edges", "mean_luma", "backlight_score", "watermark_fraction")},
            "overlays": len(a["overlays_masked"]), "faces": len(a["faces"]), "text": a["text"],
            "qr_decoded_any": bool(a["qr"]["codes"])}
        if c["label"].get("pass") == "valid":
            row["qr_base"] = _valid_decoded(a["qr"]["codes"], pk, now)
            row["qr_improved"], row["ops"] = (True, []) if row["qr_base"] else improve_chain(img, a, pk, now)
        if c["truth"].get("face_box"):
            row["face_base"] = face_hit(a["faces"], c["truth"]["face_box"])
            q = a["quality"]["issues"]
            e = a["_work"]
            if "dark" in q:
                e = enhance.gamma(e)[0]
            if "backlit" in q:
                e = enhance.clahe(e)
            row["face_improved"] = row["face_base"] or (e is not a["_work"] and face_hit(presence.detect_faces(e), c["truth"]["face_box"]))
            row["face_best_score"] = best_face_score(a["_work"], c["truth"]["face_box"])
        elif c["group"] == "empty":
            row["face_best_score"] = best_face_score(a["_work"], None)
        if c["truth"].get("sign_lines"):
            lines = c["truth"]["sign_lines"]
            hit = 0
            for ln in lines:
                x, y, w, h = ln["box"]
                if any(x - 8 <= r["box"][0] + r["box"][2] / 2 <= x + w + 8 and y - 8 <= r["box"][1] + r["box"][3] / 2 <= y + h + 8
                       for r in a["text_regions"]):
                    hit += 1
            row["lines_total"], row["lines_detected"] = len(lines), hit
        rows.append(row)
        print(f"\r{len(rows)}/{len(m['cases'])}", end="", file=sys.stderr)
    print(file=sys.stderr)
    out = {"env": env_info(), "eval_now": m["eval_now"], "rows": rows,
           "timing_ms": {k: {"p50": p(v, 50), "p95": p(v, 95), "n": len(v)} for k, v in timings.items()}}
    (EVAL / "perception.json").write_text(json.dumps(out, indent=1))
    print("wrote docs/eval/perception.json")


# ------------------------------------------------------------------ agent

def run_case(c, pk, now, planner_factory):
    frames = [cv.imread(str(SET / f)) for f in c["frames"]]
    used = UsedPasses()
    if c["label"].get("preconsume_pid"):
        used.consume(c["label"]["preconsume_pid"], now - 3600)
    s = Session(Camera(frames, "eval set frame"), pk, used, clock=lambda: now, emit_trace=False, source="eval:" + c["id"])
    t = time.perf_counter()
    r = loop.run(s, planner_factory())
    wall = round((time.perf_counter() - t) * 1000, 1)
    calls = [e["name"] for e in s.trace.events if e["type"] == "tool_call"]
    replanned = any(e["type"] == "tool_call" and e["name"] in ("improve_view", "request_new_snapshot") and e["evidence"]
                    and (e["evidence"].get("quality_issues") or e["evidence"].get("qr_located_not_decoded")) for e in s.trace.events)
    return {"id": c["id"], "group": c["group"], "expect": c["expect"], "label": c["label"], "decision": r["decision"],
            "reason": r["reason"], "end": r["end"], "calls": calls, "guard_refusals": s.guard_refusals,
            "safety_net": any(e["type"] == "safety_net" for e in s.trace.events),
            "tool_errors": s.counts["tool_errors"], "replanned_from_perception": replanned,
            "verifications": {k: v["result"]["reason"] for k, v in s.verifications.items()},
            "family_reasons": [f["reason"] for f in s.family_requests],
            "model_ms": s.stage_ms.get("model.converse", []), "perception_ms": s.stage_ms.get("perception.total", []),
            "tokens": r["usage"], "wall_ms": wall, "summary": r["summary"][:300]}, s.trace.events


def agent(args):
    m, pk = load()
    now = m["eval_now"] + 5
    cases = m["cases"]
    if args.only:
        cases = [c for c in cases if c["group"] in args.only.split(",")]
    if args.planner == "nova":
        shared = loop.BedrockPlanner()
        factory = lambda: shared  # noqa: E731  (stateless; boto3 clients are thread-safe)
    else:
        factory = loop.ScriptedPlanner
    results, traces = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (r, ev) in enumerate(ex.map(lambda c: run_case(c, pk, now, factory), cases)):
            results.append(r)
            traces.extend(ev)
            print(f"\r{i + 1}/{len(cases)} {r['id']:16s} {r['decision']:12s}", end="", file=sys.stderr)
    print(file=sys.stderr)
    name = f"agent_{args.planner}"
    (EVAL / f"{name}.json").write_text(json.dumps({"env": env_info(), "planner": factory().__class__.__name__ if args.planner != "nova" else loop.BedrockPlanner.name,
                                                   "results": results}, indent=1))
    with (EVAL / f"{name}_traces.jsonl").open("w") as f:
        for e in traces:
            f.write(json.dumps(e, default=str) + "\n")
    print(f"wrote docs/eval/{name}.json and traces")


# ------------------------------------------------------------------ lambda latency

def lambda_latency(args):
    import boto3

    from homepulse_vision.scenarios import SCENARIOS

    client = boto3.client("lambda", region_name="us-east-1")
    rows = []
    for rep in range(args.reps):
        for i, sc in enumerate(SCENARIOS):
            ev = {"version": "2.0", "routeKey": "$default", "rawPath": "/api/run", "rawQueryString": "",
                  "headers": {"host": "eval", "content-type": "application/json"},
                  "requestContext": {"http": {"method": "POST", "path": "/api/run", "protocol": "HTTP/1.1",
                                              "sourceIp": f"198.51.100.{(rep * 20 + i) % 250 + 1}", "userAgent": "run_eval"},
                                     "accountId": "anonymous", "apiId": "eval", "domainName": "eval", "requestId": "eval",
                                     "routeKey": "$default", "stage": "$default", "time": "", "timeEpoch": 0},
                  "body": json.dumps({"scenario": sc.id}), "isBase64Encoded": False}
            t = time.perf_counter()
            resp = client.invoke(FunctionName="homepulse-vision", Payload=json.dumps(ev).encode())
            rt = round((time.perf_counter() - t) * 1000, 1)
            out = json.loads(resp["Payload"].read())
            body = json.loads(out["body"])
            if out.get("statusCode") != 200:
                rows.append({"scenario": sc.id, "status": out.get("statusCode"), "error": body})
                continue
            rows.append({"scenario": sc.id, "status": 200, "decision": body["decision"], "expected": body["expected"],
                         "client_roundtrip_ms": rt, "server_wall_ms": body["wall_ms"], "stage_ms": body["stage_ms"]})
            print(f"{sc.id:16s} {body['decision']:12s} rt={rt}ms", file=sys.stderr)
            time.sleep(args.sleep)
    (EVAL / "lambda_latency.json").write_text(json.dumps({"date": time.strftime("%Y-%m-%d %H:%M %Z"), "function": "homepulse-vision",
                                                          "memory_mb": 2048, "arch": "arm64", "rows": rows}, indent=1))
    print("wrote docs/eval/lambda_latency.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("perception")
    a = sub.add_parser("agent")
    a.add_argument("--planner", choices=["nova", "scripted"], default="nova")
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--only", default="")
    lp = sub.add_parser("lambda")
    lp.add_argument("--reps", type=int, default=2)
    lp.add_argument("--sleep", type=float, default=1.0)
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "perception":
        perception(args)
    elif args.cmd == "agent":
        agent(args)
    elif args.cmd == "lambda":
        lambda_latency(args)
    else:
        from report_eval import report  # noqa: E402

        report()
