"""Export one agent run's trace from CloudWatch Logs (written by the Lambda) to traces/.

  python scripts/export_cloudwatch_trace.py [scenario_id] [hours_back]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parent.parent
GROUP = "/aws/lambda/homepulse-vision"


def fetch(logs, pattern: str, start: int) -> list[dict]:
    out, kw = [], {}
    while True:
        r = logs.filter_log_events(logGroupName=GROUP, startTime=start, filterPattern=pattern, **kw)
        out += r["events"]
        if "nextToken" not in r:
            return out
        kw["nextToken"] = r["nextToken"]


def main() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "nurse_motion"
    hours = float(sys.argv[2]) if len(sys.argv) > 2 else 2
    logs = boto3.client("logs", region_name="us-east-1")
    start = int((time.time() - hours * 3600) * 1000)
    first = fetch(logs, f'"scenario:{scenario}"', start)
    if not first:
        sys.exit(f"no run of {scenario} in the last {hours} h")
    sid = json.loads(first[-1]["message"])["hpv_trace"]["session"]
    ev = sorted(fetch(logs, f'"{sid}"', start), key=lambda e: (e["timestamp"], json.loads(e["message"])["hpv_trace"]["seq"]))
    out = ROOT / "traces" / f"04-cloudwatch-lambda-{scenario}.jsonl"
    with out.open("w") as f:
        for e in ev:
            f.write(json.dumps({"cloudwatch_timestamp": e["timestamp"], "log_group": GROUP, "log_stream": e["logStreamName"],
                                **json.loads(e["message"])}) + "\n")
    print(out, len(ev), "events, session", sid)


if __name__ == "__main__":
    main()
