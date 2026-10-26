"""Structured trace: one JSON object per step, printed as a JSON line (CloudWatch Logs on Lambda)."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone


class Trace:
    def __init__(self, session_id: str, emit: bool | None = None) -> None:
        self.session_id = session_id
        self.events: list[dict] = []
        self.t0 = time.perf_counter()
        self.emit = os.environ.get("HPV_TRACE_STDOUT", "1") == "1" if emit is None else emit

    def add(self, type_: str, **data) -> dict:
        ev = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "t_ms": round((time.perf_counter() - self.t0) * 1000, 1),
            "seq": len(self.events),
            "session": self.session_id,
            "type": type_,
            **data,
        }
        self.events.append(ev)
        if self.emit:
            sys.stdout.write(json.dumps({"hpv_trace": ev}, default=str) + "\n")
            sys.stdout.flush()
        return ev
