"""In-memory sliding-window rate limits: per client IP and a global cap per instance.

Lambda reserved concurrency bounds the number of instances, so the global cap
times that concurrency bounds total agent runs (and Bedrock spend).
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self) -> None:
        env = os.environ.get
        self.rules_ip = [(60, int(env("HPV_RL_IP_PER_MIN", 6))), (3600, int(env("HPV_RL_IP_PER_HOUR", 40)))]
        self.rules_global = [(60, int(env("HPV_RL_GLOBAL_PER_MIN", 30))), (86400, int(env("HPV_RL_GLOBAL_PER_DAY", 1500)))]
        self._ip: dict[str, deque] = defaultdict(deque)
        self._global: deque = deque()
        self._lock = threading.Lock()

    @staticmethod
    def _over(q: deque, rules, now: float) -> int | None:
        longest = max(w for w, _ in rules)
        while q and now - q[0] > longest:
            q.popleft()
        for window, limit in rules:
            n = sum(1 for t in q if now - t <= window)
            if n >= limit:
                oldest = next(t for t in q if now - t <= window)
                return int(window - (now - oldest)) + 1
        return None

    def check(self, ip: str, now: float | None = None) -> tuple[bool, int, str]:
        """Returns (allowed, retry_after_s, scope)."""
        now = time.time() if now is None else now
        with self._lock:
            q = self._ip[ip]
            wait = self._over(q, self.rules_ip, now)
            if wait:
                return False, wait, "ip"
            wait = self._over(self._global, self.rules_global, now)
            if wait:
                return False, wait, "global"
            q.append(now)
            self._global.append(now)
            if len(self._ip) > 10000:  # bound memory under address spraying
                for k in [k for k, v in self._ip.items() if not v][:5000]:
                    del self._ip[k]
            return True, 0, ""
