"""Security middleware: optional API-key auth, per-IP token-bucket rate
limiting, and hardening headers. Kept dependency-free on purpose."""
from __future__ import annotations

import hmac
import threading
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings

_PUBLIC_PATHS = {"/health"}


class _TokenBucket:
    def __init__(self, rate_per_minute: int):
        self.capacity = float(rate_per_minute)
        self.refill_per_s = rate_per_minute / 60.0
        self.buckets: dict[str, tuple[float, float]] = {}  # ip -> (tokens, last_ts)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            tokens, last = self.buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill_per_s)
            if tokens < 1.0:
                self.buckets[key] = (tokens, now)
                return False
            self.buckets[key] = (tokens - 1.0, now)
            # Opportunistic cleanup so the map cannot grow unbounded.
            if len(self.buckets) > 10_000:
                cutoff = now - 300
                self.buckets = {k: v for k, v in self.buckets.items() if v[1] > cutoff}
            return True


class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.bucket = _TokenBucket(settings.rate_limit_per_minute)

    async def dispatch(self, request: Request, call_next):
        if request.url.path not in _PUBLIC_PATHS:
            client_ip = request.client.host if request.client else "unknown"
            if not self.bucket.allow(client_ip):
                return JSONResponse({"detail": "Rate limit exceeded."}, status_code=429)

            if settings.api_key is not None:
                provided = request.headers.get("x-api-key", "")
                if not hmac.compare_digest(provided, settings.api_key):
                    return JSONResponse({"detail": "Invalid or missing API key."}, status_code=401)

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = response.headers.get("Cache-Control", "no-store")
        return response
