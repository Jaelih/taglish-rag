"""Client-side rate limiting for the Gemini backend.

Google AI Studio's free tier caps gemini-3.1-flash-lite at 15 requests per
minute, and a full generation eval fires 180 of them (90 eval items x one
answer + one judge call), so an unthrottled run reliably dies on a 429
partway through.

This is a sliding-window limiter rather than a "sleep 60s after every 15"
batch pause: it records the timestamp of each call and, once 15 sit inside
the trailing 60s window, sleeps only until the oldest falls out. Same
guarantee, but the time the API already spent answering counts toward the
window instead of being paid for twice.

Waits are logged to stdout with flush=True (the repo's convention -- runs
are captured by shell redirection, cf. results/ablation_run.log) so a long
throttled run is legible in the terminal instead of looking hung.
"""
from __future__ import annotations

import threading
import time
from collections import deque

from taglish_rag.config import env

DEFAULT_MAX_RPM = 15  # Google AI Studio free tier for gemini-3.1-flash-lite
DEFAULT_PERIOD = 60.0

#: Waits longer than this get an intermediate countdown line, so the terminal
#: keeps moving during the pause.
COUNTDOWN_THRESHOLD = 10.0
COUNTDOWN_INTERVAL = 10.0


class RateLimiter:
    """Sliding-window limiter: at most `max_calls` acquires per `period` seconds.

    `clock`/`sleep` are injectable so the tests can exercise the window
    arithmetic without spending real wall-clock time.
    """

    def __init__(
        self,
        max_calls: int = DEFAULT_MAX_RPM,
        period: float = DEFAULT_PERIOD,
        name: str = "rate-limit",
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        self.max_calls = max_calls
        self.period = period
        self.name = name
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._total = 0
        self._lock = threading.Lock()

    @property
    def total_calls(self) -> int:
        return self._total

    def acquire(self) -> None:
        """Block until another call fits inside the window, then record it."""
        with self._lock:
            self._prune()
            if len(self._calls) >= self.max_calls:
                wait = self._calls[0] + self.period - self._clock()
                if wait > 0:
                    self._log_and_wait(wait)
                self._prune()
            self._calls.append(self._clock())
            self._total += 1

    def _prune(self) -> None:
        cutoff = self._clock() - self.period
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()

    def _log_and_wait(self, wait: float) -> None:
        print(
            f"[{self.name}] {len(self._calls)}/{self.max_calls} requests in the last "
            f"{self.period:.0f}s - waiting {wait:.1f}s before request "
            f"{self._total + 1}",
            flush=True,
        )
        remaining = wait
        while remaining > COUNTDOWN_THRESHOLD:
            self._sleep(COUNTDOWN_INTERVAL)
            remaining -= COUNTDOWN_INTERVAL
            if remaining > 0:
                print(f"[{self.name}]   {remaining:.0f}s remaining...", flush=True)
        if remaining > 0:
            self._sleep(remaining)
        print(f"[{self.name}] resuming (request {self._total + 1})", flush=True)


_shared: RateLimiter | None = None
_shared_lock = threading.Lock()


def _configured_max_rpm() -> int:
    """Free tier by default; override with GEMINI_MAX_RPM for a paid key."""
    raw = env("GEMINI_MAX_RPM")
    if not raw:
        return DEFAULT_MAX_RPM
    try:
        value = int(raw)
    except ValueError:
        print(
            f"[rate-limit] ignoring invalid GEMINI_MAX_RPM={raw!r}, "
            f"using {DEFAULT_MAX_RPM}",
            flush=True,
        )
        return DEFAULT_MAX_RPM
    return value if value >= 1 else DEFAULT_MAX_RPM


def get_gemini_limiter() -> RateLimiter:
    """Process-wide limiter shared by every GeminiGenerator.

    get_generator() hands back a fresh client each call, so the limiter has to
    live at module scope or two generators would each get their own 15 RPM.
    """
    global _shared
    with _shared_lock:
        if _shared is None:
            max_rpm = _configured_max_rpm()
            _shared = RateLimiter(max_calls=max_rpm, name="rate-limit")
            print(
                f"[rate-limit] throttling Gemini calls to {max_rpm}/min "
                f"(set GEMINI_MAX_RPM to change)",
                flush=True,
            )
        return _shared
