"""Answer generation backed by Google AI Studio (Gemini Flash), which has
a genuinely free tier with no card required. `GOOGLE_API_KEY` must be set
in .env -- retrieval, ingestion, and the retrieval eval all still run
fully offline without it, but answer generation, the CRAG loop's generate
step, and the LLM-as-judge all need a live key.

Every Gemini request in the repo goes through GeminiGenerator.generate, so
that is where the free tier's 15 RPM cap is enforced (see
taglish_rag.ratelimit) and where 429s are retried -- one chokepoint covers
naive RAG, the CRAG agent, the judge, and the Gradio app.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from taglish_rag.config import env
from taglish_rag.ratelimit import RateLimiter, get_gemini_limiter

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about Philippine government "
    "services (BIR, PhilHealth, Pag-IBIG) using ONLY the provided context. "
    "If the context does not contain the answer, say clearly that you don't know "
    "rather than guessing. Cite which source titles you used."
)

MAX_ATTEMPTS = 3
#: Fallback backoffs when the server doesn't tell us how long to wait. Long
#: enough to clear a per-minute quota window rather than hammering it again.
BACKOFF_SECONDS = [30.0, 60.0]

#: Substrings that mark an error as worth retrying: quota exhaustion and
#: transient transport failures. Matched against the exception's type name and
#: message rather than caught by class, since google.api_core isn't a declared
#: dependency and the SDK's exception surface varies by version.
RETRYABLE_MARKERS = (
    "429",
    "resource_exhausted",
    "resourceexhausted",
    "rate limit",
    "quota",
    "503",
    "unavailable",
    "500",
    "internal error",
    "deadline",
    "timeout",
)


@dataclass
class GenerationResult:
    answer: str
    backend: str
    model: str


class Generator:
    #: Short provider label stamped into GenerationResult and into eval
    #: result files, so a run always records which backend produced it.
    backend: str = "unknown"

    def generate(self, question: str, context: str) -> GenerationResult:
        raise NotImplementedError


def _is_retryable(exc: Exception) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(marker in blob for marker in RETRYABLE_MARKERS)


def _server_retry_delay(exc: Exception) -> float | None:
    """Gemini 429s carry a RetryInfo hint ('retryDelay': '37s'); honor it when
    present so we wait exactly as long as the server asked and no longer."""
    match = re.search(r"retry[-_ ]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s?", str(exc), re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


class GeminiGenerator(Generator):
    backend = "gemini"

    def __init__(self, model: str = "gemini-3.1-flash-lite", limiter: RateLimiter | None = None):
        from google import genai

        api_key = env("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        self._client = genai.Client(api_key=api_key)
        self._model_name = model
        self._limiter = limiter or get_gemini_limiter()

    def generate(self, question: str, context: str) -> GenerationResult:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
        )
        contents = f"Context:\n{context}\n\nQuestion: {question}"

        for attempt in range(1, MAX_ATTEMPTS + 1):
            # Retries go through the limiter too, so a retried call still
            # counts against the 15/min window.
            self._limiter.acquire()
            try:
                resp = self._client.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001 - re-raised unless retryable
                if attempt == MAX_ATTEMPTS or not _is_retryable(exc):
                    raise
                delay = _server_retry_delay(exc) or BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)]
                print(
                    f"[gemini] {type(exc).__name__} on attempt {attempt}/{MAX_ATTEMPTS} "
                    f"- retrying in {delay:.0f}s ({exc})",
                    flush=True,
                )
                time.sleep(delay)
                continue
            return GenerationResult(answer=resp.text, backend=self.backend, model=self._model_name)

        raise RuntimeError("unreachable: generate loop exhausted without returning or raising")


def get_generator() -> Generator:
    return GeminiGenerator()
