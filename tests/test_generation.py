import pytest
from conftest import StubGenerator

from taglish_rag.generation.generator import (
    GeminiGenerator,
    _is_retryable,
    _server_retry_delay,
)
from taglish_rag.generation.judge import LLMJudge, _extract_json
from taglish_rag.ratelimit import RateLimiter


def test_extract_json_handles_surrounding_prose():
    text = 'Sure, here is my answer:\n{"groundedness": 0.8, "is_refusal": false}\nHope that helps!'
    parsed = _extract_json(text)
    assert parsed == {"groundedness": 0.8, "is_refusal": False}


def test_extract_json_returns_none_for_garbage():
    assert _extract_json("not json at all") is None


def test_llm_judge_marks_unparseable_response():
    judge = LLMJudge(StubGenerator("not json"))
    score = judge.score("q", "ctx", "ref", "ans")
    assert score.parse_ok is False
    assert score.groundedness is None


def test_llm_judge_parses_well_formed_response():
    judge = LLMJudge(
        StubGenerator(
            '{"groundedness": 0.9, "correctness": 1.0, "citation_accuracy": 0.5, '
            '"is_refusal": false, "rationale": "looks right"}'
        )
    )
    score = judge.score("q", "ctx", "ref", "ans")
    assert score.parse_ok is True
    assert score.groundedness == 0.9
    assert score.is_refusal is False


# --- rate limiting -----------------------------------------------------


class FakeClock:
    """Deterministic stand-in for time.monotonic/time.sleep, so the window
    arithmetic is tested without spending real wall-clock time."""

    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _limiter(clock, max_calls=15, period=60.0):
    return RateLimiter(max_calls=max_calls, period=period, clock=clock, sleep=clock.sleep)


def test_calls_under_the_cap_never_sleep():
    clock = FakeClock()
    limiter = _limiter(clock)
    for _ in range(15):
        limiter.acquire()
    assert clock.slept == []


def test_call_past_the_cap_waits_for_the_oldest_to_expire():
    clock = FakeClock()
    limiter = _limiter(clock)
    for _ in range(15):
        limiter.acquire()
        clock.now += 1.0  # 15 calls spread over 15s
    limiter.acquire()
    # oldest call was 15s ago, so only the remaining 45s of its window is owed
    assert sum(clock.slept) == pytest.approx(45.0)


def test_slow_caller_never_sleeps_because_old_calls_fall_out_of_the_window():
    clock = FakeClock()
    limiter = _limiter(clock)
    for _ in range(40):
        limiter.acquire()
        clock.now += 5.0  # 12/min, comfortably under the cap
    assert clock.slept == []


def test_long_wait_is_broken_into_countdown_chunks():
    clock = FakeClock()
    limiter = _limiter(clock)
    for _ in range(15):
        limiter.acquire()
    limiter.acquire()  # full 60s owed
    assert sum(clock.slept) == pytest.approx(60.0)
    assert len(clock.slept) > 1  # countdown lines, not one silent 60s block


def test_limiter_rejects_a_zero_cap():
    with pytest.raises(ValueError):
        RateLimiter(max_calls=0)


# --- Gemini retry/backoff ----------------------------------------------


def test_retryable_markers_match_quota_and_transport_errors():
    assert _is_retryable(Exception("429 RESOURCE_EXHAUSTED: quota exceeded"))
    assert _is_retryable(Exception("503 UNAVAILABLE"))
    assert not _is_retryable(ValueError("GOOGLE_API_KEY not set"))
    assert not _is_retryable(Exception("400 INVALID_ARGUMENT"))


def test_server_retry_delay_is_honored_when_present():
    exc = Exception("{'error': {'code': 429}, 'retryDelay': '37s'}")
    assert _server_retry_delay(exc) == 37.0
    assert _server_retry_delay(Exception("429 too many requests")) is None


class _FlakyModels:
    """Stands in for client.models: fails `failures` times, then answers."""

    def __init__(self, failures: int, exc: Exception):
        self.failures = failures
        self.exc = exc
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return type("Resp", (), {"text": "recovered answer"})()


def _stub_gemini(models):
    """Build a GeminiGenerator without __init__, which would need a live key."""
    gen = GeminiGenerator.__new__(GeminiGenerator)
    gen._client = type("Client", (), {"models": models})()
    gen._model_name = "test-model"
    clock = FakeClock()
    gen._limiter = _limiter(clock)
    return gen


def test_generate_retries_a_429_and_then_succeeds(monkeypatch):
    monkeypatch.setattr("taglish_rag.generation.generator.time.sleep", lambda s: None)
    models = _FlakyModels(2, Exception("429 RESOURCE_EXHAUSTED: quota exceeded"))
    result = _stub_gemini(models).generate("q", "ctx")
    assert result.answer == "recovered answer"
    assert models.calls == 3


def test_generate_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr("taglish_rag.generation.generator.time.sleep", lambda s: None)
    models = _FlakyModels(99, Exception("429 RESOURCE_EXHAUSTED"))
    with pytest.raises(Exception, match="429"):
        _stub_gemini(models).generate("q", "ctx")
    assert models.calls == 3


def test_generate_does_not_retry_a_non_retryable_error(monkeypatch):
    monkeypatch.setattr("taglish_rag.generation.generator.time.sleep", lambda s: None)
    models = _FlakyModels(99, ValueError("400 INVALID_ARGUMENT"))
    with pytest.raises(ValueError):
        _stub_gemini(models).generate("q", "ctx")
    assert models.calls == 1
