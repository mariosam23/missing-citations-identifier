"""Unit tests for the model-rotating Gemini client (no network)."""

from __future__ import annotations

import pytest

from llm.rotating_client import RotatingGeminiClient, is_rate_limit_error


class _RateLimitError(Exception):
    """Mimics google.genai's 429 ClientError for detection purposes."""

    code = 429

    def __init__(self) -> None:
        super().__init__("429 RESOURCE_EXHAUSTED quota exceeded")


class FakeMember:
    """A rotation member that replays a scripted sequence of outcomes."""

    def __init__(self, name: str, *outcomes: object) -> None:
        self._name = name
        self._outcomes = list(outcomes)
        self.calls = 0

    @property
    def model_name(self) -> str:
        return self._name

    def complete(
        self, system: str, user: str, response_mime_type: str | None = None
    ) -> str:
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return str(outcome)


def test_is_rate_limit_error_detects_429() -> None:
    assert is_rate_limit_error(_RateLimitError()) is True
    assert is_rate_limit_error(ValueError("nope")) is False


def test_rotates_to_next_model_on_rate_limit() -> None:
    a = FakeMember("a", _RateLimitError())
    b = FakeMember("b", "from-b")
    client = RotatingGeminiClient([a, b], sleep=lambda _s: None)

    assert client.complete("s", "u") == "from-b"
    assert a.calls == 1 and b.calls == 1


def test_successful_calls_spread_round_robin() -> None:
    a = FakeMember("a", "from-a")
    b = FakeMember("b", "from-b")
    client = RotatingGeminiClient([a, b], sleep=lambda _s: None)

    assert client.complete("s", "u") == "from-a"  # index 0
    assert client.complete("s", "u") == "from-b"  # advanced to index 1
    assert a.calls == 1 and b.calls == 1


def test_non_rate_limit_error_propagates() -> None:
    a = FakeMember("a", ValueError("real bug"))
    b = FakeMember("b", "unused")
    client = RotatingGeminiClient([a, b], sleep=lambda _s: None)

    with pytest.raises(ValueError, match="real bug"):
        client.complete("s", "u")
    assert b.calls == 0


def test_cooldown_then_retry_when_all_limited_once() -> None:
    a = FakeMember("a", _RateLimitError(), "recovered-a")
    b = FakeMember("b", _RateLimitError())
    slept: list[float] = []
    client = RotatingGeminiClient(
        [a, b], max_cycles=2, cooldown_seconds=7.0, sleep=slept.append
    )

    assert client.complete("s", "u") == "recovered-a"
    assert slept == [7.0]  # cooled down once after the first full cycle failed


def test_raises_last_error_when_all_cycles_exhausted() -> None:
    a = FakeMember("a", _RateLimitError())
    b = FakeMember("b", _RateLimitError())
    client = RotatingGeminiClient(
        [a, b], max_cycles=1, sleep=lambda _s: None
    )

    with pytest.raises(_RateLimitError):
        client.complete("s", "u")


def test_model_name_joins_members() -> None:
    client = RotatingGeminiClient(
        [FakeMember("a", "x"), FakeMember("b", "y")], sleep=lambda _s: None
    )
    assert client.model_name == "a+b"


def test_empty_client_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        RotatingGeminiClient([])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
