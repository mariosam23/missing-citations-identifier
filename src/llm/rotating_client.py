"""Round-robin Gemini client that rotates models to spread rate-limit load.

Per-model quotas are not generous, so a single model stalls a long document or a
large evaluation. This wrapper holds several models and:

* spreads successful calls across them round-robin (load balancing), and
* on a ``429 RESOURCE_EXHAUSTED`` (rate/quota), rotates to the next model and
  retries the same request instead of failing it.

Only quota/rate errors trigger rotation; any other error propagates unchanged so
real bugs are not masked. It satisfies the same ``complete(...)`` interface the
``CitationNeedIdentifier`` depends on, so it is a drop-in for ``GeminiClient``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Protocol

from llm.gemini_client import GeminiClient
from utils.logger import logger

_RATE_LIMIT_STATUS = 429
_DEFAULT_MAX_TOKENS = 4096


class CompletionClient(Protocol):
    """Minimal text-completion interface a rotating member must provide."""

    def complete(
        self,
        system: str,
        user: str,
        response_mime_type: str | None = None,
    ) -> str: ...

    @property
    def model_name(self) -> str: ...


def is_rate_limit_error(exc: BaseException) -> bool:
    """Return True for a Gemini quota/rate-limit (HTTP 429) error."""
    for attr in ("code", "status_code"):
        if getattr(exc, attr, None) == _RATE_LIMIT_STATUS:
            return True
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or f"{_RATE_LIMIT_STATUS}" in text


class RotatingGeminiClient:
    """Cycle through several models, rotating on rate-limit errors."""

    def __init__(
        self,
        clients: Sequence[CompletionClient],
        *,
        max_cycles: int = 2,
        cooldown_seconds: float = 20.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        members = list(clients)
        if not members:
            raise ValueError("RotatingGeminiClient needs at least one client")
        self._clients = members
        self._index = 0
        self._max_cycles = max(1, max_cycles)
        self._cooldown_seconds = cooldown_seconds
        self._sleep = sleep

    @classmethod
    def from_models(
        cls,
        models: Sequence[str],
        *,
        temperature: float = 0.0,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        max_cycles: int = 2,
        cooldown_seconds: float = 20.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> RotatingGeminiClient:
        names = [m for m in models if m]
        if not names:
            raise ValueError("RotatingGeminiClient.from_models needs >=1 model")
        clients = [
            GeminiClient(model=name, temperature=temperature, max_tokens=max_tokens)
            for name in names
        ]
        return cls(
            clients,
            max_cycles=max_cycles,
            cooldown_seconds=cooldown_seconds,
            sleep=sleep,
        )

    @property
    def model_name(self) -> str:
        return "+".join(client.model_name for client in self._clients)

    def complete(
        self,
        system: str,
        user: str,
        response_mime_type: str | None = None,
    ) -> str:
        count = len(self._clients)
        last_error: BaseException | None = None

        for cycle in range(self._max_cycles):
            for _ in range(count):
                client = self._clients[self._index]
                try:
                    result = client.complete(system, user, response_mime_type)
                except Exception as exc:  # noqa: BLE001 - re-raised unless 429
                    if not is_rate_limit_error(exc):
                        raise
                    last_error = exc
                    logger.warning(
                        "model %s is rate-limited; rotating to the next model",
                        client.model_name,
                    )
                    self._advance()
                    continue
                self._advance()
                return result

            if cycle + 1 < self._max_cycles:
                logger.warning(
                    "all %d models rate-limited; cooling down %.0fs",
                    count,
                    self._cooldown_seconds,
                )
                self._sleep(self._cooldown_seconds)

        assert last_error is not None  # only reachable after a rotation
        raise last_error

    def _advance(self) -> None:
        self._index = (self._index + 1) % len(self._clients)
