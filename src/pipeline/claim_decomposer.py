"""Stage 5 — LLM-backed claim decomposition."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from entities import AggregationStrategy, Decomposition, Subclaim
from prompts import DECOMPOSER_SYSTEM_PROMPT, DECOMPOSER_USER_PROMPT_TEMPLATE

from utils import logger


class CompletionClient(Protocol):
    def complete(
        self,
        system: str,
        user: str,
        response_mime_type: str | None = None,
    ) -> str:
        ...


class ClaimDecomposer:
    """Split complex scientific claims into atomic retrieval subclaims."""

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        client: CompletionClient | None = None,
    ) -> None:
        if client is None:
            from llm.genai_client import LLMClient

            client = LLMClient(model=model, temperature=0.1, max_tokens=2048)
        self.client = client

    def decompose(self, claim: str) -> Decomposition:
        """Return a decomposition, falling back to the original claim on LLM issues."""
        claim = claim.strip()
        if not claim:
            raise ValueError("claim must not be empty")

        user_prompt = DECOMPOSER_USER_PROMPT_TEMPLATE.format(claim=claim)
        try:
            response = self.client.complete(
                DECOMPOSER_SYSTEM_PROMPT,
                user_prompt,
                response_mime_type="application/json",
            )
            return self._decomposition_from_response(claim, response)
        except Exception as exc:
            logger.warning("Claim decomposition failed; using original claim: %s", exc)
            return self.single_claim(claim)

    @classmethod
    def _decomposition_from_response(cls, original_text: str, response: str) -> Decomposition:
        parsed = cls._parse_json(response)
        if not isinstance(parsed, dict):
            raise ValueError("Expected decomposer response to be a JSON object")

        raw_subclaims = parsed.get("subclaims")
        if not isinstance(raw_subclaims, list):
            raise ValueError("Decomposer response missing subclaims list")

        subclaims = cls._parse_subclaims(raw_subclaims)
        aggregation = cls._parse_strategy(parsed.get("aggregation"))
        return Decomposition(
            original_text=original_text,
            subclaims=tuple(subclaims),
            aggregation=aggregation,
        )

    @staticmethod
    def single_claim(claim: str) -> Decomposition:
        return Decomposition(
            original_text=claim,
            subclaims=(Subclaim(text=claim, importance=1.0),),
            aggregation=AggregationStrategy.WEIGHTED,
        )

    @staticmethod
    def _parse_json(response: str) -> Any:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            json_match = re.search(r"\{[\s\S]*\}", cleaned)
            if not json_match:
                raise ValueError(f"Could not parse decomposer response as JSON: {response}")
            return json.loads(json_match.group())

    @classmethod
    def _parse_subclaims(cls, raw_subclaims: list[Any]) -> list[Subclaim]:
        parsed: list[tuple[str, float]] = []
        for item in raw_subclaims[:4]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            try:
                importance = float(item.get("importance", 1.0))
            except (TypeError, ValueError):
                importance = 1.0
            parsed.append((text, max(importance, 0.0)))

        if not parsed:
            raise ValueError("Decomposer returned no usable subclaims")

        return [
            Subclaim(text=text, importance=importance)
            for text, importance in cls._normalize_importance(parsed)
        ]

    @staticmethod
    def _normalize_importance(subclaims: list[tuple[str, float]]) -> list[tuple[str, float]]:
        total = sum(importance for _, importance in subclaims)
        if total <= 0.0:
            equal_weight = 1.0 / len(subclaims)
            return [(text, equal_weight) for text, _ in subclaims]
        return [(text, importance / total) for text, importance in subclaims]

    @staticmethod
    def _parse_strategy(raw: Any) -> AggregationStrategy:
        try:
            return AggregationStrategy(str(raw or "WEIGHTED").upper())
        except ValueError:
            return AggregationStrategy.WEIGHTED
