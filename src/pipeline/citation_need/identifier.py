"""Binary, LLM-based citation-need identifier (few-shot, paragraph-level).

For each sentence in a draft the identifier answers a single question — *should
this sentence cite a source?* — via in-context learning with a fixed few-shot
prompt. The design choices that make this more than a bare API call:

* **Paragraph-level context.** A whole paragraph is sent in one request with its
  candidate sentences tagged ``[0]``, ``[1]``, … so the model resolves pronouns
  ("this method") and tells the authors' own contribution apart from reported
  prior findings — the single hardest source of false positives.
* **Cheap local prefilter.** Sentences that already contain an explicit citation
  (``\\cite{...}``, ``[1]``, ``(Author, 2020)``) or are too short never reach the
  API, so we never re-flag a cited sentence and we keep the call count down.
* **Exact offsets.** Segmentation reuses the offset-preserving spans from
  ``pipeline.missing_citations.detector`` (battle-tested across CRLF and
  hard-wrapped LaTeX), so every result carries character offsets the editor can
  act on; the LLM is never trusted to report positions.

The returned ``confidence`` is the model's *self-reported* certainty. It is not a
calibrated probability — treat it only as a thresholding/ranking signal.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby
from typing import Protocol

from pydantic import BaseModel, ValidationError, field_validator

from llm.gemini_client import GeminiClient
from llm.rotating_client import RotatingGeminiClient
from pipeline.missing_citations.detector import (
    SentenceSpan,
    has_explicit_citation,
    iter_sentence_spans,
)
from utils.config import config
from utils.logger import logger

_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_JSON_MIME = "application/json"

# The decision criteria are shared by the single-paragraph prompt (production
# /analyze path) and the batch prompt (evaluation path) so both judge by the
# same rule.
_CITATION_CRITERIA = """\
A sentence NEEDS a citation (needs_citation = true) when it states something \
that rests on an external source, such as:
- a specific prior finding, result, or empirical claim attributed to others;
- a statistic or quantitative figure that is not the authors' own measurement;
- an existing method, model, dataset, tool, or benchmark created by others;
- an established concept or definition presented as a known fact.

A sentence does NOT need a citation (needs_citation = false) when it:
- describes the authors' OWN work, contribution, results, or methodology \
("we propose", "in this paper", "our model", "we observe that ...");
- is broad, uncontroversial common knowledge;
- is a transition, a definition of the paper's own terms, or a statement about \
the paper's structure."""

SYSTEM_PROMPT = (
    "You are an expert academic reviewer deciding which sentences in a draft "
    "need a citation to a source.\n\n"
    "You will receive ONE paragraph. Some sentences are tagged with an index "
    "in square brackets, like [0] or [1]. For EACH indexed sentence, decide "
    "whether it needs a citation. Use the whole paragraph for context (resolve "
    'references such as "this method" or "these results").\n\n'
    + _CITATION_CRITERIA
    + "\n\n"
    "Return ONLY a JSON array, one object per indexed sentence:\n"
    '[{"index": <int>, "needs_citation": <true|false>, '
    '"confidence": <0.0-1.0>}]\n'
    "where confidence is your certainty in the decision.\n\n"
    "Examples.\n\n"
    "Paragraph:\n"
    "[0] Residual connections were shown to enable the training of "
    "substantially deeper convolutional networks. The idea has since become "
    "standard. [1] In this paper, we adapt it to graph-structured inputs.\n"
    "JSON:\n"
    '[{"index": 0, "needs_citation": true, "confidence": 0.92}, '
    '{"index": 1, "needs_citation": false, "confidence": 0.95}]\n\n'
    "Paragraph:\n"
    "[0] BERT reaches a GLUE score of 80.5, surpassing previous systems by a "
    "wide margin. [1] We fine-tune it for three epochs with a batch size of "
    "32. [2] Optimisation is a core part of machine learning.\n"
    "JSON:\n"
    '[{"index": 0, "needs_citation": true, "confidence": 0.95}, '
    '{"index": 1, "needs_citation": false, "confidence": 0.9}, '
    '{"index": 2, "needs_citation": false, "confidence": 0.85}]'
)

# Batch prompt: several INDEPENDENT target sentences (each with optional
# surrounding context) judged in one call. Used by the evaluation to keep the
# request count low enough for restrictive free-tier quotas.
BATCH_SYSTEM_PROMPT = (
    "You are an expert academic reviewer deciding which sentences need a "
    "citation to a source.\n\n"
    "You will receive several INDEPENDENT items, each tagged '### Item <i>'. "
    "Every item has a TARGET sentence and may include the sentence Before "
    "and/or After it for context. For EACH item decide whether its TARGET "
    "sentence needs a citation. The items are unrelated to each other.\n\n"
    + _CITATION_CRITERIA
    + "\n\n"
    "Return ONLY a JSON array, one object per item, using the item index:\n"
    '[{"index": <int>, "needs_citation": <true|false>, '
    '"confidence": <0.0-1.0>}]\n'
    "where confidence is your certainty in the decision."
)


@dataclass(frozen=True, slots=True)
class CitationNeedResult:
    """The binary citation-need decision for a single sentence."""

    sentence_id: str
    text: str
    start_offset: int
    end_offset: int
    paragraph_index: int
    section_type: str | None
    needs_citation: bool
    confidence: float


@dataclass(frozen=True, slots=True)
class CitationNeedQuery:
    """One independent target sentence to classify, with optional context."""

    target: str
    previous: str | None = None
    next: str | None = None


@dataclass(frozen=True, slots=True)
class CitationNeedJudgement:
    """A binary decision for one ``CitationNeedQuery``.

    ``sent`` is True when the target passed the local prefilter and was sent to
    the LLM; ``answered`` is True when the LLM actually returned a decision for
    it. A target skipped by the prefilter (already cited or too short) has both
    False and ``needs_citation`` False. A sent target the LLM omitted has
    ``sent`` True / ``answered`` False (counted as a miss, not as a real answer).
    """

    needs_citation: bool
    confidence: float
    sent: bool
    answered: bool


class LLMTextClient(Protocol):
    """Minimal text-completion interface the identifier depends on."""

    def complete(
        self,
        system: str,
        user: str,
        response_mime_type: str | None = None,
    ) -> str: ...


class _Decision(BaseModel):
    """One parsed per-sentence decision from the model's JSON reply."""

    index: int
    needs_citation: bool
    confidence: float = 0.5

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        return min(1.0, max(0.0, value))


class CitationNeedIdentifier:
    """Identify sentences that should cite a source, one paragraph per call."""

    def __init__(
        self,
        client: LLMTextClient | None = None,
        *,
        model_name: str | None = None,
        models: Sequence[str] | None = None,
        min_words: int | None = None,
        max_sentences: int | None = None,
    ) -> None:
        self._client, self._model_name = self._resolve_client(
            client, model_name, models
        )
        self._min_words = (
            min_words if min_words is not None else config.CITATION_NEED_MIN_WORDS
        )
        self._max_sentences = (
            max_sentences
            if max_sentences is not None
            else config.CITATION_NEED_MAX_SENTENCES
        )

    @staticmethod
    def _resolve_client(
        client: LLMTextClient | None,
        model_name: str | None,
        models: Sequence[str] | None,
    ) -> tuple[LLMTextClient, str]:
        """Pick the LLM client: injected > single model > model rotation.

        ``model_name`` pins a single model (clean per-model evaluation);
        otherwise the configured rotation spreads load across several models.
        """
        if client is not None:
            name = model_name or str(getattr(client, "model_name", "injected"))
            return client, name
        if model_name is not None:
            return GeminiClient(model=model_name, temperature=0.0), model_name
        rotation = list(models) if models else config.gemini_model_rotation()
        rotating = RotatingGeminiClient.from_models(rotation, temperature=0.0)
        return rotating, rotating.model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def analyze(
        self,
        text: str,
        *,
        max_sentences: int | None = None,
    ) -> list[CitationNeedResult]:
        """Return a binary citation-need decision for each evaluated sentence.

        Already-cited and too-short sentences are filtered out before any API
        call and are absent from the result.
        """
        limit = max_sentences if max_sentences is not None else self._max_sentences
        spans = iter_sentence_spans(text, max_sentences=limit)

        results: list[CitationNeedResult] = []
        for _, paragraph_iter in groupby(spans, key=lambda s: s.paragraph_index):
            results.extend(self._analyze_paragraph(list(paragraph_iter)))
        return results

    def classify(
        self,
        queries: Sequence[CitationNeedQuery],
        *,
        batch_size: int = 10,
    ) -> list[CitationNeedJudgement]:
        """Classify independent target sentences, several per LLM call.

        Returns one judgement per query, in order. Targets that fail the local
        prefilter (already cited or too short) are decided without an API call;
        the rest are grouped into batches of ``batch_size`` to keep the request
        count low. This is the evaluation path; production uses :meth:`analyze`.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        judgements: list[CitationNeedJudgement | None] = [None] * len(queries)
        candidates: list[tuple[int, CitationNeedQuery]] = []
        for position, query in enumerate(queries):
            if self._target_is_candidate(query.target):
                candidates.append((position, query))
            else:
                judgements[position] = CitationNeedJudgement(
                    needs_citation=False, confidence=0.0, sent=False, answered=False
                )

        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start : start + batch_size]
            decisions = self._classify_chunk([query for _, query in chunk])
            for local_index, (position, _query) in enumerate(chunk):
                decision = decisions.get(local_index)
                judgements[position] = CitationNeedJudgement(
                    needs_citation=decision.needs_citation if decision else False,
                    confidence=decision.confidence if decision else 0.0,
                    sent=True,
                    answered=decision is not None,
                )

        return [j for j in judgements if j is not None]

    def _classify_chunk(
        self, queries: list[CitationNeedQuery]
    ) -> dict[int, _Decision]:
        user_prompt = self._build_batch_prompt(queries)
        try:
            raw = self._client.complete(
                system=BATCH_SYSTEM_PROMPT,
                user=user_prompt,
                response_mime_type=_JSON_MIME,
            )
        except Exception:  # noqa: BLE001 - one bad batch must not kill the run
            logger.warning(
                "citation-need batch call failed for %d items; skipping",
                len(queries),
                exc_info=True,
            )
            return {}
        return self._parse_decisions(raw)

    @staticmethod
    def _build_batch_prompt(queries: list[CitationNeedQuery]) -> str:
        blocks: list[str] = []
        for index, query in enumerate(queries):
            lines = [f"### Item {index}"]
            if query.previous:
                lines.append(f"Before: {query.previous}")
            lines.append(f"TARGET: {query.target}")
            if query.next:
                lines.append(f"After: {query.next}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _target_is_candidate(self, target: str) -> bool:
        if has_explicit_citation(target):
            return False
        return len(_WORD_PATTERN.findall(target)) >= self._min_words

    def _analyze_paragraph(
        self, spans: list[SentenceSpan]
    ) -> list[CitationNeedResult]:
        candidate_ids = {
            span.sentence_id for span in spans if self._is_candidate(span)
        }
        if not candidate_ids:
            return []

        user_prompt, index_map = self._build_user_prompt(spans, candidate_ids)
        try:
            raw = self._client.complete(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                response_mime_type=_JSON_MIME,
            )
        except Exception:  # noqa: BLE001 - one bad paragraph must not kill the doc
            logger.warning(
                "citation-need LLM call failed for paragraph %s; skipping",
                spans[0].paragraph_index,
                exc_info=True,
            )
            return []

        logger.debug("citation-need raw response: %s", raw)
        decisions = self._parse_decisions(raw)
        if not decisions:
            logger.warning(
                "citation-need reply had no usable decisions for paragraph %s; "
                "skipping",
                spans[0].paragraph_index,
            )
            return []

        results: list[CitationNeedResult] = []
        for index, span in index_map.items():
            decision = decisions.get(index)
            results.append(
                CitationNeedResult(
                    sentence_id=span.sentence_id,
                    text=span.text,
                    start_offset=span.start_offset,
                    end_offset=span.end_offset,
                    paragraph_index=span.paragraph_index,
                    section_type=span.section_type,
                    needs_citation=decision.needs_citation if decision else False,
                    confidence=decision.confidence if decision else 0.0,
                )
            )
        return results

    def _is_candidate(self, span: SentenceSpan) -> bool:
        return self._target_is_candidate(span.text)

    def _build_user_prompt(
        self,
        spans: list[SentenceSpan],
        candidate_ids: set[str],
    ) -> tuple[str, dict[int, SentenceSpan]]:
        """Render the paragraph with candidate sentences tagged ``[i]``.

        Non-candidate sentences stay in the text (unindexed) so the model still
        sees the full narrative, but no decision is requested for them.
        """
        parts: list[str] = []
        index_map: dict[int, SentenceSpan] = {}
        index = 0
        for span in spans:
            if span.sentence_id in candidate_ids:
                parts.append(f"[{index}] {span.text}")
                index_map[index] = span
                index += 1
            else:
                parts.append(span.text)
        return " ".join(parts), index_map

    def _parse_decisions(self, raw: str) -> dict[int, _Decision]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("citation-need reply was not valid JSON: %r", raw[:200])
            return {}

        decisions: dict[int, _Decision] = {}
        for item in _coerce_to_list(payload):
            try:
                decision = _Decision.model_validate(item)
            except ValidationError:
                logger.warning("citation-need item failed validation: %r", item)
                continue
            decisions[decision.index] = decision
        return decisions


def _coerce_to_list(payload: object) -> list[object]:
    """Accept a bare JSON array or a ``{"...": [...]}`` wrapper."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                return value
    return []


_identifier: CitationNeedIdentifier | None = None


def get_identifier() -> CitationNeedIdentifier:
    """Return the process-wide identifier singleton (lazily constructed)."""
    global _identifier
    if _identifier is None:
        _identifier = CitationNeedIdentifier()
    return _identifier
