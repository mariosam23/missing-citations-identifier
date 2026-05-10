import json
import re
import time
from dataclasses import replace

from utils import logger

from llm.genai_client import LLMClient
from prompts import CLASSIFIER_SYSTEM_PROMPT, CLASSIFIER_USER_PROMPT_TEMPLATE
from entities import SentenceRecord, CitationIntent
from utils.config import config


INTENT_MAP = {
    "BACKGROUND": CitationIntent.BACKGROUND,
    "METHOD": CitationIntent.METHOD,
    "RESULT": CitationIntent.RESULT,
    "OTHER": CitationIntent.OTHER,
}

class GeminiClassifier:
    def __init__(
        self,
        model: str | None = None,
        batch_size: int = 10,
        delay_between_calls_seconds: float = 30.0,
        system_prompt: str | None = None,
    ):
        model = model or config.CLASSIFIER_MODEL
        self.client = LLMClient(model=model, temperature=0.1, max_tokens=8000)
        self.batch_size = batch_size
        self.delay_between_calls_seconds = delay_between_calls_seconds
        self.system_prompt = system_prompt or CLASSIFIER_SYSTEM_PROMPT

    def classify_sentences(self, sentences: list[SentenceRecord], paper_title: str, paper_abstract: str) -> list[SentenceRecord]:
        """Classify a list of sentences for citation worthiness and intent.

        Returns a new list of `SentenceRecord`s with classification fields
        populated. The input list and its elements are not modified.
        """
        if not sentences:
            logger.info("No sentences require classification (list is empty).")
            return list(sentences)

        updated: list[SentenceRecord] = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            batch_number = (i // self.batch_size) + 1
            total_batches = (len(sentences) + self.batch_size - 1) // self.batch_size
            logger.info("Sending batch %d/%d to Gemini...", batch_number, total_batches)
            updated.extend(self._classify_batch(batch, paper_title, paper_abstract))

            if i + self.batch_size < len(sentences) and self.delay_between_calls_seconds > 0:
                logger.info(
                    "Waiting %d seconds before next Gemini API call...", int(self.delay_between_calls_seconds)
                )
                time.sleep(self.delay_between_calls_seconds)
        return updated

    def _classify_batch(self, batch: list[SentenceRecord], paper_title: str, paper_abstract: str) -> list[SentenceRecord]:
        """Classify a batch of sentences using the LLM and return updated copies."""
        try:
            classifications = self._request_batch_classification(batch, paper_title, paper_abstract)
        except ValueError as exc:
            if len(batch) == 1:
                raise

            split_point = max(1, len(batch) // 2)
            logger.warning(
                "Batch of %d sentences failed (%s). Retrying as chunks of %d and %d...",
                len(batch), exc, split_point, len(batch) - split_point
            )
            first = self._classify_batch(batch[:split_point], paper_title, paper_abstract)
            second = self._classify_batch(batch[split_point:], paper_title, paper_abstract)
            return first + second

        return self._apply_classifications(batch, classifications)

    def _request_batch_classification(
        self,
        batch: list[SentenceRecord],
        paper_title: str,
        paper_abstract: str
    ) -> list[dict]:
        """Request classifications for a batch and validate the response shape."""
        sentences_text = "\n".join(f"{j}. {s.text}" for j, s in enumerate(batch))

        user_prompt = CLASSIFIER_USER_PROMPT_TEMPLATE.format(
            title=paper_title,
            abstract=paper_abstract,
            sentences=sentences_text,
        )

        response = self.client.complete(
            self.system_prompt,
            user_prompt,
            response_mime_type="application/json",
        )
        logger.debug("Gemini raw response:\n%s", response)

        classifications = self._parse_classifications(response)
        self._validate_classifications(classifications, len(batch))
        return classifications

    @staticmethod
    def _apply_classifications(
        batch: list[SentenceRecord], classifications: list[dict]
    ) -> list[SentenceRecord]:
        """Return new sentence records with parsed classification fields applied."""
        from entities.sentence_record import CitationState
        STATE_MAP = {
            "MISSING_CITATION": CitationState.MISSING_CITATION,
            "COVERED_BY_BLOCK": CitationState.COVERED_BY_BLOCK,
            "HAS_CITATION": CitationState.HAS_CITATION,
            "NOT_CITATION_WORTHY": CitationState.NOT_CITATION_WORTHY,
        }

        updates: dict[int, dict] = {}
        for cls in classifications:
            idx = cls["sentence_index"]
            state_str = cls.get("citation_state", "NOT_CITATION_WORTHY")
            intent_str = cls.get("citation_intent", "OTHER")
            urgency = float(cls.get("urgency_of_citation", 0.5))
            updates[idx] = {
                "citation_state": STATE_MAP.get(state_str, CitationState.NOT_CITATION_WORTHY),
                "citation_intent": INTENT_MAP.get(intent_str),
                "worthiness_score": urgency,
            }

        return [replace(sentence, **updates[i]) for i, sentence in enumerate(batch)]

    @staticmethod
    def _validate_classifications(classifications: list[dict], expected_count: int) -> None:
        """Ensure the model returned one well-formed classification per sentence."""
        if len(classifications) != expected_count:
            raise ValueError(
                f"Gemini returned {len(classifications)} classifications for "
                f"{expected_count} sentences."
            )

        seen_indices: set[int] = set()
        for cls in classifications:
            idx = cls.get("sentence_index")
            if not isinstance(idx, int):
                raise ValueError(f"Invalid sentence_index in Gemini response: {cls}")
            if idx < 0 or idx >= expected_count:
                raise ValueError(
                    f"Gemini returned out-of-range sentence_index {idx} "
                    f"for batch size {expected_count}."
                )
            if idx in seen_indices:
                raise ValueError(f"Gemini returned duplicate sentence_index {idx}.")
            seen_indices.add(idx)

        missing_indices = sorted(set(range(expected_count)) - seen_indices)
        if missing_indices:
            raise ValueError(f"Gemini omitted sentence_index values: {missing_indices}")

    @staticmethod
    def _parse_classifications(response: str) -> list[dict]:
        """Parse Gemini output, tolerating fenced JSON while surfacing truncation clearly."""
        cleaned = response.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        truncation_error = ValueError(
            "Gemini returned a truncated JSON array. "
            "Reduce batch size or response length."
        )

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as outer_exc:
            json_match = re.search(r"\[[\s\S]*\]", cleaned)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                except json.JSONDecodeError as inner_exc:
                    if GeminiClassifier._is_truncation_error(inner_exc, json_match.group()):
                        raise truncation_error from inner_exc
                    raise ValueError(
                        "Gemini returned malformed JSON. "
                        "Reduce batch size or response length."
                    ) from inner_exc
            elif GeminiClassifier._is_truncation_error(outer_exc, cleaned):
                raise truncation_error from outer_exc
            else:
                raise ValueError(f"Could not parse LLM response as JSON: {response}") from outer_exc

        if not isinstance(parsed, list):
            raise ValueError(f"Expected a JSON array from Gemini, got: {type(parsed).__name__}")

        return parsed

    @staticmethod
    def _is_truncation_error(exc: json.JSONDecodeError, source: str) -> bool:
        """Distinguish end-of-input truncation from a syntactic error mid-document.

        ``json`` reports the failure offset in ``exc.pos``; if it lands at (or
        adjacent to) the end of the input, or the message names an unterminated
        token, the response was almost certainly cut off rather than malformed.
        """
        if "Unterminated" in exc.msg:
            return True
        return exc.pos >= len(source.rstrip()) - 1
