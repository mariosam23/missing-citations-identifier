"""Unit tests for the binary LLM citation-need identifier (no network).

A fake client stands in for Gemini so the tests exercise segmentation, the
prefilter, offset integrity, and defensive JSON parsing deterministically.
"""

from __future__ import annotations

import json

import pytest

from pipeline.citation_need.identifier import (
    CitationNeedIdentifier,
    CitationNeedQuery,
)


def _reply(*needs: bool) -> str:
    """Build a batch JSON reply with one object per item index."""
    return json.dumps(
        [
            {"index": i, "needs_citation": n, "confidence": 0.9}
            for i, n in enumerate(needs)
        ]
    )


class FakeClient:
    """Returns canned JSON replies in order and records the prompts it saw."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def complete(
        self,
        system: str,
        user: str,
        response_mime_type: str | None = None,
    ) -> str:
        self.prompts.append(user)
        if not self._responses:
            raise AssertionError("unexpected extra LLM call")
        return self._responses.pop(0)


def _identifier(*responses: str) -> tuple[CitationNeedIdentifier, FakeClient]:
    client = FakeClient(*responses)
    return CitationNeedIdentifier(client=client), client


def test_offsets_are_exact_slices_of_the_document() -> None:
    doc = "Alpha beta gamma delta epsilon. Zeta eta theta iota kappa."
    identifier, _ = _identifier(
        '[{"index": 0, "needs_citation": true, "confidence": 0.8},'
        ' {"index": 1, "needs_citation": false, "confidence": 0.3}]'
    )

    results = identifier.analyze(doc)

    assert len(results) == 2
    for result in results:
        assert doc[result.start_offset : result.end_offset] == result.text
    assert results[0].needs_citation is True
    assert results[1].needs_citation is False


def test_already_cited_sentence_is_never_sent_to_the_llm() -> None:
    doc = (
        "We build on prior work \\cite{smith2020}. "
        "The method improves accuracy by ten points."
    )
    identifier, client = _identifier(
        '[{"index": 0, "needs_citation": true, "confidence": 0.9}]'
    )

    results = identifier.analyze(doc)

    # Only the uncited sentence is evaluated and returned.
    assert len(results) == 1
    assert results[0].text.startswith("The method improves")
    # The cited sentence appears as context but is not tagged for a decision.
    prompt = client.prompts[0]
    assert "[0] The method improves" in prompt
    assert "[1]" not in prompt
    assert "\\cite{smith2020}" in prompt


def test_short_sentences_are_filtered_before_any_call() -> None:
    identifier, client = _identifier()  # no responses => no call expected

    results = identifier.analyze("Too short here.")

    assert results == []
    assert client.prompts == []


def test_malformed_json_does_not_crash_the_document() -> None:
    doc = "This established result rests on earlier published findings clearly."
    identifier, _ = _identifier("this is not json")

    assert identifier.analyze(doc) == []


def test_wrapped_list_and_out_of_range_confidence_are_handled() -> None:
    doc = "This established result rests on earlier published findings clearly."
    identifier, _ = _identifier(
        '{"decisions": [{"index": 0, "needs_citation": true,'
        ' "confidence": 1.7}]}'
    )

    results = identifier.analyze(doc)

    assert len(results) == 1
    assert results[0].confidence == 1.0


def test_missing_index_defaults_to_no_citation() -> None:
    doc = "Alpha beta gamma delta epsilon. Zeta eta theta iota kappa."
    identifier, _ = _identifier(
        '[{"index": 0, "needs_citation": true, "confidence": 0.7}]'
    )

    results = identifier.analyze(doc)

    assert len(results) == 2
    assert results[0].needs_citation is True
    assert results[1].needs_citation is False
    assert results[1].confidence == 0.0


def test_each_paragraph_is_one_call() -> None:
    doc = (
        "Para one sentence alpha beta gamma.\n\n"
        "Para two sentence delta epsilon zeta."
    )
    identifier, client = _identifier(
        '[{"index": 0, "needs_citation": true, "confidence": 0.6}]',
        '[{"index": 0, "needs_citation": false, "confidence": 0.6}]',
    )

    results = identifier.analyze(doc)

    assert len(client.prompts) == 2
    assert len(results) == 2
    assert results[0].paragraph_index != results[1].paragraph_index
    assert results[0].needs_citation is True
    assert results[1].needs_citation is False


def test_classify_maps_decisions_in_order() -> None:
    identifier, client = _identifier(_reply(True, False, True))
    queries = [
        CitationNeedQuery("Prior work reports a large effect on accuracy here."),
        CitationNeedQuery("In this paper we propose a new lighter variant model."),
        CitationNeedQuery("BERT is widely used across many language understanding."),
    ]

    judgements = identifier.classify(queries, batch_size=10)

    assert [j.needs_citation for j in judgements] == [True, False, True]
    assert all(j.sent and j.answered for j in judgements)
    assert len(client.prompts) == 1  # all three fit in one call


def test_classify_prefilters_cited_and_short_without_calling() -> None:
    identifier, client = _identifier(_reply(True))
    queries = [
        CitationNeedQuery("This established result rests on earlier findings now."),
        CitationNeedQuery("We build on prior work \\cite{smith2020} directly here."),
        CitationNeedQuery("Too short indeed."),
    ]

    judgements = identifier.classify(queries, batch_size=10)

    # Only the first query is a candidate; the prompt holds exactly one item.
    assert len(client.prompts) == 1
    assert "### Item 0" in client.prompts[0]
    assert "### Item 1" not in client.prompts[0]
    assert judgements[0].sent and judgements[0].answered
    assert judgements[0].needs_citation is True
    # Cited + short are skipped: not sent, not flagged.
    assert judgements[1].sent is False and judgements[1].needs_citation is False
    assert judgements[2].sent is False and judgements[2].needs_citation is False


def test_classify_splits_into_batches() -> None:
    identifier, client = _identifier(
        _reply(*([True] * 10)),
        _reply(*([True] * 10)),
        _reply(*([True] * 5)),
    )
    queries = [
        CitationNeedQuery(f"Prior work number {i} reports an important finding now.")
        for i in range(25)
    ]

    judgements = identifier.classify(queries, batch_size=10)

    assert len(client.prompts) == 3  # 10 + 10 + 5
    assert len(judgements) == 25
    assert all(j.answered for j in judgements)


def test_classify_failed_batch_marks_items_sent_but_unanswered() -> None:
    identifier, _ = _identifier("not valid json at all")
    queries = [
        CitationNeedQuery("Prior work reports a substantial improvement in scores.")
    ]

    judgements = identifier.classify(queries, batch_size=10)

    assert judgements[0].sent is True
    assert judgements[0].answered is False
    assert judgements[0].needs_citation is False


def test_classify_missing_index_is_unanswered() -> None:
    identifier, _ = _identifier(_reply(True))  # only index 0 returned
    queries = [
        CitationNeedQuery("Prior work reports a large measured effect on accuracy."),
        CitationNeedQuery("Existing methods are commonly evaluated on this benchmark."),
    ]

    judgements = identifier.classify(queries, batch_size=10)

    assert judgements[0].answered is True and judgements[0].needs_citation is True
    assert judgements[1].sent is True and judgements[1].answered is False
    assert judgements[1].needs_citation is False


def test_classify_rejects_bad_batch_size() -> None:
    identifier, _ = _identifier()
    queries = [CitationNeedQuery("anything at all here please now")]
    with pytest.raises(ValueError, match="batch_size"):
        identifier.classify(queries, batch_size=0)


# --- input-sanitization filter -------------------------------------------------


def test_analyze_sends_cleaned_text_but_keeps_original_offsets() -> None:
    # Trailing "in ]" is citation-stripping residue: cleaned out of the prompt,
    # but the returned span must stay an exact slice of the original document.
    doc = "The established result rests on earlier published findings in 2004 in ]."
    identifier, client = _identifier(
        '[{"index": 0, "needs_citation": true, "confidence": 0.8}]'
    )

    results = identifier.analyze(doc)

    assert len(results) == 1
    assert doc[results[0].start_offset : results[0].end_offset] == results[0].text
    assert "]" in results[0].text  # original residue preserved in the result
    assert "in ]" not in client.prompts[0]  # but the LLM saw cleaned prose


def test_sanitizer_off_reproduces_unfiltered_prompt() -> None:
    doc = "The established result rests on earlier published findings in 2004 in ]."
    client = FakeClient(
        '[{"index": 0, "needs_citation": true, "confidence": 0.8}]'
    )
    identifier = CitationNeedIdentifier(client=client, enable_sanitizer=False)

    identifier.analyze(doc)

    assert "in ]" in client.prompts[0]


def test_classify_filter_skips_list_item_baseline_sends_it() -> None:
    query = CitationNeedQuery(
        "3. Subsample the remaining libraries without replacement to size N."
    )

    on_client = FakeClient()  # no responses => no call expected
    on = CitationNeedIdentifier(client=on_client, enable_sanitizer=True)
    on_judgements = on.classify([query], batch_size=10)
    assert on_client.prompts == []
    assert on_judgements[0].sent is False
    assert on_judgements[0].needs_citation is False

    off_client = FakeClient(_reply(False))
    off = CitationNeedIdentifier(client=off_client, enable_sanitizer=False)
    off_judgements = off.classify([query], batch_size=10)
    assert len(off_client.prompts) == 1
    assert off_judgements[0].sent is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
