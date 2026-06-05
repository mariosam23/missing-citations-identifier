"""Unit tests for the pre-LLM input-sanitization filter (no network)."""

from __future__ import annotations

import pytest

from pipeline.citation_need.sanitize import clean_for_llm, is_classifiable


@pytest.mark.parametrize(
    ("raw", "expected_absent"),
    [
        # Orphaned close bracket left by a stripped marker.
        ("related to inductive rule learning algorithms ], or mainly on", " ]"),
        # Dangling connective pointing at a removed marker.
        ("the most cited survey on concept drift was published back in 2004 in ].", "in ]"),
        # Dangling open "(e.g.,".
        ("emergence of born globals in numerous nations (e.g., indicate that", "(e.g.,"),
    ],
)
def test_clean_removes_citation_residue(raw: str, expected_absent: str) -> None:
    cleaned = clean_for_llm(raw)
    assert expected_absent not in cleaned
    # Real words survive.
    assert "learning" in cleaned or "drift" in cleaned or "globals" in cleaned


def test_clean_strips_leading_figure_token() -> None:
    cleaned = clean_for_llm("2c Interestingly, the PPF demonstrated two timescales.")
    assert cleaned.startswith("Interestingly")
    assert "2c" not in cleaned.split()[0]


def test_clean_collapses_whitespace_and_space_before_punct() -> None:
    assert clean_for_llm("alpha   beta  gamma  .") == "alpha beta gamma."


def test_clean_is_idempotent() -> None:
    raw = "techniques ], used widely in the field of study today."
    once = clean_for_llm(raw)
    assert clean_for_llm(once) == once


def test_clean_preserves_real_parentheticals() -> None:
    raw = "The model (a deep network) outperforms the baseline by a wide margin."
    assert clean_for_llm(raw) == raw


def test_is_classifiable_accepts_real_prose() -> None:
    assert is_classifiable(
        "Distributional word vectors define the compositional similarity of strings."
    )


def test_is_classifiable_rejects_numbered_list_item() -> None:
    assert not is_classifiable(
        "3. Subsample the remaining libraries without replacement to size N."
    )


def test_is_classifiable_rejects_too_short() -> None:
    assert not is_classifiable("Too short here.")


def test_is_classifiable_rejects_symbol_heavy_line() -> None:
    assert not is_classifiable("x = 2 * (a + b) / (c - d) ^ 2 + 3 * e - f ...")


def test_is_classifiable_rejects_all_caps_fragment() -> None:
    # No lowercase word -> looks like a heading/label, not prose.
    assert not is_classifiable("TABLE OF RESULTS FOR ALL EXPERIMENTAL CONDITIONS")


def test_is_classifiable_respects_min_words_override() -> None:
    assert is_classifiable("alpha beta gamma", min_words=3)
    assert not is_classifiable("alpha beta gamma", min_words=4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
