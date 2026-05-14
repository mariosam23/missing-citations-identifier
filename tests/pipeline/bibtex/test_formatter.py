"""Unit tests for ``pipeline.bibtex.formatter``.

Covers entry-type detection, author formatting, LaTeX escaping, and
full BibTeX entry generation with snapshot comparisons.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from pipeline.bibtex.formatter import (
    detect_entry_type,
    escape_latex,
    format_authors,
    paper_to_bibtex,
)


def _make_paper(
    *,
    canonical_title: str = "Untitled",
    authors: dict[str, Any] | None = None,
    first_author: str | None = None,
    year: int | None = None,
    venue: str | None = None,
    doi: str | None = None,
    arxiv_id: str | None = None,
    url: str | None = None,
    source: str | None = None,
) -> Any:
    """Create a lightweight Paper-like object for testing."""
    return SimpleNamespace(
        canonical_title=canonical_title,
        authors=authors,
        first_author=first_author,
        year=year,
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
        source=source,
    )


# ── detect_entry_type ─────────────────────────────────────────────────


class TestDetectEntryType:
    def test_proceedings_venue(self) -> None:
        paper = _make_paper(venue="Advances in Neural Information Processing Systems")
        assert detect_entry_type(paper) == "inproceedings"

    def test_conference_venue(self) -> None:
        paper = _make_paper(venue="International Conference on Machine Learning")
        assert detect_entry_type(paper) == "inproceedings"

    def test_workshop_venue(self) -> None:
        paper = _make_paper(venue="Workshop on Representation Learning")
        assert detect_entry_type(paper) == "inproceedings"

    def test_journal_venue(self) -> None:
        paper = _make_paper(venue="Journal of Machine Learning Research")
        assert detect_entry_type(paper) == "article"

    def test_transactions_venue(self) -> None:
        paper = _make_paper(venue="IEEE Transactions on Neural Networks")
        assert detect_entry_type(paper) == "article"

    def test_arxiv_only_misc(self) -> None:
        paper = _make_paper(venue=None, arxiv_id="2301.01234", source="arxiv")
        assert detect_entry_type(paper) == "misc"

    def test_arxiv_from_source(self) -> None:
        paper = _make_paper(venue=None, arxiv_id=None, source="arxiv")
        assert detect_entry_type(paper) == "misc"

    def test_unknown_venue_defaults_article(self) -> None:
        paper = _make_paper(venue="AAAI")
        assert detect_entry_type(paper) == "article"

    def test_no_venue_no_arxiv_defaults_misc(self) -> None:
        paper = _make_paper(venue=None, source=None)
        assert detect_entry_type(paper) == "misc"


# ── format_authors ────────────────────────────────────────────────────


class TestFormatAuthors:
    def test_openalex_display_name_list(self) -> None:
        paper = _make_paper(
            authors={
                "list": [
                    {"name": "Ashish Vaswani"},
                    {"name": "Noam Shazeer"},
                ]
            },
            first_author="Vaswani",
        )
        result = format_authors(paper)
        assert result == "Vaswani, Ashish and Shazeer, Noam"

    def test_family_given_shape(self) -> None:
        paper = _make_paper(
            authors={
                "list": [
                    {"family": "Mikolov", "given": "Tomas"},
                    {"family": "Chen", "given": "Kai"},
                ]
            },
            first_author="Mikolov",
        )
        result = format_authors(paper)
        assert result == "Mikolov, Tomas and Chen, Kai"

    def test_already_bibtex_order(self) -> None:
        paper = _make_paper(
            authors={"list": ["Vaswani, Ashish"]},
            first_author="Vaswani",
        )
        result = format_authors(paper)
        assert result == "Vaswani, Ashish"

    def test_single_name_token(self) -> None:
        paper = _make_paper(
            authors={"list": ["Madonna"]},
            first_author="Madonna",
        )
        result = format_authors(paper)
        assert result == "Madonna"

    def test_fallback_to_first_author(self) -> None:
        paper = _make_paper(authors=None, first_author="Vaswani")
        result = format_authors(paper)
        assert result == "Vaswani"

    def test_anonymous_fallback(self) -> None:
        paper = _make_paper(authors=None, first_author=None)
        result = format_authors(paper)
        assert result == "Anonymous"


# ── escape_latex ──────────────────────────────────────────────────────


class TestEscapeLatex:
    def test_ampersand(self) -> None:
        assert "&" not in escape_latex("Foo & Bar") or r"\&" in escape_latex(
            "Foo & Bar"
        )

    def test_accented_character(self) -> None:
        result = escape_latex("Ré")
        # pylatexenc should convert é → some LaTeX escape
        assert "é" not in result or "\\" in result


# ── paper_to_bibtex full entry ────────────────────────────────────────


class TestPaperToBibtex:
    def test_inproceedings_vaswani(self) -> None:
        paper = _make_paper(
            canonical_title="Attention is All You Need",
            authors={
                "list": [
                    {"name": "Ashish Vaswani"},
                    {"name": "Noam Shazeer"},
                    {"name": "Niki Parmar"},
                ]
            },
            first_author="Vaswani",
            year=2017,
            venue="Advances in Neural Information Processing Systems",
            doi="10.5555/3295222.3295349",
            url="https://arxiv.org/abs/1706.03762",
        )
        result = paper_to_bibtex(paper, "vaswani2017attention")
        assert result.startswith("@inproceedings{vaswani2017attention,")
        assert "title = {Attention is All You Need}" in result
        assert "author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki}" in result
        assert "year = {2017}" in result
        assert "booktitle = {Advances in Neural Information Processing Systems}" in result
        assert "doi = {10.5555/3295222.3295349}" in result

    def test_misc_arxiv(self) -> None:
        paper = _make_paper(
            canonical_title="Efficient Estimation of Word Representations",
            authors={
                "list": [
                    {"name": "Tomas Mikolov"},
                    {"name": "Kai Chen"},
                ]
            },
            first_author="Mikolov",
            year=2013,
            venue=None,
            arxiv_id="1301.3781",
            source="arxiv",
        )
        result = paper_to_bibtex(paper, "mikolov2013efficient")
        assert result.startswith("@misc{mikolov2013efficient,")
        assert "eprint = {1301.3781}" in result
        assert "archivePrefix = {arXiv}" in result

    def test_article_journal(self) -> None:
        paper = _make_paper(
            canonical_title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors={"list": [{"name": "Jacob Devlin"}]},
            first_author="Devlin",
            year=2019,
            venue="Journal of Machine Learning Research",
            doi="10.1234/test",
        )
        result = paper_to_bibtex(paper, "devlin2019bert")
        assert result.startswith("@article{devlin2019bert,")
        assert "journal = {Journal of Machine Learning Research}" in result

    def test_special_characters_in_title(self) -> None:
        paper = _make_paper(
            canonical_title='Foo & Bar: A "Tést" with 100% Special #Chars',
            authors={"list": [{"name": "Test Author"}]},
            first_author="Author",
            year=2024,
        )
        result = paper_to_bibtex(paper, "author2024foo")
        # Should not contain raw & or # (LaTeX-significant)
        title_line = [ln for ln in result.splitlines() if "title" in ln][0]
        # The raw & and # should be escaped
        assert " & " not in title_line
        assert " #" not in title_line

    def test_arxiv_url_stripped_from_eprint(self) -> None:
        paper = _make_paper(
            canonical_title="Test",
            authors={"list": [{"name": "A B"}]},
            first_author="B",
            year=2023,
            arxiv_id="https://arxiv.org/abs/2301.01234",
            source="arxiv",
        )
        result = paper_to_bibtex(paper, "b2023test")
        assert "eprint = {2301.01234}" in result

    def test_no_year(self) -> None:
        paper = _make_paper(
            canonical_title="No Year Paper",
            authors={"list": [{"name": "Jane Doe"}]},
            first_author="Doe",
            year=None,
        )
        result = paper_to_bibtex(paper, "doeundated")
        assert "year" not in result

    def test_entry_ends_with_newline(self) -> None:
        paper = _make_paper(
            canonical_title="Test",
            authors={"list": [{"name": "A B"}]},
            first_author="B",
            year=2023,
        )
        result = paper_to_bibtex(paper, "b2023test")
        assert result.endswith("}\n")

    def test_key_sanitisation(self) -> None:
        """Keys with accented or special chars are cleaned."""
        paper = _make_paper(
            canonical_title="Test",
            authors={"list": [{"name": "A B"}]},
            first_author="B",
            year=2023,
        )
        result = paper_to_bibtex(paper, "müller2023über")
        assert "@misc{muller2023uber," in result
