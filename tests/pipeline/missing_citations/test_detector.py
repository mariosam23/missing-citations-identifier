from __future__ import annotations

from pipeline.missing_citations.detector import (
    CitationNeedLabel,
    has_explicit_citation,
    iter_sentence_spans,
    scan_document,
)


def test_sentence_offsets_preserve_original_text() -> None:
    text = "Intro text.\n\nPrevious work has been shown to improve BERT."

    spans = iter_sentence_spans(text)

    assert len(spans) == 2
    for span in spans:
        assert text[span.start_offset : span.end_offset] == span.text


def test_sentence_splitter_handles_multiple_sentences_in_one_paragraph() -> None:
    text = "First sentence. Previous work has been shown to improve BERT."

    spans = iter_sentence_spans(text)

    assert [span.text for span in spans] == [
        "First sentence.",
        "Previous work has been shown to improve BERT.",
    ]


def test_detects_latex_markdown_and_author_year_citations() -> None:
    assert has_explicit_citation("Transformers are effective \\citep{vaswani2017}.")
    assert has_explicit_citation("Transformers are effective [@vaswani2017].")
    assert has_explicit_citation("Transformers are effective (Vaswani et al., 2017).")


def test_explicit_citation_precedes_missing_citation_cues() -> None:
    detections = scan_document(
        "Previous work has been shown to improve Transformer models \\cite{foo}."
    )

    assert detections[0].label == CitationNeedLabel.HAS_CITATION


def test_own_contribution_sentence_is_suppressed() -> None:
    detections = scan_document(
        "We propose a retrieval pipeline and evaluate it on thesis examples."
    )

    assert detections[0].label == CitationNeedLabel.NOT_CITATION_WORTHY


def test_prior_work_cue_promotes_missing_citation() -> None:
    detections = scan_document(
        "Previous work has been shown to improve Transformer pretraining."
    )

    assert detections[0].label == CitationNeedLabel.MISSING_CITATION


def test_named_model_with_sota_claim_needs_citation() -> None:
    detections = scan_document(
        "The model known as BERT has achieved state-of-the-art results "
        "in many NLP tasks."
    )

    assert detections[0].label == CitationNeedLabel.MISSING_CITATION
    assert detections[0].confidence >= 0.55


def test_latex_document_extracts_prose_sentence() -> None:
    doc = r"""\documentclass{article}
\usepackage{hyperref}
\begin{document}
\section*{Introduction}
The model known as BERT has achieved state-of-the-art results in many NLP tasks.
\end{document}
"""
    spans = iter_sentence_spans(doc)
    bert_spans = [span for span in spans if "BERT" in span.text]

    assert len(bert_spans) == 1
    assert bert_spans[0].text == (
        "The model known as BERT has achieved state-of-the-art results "
        "in many NLP tasks."
    )
    assert bert_spans[0].section_type == "introduction"

    detections = scan_document(doc)
    missing = [
        detection
        for detection in detections
        if detection.label == CitationNeedLabel.MISSING_CITATION
    ]
    assert any("BERT" in detection.sentence.text for detection in missing)


def test_latex_offsets_survive_crlf_line_wraps() -> None:
    # Windows-authored .tex sends CRLF; a hard-wrapped sentence spans two lines.
    doc = (
        "\\documentclass{article}\r\n"
        "\\begin{document}\r\n"
        "\\section{Background}\r\n"
        "Previous work has been shown to improve\r\n"
        "Transformer pretraining across many tasks.\r\n"
        "\\end{document}\r\n"
    )

    span = next(s for s in iter_sentence_spans(doc) if "Transformer" in s.text)

    # The reported offsets must address the real sentence — terminal period
    # included — not a copy that has drifted by one char per CRLF break.
    assert doc[span.start_offset : span.end_offset] == span.text
    assert span.text == (
        "Previous work has been shown to improve\r\n"
        "Transformer pretraining across many tasks."
    )


def test_latex_offsets_survive_indented_wraps() -> None:
    doc = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\section{Background}\n"
        "Previous work has been shown to improve\n"
        "    Transformer pretraining across many tasks.\n"
        "\\end{document}\n"
    )

    span = next(s for s in iter_sentence_spans(doc) if "Transformer" in s.text)

    assert doc[span.start_offset : span.end_offset] == span.text
    assert span.text == (
        "Previous work has been shown to improve\n"
        "    Transformer pretraining across many tasks."
    )


def test_uncited_continuation_near_cited_sentence_is_covered_by_block() -> None:
    detections = scan_document(
        "Transformers are common in NLP \\cite{vaswani2017}. "
        "The model is efficient on several downstream tasks."
    )

    assert detections[0].label == CitationNeedLabel.HAS_CITATION
    assert detections[1].label == CitationNeedLabel.COVERED_BY_BLOCK


def test_strong_prior_work_cue_overrides_block_coverage() -> None:
    detections = scan_document(
        "Transformers are common in NLP \\cite{vaswani2017}. "
        "Previous work has been shown to improve Transformer pretraining."
    )

    assert detections[0].label == CitationNeedLabel.HAS_CITATION
    assert detections[1].label == CitationNeedLabel.MISSING_CITATION
