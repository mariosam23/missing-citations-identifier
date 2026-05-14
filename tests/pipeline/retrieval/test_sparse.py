"""Integration tests for ``pipeline.retrieval.sparse``.

Needs a live Postgres with the Phase 6 sparse schema applied (the ``pg_session``
fixture skips otherwise). Each test inserts a handful of citation contexts in a
transaction that is rolled back at teardown, so nothing persists.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from pipeline.retrieval.dense import ContextSource
from pipeline.retrieval.sparse import retrieve_sparse

# High bases keep synthetic rows clear of real corpus IDs even mid-transaction.
_PID_BASE = 90_000_000
_CID_BASE = 90_000_000

_SENTENCES: tuple[str, ...] = (
    "We fine-tune the model with LoRA for parameter-efficient adaptation.",
    "Word embeddings capture the distributional semantics of tokens.",
    "The transformer architecture relies on multi-head self-attention.",
    "We evaluate on the GLUE benchmark across nine language tasks.",
    "Reinforcement learning from human feedback aligns the policy with intent.",
)


def _insert_paper(session: Session, paper_id: int, title: str) -> None:
    session.execute(
        text(
            "INSERT INTO papers (paper_id, canonical_title, normalized_title) "
            "VALUES (:pid, :title, :ntitle)"
        ),
        {"pid": paper_id, "title": title, "ntitle": title.lower()},
    )


def _insert_context(
    session: Session,
    context_id: int,
    cited_paper_id: int,
    citing_paper_id: int | None,
    sentence: str,
    citing_year: int | None = None,
) -> None:
    session.execute(
        text(
            "INSERT INTO citation_contexts "
            "(context_id, cited_paper_id, citing_paper_id, citing_year, "
            " sentence_with_markers, sentence_without_markers) "
            "VALUES (:cid, :cited, :citing, :year, :s, :s)"
        ),
        {
            "cid": context_id,
            "cited": cited_paper_id,
            "citing": citing_paper_id,
            "year": citing_year,
            "s": sentence,
        },
    )


@pytest.fixture
def five_contexts(pg_session: Session) -> Session:
    """Insert five papers + one citation context each; return the session."""
    citing_paper = _PID_BASE  # reuse the first paper as the citing paper
    for i in range(len(_SENTENCES)):
        _insert_paper(pg_session, _PID_BASE + i, f"Synthetic paper {i}")
    for i, sentence in enumerate(_SENTENCES):
        _insert_context(
            pg_session,
            _CID_BASE + i,
            cited_paper_id=_PID_BASE + i,
            citing_paper_id=citing_paper,
            sentence=sentence,
            citing_year=2020,
        )
    pg_session.flush()
    return pg_session


def _ours(results: list) -> list:
    """Keep only the synthetic rows — the real corpus shares the table."""
    return [c for c in results if c.context_id >= _CID_BASE]


class TestRetrieveSparse:
    def test_acronym_query_tops_the_matching_context(
        self, five_contexts: Session
    ) -> None:
        results = _ours(retrieve_sparse(five_contexts, "LoRA"))

        assert len(results) == 1
        hit = results[0]
        assert hit.context_id == _CID_BASE  # the LoRA sentence
        assert hit.cited_paper_id == _PID_BASE
        assert hit.source is ContextSource.SPARSE
        assert hit.similarity > 0.0  # ts_rank_cd score

    def test_english_stemming_matches_inflected_form(
        self, five_contexts: Session
    ) -> None:
        # Query "embedding"; the sentence says "embeddings" — Snowball stems both.
        results = _ours(retrieve_sparse(five_contexts, "embedding"))

        assert {c.context_id for c in results} == {_CID_BASE + 1}

    def test_english_config_strips_stopwords(
        self, five_contexts: Session
    ) -> None:
        """The 'english' branch contributes nothing for a stop-word-only query.

        ``plainto_tsquery('english', ...)`` reduces a stop-word-only string to an
        empty tsquery (0 nodes), so ``sentence_tsv_english`` matches nothing. The
        'simple' branch has no stop-word list and *can* still match — fusion then
        treats those as low-signal, which is acceptable.
        """
        nodes = five_contexts.execute(
            text("SELECT numnode(plainto_tsquery('english', 'in the the to of'))")
        ).scalar_one()
        assert nodes == 0

    def test_target_year_filter_excludes_newer_contexts(
        self, five_contexts: Session
    ) -> None:
        # All synthetic contexts are citing_year=2020.
        assert _ours(retrieve_sparse(five_contexts, "LoRA", target_year=2019)) == []
        assert len(_ours(retrieve_sparse(five_contexts, "LoRA", target_year=2020))) == 1

    def test_english_and_simple_configs_differ(
        self, five_contexts: Session
    ) -> None:
        """The two tsvector columns use genuinely different configs.

        'english' applies Snowball stemming ("zqxnetworks" → "zqxnetwork");
        'simple' does not. A stemmed query form therefore matches the english
        column but not the simple one — and ``retrieve_sparse`` recovers it
        anyway because it ORs across both columns.
        """
        session = five_contexts
        _insert_context(
            session,
            _CID_BASE + 99,
            cited_paper_id=_PID_BASE,
            citing_paper_id=_PID_BASE,
            sentence="We benchmark several zqxnetworks in the study.",
            citing_year=2020,
        )
        session.flush()

        english_match, simple_match = session.execute(
            text(
                "SELECT "
                "  sentence_tsv_english @@ plainto_tsquery('english', 'zqxnetwork'), "
                "  sentence_tsv_simple  @@ plainto_tsquery('simple',  'zqxnetwork') "
                "FROM citation_contexts WHERE context_id = :cid"
            ),
            {"cid": _CID_BASE + 99},
        ).one()
        assert english_match is True  # 'zqxnetworks' stems to 'zqxnetwork'
        assert simple_match is False  # 'simple' keeps the unstemmed token

        # retrieve_sparse ORs both columns, so the stemmed query still finds it.
        recovered = _ours(retrieve_sparse(session, "zqxnetwork"))
        assert _CID_BASE + 99 in {c.context_id for c in recovered}
