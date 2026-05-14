"""Integration test for hybrid retrieval on the ``/recommend`` route.

Asserts that a query carrying a unique acronym surfaces the paper whose
contexts contain that acronym — the core Phase 6 win: the sparse branch
recovers a lexically-exact token, RRF fuses it with the dense branch, and the
aggregator ranks the paper into the top-3.

Needs a live Postgres with the Phase 6 sparse schema (``pg_session`` skips
otherwise) and the embedding model available locally (skips if it cannot
load). Synthetic rows are inserted in a transaction rolled back at teardown.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

# A token deliberately absent from any real corpus sentence, so the sparse
# branch matches *only* our synthetic contexts.
_ACRONYM = "Zqxnet"
_PID_BASE = 91_000_000
_CID_BASE = 91_000_000

# Three on-topic contexts for the target paper, each from a distinct citing
# paper, so the aggregator sees broad, consistent evidence.
_TARGET_SENTENCES: tuple[str, ...] = (
    f"We adopt {_ACRONYM} for efficient long-sequence modeling in our pipeline.",
    f"{_ACRONYM} reduces memory while keeping accuracy on long-sequence tasks.",
    f"Building on {_ACRONYM}, we extend efficient sequence modeling to new domains.",
)
_QUERY = f"We use {_ACRONYM} for efficient long-sequence modeling."


def _insert_paper(session: Session, paper_id: int, title: str) -> None:
    session.execute(
        text(
            "INSERT INTO papers "
            "(paper_id, canonical_title, normalized_title, first_author, year) "
            "VALUES (:pid, :title, :ntitle, :author, :year)"
        ),
        {
            "pid": paper_id,
            "title": title,
            "ntitle": title.lower(),
            "author": "Tester",
            "year": 2021,
        },
    )


def _insert_context(
    session: Session, context_id: int, cited_paper_id: int, citing_paper_id: int,
    sentence: str,
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
            "year": 2021,
            "s": sentence,
        },
    )


def _insert_embedding(session: Session, context_id: int, embedding) -> None:
    session.execute(
        text(
            "INSERT INTO citation_context_embeddings "
            "(context_id, embedding, model_name) "
            "VALUES (:cid, CAST(:emb AS vector), :model)"
        ),
        {"cid": context_id, "emb": embedding, "model": "test"},
    )


@pytest.fixture
def hybrid_corpus(pg_session: Session) -> Session:
    """Insert one target paper + 3 citing papers + 3 embedded contexts."""
    try:
        from pipeline.embedding.embedder import encode_texts
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Embedder unavailable: {exc}")

    try:
        embeddings = encode_texts(list(_TARGET_SENTENCES))
    except Exception as exc:  # pragma: no cover - model download/load failure
        pytest.skip(f"Embedding model could not be loaded: {exc}")

    target_id = _PID_BASE
    _insert_paper(pg_session, target_id, f"{_ACRONYM}: Efficient Sequence Modeling")
    for i, sentence in enumerate(_TARGET_SENTENCES):
        citing_id = _PID_BASE + 1 + i
        _insert_paper(pg_session, citing_id, f"Citing paper {i}")
        context_id = _CID_BASE + i
        _insert_context(pg_session, context_id, target_id, citing_id, sentence)
        _insert_embedding(pg_session, context_id, embeddings[i])
    pg_session.flush()
    return pg_session


@pytest.fixture
def client(hybrid_corpus: Session) -> Iterator:
    from fastapi.testclient import TestClient

    from api.deps import db_session
    from api.main import app

    def _override() -> Iterator[Session]:
        yield hybrid_corpus

    app.dependency_overrides[db_session] = _override
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(db_session, None)


def test_unique_acronym_query_ranks_target_in_top_3(client) -> None:
    response = client.post("/recommend", json={"text": _QUERY, "top_k": 10})

    assert response.status_code == 200
    candidates = response.json()["candidates"]
    assert candidates, "expected at least one candidate"

    paper_ids = [c["paper_id"] for c in candidates]
    assert _PID_BASE in paper_ids, "target paper missing from the recommendations"
    assert paper_ids.index(_PID_BASE) < 3, (
        f"target paper ranked #{paper_ids.index(_PID_BASE) + 1}, expected top-3"
    )

    target = next(c for c in candidates if c["paper_id"] == _PID_BASE)
    assert any(
        _ACRONYM.lower() in ev["sentence"].lower() for ev in target["evidence"]
    ), "target evidence should quote the acronym-bearing sentences"
