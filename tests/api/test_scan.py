from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from uuid import UUID

from fastapi.testclient import TestClient

from api.deps import db_session
from api.main import app
from api.schemas import Candidate, Evidence
from api.services.candidate_builder import CandidateBuildResult


def _override_session() -> Iterator[object]:
    yield SimpleNamespace()


def test_scan_returns_offsets_and_candidates(monkeypatch) -> None:
    from api.routes import recommend as route

    event_id = UUID("00000000-0000-0000-0000-000000000001")
    candidate = Candidate(
        paper_id=1,
        title="Attention Is All You Need",
        authors=["Vaswani, Ashish"],
        year=2017,
        venue="NeurIPS",
        citation_key="vaswani2017attention",
        score=0.82,
        evidence=[
            Evidence(
                sentence="Transformer pretraining improves language modeling.",
                citing_year=2020,
                similarity=0.82,
            )
        ],
        bibtex="@inproceedings{vaswani2017attention}",
    )

    monkeypatch.setattr(
        route,
        "_rank_candidates",
        lambda *args, **kwargs: [
            SimpleNamespace(score=0.82),
            SimpleNamespace(score=0.62),
        ],
    )
    monkeypatch.setattr(
        route,
        "build_candidates",
        lambda *args, **kwargs: CandidateBuildResult(
            candidates=[candidate],
            results_to_log=[],
        ),
    )
    monkeypatch.setattr(route, "_log_recommendation", lambda **kwargs: event_id)

    app.dependency_overrides[db_session] = _override_session
    try:
        client = TestClient(app)
        text = "Previous work has been shown to improve Transformer pretraining."

        response = client.post(
            "/scan",
            json={
                "text": text,
                "top_k": 5,
                "max_sentences": 20,
                "min_confidence": 0.55,
            },
        )
    finally:
        app.dependency_overrides.pop(db_session, None)

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["text"] == text
    assert text[items[0]["start_offset"] : items[0]["end_offset"]] == text
    assert items[0]["label"] == "MISSING_CITATION"
    assert items[0]["candidates"][0]["paper_id"] == 1
    assert items[0]["recommendation_event_id"] == str(event_id)


def test_scan_ignores_already_cited_sentences(monkeypatch) -> None:
    from api.routes import recommend as route

    monkeypatch.setattr(route, "_rank_candidates", lambda *args, **kwargs: [])

    app.dependency_overrides[db_session] = _override_session
    try:
        client = TestClient(app)
        response = client.post(
            "/scan",
            json={
                "text": (
                    "Previous work has been shown to improve Transformer "
                    "pretraining \\cite{vaswani2017}."
                ),
            },
        )
    finally:
        app.dependency_overrides.pop(db_session, None)

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_scan_keeps_strong_rule_hit_with_moderate_retrieval(monkeypatch) -> None:
    from api.routes import recommend as route

    candidate = Candidate(
        paper_id=42,
        title="BERT: Pre-training of Deep Bidirectional Transformers",
        authors=["Devlin, Jacob"],
        year=2019,
        venue="NAACL",
        citation_key="devlin2019bert",
        score=0.58,
        evidence=[
            Evidence(
                sentence="BERT sets new state-of-the-art on GLUE.",
                citing_year=2020,
                similarity=0.58,
            )
        ],
        bibtex="@inproceedings{devlin2019bert}",
    )

    monkeypatch.setattr(
        route,
        "_rank_candidates",
        lambda *args, **kwargs: [
            SimpleNamespace(score=0.58),
            SimpleNamespace(score=0.52),
        ],
    )
    monkeypatch.setattr(
        route,
        "build_candidates",
        lambda *args, **kwargs: CandidateBuildResult(
            candidates=[candidate],
            results_to_log=[],
        ),
    )
    monkeypatch.setattr(route, "_log_recommendation", lambda **kwargs: None)

    text = (
        "The model known as BERT has achieved state-of-the-art results "
        "in many NLP tasks."
    )

    app.dependency_overrides[db_session] = _override_session
    try:
        client = TestClient(app)
        response = client.post(
            "/scan",
            json={"text": text, "top_k": 5, "min_confidence": 0.55},
        )
    finally:
        app.dependency_overrides.pop(db_session, None)

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["text"] == text
    assert items[0]["confidence"] >= 0.55


def test_scan_returns_strong_rule_hit_without_candidates(monkeypatch) -> None:
    from api.routes import recommend as route

    monkeypatch.setattr(route, "_rank_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        route,
        "build_candidates",
        lambda *args, **kwargs: CandidateBuildResult(
            candidates=[],
            results_to_log=[],
        ),
    )

    text = (
        "The model known as BERT has achieved state-of-the-art results "
        "in many NLP tasks."
    )

    app.dependency_overrides[db_session] = _override_session
    try:
        client = TestClient(app)
        response = client.post(
            "/scan",
            json={"text": text, "top_k": 5, "min_confidence": 0.55},
        )
    finally:
        app.dependency_overrides.pop(db_session, None)

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["text"] == text
    assert items[0]["candidates"] == []
