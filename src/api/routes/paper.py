"""GET /paper/{paper_id}/bibtex — return a single BibTeX entry as plain text.

Intended for "Copy BibTeX" flows (webview button, direct browser access)
where the caller already knows the paper ID and does not need the full
``/recommend`` response.
"""

from __future__ import annotations

import unicodedata
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.deps import db_session
from database.postgres.tables.papers import Paper
from pipeline.bibtex.formatter import paper_to_bibtex

router = APIRouter(prefix="/paper", tags=["paper"])

DbSession = Annotated[Session, Depends(db_session)]


def _build_default_key(paper: Paper) -> str:
    """Minimal citation key when the caller does not supply one."""
    surname_raw = paper.first_author or "anon"
    nfkd = unicodedata.normalize("NFKD", surname_raw)
    surname = (
        nfkd.encode("ascii", "ignore")
        .decode("ascii")
        .replace(",", "")
        .replace(" ", "")
        .lower()
    )
    year = str(paper.year) if paper.year else "nodate"
    return f"{surname}{year}"


@router.get(
    "/{paper_id}/bibtex",
    response_class=PlainTextResponse,
    summary="Return a BibTeX entry for a paper.",
)
def get_bibtex(paper_id: int, session: DbSession) -> PlainTextResponse:
    paper = session.execute(
        select(Paper).where(Paper.paper_id == paper_id)
    ).scalar_one_or_none()

    if paper is None:
        raise HTTPException(status_code=404, detail="paper not found")

    key = _build_default_key(paper)
    bibtex = paper_to_bibtex(paper, key)
    return PlainTextResponse(content=bibtex, media_type="text/plain")
