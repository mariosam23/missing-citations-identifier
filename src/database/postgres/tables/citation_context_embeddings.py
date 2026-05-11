from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from utils.config import config

from ..base import Base


class CitationContextEmbedding(Base):
    """Dense embedding for a citation context. Schema §6.5.

    Dimensionality is fixed at model selection time (bge-base-en-v1.5 → 768).
    The HNSW index is built in a follow-up Alembic revision after bulk insert.
    """

    __tablename__ = "citation_context_embeddings"

    context_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("citation_contexts.context_id"),
        primary_key=True,
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(config.EMBEDDER_DIM))
    model_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
