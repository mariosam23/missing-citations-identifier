"""Initial schema: pgvector extension + 5 CCDB tables.

The HNSW index on ``citation_context_embeddings.embedding`` is deferred to a
later revision (Phase 3) so we can bulk-load embeddings before paying the
index build cost.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-05-11
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 768


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "papers",
        sa.Column("paper_id", sa.BigInteger(), primary_key=True),
        sa.Column("canonical_title", sa.Text(), nullable=False),
        sa.Column("normalized_title", sa.Text(), nullable=False),
        sa.Column("authors", JSONB(), nullable=True),
        sa.Column("first_author", sa.Text(), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("venue", sa.Text(), nullable=True),
        sa.Column("doi", sa.Text(), nullable=True),
        sa.Column("arxiv_id", sa.Text(), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column(
            "is_survey", sa.Boolean(), nullable=True, server_default=sa.text("false")
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=True,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_papers_normalized_title", "papers", ["normalized_title"])
    op.create_index("idx_papers_year", "papers", ["year"])
    op.create_index(
        "idx_papers_first_author_year", "papers", ["first_author", "year"]
    )
    op.create_index("idx_papers_doi", "papers", ["doi"])
    op.create_index("idx_papers_arxiv_id", "papers", ["arxiv_id"])

    op.create_table(
        "source_documents",
        sa.Column("doc_id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "paper_id",
            sa.BigInteger(),
            sa.ForeignKey("papers.paper_id"),
            nullable=True,
        ),
        sa.Column("source_path", sa.Text(), nullable=True),
        sa.Column("source_type", sa.Text(), nullable=True),
        sa.Column("parse_status", sa.Text(), nullable=True),
        sa.Column("parse_quality_score", sa.Float(), nullable=True),
        sa.Column("raw_metadata", JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=True,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "references",
        sa.Column("reference_id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "citing_paper_id",
            sa.BigInteger(),
            sa.ForeignKey("papers.paper_id"),
            nullable=True,
        ),
        sa.Column("raw_reference_text", sa.Text(), nullable=True),
        sa.Column("ref_key", sa.Text(), nullable=True),
        sa.Column(
            "cited_paper_id",
            sa.BigInteger(),
            sa.ForeignKey("papers.paper_id"),
            nullable=True,
        ),
        sa.Column("parsed_title", sa.Text(), nullable=True),
        sa.Column("parsed_authors", JSONB(), nullable=True),
        sa.Column("parsed_first_author", sa.Text(), nullable=True),
        sa.Column("parsed_year", sa.Integer(), nullable=True),
        sa.Column("parsed_venue", sa.Text(), nullable=True),
        sa.Column("doi", sa.Text(), nullable=True),
        sa.Column("arxiv_id", sa.Text(), nullable=True),
        sa.Column("resolution_confidence", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=True,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "citation_contexts",
        sa.Column("context_id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "citing_paper_id",
            sa.BigInteger(),
            sa.ForeignKey("papers.paper_id"),
            nullable=True,
        ),
        sa.Column(
            "cited_paper_id",
            sa.BigInteger(),
            sa.ForeignKey("papers.paper_id"),
            nullable=True,
        ),
        sa.Column(
            "reference_id",
            sa.BigInteger(),
            sa.ForeignKey("references.reference_id"),
            nullable=True,
        ),
        sa.Column("section_name", sa.Text(), nullable=True),
        sa.Column("section_type", sa.Text(), nullable=True),
        sa.Column("paragraph_index", sa.Integer(), nullable=True),
        sa.Column("sentence_index", sa.Integer(), nullable=True),
        sa.Column("sentence_with_markers", sa.Text(), nullable=False),
        sa.Column("sentence_without_markers", sa.Text(), nullable=False),
        sa.Column("left_context", sa.Text(), nullable=True),
        sa.Column("right_context", sa.Text(), nullable=True),
        sa.Column("marker_text", sa.Text(), nullable=True),
        sa.Column("marker_start_char", sa.Integer(), nullable=True),
        sa.Column("marker_end_char", sa.Integer(), nullable=True),
        sa.Column("citation_group_id", sa.Text(), nullable=True),
        sa.Column("citation_group_size", sa.Integer(), nullable=True),
        sa.Column("local_window_text", sa.Text(), nullable=True),
        sa.Column("context_text_for_embedding", sa.Text(), nullable=True),
        sa.Column("citing_year", sa.Integer(), nullable=True),
        sa.Column("cited_year", sa.Integer(), nullable=True),
        sa.Column("age_at_citation", sa.Integer(), nullable=True),
        sa.Column("citation_role", sa.Text(), nullable=True),
        sa.Column("citation_role_confidence", sa.Float(), nullable=True),
        sa.Column("extraction_confidence", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=True,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "idx_contexts_cited_paper", "citation_contexts", ["cited_paper_id"]
    )
    op.create_index(
        "idx_contexts_citing_paper", "citation_contexts", ["citing_paper_id"]
    )
    op.create_index(
        "idx_contexts_years", "citation_contexts", ["citing_year", "cited_year"]
    )
    op.create_index(
        "idx_contexts_section_type", "citation_contexts", ["section_type"]
    )

    op.create_table(
        "citation_context_embeddings",
        sa.Column(
            "context_id",
            sa.BigInteger(),
            sa.ForeignKey("citation_contexts.context_id"),
            primary_key=True,
        ),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("model_name", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=True,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("citation_context_embeddings")
    op.drop_index("idx_contexts_section_type", table_name="citation_contexts")
    op.drop_index("idx_contexts_years", table_name="citation_contexts")
    op.drop_index("idx_contexts_citing_paper", table_name="citation_contexts")
    op.drop_index("idx_contexts_cited_paper", table_name="citation_contexts")
    op.drop_table("citation_contexts")
    op.drop_table("references")
    op.drop_table("source_documents")
    op.drop_index("idx_papers_arxiv_id", table_name="papers")
    op.drop_index("idx_papers_doi", table_name="papers")
    op.drop_index("idx_papers_first_author_year", table_name="papers")
    op.drop_index("idx_papers_year", table_name="papers")
    op.drop_index("idx_papers_normalized_title", table_name="papers")
    op.drop_table("papers")
    op.execute("DROP EXTENSION IF EXISTS vector")
