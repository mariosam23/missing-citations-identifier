"""Phase 9 feedback logging: recommendation + feedback event tables.

Adds three plaintext logging tables (no encryption, no hashing — see the
Phase 9 plan):

* ``recommendation_events``  — one row per ``/recommend`` call.
* ``recommendation_results`` — the candidates shown for each event.
* ``feedback_events``        — user interactions (accept, thumbs, reject, …).

``feedback_events`` carries a ``UNIQUE NULLS NOT DISTINCT`` constraint so the
``/feedback`` route can upsert and make repeated identical feedback idempotent.
``NULLS NOT DISTINCT`` requires Postgres 15+ (the project runs pg17).

Revision ID: 0012_feedback_tables
Revises: 0011_rebuild_hnsw_bge_large
Create Date: 2026-05-23
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0012_feedback_tables"
down_revision: str | None = "0011_rebuild_hnsw_bge_large"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UQ_FEEDBACK = "uq_feedback_event_result_type"


def upgrade() -> None:
    op.create_table(
        "recommendation_events",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("document_path", sa.Text(), nullable=True),
        sa.Column("target_year", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=True,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "recommendation_results",
        sa.Column("result_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "event_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("recommendation_events.event_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "paper_id",
            sa.BigInteger(),
            sa.ForeignKey("papers.paper_id"),
            nullable=False,
        ),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("score", sa.Double(), nullable=False),
        sa.Column("citation_key", sa.Text(), nullable=False),
        sa.Column(
            "shown_at",
            sa.DateTime(),
            nullable=True,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "idx_rec_results_event", "recommendation_results", ["event_id"]
    )
    op.create_index(
        "idx_rec_results_paper", "recommendation_results", ["paper_id"]
    )

    op.create_table(
        "feedback_events",
        sa.Column("feedback_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "event_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("recommendation_events.event_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "result_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("recommendation_results.result_id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("feedback_type", sa.Text(), nullable=False),
        sa.Column("feedback_value", sa.Integer(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=True,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_feedback_event", "feedback_events", ["event_id"])
    op.create_index("idx_feedback_result", "feedback_events", ["result_id"])
    # NULLS NOT DISTINCT so event-level feedback (result_id IS NULL) also dedups.
    op.execute(
        f"ALTER TABLE feedback_events "
        f"ADD CONSTRAINT {_UQ_FEEDBACK} "
        f"UNIQUE NULLS NOT DISTINCT (event_id, result_id, feedback_type)"
    )


def downgrade() -> None:
    # Children first; dropping a table also drops its indexes and constraints.
    op.drop_table("feedback_events")
    op.drop_table("recommendation_results")
    op.drop_table("recommendation_events")
