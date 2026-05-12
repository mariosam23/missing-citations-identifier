"""Add resolution_method column to references table.

Records which matching strategy populated ``cited_paper_id``.
Invaluable for debugging resolution quality and auditing hit rates.

Revision ID: 0002_add_resolution_method
Revises: 0001_initial_schema
Create Date: 2026-05-11
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002_add_resolution_method"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "references",
        sa.Column("resolution_method", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("references", "resolution_method")