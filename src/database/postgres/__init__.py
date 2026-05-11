"""Postgres ORM models and engine wiring.

Importing this package ensures every table is registered on ``Base.metadata``
so Alembic autogenerate sees the full schema.
"""

from . import tables  # noqa: F401  (side-effect: register models on Base.metadata)
from .base import Base

__all__ = ["Base"]
