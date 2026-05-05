"""Qdrant-backed storage helpers for the paper collection."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qdrant_client.http.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from utils import logger

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DEFAULT_DENSE_DIM = 1024
DEFAULT_QDRANT_URL = "http://localhost:6333"


@dataclass(frozen=True)
class PaperVector:
    """Encoded paper payload ready to be persisted in Qdrant."""

    point_id: str
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]
    payload: dict


class QdrantPaperStore:
    """Small adapter that owns Qdrant client operations for paper vectors."""

    def __init__(
        self,
        qdrant_client: Any,
        collection: str = "papers",
        dense_dim: int = DEFAULT_DENSE_DIM,
    ) -> None:
        self.client = qdrant_client
        self.collection = collection
        self.dense_dim = dense_dim

    def create_collection_if_missing(self) -> bool:
        """Create the configured paper collection if it does not exist."""
        if self.collection_exists():
            logger.info("Collection %r already exists - skipping creation.", self.collection)
            return False

        logger.info(
            "Creating Qdrant collection %r (dense_dim=%d).",
            self.collection,
            self.dense_dim,
        )
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=self.dense_dim,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        logger.info("Collection %r created successfully.", self.collection)
        return True

    def collection_exists(self) -> bool:
        """Return True when the configured collection exists."""
        existing = {c.name for c in self.client.get_collections().collections}
        return self.collection in existing

    def count(self) -> int:
        """Return the number of indexed points in the configured collection."""
        result = self.client.count(collection_name=self.collection, exact=True)
        return int(result.count)

    def existing_point_ids(self, point_ids: list[str], batch_size: int = 64) -> set[str]:
        """Return the subset of point IDs already present in Qdrant."""
        existing: set[str] = set()
        for i in range(0, len(point_ids), batch_size):
            records = self.client.retrieve(
                collection_name=self.collection,
                ids=point_ids[i : i + batch_size],
                with_payload=False,
                with_vectors=False,
            )
            existing.update(str(record.id) for record in records)
        return existing

    def upsert_paper_vectors(self, vectors: list[PaperVector]) -> int:
        """Persist encoded paper vectors and payloads to Qdrant."""
        points = [
            PointStruct(
                id=vector.point_id,
                vector={
                    DENSE_VECTOR_NAME: vector.dense,
                    SPARSE_VECTOR_NAME: SparseVector(
                        indices=vector.sparse_indices,
                        values=vector.sparse_values,
                    ),
                },
                payload=vector.payload,
            )
            for vector in vectors
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)


def create_qdrant_client(url: str | None = None) -> "QdrantClient":
    """Create a Qdrant client using the project default URL fallback."""
    from qdrant_client import QdrantClient

    return QdrantClient(url=url or DEFAULT_QDRANT_URL)


def count_collections(qdrant_client: Any) -> int:
    """Return the number of collections available through the client."""
    return len(qdrant_client.get_collections().collections)


def require_collection_point_count(
    qdrant_client: Any,
    collection: str,
) -> int:
    """Return collection point count or raise a RuntimeError with context."""
    try:
        collection_info = qdrant_client.get_collection(collection)
        num_papers = int(collection_info.points_count or 0)
    except Exception as exc:
        raise RuntimeError(f"Failed to access collection {collection!r}: {exc}") from exc

    if num_papers == 0:
        raise RuntimeError(
            f"Collection {collection!r} is empty. Please index papers first "
            "(see src/indexer.py)."
        )
    return num_papers
