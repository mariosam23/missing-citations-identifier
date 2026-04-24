"""EmbeddingIndex: build and manage the Qdrant paper collection.

Responsibilities
----------------
* Create the Qdrant collection with the correct dense + sparse named vectors.
* Upsert paper records (title + abstract) into the collection with metadata
  stored in the point payload for later retrieval.
* Provide a lightweight existence check so repeated runs skip already-indexed
  papers.

The collection uses **two named vectors**:

* ``"dense"``  — 1024-dim float32 vectors from ``intfloat/multilingual-e5-large-instruct``
* ``"sparse"`` — sparse SPLADE vectors from ``prithivida/Splade_PP_en_v1``

Usage
-----
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    from fastembed import SparseTextEmbedding

    client = QdrantClient(url=settings.QDRANT_URL)
    dense  = SentenceTransformer(settings.DENSE_MODEL)
    sparse = SparseTextEmbedding(model_name=settings.SPARSE_MODEL)

    index = EmbeddingIndex(client, dense, sparse)
    index.create_collection_if_missing()
    index.upsert_papers(papers)          # list[dict] from Postgres / OpenAlex
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from qdrant_client.http.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    SparseVector,
    VectorParams,
)

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instruction prefix for E5-instruct passage encoding.
# Papers are stored as "passage" embeddings (no prefix needed for E5-large).
# ---------------------------------------------------------------------------
_PASSAGE_PREFIX = "passage: "

# Default dense vector dimensionality for multilingual-e5-large-instruct.
_DENSE_DIM = 1024

# Number of papers to upsert in a single Qdrant batch call.
_DEFAULT_BATCH_SIZE = 64

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"


class EmbeddingIndex:
    """Build and manage the Qdrant hybrid-search collection for academic papers.

    Parameters
    ----------
    qdrant_client:
        An initialised ``QdrantClient`` instance.
    dense_model:
        A ``SentenceTransformer`` (or compatible) model.
    sparse_model:
        A ``fastembed.SparseTextEmbedding`` (or compatible) model.
    collection:
        Name of the Qdrant collection. Matches ``settings.QDRANT_COLLECTION_NAME``.
    dense_dim:
        Dimensionality of the dense vector. Override if you swap the embedding
        model for a different size.
    batch_size:
        Number of papers per Qdrant upsert call.
    """

    def __init__(
        self,
        qdrant_client: "QdrantClient",
        dense_model,
        sparse_model,
        collection: str = "papers",
        dense_dim: int = _DENSE_DIM,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self.client = qdrant_client
        self.dense = dense_model
        self.sparse = sparse_model
        self.collection = collection
        self.dense_dim = dense_dim
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection_if_missing(self) -> bool:
        """Create the Qdrant collection with named dense + sparse vectors.

        Returns
        -------
        bool
            ``True`` if the collection was created, ``False`` if it already
            existed (no action taken).
        """
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            logger.info("Collection %r already exists — skipping creation.", self.collection)
            return False

        logger.info(
            "Creating Qdrant collection %r (dense_dim=%d).", self.collection, self.dense_dim
        )
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                _DENSE_VECTOR_NAME: VectorParams(
                    size=self.dense_dim,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                _SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        logger.info("Collection %r created successfully.", self.collection)
        return True

    def collection_exists(self) -> bool:
        """Return True if the collection already exists in Qdrant."""
        existing = {c.name for c in self.client.get_collections().collections}
        return self.collection in existing

    def count(self) -> int:
        """Return the number of indexed points in the collection."""
        result = self.client.count(collection_name=self.collection, exact=True)
        return result.count

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def upsert_papers(self, papers: list[dict]) -> int:
        """Embed and upsert a list of paper dicts into Qdrant.

        Parameters
        ----------
        papers:
            Each dict must have at minimum:
            * ``"paper_id"`` (str) — Semantic Scholar or OpenAlex ID
            * ``"title"``    (str)
            * ``"abstract"`` (str)

            Optional keys stored in payload: ``"year"``, ``"venue"``,
            ``"cited_by_count"``.

        Returns
        -------
        int
            Number of papers successfully upserted.
        """
        papers = self._filter_unindexed_papers(papers)
        if not papers:
            logger.info("No new papers to index for collection %r.", self.collection)
            return 0

        total = 0
        for i in range(0, len(papers), self.batch_size):
            batch = papers[i : i + self.batch_size]
            n = self._upsert_batch(batch)
            total += n
            logger.info(
                "Upserted batch %d/%d (%d papers, running total=%d).",
                i // self.batch_size + 1,
                (len(papers) + self.batch_size - 1) // self.batch_size,
                n,
                total,
            )
        return total

    def _upsert_batch(self, papers: list[dict]) -> int:
        """Encode and upsert a single batch of papers."""
        texts = [self._paper_text(p) for p in papers]

        # Dense embeddings (batch)
        dense_vecs = self.dense.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        # Sparse embeddings (generator)
        sparse_embeddings = list(self.sparse.embed(texts))

        points: list[PointStruct] = []
        for paper, dense_vec, sparse_emb in zip(papers, dense_vecs, sparse_embeddings):
            point_id = self._stable_id(paper["paper_id"])
            payload = {
                "paper_id": paper["paper_id"],
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "cited_by_count": paper.get("cited_by_count"),
            }
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        _DENSE_VECTOR_NAME: self._as_list(dense_vec),
                        _SPARSE_VECTOR_NAME: SparseVector(
                            indices=self._as_list(sparse_emb.indices),
                            values=self._as_list(sparse_emb.values),
                        ),
                    },
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _paper_text(paper: dict) -> str:
        """Combine title and abstract into a single string for embedding."""
        title = paper.get("title") or ""
        abstract = paper.get("abstract") or ""
        return f"{_PASSAGE_PREFIX}{title}. {abstract}".strip()

    @staticmethod
    def _stable_id(paper_id: str) -> str:
        """Convert a paper_id string to a stable UUID string for Qdrant.

        Qdrant point IDs must be either unsigned integers or UUID strings.
        We use UUID5 with a fixed namespace so the same paper_id always maps
        to the same point, enabling safe re-upserts.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, paper_id))

    @staticmethod
    def _as_list(values) -> list:
        """Convert ndarray-like outputs to a plain Python list."""
        if hasattr(values, "tolist"):
            return values.tolist()
        return list(values)

    def _filter_unindexed_papers(self, papers: list[dict]) -> list[dict]:
        """Drop papers that are already present in Qdrant.

        This keeps reruns incremental instead of re-embedding the full corpus.
        """
        if not papers or not self.collection_exists():
            return papers

        indexed_ids = self._existing_point_ids(
            [self._stable_id(paper["paper_id"]) for paper in papers]
        )
        if not indexed_ids:
            return papers

        filtered = [
            paper
            for paper in papers
            if self._stable_id(paper["paper_id"]) not in indexed_ids
        ]
        skipped = len(papers) - len(filtered)
        if skipped:
            logger.info(
                "Skipping %d already-indexed papers in collection %r.",
                skipped,
                self.collection,
            )
        return filtered

    def _existing_point_ids(self, point_ids: list[str]) -> set[str]:
        """Return the subset of point IDs that already exist in Qdrant."""
        existing: set[str] = set()
        for i in range(0, len(point_ids), self.batch_size):
            records = self.client.retrieve(
                collection_name=self.collection,
                ids=point_ids[i : i + self.batch_size],
                with_payload=False,
                with_vectors=False,
            )
            existing.update(str(record.id) for record in records)
        return existing
