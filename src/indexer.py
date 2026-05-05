"""EmbeddingIndex: build and manage the Qdrant paper collection.

The indexer owns paper text preparation, embedding, batching, and stable point
IDs. Qdrant-specific collection and point operations live in
``database.qdrant``.
"""

import uuid
from typing import TYPE_CHECKING

from database.qdrant import DEFAULT_DENSE_DIM, PaperVector, QdrantPaperStore
from utils import logger

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


_PASSAGE_PREFIX = "passage: "
_DEFAULT_BATCH_SIZE = 64


class EmbeddingIndex:
    """Build and manage the hybrid-search index for academic papers."""

    def __init__(
        self,
        qdrant_client: "QdrantClient | None",
        dense_model,
        sparse_model,
        collection: str = "papers",
        dense_dim: int = DEFAULT_DENSE_DIM,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        store: QdrantPaperStore | None = None,
    ) -> None:
        if store is None and qdrant_client is None:
            raise ValueError("Either qdrant_client or store must be provided")

        self.store = store or QdrantPaperStore(
            qdrant_client=qdrant_client,
            collection=collection,
            dense_dim=dense_dim,
        )
        self.dense = dense_model
        self.sparse = sparse_model
        self.collection = self.store.collection
        self.dense_dim = self.store.dense_dim
        self.batch_size = batch_size

    def create_collection_if_missing(self) -> bool:
        """Create the Qdrant collection with named dense + sparse vectors."""
        return self.store.create_collection_if_missing()

    def collection_exists(self) -> bool:
        """Return True if the collection already exists in Qdrant."""
        return self.store.collection_exists()

    def count(self) -> int:
        """Return the number of indexed points in the collection."""
        return self.store.count()

    def upsert_papers(self, papers: list[dict]) -> int:
        """Embed and upsert a list of paper dicts into Qdrant."""
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
        texts = [self._paper_text(paper) for paper in papers]
        dense_vecs = self.dense.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sparse_embeddings = list(self.sparse.embed(texts))

        vectors: list[PaperVector] = []
        for paper, dense_vec, sparse_emb in zip(papers, dense_vecs, sparse_embeddings):
            vectors.append(
                PaperVector(
                    point_id=self._stable_id(paper["paper_id"]),
                    dense=self._as_list(dense_vec),
                    sparse_indices=self._as_list(sparse_emb.indices),
                    sparse_values=self._as_list(sparse_emb.values),
                    payload={
                        "paper_id": paper["paper_id"],
                        "title": paper.get("title", ""),
                        "year": paper.get("year"),
                        "venue": paper.get("venue"),
                        "cited_by_count": paper.get("cited_by_count"),
                    },
                )
            )

        return self.store.upsert_paper_vectors(vectors)

    @staticmethod
    def _paper_text(paper: dict) -> str:
        """Combine title and abstract into a single string for embedding."""
        title = paper.get("title") or ""
        abstract = paper.get("abstract") or ""
        return f"{_PASSAGE_PREFIX}{title}. {abstract}".strip()

    @staticmethod
    def _stable_id(paper_id: str) -> str:
        """Convert a paper_id string to a stable UUID string for Qdrant."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, paper_id))

    @staticmethod
    def _as_list(values) -> list:
        """Convert ndarray-like outputs to a plain Python list."""
        if hasattr(values, "tolist"):
            return values.tolist()
        return list(values)

    def _filter_unindexed_papers(self, papers: list[dict]) -> list[dict]:
        """Drop papers that are already present in Qdrant."""
        if not papers or not self.collection_exists():
            return papers

        id_map = {self._stable_id(paper["paper_id"]): paper for paper in papers}
        indexed_ids = self._existing_point_ids(list(id_map.keys()))
        if not indexed_ids:
            return papers

        filtered = [paper for sid, paper in id_map.items() if sid not in indexed_ids]
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
        return self.store.existing_point_ids(point_ids, batch_size=self.batch_size)
