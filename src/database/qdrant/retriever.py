"""Hybrid Qdrant retrieval over the indexed paper collection."""

from typing import TYPE_CHECKING

from qdrant_client.http.models import Fusion, FusionQuery, Prefetch, SparseVector

from entities.retrieval_result import RetrievalResult
from utils import logger

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


_QUERY_PREFIX = (
    "Instruct: Given a scientific claim, retrieve research papers whose "
    "title and abstract provide evidence or prior work supporting this claim\n"
    "Query: "
)


class HybridRetriever:
    """Embed a query and search Qdrant with dense + sparse RRF fusion."""

    def __init__(
        self,
        qdrant_client: "QdrantClient",
        dense_model,
        sparse_model,
        collection: str = "papers",
        prefetch_limit: int = 50,
    ) -> None:
        self.client = qdrant_client
        self.dense = dense_model
        self.sparse = sparse_model
        self.collection = collection
        self.prefetch_limit = prefetch_limit

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant papers for query."""
        dense_vec = self._encode_dense(query)
        sparse_indices, sparse_values = self._encode_sparse(query)

        logger.debug(
            "Querying Qdrant collection=%r top_k=%d dense_dim=%d sparse_nnz=%d",
            self.collection,
            top_k,
            len(dense_vec),
            len(sparse_indices),
        )

        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=self.prefetch_limit,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using="sparse",
                    limit=self.prefetch_limit,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        results = [self._point_to_result(point) for point in response.points]
        logger.debug("Retrieved %d results for query=%r", len(results), query[:80])
        return results

    def _encode_dense(self, query: str) -> list[float]:
        """Encode query with the instruction-prefixed E5 model."""
        embedding = self.dense.encode(
            [f"{_QUERY_PREFIX}{query}"],
            normalize_embeddings=True,
        )
        return self._as_list(embedding[0])

    def _encode_sparse(self, query: str) -> tuple[list[int], list[float]]:
        """Encode query with SPLADE and return indices plus values."""
        sparse_embedding = next(iter(self.sparse.embed([query])))
        return (
            self._as_list(sparse_embedding.indices),
            self._as_list(sparse_embedding.values),
        )

    @staticmethod
    def _as_list(values) -> list:
        """Convert ndarray-like outputs to a plain Python list."""
        if hasattr(values, "tolist"):
            return values.tolist()
        return list(values)

    @staticmethod
    def _point_to_result(point) -> RetrievalResult:
        """Map a Qdrant scored point to the project retrieval contract."""
        payload: dict = point.payload or {}
        return RetrievalResult(
            paper_id=payload.get("paper_id", str(point.id)),
            title=payload.get("title", ""),
            score=float(point.score),
            year=payload.get("year"),
            venue=payload.get("venue"),
            cited_by_count=payload.get("cited_by_count"),
        )
