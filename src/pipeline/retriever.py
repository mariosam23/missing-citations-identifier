"""Hybrid Qdrant retrieval over the indexed paper collection."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from qdrant_client.http.models import Fusion, FusionQuery, Prefetch, QueryRequest, SparseVector

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
        results = self.retrieve_batch([query], top_k=top_k)
        return results[0] if results else []

    def retrieve_batch(
        self, queries: Sequence[str], top_k: int = 10
    ) -> list[list[RetrievalResult]]:
        """Retrieve top-k for many queries in one round trip.

        Encodes all queries together (a single forward pass through both the
        dense and sparse encoders) and issues a single Qdrant ``query_batch_points``
        call instead of N individual queries. Returns one result list per query
        in the same order as ``queries``.
        """
        if not queries:
            return []

        dense_vecs = self._encode_dense_batch(queries)
        sparse_pairs = self._encode_sparse_batch(queries)

        logger.debug(
            "Batch querying Qdrant collection=%r n_queries=%d top_k=%d",
            self.collection,
            len(queries),
            top_k,
        )

        requests = [
            QueryRequest(
                prefetch=[
                    Prefetch(
                        query=dense_vec,
                        using="dense",
                        limit=self.prefetch_limit,
                    ),
                    Prefetch(
                        query=SparseVector(indices=indices, values=values),
                        using="sparse",
                        limit=self.prefetch_limit,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
            for dense_vec, (indices, values) in zip(dense_vecs, sparse_pairs)
        ]

        responses = self.client.query_batch_points(
            collection_name=self.collection,
            requests=requests,
        )

        return [
            [self._point_to_result(point) for point in response.points]
            for response in responses
        ]

    def _encode_dense(self, query: str) -> list[float]:
        """Encode a single query with the instruction-prefixed E5 model."""
        return self._encode_dense_batch([query])[0]

    def _encode_dense_batch(self, queries: Sequence[str]) -> list[list[float]]:
        """Encode many queries in one forward pass."""
        embeddings = self.dense.encode(
            [f"{_QUERY_PREFIX}{q}" for q in queries],
            normalize_embeddings=True,
        )
        return [self._as_list(emb) for emb in embeddings]

    def _encode_sparse(self, query: str) -> tuple[list[int], list[float]]:
        """Encode a single query with SPLADE and return indices plus values."""
        return self._encode_sparse_batch([query])[0]

    def _encode_sparse_batch(
        self, queries: Sequence[str]
    ) -> list[tuple[list[int], list[float]]]:
        """Encode many queries with SPLADE; preserves input order."""
        embeddings = list(self.sparse.embed(list(queries)))
        return [
            (self._as_list(emb.indices), self._as_list(emb.values))
            for emb in embeddings
        ]

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
            abstract=payload.get("abstract"),
        )
