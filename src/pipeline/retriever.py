"""Hybrid Qdrant retrieval over the indexed paper collection."""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from qdrant_client.http.models import (
    Fusion,
    FusionQuery,
    IsEmptyCondition,
    PayloadField,
    Prefetch,
    QueryRequest,
    SparseVector,
)
from qdrant_client.models import Filter, FieldCondition, Range

from entities.retrieval_result import RetrievalResult
from utils import logger

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import ScoredPoint


# ndarray, list[float], list[int] — anything `_as_list` can normalize.
ArrayLike = Any


class DenseEncoder(Protocol):
    """SentenceTransformer-compatible dense encoder.

    Must expose ``.encode(texts, normalize_embeddings=...)`` returning an
    iterable of vectors (ndarray rows or plain sequences).
    """

    def encode(
        self,
        sentences: Sequence[str],
        normalize_embeddings: bool = ...,
    ) -> Iterable[ArrayLike]: ...


class SparseEmbedding(Protocol):
    """One SPLADE-style sparse vector with parallel index/value arrays."""

    indices: ArrayLike
    values: ArrayLike


class SparseEncoder(Protocol):
    """fastembed ``SparseTextEmbedding``-compatible sparse encoder.

    Must expose ``.embed(texts)`` yielding objects with ``.indices`` and
    ``.values`` attributes.
    """

    def embed(self, texts: list[str]) -> Iterable[SparseEmbedding]: ...


_QUERY_PREFIX = (
    "Instruct: Retrieve the most relevant scientific paper that should "
    "be cited to support the following text snippet from a research paper\n"
    "Query: "
)


class HybridRetriever:
    """Embed a query and search Qdrant with dense + sparse RRF fusion."""

    def __init__(
        self,
        qdrant_client: "QdrantClient",
        dense_model: DenseEncoder,
        sparse_model: SparseEncoder,
        collection: str = "papers",
        prefetch_limit: int = 50,
    ) -> None:
        self.client = qdrant_client
        self.dense = dense_model
        self.sparse = sparse_model
        self.collection = collection
        self.prefetch_limit = prefetch_limit

    def retrieve(self, query: str, top_k: int = 10, max_year: int | None = None) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant papers for query."""
        return self.retrieve_batch([query], top_k=top_k, max_year=max_year)[0]

    def retrieve_batch(
        self, queries: Sequence[str], top_k: int = 10, max_year: int | None = None
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

        query_filter = self._build_year_filter(max_year)
        effective_prefetch = max(self.prefetch_limit, top_k)

        requests = [
            QueryRequest(
                prefetch=[
                    Prefetch(
                        query=dense_vec,
                        using="dense",
                        limit=effective_prefetch,
                    ),
                    Prefetch(
                        query=SparseVector(indices=indices, values=values),
                        using="sparse",
                        limit=effective_prefetch,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
                filter=query_filter
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

    def probe_dense_cosine_batch(
        self,
        queries: Sequence[str],
        top_k: int = 10,
        max_year: int | None = None,
    ) -> list[list[RetrievalResult]]:
        """Dense-only retrieval for urgency probing.

        Bypasses RRF fusion so ``RetrievalResult.score`` is the raw cosine
        similarity in [0, 1] (dense vectors are normalized at index and query
        time). The urgency scorer uses this to threshold support against a
        meaningful similarity scale, instead of an opaque fusion score.
        """
        if not queries:
            return []

        dense_vecs = self._encode_dense_batch(queries)
        query_filter = self._build_year_filter(max_year)

        requests = [
            QueryRequest(
                query=dense_vec,
                using="dense",
                limit=top_k,
                with_payload=True,
                filter=query_filter,
            )
            for dense_vec in dense_vecs
        ]

        responses = self.client.query_batch_points(
            collection_name=self.collection,
            requests=requests,
        )

        return [
            [self._point_to_result(point) for point in response.points]
            for response in responses
        ]

    def _build_year_filter(self, max_year: int | None) -> Filter | None:
        """Build a year filter that allows papers with missing/null year.

        Qdrant's numeric ``Range`` excludes points where the field is absent
        or null. We wrap the range in a ``should`` so that a paper with
        ``year=None`` (allowed by the indexer) still passes the filter.
        """
        if max_year is None:
            return None
        return Filter(
            should=[
                FieldCondition(key="year", range=Range(lte=max_year)),
                IsEmptyCondition(is_empty=PayloadField(key="year")),
            ]
        )

    def _encode_dense_batch(self, queries: Sequence[str]) -> list[list[float]]:
        """Encode many queries in one forward pass."""
        embeddings = self.dense.encode(
            [f"{_QUERY_PREFIX}{q}" for q in queries],
            normalize_embeddings=True,
        )
        return [self._as_list(emb) for emb in embeddings]

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
    def _as_list(values: ArrayLike) -> list:
        """Convert ndarray-like outputs to a plain Python list."""
        if hasattr(values, "tolist"):
            return values.tolist()
        return list(values)

    @staticmethod
    def _point_to_result(point: "ScoredPoint") -> RetrievalResult:
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
