"""Stage 4A — Hybrid Retriever (V0 baseline).

Embeds a query sentence with a dense E5 model and a sparse SPLADE model,
then issues a single Qdrant prefetch + RRF fusion query to get top-k results.

Usage
-----
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    from fastembed import SparseTextEmbedding

    client = QdrantClient(url=settings.QDRANT_URL)
    dense  = SentenceTransformer(settings.DENSE_MODEL)
    sparse = SparseTextEmbedding(model_name=settings.SPARSE_MODEL)

    retriever = HybridRetriever(client, dense, sparse)
    results   = retriever.retrieve("attention mechanisms in transformers", top_k=10)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qdrant_client.http.models import Fusion, FusionQuery, Prefetch, SparseVector

from entities.retrieval_result import RetrievalResult

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instruction prefix used for E5-instruct models.
# The model was trained with task-specific prefixes; using the correct one
# substantially improves recall compared to bare query strings.
# ---------------------------------------------------------------------------
_QUERY_PREFIX = (
    "Instruct: Given a scientific claim, retrieve research papers whose "
    "title and abstract provide evidence or prior work supporting this claim\n"
    "Query: "
)


class HybridRetriever:
    """Embed a sentence, search Qdrant (dense E5 + sparse SPLADE, RRF fusion).

    Parameters
    ----------
    qdrant_client:
        An initialised ``QdrantClient`` instance.
    dense_model:
        A ``SentenceTransformer`` (or compatible) model that implements
        ``encode(texts, normalize_embeddings) -> np.ndarray``.
    sparse_model:
        A ``fastembed.SparseTextEmbedding`` (or compatible) model that
        implements ``embed(texts) -> Iterable[SparseEmbedding]``.
        Each ``SparseEmbedding`` must have ``.indices`` and ``.values``
        array attributes.
    collection:
        Name of the Qdrant collection to search against.
    prefetch_limit:
        Number of candidates fetched per modality before fusion (50 is a
        safe default; increase if recall is saturating at top-k).
    """

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant papers for *query*.

        Parameters
        ----------
        query:
            The retrieval query — typically a citation-stripped sentence from
            a scientific paper (``SentenceRecord.get_retrieval_text()``).
        top_k:
            Number of results to return after RRF fusion.

        Returns
        -------
        list[RetrievalResult]
            Ranked list of results (best first).
        """
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

        results = [self._point_to_result(p) for p in response.points]
        logger.debug("Retrieved %d results for query=%r", len(results), query[:80])
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_dense(self, query: str) -> list[float]:
        """Encode *query* with the instruction-prefixed E5 model."""
        prefixed = f"{_QUERY_PREFIX}{query}"
        embedding = self.dense.encode([prefixed], normalize_embeddings=True)
        return self._as_list(embedding[0])

    def _encode_sparse(self, query: str) -> tuple[list[int], list[float]]:
        """Encode *query* with SPLADE and return (indices, values)."""
        # fastembed returns a generator; consume one element
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
        """Map a Qdrant ``ScoredPoint`` to a ``RetrievalResult``.

        Qdrant payload keys are set at index time by ``EmbeddingIndex``
        (see ``src/indexer.py``).  Missing keys fall back to ``None``.
        """
        payload: dict = point.payload or {}
        return RetrievalResult(
            paper_id=payload.get("paper_id", str(point.id)),
            title=payload.get("title", ""),
            score=float(point.score),
            year=payload.get("year"),
            venue=payload.get("venue"),
            cited_by_count=payload.get("cited_by_count"),
        )
