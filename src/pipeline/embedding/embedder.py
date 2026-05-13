"""Process-wide SentenceTransformer singleton.

The model (BAAI/bge-base-en-v1.5, ~440 MB on disk) is loaded lazily on first
access so that importing this module — or the FastAPI app that depends on it —
does not pull in torch and the model weights at startup.

bge-base-en-v1.5 does **not** require a query/passage instruction prefix
(unlike the original BGE-v1), so the same encoder is used for both the bulk
embedding script and runtime query encoding. Embeddings are L2-normalized
(``normalize_embeddings=True``) so cosine similarity equals the dot product
and pgvector's ``vector_cosine_ops`` index matches the SQL we run against it.

Device selection: if a CUDA GPU is available, the model is loaded on
``cuda:0`` automatically (a 3060-class card cuts bulk embed time from
~30 min to ~2 min on this corpus). Override with the ``EMBEDDER_DEVICE``
env var (e.g. ``cpu``, ``cuda:1``, ``mps``).
"""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

from utils.config import config
from utils.logger import logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None
_lock = Lock()


def _resolve_device() -> str:
    """Return the torch device string to load the model onto.

    Honors ``config.EMBEDDER_DEVICE`` when set; otherwise picks CUDA when
    available, then MPS (Apple Silicon), falling back to CPU.
    """
    configured = getattr(config, "EMBEDDER_DEVICE", "") or ""
    if configured:
        return configured
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_embedder() -> SentenceTransformer:
    """Return the singleton SentenceTransformer, loading on first call."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            device = _resolve_device()
            logger.info(
                "loading embedder model=%s dim=%d device=%s",
                config.EMBEDDER_MODEL_NAME,
                config.EMBEDDER_DIM,
                device,
            )
            _model = SentenceTransformer(
                config.EMBEDDER_MODEL_NAME, device=device
            )
    return _model


def encode_texts(
    texts: list[str],
    *,
    batch_size: int | None = None,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Encode a batch of texts, returning an ``(n, dim)`` float32 array.

    Always L2-normalizes — required for ``vector_cosine_ops`` to behave as
    a dot-product index.
    """
    model = get_embedder()
    bs = batch_size or config.EMBEDDER_BATCH_SIZE
    vectors = model.encode(
        texts,
        batch_size=bs,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def encode_query(text: str) -> np.ndarray:
    """Encode a single query string. Returns a 1-D ``(dim,)`` float32 array."""
    return encode_texts([text])[0]
