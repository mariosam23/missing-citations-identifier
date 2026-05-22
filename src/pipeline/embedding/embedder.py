"""Process-wide SentenceTransformer singleton.

The model (``BAAI/bge-large-en-v1.5``, ~1.34 GB at fp16) is loaded lazily
on first access so that importing this module — or the FastAPI app that
depends on it — does not pull in torch and the model weights at startup.

bge-large-en-v1.5 has a native dimension of 1024 (``truncate_dim=config.EMBEDDER_DIM``),
which is plenty for sentence-level citation-context retrieval and fits well
inside pgvector's HNSW dim limit.
BGE-large performs best on this asymmetric sentence-retrieval task when
queries are wrapped in the query instruction prefix, and database contexts
are embedded raw.
``encode_texts(..., is_query=True)`` and ``encode_query`` apply the
prompt configured by ``config.EMBEDDER_QUERY_PROMPT_NAME``; the corpus
embedding path leaves ``is_query=False`` so passages stay prompt-free.
Embeddings are L2-normalized (``normalize_embeddings=True``) so cosine
similarity equals the dot product and pgvector's ``vector_cosine_ops``
index matches the SQL we run against it.

We load it in fp16 (``torch_dtype="float16"``) or bfloat16 for speed and
memory efficiency on CUDA; on CPU torch will silently upcast to fp32.

Device selection: if a CUDA GPU is available, the model is loaded on
``cuda:0`` automatically. Override with the ``EMBEDDER_DEVICE`` env var
(e.g. ``cpu``, ``cuda:1``, ``mps``).
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
            # Use float16 or bfloat16 for speed and memory efficiency on CUDA.
            model_kwargs = {}
            if "cuda" in device:
                import torch
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    model_kwargs = {"torch_dtype": torch.bfloat16}
                else:
                    model_kwargs = {"torch_dtype": torch.float16}

            _model = SentenceTransformer(
                config.EMBEDDER_MODEL_NAME,
                device=device,
                trust_remote_code=False,
                truncate_dim=config.EMBEDDER_DIM,
                model_kwargs=model_kwargs,
            )
            # If the model uses query prompts but they are empty/missing, set them.
            if "bge-large" in config.EMBEDDER_MODEL_NAME or "bge-small" in config.EMBEDDER_MODEL_NAME:
                if "query" not in _model.prompts or not _model.prompts["query"]:
                    _model.prompts["query"] = "Represent this sentence for searching relevant passages: "
    return _model


def encode_texts(
    texts: list[str],
    *,
    batch_size: int | None = None,
    show_progress_bar: bool = False,
    is_query: bool = False,
) -> np.ndarray:
    """Encode a batch of texts, returning an ``(n, dim)`` float32 array.

    Always L2-normalizes — required for ``vector_cosine_ops`` to behave as
    a dot-product index. Set ``is_query=True`` to wrap inputs with the
    BGE-large query instruction prefix (``config.EMBEDDER_QUERY_PROMPT_NAME``).
    """
    model = get_embedder()
    bs = batch_size or config.EMBEDDER_BATCH_SIZE
    prompt_name = config.EMBEDDER_QUERY_PROMPT_NAME if is_query else ""
    vectors = model.encode(
        texts,
        batch_size=bs,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        prompt_name=prompt_name or None,
    )
    return np.asarray(vectors, dtype=np.float32)


def encode_query(text: str) -> np.ndarray:
    """Encode a single query string. Returns a 1-D ``(dim,)`` float32 array."""
    return encode_texts([text], is_query=True)[0]
