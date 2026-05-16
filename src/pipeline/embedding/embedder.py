"""Process-wide SentenceTransformer singleton.

The model (``dunzhang/stella_en_1.5B_v5``, ~3 GB at fp16) is loaded lazily
on first access so that importing this module — or the FastAPI app that
depends on it — does not pull in torch and the model weights at startup.

stella_en_1.5B_v5 has a matryoshka head that emits embeddings at any of
``{512, 768, 1024, 2048, 4096, 8192}`` dims natively; we cut to 1024
(``truncate_dim=config.EMBEDDER_DIM``) which is plenty for sentence-level
citation-context retrieval and fits well inside pgvector's HNSW dim limit.
Stella is **asymmetric**: queries must be wrapped in an instruction
prefix (``s2s_query``/``s2p_query``) while passages are encoded raw.
``encode_texts(..., is_query=True)`` and ``encode_query`` apply the
prompt configured by ``config.EMBEDDER_QUERY_PROMPT_NAME``; the corpus
embedding path leaves ``is_query=False`` so passages stay prompt-free.
Embeddings are L2-normalized
(``normalize_embeddings=True``) so cosine similarity equals the dot
product and pgvector's ``vector_cosine_ops`` index matches the SQL we
run against it.

The model ships custom modeling code, so it must be loaded with
``trust_remote_code=True``. We load it in fp16 (``torch_dtype="float16"``)
so it fits comfortably on a T4-class GPU at batch 16–64; on CPU torch
will silently upcast to fp32.

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
            # Stella's Qwen2 attention is fp16-unstable: long inputs
            # (e.g. with the s2s_query instruction prefix) reliably emit
            # NaN embeddings on Turing-class GPUs. Always load in fp32.
            model_kwargs: dict[str, str] = {"torch_dtype": "float32"}
            _model = SentenceTransformer(
                config.EMBEDDER_MODEL_NAME,
                device=device,
                trust_remote_code=True,
                truncate_dim=config.EMBEDDER_DIM,
                model_kwargs=model_kwargs,
            )
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
    Stella query instruction prefix (``config.EMBEDDER_QUERY_PROMPT_NAME``).
    """
    model = get_embedder()
    bs = batch_size or config.EMBEDDER_BATCH_SIZE
    encode_kwargs: dict[str, object] = {
        "batch_size": bs,
        "normalize_embeddings": True,
        "show_progress_bar": show_progress_bar,
        "convert_to_numpy": True,
    }
    prompt_name = config.EMBEDDER_QUERY_PROMPT_NAME if is_query else ""
    if prompt_name:
        encode_kwargs["prompt_name"] = prompt_name
    vectors = model.encode(texts, **encode_kwargs)
    return np.asarray(vectors, dtype=np.float32)


def encode_query(text: str) -> np.ndarray:
    """Encode a single query string. Returns a 1-D ``(dim,)`` float32 array."""
    return encode_texts([text], is_query=True)[0]
