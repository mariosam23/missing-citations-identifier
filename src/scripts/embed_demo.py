"""Minimal demonstration of how a sentence is turned into an embedding.

The script encodes one "query" sentence (the one an author is currently
writing) and two stored "citing" sentences with the same model used by the
system, ``BAAI/bge-large-en-v1.5``. It then prints the cosine similarity
between the query and each stored sentence, showing that a paraphrase lands
close to the query while an unrelated sentence does not.

Because the embeddings are L2-normalized, cosine similarity reduces to a plain
dot product. Run with::

    python -m src.scripts.embed_demo
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-large-en-v1.5"
# BGE encodes queries (not passages) behind a short instruction prefix.
QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

QUERY_SENTENCE = "Contextual word representations are obtained with a pretrained Transformer."
STORED_SENTENCES: tuple[str, ...] = (
    "Tokens are encoded with a pre-trained Transformer model.",
    "The dataset was annotated by three independent reviewers.",
)


def encode(model: SentenceTransformer, text: str, *, is_query: bool) -> np.ndarray:
    """Return the L2-normalized embedding of ``text`` as a 1-D float array."""
    prompt = QUERY_PROMPT if is_query else None
    vector: np.ndarray = model.encode(
        text,
        prompt=prompt,
        normalize_embeddings=True,
    )
    return vector


def main() -> None:
    """Encode the example sentences and report cosine similarities."""
    model = SentenceTransformer(MODEL_NAME)

    query_vector = encode(model, QUERY_SENTENCE, is_query=True)
    logger.info("Query: %s", QUERY_SENTENCE)
    logger.info("Embedding dimension: %d", query_vector.shape[0])
    logger.info("First values: %s ...\n", np.round(query_vector[:5], 4).tolist())

    for sentence in STORED_SENTENCES:
        stored_vector = encode(model, sentence, is_query=False)
        # Both vectors are unit-length, so the dot product is the cosine.
        cosine = float(np.dot(query_vector, stored_vector))
        logger.info("cos = %.3f  <-  %s", cosine, sentence)


if __name__ == "__main__":
    main()
