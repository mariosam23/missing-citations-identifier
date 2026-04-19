from typing import Optional

import spacy
from spacy.language import Language

# Cached spaCy model used for sentence segmentation and light text processing
_sentence_nlp: Optional[Language] = None

def get_sentence_nlp() -> Language:
    """Return a cached spaCy `Language` instance tuned for sentence extraction.

    This accessor is intended for callers that perform sentence segmentation
    and simple token-level processing (e.g. `doc.sents`). It loads
    `en_core_web_sm` on first use and returns the same instance thereafter.
    """
    global _sentence_nlp

    if _sentence_nlp is None:
        _sentence_nlp = spacy.load("en_core_web_sm")
    return _sentence_nlp
