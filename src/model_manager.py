from typing import Optional

import spacy
from spacy.language import Language

# Global spacy model to avoid reloading
_nlp: Optional[Language] = None

def get_nlp() -> Language:
    """Load the spacy model"""
    global _nlp

    if _nlp is None:
        _nlp = spacy.load("en_core_sci_scibert")
    return _nlp
