from pipeline.sentence_extractor import (
    _build_author_year_index,
    _resolve_author_year_citations,
)


def test_author_year_suffix_resolves_exact_bibliography_entry() -> None:
    bibliography = {
        "b34": "Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, "
        "Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018a. "
        "Deep contextualized word representations. In NAACL.",
        "b35": "Matthew Peters, Mark Neumann, Luke Zettlemoyer, and Wen-tau Yih. "
        "2018b. Dissecting contextual word embeddings: Architecture and representation.",
    }
    index = _build_author_year_index(bibliography)

    assert _resolve_author_year_citations("Peters et al., 2018a", index) == ["b34"]
    assert _resolve_author_year_citations("Peters et al., 2018b", index) == ["b35"]


def test_unsuffixed_author_year_keeps_ambiguous_same_year_variants() -> None:
    bibliography = {
        "b34": "Matthew Peters, Mark Neumann, Mohit Iyyer. 2018a. "
        "Deep contextualized word representations.",
        "b35": "Matthew Peters, Mark Neumann, Luke Zettlemoyer. 2018b. "
        "Dissecting contextual word embeddings.",
    }
    index = _build_author_year_index(bibliography)

    assert _resolve_author_year_citations("Peters et al., 2018", index) == ["b34", "b35"]
