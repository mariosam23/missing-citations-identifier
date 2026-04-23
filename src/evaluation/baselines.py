"""BM25 lexical baseline and helpers to load external rankings from disk."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmarks.common import BenchmarkExample


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


@dataclass(frozen=True)
class PaperDocument:
    """Minimal corpus row for lexical ranking (OpenAlex-style `paperId`)."""

    paper_id: str
    title: str
    abstract: str = ""

    @property
    def body(self) -> str:
        parts = [self.title or "", self.abstract or ""]
        return " ".join(p for p in parts if p).strip()

    @classmethod
    def from_orm(cls, row: Any) -> "PaperDocument":
        """Build from a SQLAlchemy `Paper` row or any object with ``paperId``, ``title``, ``abstract``."""
        paper_id = getattr(row, "paperId", None) or getattr(row, "paper_id", "") or ""
        title = getattr(row, "title", None) or ""
        abstract = getattr(row, "abstract", None) or ""
        return cls(paper_id=str(paper_id), title=str(title), abstract=str(abstract))


class OkapiBM25:
    """In-memory Okapi BM25 over a fixed corpus (no external dependencies)."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._corpus_tokens = [list(doc) for doc in corpus_tokens]
        self._doc_freqs: list[dict[str, int]] = []
        self._doc_lens: list[int] = []
        df: dict[str, int] = defaultdict(int)
       
        for doc in self._corpus_tokens:
            tf: dict[str, int] = defaultdict(int)
       
            for t in doc:
                tf[t] += 1
       
            self._doc_freqs.append(dict(tf))
            self._doc_lens.append(len(doc))
       
            for t in tf:
                df[t] += 1
       
        self._avgdl = sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0.0
        self._N = len(self._corpus_tokens)
        self._idf: dict[str, float] = {}
       
        for t, f in df.items():
            self._idf[t] = math.log(1.0 + (self._N - f + 0.5) / (f + 0.5))

    def scores(self, query_tokens: Sequence[str]) -> list[float]:
        if not self._corpus_tokens:
            return []
        q_tf: dict[str, int] = defaultdict(int)
        for t in query_tokens:
            q_tf[t] += 1

        scores = [0.0] * self._N
      
        for i, doc_tf in enumerate(self._doc_freqs):
 
            dl = self._doc_lens[i]
            denom_norm = self.k1 * (1 - self.b + self.b * dl / self._avgdl) if self._avgdl > 0 else self.k1
            s = 0.0
 
            for t, qfreq in q_tf.items():
                if t not in doc_tf:
                    continue
 
                idf = self._idf.get(t, 0.0)
                f = doc_tf[t]
                num = f * (self.k1 + 1)
                den = f + denom_norm
                s += idf * (num / den) * qfreq
 
            scores[i] = s
 
        return scores


class BM25Baseline:
    """Rank `PaperDocument` rows with BM25 given a natural-language query."""

    def __init__(self, papers: Sequence[PaperDocument], **bm25_kwargs: float):
        tokens = [_tokenize(p.body) for p in papers]
        self._papers = list(papers)
        self._engine = OkapiBM25(tokens, k1=float(bm25_kwargs.get("k1", 1.5)), b=float(bm25_kwargs.get("b", 0.75)))

    def rank(self, query: str, top_k: int = 50) -> list[str]:
        q = _tokenize(query)
     
        scores = self._engine.scores(q)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[str] = []
     
        for i in order[:top_k]:
            out.append(self._papers[i].paper_id)
        return out

    def predict_fn(self, top_k: int = 50):
        """Return a `PredictFn` closure for `RetrievalEvaluator`."""

        def _predict(example: BenchmarkExample) -> list[str]:
            return self.rank(example.query_text, top_k=top_k)

        return _predict


def load_rankings_json(path: str | Path) -> dict[str, list[str]]:
    """Load a JSON mapping of example_id -> ranked paper ids.

    Supported shapes:
    - ``{"exp-1": ["W1", "W2"], ...}``
    - ``[{"example_id": "...", "ranked_paper_ids": ["..."]}, ...]``
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        out: dict[str, list[str]] = {}
        for k, v in raw.items():
            if not isinstance(k, str):
                continue
            if isinstance(v, list):
                out[k] = [str(x) for x in v]
        return out
    if isinstance(raw, list):
        out_list: dict[str, list[str]] = {}
        for row in raw:
            if not isinstance(row, Mapping):
                continue
            eid = row.get("example_id") or row.get("id")
            ranks = row.get("ranked_paper_ids") or row.get("ranking") or row.get("paper_ids")
            if isinstance(eid, str) and isinstance(ranks, list):
                out_list[eid] = [str(x) for x in ranks]
        return out_list
    raise ValueError(f"Unsupported rankings JSON structure: {type(raw).__name__}")


class ExternalRankingBaseline:
    """Serve rankings from a precomputed lookup table (e.g. another system’s output)."""

    def __init__(self, rankings: Mapping[str, Sequence[str]], *, default: Sequence[str] | None = None):
        self._rankings = {k: list(v) for k, v in rankings.items()}
        self._default = list(default) if default is not None else []

    @classmethod
    def from_json(cls, path: str | Path, *, default: Sequence[str] | None = None) -> "ExternalRankingBaseline":
        return cls(load_rankings_json(path), default=default)

    def rank(self, example_id: str, top_k: int | None = None) -> list[str]:
        ranked = self._rankings.get(example_id, self._default)
        if top_k is None:
            return list(ranked)
        return ranked[:top_k]

    def predict_fn(self, top_k: int | None = 50):
        def _predict(example: BenchmarkExample) -> list[str]:
            return self.rank(example.example_id, top_k=top_k)

        return _predict
