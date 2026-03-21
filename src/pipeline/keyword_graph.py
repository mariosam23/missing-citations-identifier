from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

import networkx as nx
import spacy
import matplotlib.pyplot as plt
from spacy.tokens import Doc, Span
from spacy import language
from spacy import tokens

logger = logging.getLogger(__name__)

# Constants
PAGERANK_PRUNE_PERCENTILE = 0.20
MIN_NODES_FOR_PRUNING = 5
LOW_CONFIDENCE_WORD_THRESHOLD = 50
NOISE_POS = {"DET", "PUNCT", "SYM", "NUM", "X"}
GENERIC_STOPWORDS = {
    "result", "results", "method", "approach", "model", "system", "work",
    "paper", "study", "task", "problem", "datum", "set", "case", "way",
    "use", "number", "time", "type", "order", "level", "part", "form",
    "point", "end", "line", "leave", "condition", "range", "base", "step",
    "show", "give", "need", "include", "find", "make", "provide",
    "natural", "new", "large", "good", "high", "low", "different",
    "important", "significant", "possible", "specific", "general",
    "improvement", "performance", "experiment", "accuracy",
}

@dataclass
class KeywordGraph:
    """
    Data class representing a keyword graph extracted from an abstract.
    """
    graph: nx.Graph
    entity_labels: list[str] = field(default_factory=list)
    low_confidence: bool = False

    def to_dict(self) -> dict:
        """
        Convert the KeywordGraph into a dictionary representation.
        
        Returns:
            dict: The graph data in dictionary format containing nodes, edges,
                  and a low confidence flag.
        """
        return {
            "nodes": [
                {"label": n, **self.graph.nodes[n]}
                for n in self.graph.nodes()
            ],
            "edges": [
                {"source": u, "target": v, "weight": d.get("weight", 1)}
                for u, v, d in self.graph.edges(data=True)
            ],
            "low_confidence": self.low_confidence,
        }

    def visualize(self):
        plt.figure(figsize=(18, 12))
        
        # Get the networkx graph
        G = self.graph
        
        # Generate a layout for nodes - using kamada_kawai_layout which usually reduces overlap
        pos = nx.kamada_kawai_layout(G)
        
        # Fallback/Scale the layout to create more space between disconnected components
        pos = nx.spring_layout(G, pos=pos, k=2.5, iterations=100, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='#add8e6', alpha=0.8, edgecolors='#555555')
        
        # Draw edges
        edges = G.edges()
        weights = [G[u][v].get('weight', 1) * 2 for u, v in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color='#888888', alpha=0.6)
        
        # **Crucial for overlap fix**: shift the label positions slightly above the nodes
        pos_labels = {node: (coords[0], coords[1] + 0.08) for node, coords in pos.items()}
        
        # Draw labels with high alpha background to overwrite lines beneath them
        bbox_props = dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#aaaaaa", lw=1.5, alpha=0.95)
        nx.draw_networkx_labels(G, pos_labels, font_size=11, font_weight="bold", bbox=bbox_props)
        
        # Make sure to leave margins so labels don't get cut off on the edges
        plt.margins(0.2)
        
        plt.title("Keyword Graph Visualization", fontsize=18, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class KeywordGraphBuilder:
    """
    Class to build a keyword graph from academic abstracts using NLP techniques.
    """

    def __init__(self, model_name: str = "en_core_sci_scibert"):
        """
        Initialize the KeywordGraphBuilder with a SpaCy model.

        Args:
            model_name (str): The name of the SpaCy model to use. Defaults to "en_core_sci_scibert".
        """
        self._nlp = spacy.load(model_name)

    def build(self, abstract: str) -> KeywordGraph:
        """
        Build a weighted keyword graph from an academic abstract.

        Nodes are normalised scientific concepts extracted via SciSpaCy NER and
        noun-chunk parsing. Edges encode sentence co-occurrence (weight 1) or
        direct syntactic dependency (weight 2). Low-PageRank nodes are pruned
        unless the graph is already small.

        Args:
            abstract (str): The source text to build the graph from.

        Returns:
            KeywordGraph: The resulting keyword graph.
        """
        abstract = self._clean_abstract(abstract)
        doc = self._nlp(abstract)

        low_confidence = len(abstract.split()) < LOW_CONFIDENCE_WORD_THRESHOLD
        raw_spans = self._extract_raw_spans(doc)
        entity_labels = self._normalize_all(raw_spans)

        if len(entity_labels) < 3:
            low_confidence = True

        graph = nx.Graph()
        for label in entity_labels:
            graph.add_node(label, label=label)

        for source, target, weight in self._build_edges(doc, entity_labels):
            if graph.has_node(source) and graph.has_node(target):
                graph.add_edge(source, target, weight=weight)

        if not low_confidence:
            graph = self._prune_by_pagerank(graph)

        return KeywordGraph(
            graph=graph,
            entity_labels=list(graph.nodes()),
            low_confidence=low_confidence,
        )

    def _clean_abstract(self, text: str) -> str:
        """
        Fix hyphenated line-breaks and Unicode ligatures from PDF extraction.

        Args:
            text (str): The raw text extracted from a PDF.

        Returns:
            str: Cleaned text.
        """
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_raw_spans(self, doc: Doc) -> list[Span]:
        """
        Collect entity spans and noun-chunk spans from the processed document.

        Args:
            doc (Doc): Processed SpaCy document.

        Returns:
            list[Span]: Extracted entity and noun chunk spans.
        """
        spans: list[Span] = []
        spans.extend(doc.ents)
        spans.extend(doc.noun_chunks)
        return spans

    def _normalize_span(self, span: Span) -> str:
        """
        Lowercase-lemmatise a span, stripping determiners, punctuation, and numbers.

        Args:
            span (Span): A SpaCy span of tokens.

        Returns:
            str: Normalized string representation of the span.
        """
        tokens_list = [
            tok.lemma_.lower()
            for tok in span
            if tok.pos_ not in NOISE_POS and not tok.is_space
        ]
        return " ".join(tokens_list)

    def _deduplicate_longest_match(self, labels: list[str]) -> list[str]:
        """
        Keep only the longest form when one label is a substring of another.

        Args:
            labels (list[str]): List of labels.

        Returns:
            list[str]: Filtered list of unique labels.
        """
        sorted_labels = sorted(labels, key=len, reverse=True)
        kept_labels: list[str] = []
        for label in sorted_labels:
            if not any(label in longer for longer in kept_labels):
                kept_labels.append(label)
        return kept_labels

    def _is_valid_label(self, label: str) -> bool:
        """
        Check if a given label is conceptually valid (not just a stopword).

        Args:
            label (str): The label to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        if len(label) <= 2:
            return False
        if not any(c.isalpha() for c in label):
            return False
        if not label[0].isalpha():
            return False
        if label in GENERIC_STOPWORDS:
            return False
        return True

    def _normalize_all(self, raw_spans: list[Span]) -> list[str]:
        """
        Filter and normalize all input raw spans.

        Args:
            raw_spans (list[Span]): Unfiltered list of raw SpaCy spans.

        Returns:
            list[str]: A deduplicated list of valid, normalized spans.
        """
        normalized: list[str] = []
        for span in raw_spans:
            norm = self._normalize_span(span)
            if self._is_valid_label(norm):
                normalized.append(norm)

        unique_labels = list(dict.fromkeys(normalized))
        return self._deduplicate_longest_match(unique_labels)

    def _entity_tokens_in_sent(self, entity_label: str, sent: Span) -> list[spacy.tokens.Token]:
        """
        Find tokens in a sentence whose contiguous lemmas match an entity label.

        Args:
            entity_label (str): Space-separated entity lemmas.
            sent (Span): The sentence chunk to check within.

        Returns:
            list[spacy.tokens.Token]: The matched syntactic head tokens.
        """
        entity_parts = entity_label.split()
        n = len(entity_parts)
        sent_lemmas = [
            (tok, tok.lemma_.lower())
            for tok in sent
            if tok.pos_ not in NOISE_POS and not tok.is_space
        ]

        for i in range(len(sent_lemmas) - n + 1):
            window = sent_lemmas[i : i + n]
            if [lem for _, lem in window] == entity_parts:
                matched_tokens = [tok for tok, _ in window]
                head = min(matched_tokens, key=lambda t: t.i)
                for t in matched_tokens:
                    if t.head not in matched_tokens or t.head == t:
                        head = t
                        break
                return [head]
        return []

    def _shares_dependency(self, tok_a: spacy.tokens.Token, tok_b: spacy.tokens.Token) -> bool:
        """
        Check if two tokens share a direct syntactic dependency.

        Args:
            tok_a (spacy.tokens.Token): First token.
            tok_b (spacy.tokens.Token): Second token.

        Returns:
            bool: True if they share a structural dependency, False otherwise.
        """
        if tok_a.head == tok_b or tok_b.head == tok_a:
            return True
        if tok_a.head == tok_b.head:
            return True
        return False

    def _build_edges(self, doc: Doc, entity_labels: list[str]) -> list[tuple[str, str, int]]:
        """
        Find connections between concepts within sentences to form graph edges.

        Args:
            doc (Doc): The parsed SpaCy document.
            entity_labels (list[str]): Extract valid entity labels from the corpus.

        Returns:
            list[tuple[str, str, int]]: Edges represented as (source, target, weight).
        """
        edges: dict[tuple[str, str], int] = {}

        for sent in doc.sents:
            found: list[tuple[str, spacy.tokens.Token]] = []
            for label in entity_labels:
                heads = self._entity_tokens_in_sent(label, sent)
                if heads:
                    found.append((label, heads[0]))

            for i, (label_a, tok_a) in enumerate(found):
                for label_b, tok_b in found[i + 1 :]:
                    key = tuple(sorted([label_a, label_b]))
                    weight = 2 if self._shares_dependency(tok_a, tok_b) else 1
                    edges[key] = max(edges.get(key, 0), weight)

        return [(a, b, w) for (a, b), w in edges.items()]

    def _prune_by_pagerank(
        self,
        graph: nx.Graph,
        percentile: float = PAGERANK_PRUNE_PERCENTILE,
        min_nodes: int = MIN_NODES_FOR_PRUNING,
    ) -> nx.Graph:
        """
        Prune graph nodes falling below a specified PageRank threshold to remove noise.

        Args:
            graph (nx.Graph): The unpruned graph.
            percentile (float): PageRank percentile used to determine the cutoff threshold.
            min_nodes (int): Minimum nodes threshold to allow pruning.

        Returns:
            nx.Graph: A new pruned graph or the same graph if below thresholds.
        """
        if graph.number_of_nodes() <= min_nodes:
            return graph

        pr = nx.pagerank(graph, weight="weight")
        scores = sorted(pr.values())
        threshold = scores[int(len(scores) * percentile)]

        to_remove = [n for n, score in pr.items() if score <= threshold]

        if graph.number_of_nodes() - len(to_remove) < min_nodes:
            return graph

        graph.remove_nodes_from(to_remove)
        logger.debug("Pruned %d nodes below PageRank threshold %.4f", len(to_remove), threshold)
        return graph


def build_keyword_graph(abstract: str) -> KeywordGraph:
    """
    Convenience function to build a keyword graph utilizing the default builder.

    Args:
        abstract (str): Document abstract to process.

    Returns:
        KeywordGraph: The compiled keyword graph via `KeywordGraphBuilder`.
    """
    builder = KeywordGraphBuilder()
    return builder.build(abstract)

