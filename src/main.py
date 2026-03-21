import json
import logging

from pipeline.keyword_graph import build_keyword_graph
from pipeline.pdf_parser import NougatPDFParser, ParsedPaper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = NougatPDFParser("../papers/NIPS-2017-attention-is-all-you-need-Paper.pdf")
    paper = parser.parse()

    with open("paper.json", "w", encoding="utf-8") as f:
        json.dump({
            "title": paper.title,
            "abstract": paper.abstract,
            "references": paper.references
        }, f, indent=2)
    

    with open("paper.json", "r", encoding="utf-8") as f:
        paper = json.load(f)

    abstract = paper["abstract"]
    title = paper.get("title", "")
    logger.info("Paper: %s", title)
    logger.info("Abstract length: %d words", len(abstract.split()))

    kg = build_keyword_graph(abstract)

    logger.info("Low confidence: %s", kg.low_confidence)
    logger.info("Nodes (%d):", len(kg.entity_labels))
    for label in sorted(kg.entity_labels):
        deg = kg.graph.degree(label)
        logger.info("  - %-40s  degree=%d", label, deg)

    logger.info("Edges (%d):", kg.graph.number_of_edges())
    for u, v, d in kg.graph.edges(data=True):
        logger.info("  %-35s <-> %-35s  weight=%d", u, v, d["weight"])

    pr = None
    if kg.graph.number_of_nodes() > 0:
        import networkx as nx
        pr = nx.pagerank(kg.graph, weight="weight")
        logger.info("PageRank scores:")
        for node, score in sorted(pr.items(), key=lambda x: x[1], reverse=True):
            logger.info("  %-40s  %.4f", node, score)

    out = kg.to_dict()
    with open("keyword_graph_output.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved graph to keyword_graph_output.json")


if __name__ == "__main__":
    main()
