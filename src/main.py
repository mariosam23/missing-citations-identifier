from utils import logger
from pipeline.pdf_parser import GrobidPDFParser
from pipeline.sentence_extractor import extract_sentences
from pipeline.reference_resolver import ReferenceResolver

import json


def main():
    logger.info("Starting the application...")

    logger.info("Parsing the PDF and extracting information...")

    parser = GrobidPDFParser(pdf_path="../papers/BERT.pdf")
    parsed_paper = parser.parse()
    
    # sentences = extract_sentences(parsed_paper)
    # logger.info(f"Extracted {len(sentences)} sentences from the paper.")

    bib_resolver = ReferenceResolver()

    resolved_refs = []
    for ref in parsed_paper.references:
        resolved = bib_resolver.resolve(ref)
        resolved_refs.append(resolved)
    
    logger.info(f"Resolved {len(resolved_refs)} references.")
    logger.info(f"Stats: {list(bib_resolver.stats.values())}")
    print(json.dumps([r.__dict__ for r in resolved_refs], indent=2))

if __name__ == "__main__":
    main()

