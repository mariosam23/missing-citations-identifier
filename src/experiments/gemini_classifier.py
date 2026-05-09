from pipeline.classifier import GeminiClassifier
from utils.logger import logger

def test_classifier():
    import os
    
    from pipeline.pdf_parser import GrobidPDFParser
    from pipeline.sentence_extractor import extract_sentences

    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "papers", "BERT.pdf"))
    
    logger.info("Parsing %s...", pdf_path)
    parser = GrobidPDFParser(pdf_path)
    try:
        paper = parser.parse()
    except Exception as e:
        logger.error("Failed to parse paper: %s", e)
        return
        
    logger.info("Extracted Paper Title: %s", paper.title)
    
    logger.info("Extracting sentences...")
    sentences = extract_sentences(paper)
    logger.info("Total sentences extracted: %d", len(sentences))
    
    if not sentences:
        logger.warning("No sentences extracted!")
        return

    logger.info("Running classifier on test batch (first 10 non-trivial sentences)...")
    classifier = GeminiClassifier()
    
    # Skip the first few sentences as they might just be title or abstract boilerplate
    start_idx = min(20, max(0, len(sentences) - 10))
    test_batch = sentences[start_idx:start_idx+10]
    
    classified_sentences = classifier.classify_sentences(test_batch, paper.title, paper.abstract)
    
    for i, s in enumerate(classified_sentences):
        logger.info("\n--- Sentence %d ---", i+1)
        logger.info("Text: %s", s.text)
        logger.info("Citation State: %s", s.citation_state)
        logger.info("Citation Intent: %s", s.citation_intent)
        logger.info("Worthiness Score: %s", s.worthiness_score)

if __name__ == "__main__":
    test_classifier()
