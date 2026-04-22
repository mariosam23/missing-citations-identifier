import json

from llm import LLMClient
from prompts import CLASSIFIER_SYSTEM_PROMPT, CLASSIFIER_USER_PROMPT_TEMPLATE
from entities import SentenceRecord, CitationIntent
from entities import ParsedPaper


INTENT_MAP = {
    "BACKGROUND": CitationIntent.BACKGROUND,
    "METHOD": CitationIntent.METHOD,
    "RESULT": CitationIntent.RESULT,
}

# gemini-3.1-flash-lite-preview
# gemini-3-flash-preview
# gemma-4-31b-it
class GeminiClassifier:
    def __init__(self, model: str = "gemini-3-flash-preview", batch_size: int = 40):
        self.client = LLMClient(model=model, temperature=0.1, max_tokens=4096)
        self.batch_size = batch_size

    def classify_sentences(self, sentences: list[SentenceRecord], paper: ParsedPaper) -> list[SentenceRecord]:
        """Classify a list of sentences for citation worthiness and intent."""
        # Process in batches
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            self._classify_batch(batch, paper)
        return sentences

    def _classify_batch(self, batch: list[SentenceRecord], paper: ParsedPaper):
        """Classify a batch of sentences using the LLM."""
        # Prepare the sentences for the prompt
        sentences_text = "\n".join(f"{j}. {s.text}" for j, s in enumerate(batch))
        
        user_prompt = CLASSIFIER_USER_PROMPT_TEMPLATE.format(
            title=paper.title,
            abstract=paper.abstract,
            sentences=sentences_text
        )
        
        response = self.client.complete(CLASSIFIER_SYSTEM_PROMPT, user_prompt)
        
        # Parse the JSON response
        try:
            classifications = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from the response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                classifications = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse LLM response as JSON: {response}")
        
        # Update the sentences with classifications
        for cls in classifications:
            idx = cls["sentence_index"]
            if 0 <= idx < len(batch):
                sentence = batch[idx]
                citation_worthy = cls.get("citation_worthy", False)
                intent_str = cls.get("citation_intent", "OTHER")
                confidence = cls.get("confidence", 0.5)
                
                sentence.citation_worthy = citation_worthy
                sentence.citation_intent = INTENT_MAP.get(intent_str)
                sentence.worthiness_score = confidence if sentence.citation_worthy else (1 - confidence)


def test_classifier():
    import sys
    import os
    
    # Add src to sys.path so we can import properly
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    from pipeline.pdf_parser import GrobidPDFParser
    from pipeline.sentence_extractor import extract_sentences

    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "papers", "BERT.pdf"))
    
    print(f"Parsing {pdf_path}...")
    parser = GrobidPDFParser(pdf_path)
    try:
        paper = parser.parse()
    except Exception as e:
        print(f"Failed to parse paper: {e}")
        return
        
    print(f"Extracted Paper Title: {paper.title}")
    
    print("Extracting sentences...")
    sentences = extract_sentences(paper)
    print(f"Total sentences extracted: {len(sentences)}")
    
    if not sentences:
        print("No sentences extracted!")
        return

    print("Running classifier on test batch (first 10 non-trivial sentences)...")
    classifier = GeminiClassifier()
    
    # Skip the first few sentences as they might just be title or abstract boilerplate
    start_idx = min(20, max(0, len(sentences) - 10))
    test_batch = sentences[start_idx:start_idx+10]
    
    classified_sentences = classifier.classify_sentences(test_batch, paper)
    
    for i, s in enumerate(classified_sentences):
        print(f"\n--- Sentence {i+1} ---")
        print(f"Text: {s.text}")
        print(f"Citation Worthy: {s.citation_worthy}")
        print(f"Citation Intent: {s.citation_intent}")
        print(f"Worthiness Score: {s.worthiness_score}")

if __name__ == "__main__":
    test_classifier()
