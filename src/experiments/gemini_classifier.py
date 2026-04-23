import json
import re
import time

from llm.genai_client import LLMClient
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
    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        batch_size: int = 10,
        delay_between_calls_seconds: float = 60.0,
    ):
        self.client = LLMClient(model=model, temperature=0.1, max_tokens=4096)
        self.batch_size = batch_size
        self.delay_between_calls_seconds = delay_between_calls_seconds

    def classify_sentences(self, sentences: list[SentenceRecord], paper: ParsedPaper) -> list[SentenceRecord]:
        """Classify a list of sentences for citation worthiness and intent."""
        # Process in batches
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            batch_number = (i // self.batch_size) + 1
            total_batches = (len(sentences) + self.batch_size - 1) // self.batch_size
            print(f"Sending batch {batch_number}/{total_batches} to Gemini...")
            self._classify_batch(batch, paper)

            if i + self.batch_size < len(sentences) and self.delay_between_calls_seconds > 0:
                print(
                    f"Waiting {int(self.delay_between_calls_seconds)} seconds before next Gemini API call..."
                )
                time.sleep(self.delay_between_calls_seconds)
        return sentences

    def _classify_batch(self, batch: list[SentenceRecord], paper: ParsedPaper):
        """Classify a batch of sentences using the LLM."""
        try:
            classifications = self._request_batch_classification(batch, paper)
        except ValueError as exc:
            if len(batch) == 1:
                raise

            split_point = max(1, len(batch) // 2)
            print(
                f"Batch of {len(batch)} sentences failed ({exc}). "
                f"Retrying as chunks of {split_point} and {len(batch) - split_point}..."
            )
            self._classify_batch(batch[:split_point], paper)
            self._classify_batch(batch[split_point:], paper)
            return

        self._apply_classifications(batch, classifications)

    def _request_batch_classification(
        self,
        batch: list[SentenceRecord],
        paper: ParsedPaper,
    ) -> list[dict]:
        """Request classifications for a batch and validate the response shape."""
        sentences_text = "\n".join(f"{j}. {s.text}" for j, s in enumerate(batch))

        user_prompt = CLASSIFIER_USER_PROMPT_TEMPLATE.format(
            title=paper.title,
            abstract=paper.abstract,
            sentences=sentences_text,
        )

        response = self.client.complete(
            CLASSIFIER_SYSTEM_PROMPT,
            user_prompt,
            response_mime_type="application/json",
        )
        print("Gemini raw response:")
        print(response)

        classifications = self._parse_classifications(response)
        self._validate_classifications(classifications, len(batch))
        return classifications

    @staticmethod
    def _apply_classifications(batch: list[SentenceRecord], classifications: list[dict]) -> None:
        """Write parsed classifications back into the sentence records."""
        for cls in classifications:
            idx = cls["sentence_index"]
            sentence = batch[idx]
            citation_worthy = cls.get("citation_worthy", False)
            intent_str = cls.get("citation_intent", "OTHER")
            confidence = float(cls.get("confidence", 0.5))

            sentence.citation_worthy = citation_worthy
            sentence.citation_intent = INTENT_MAP.get(intent_str)
            sentence.worthiness_score = confidence if sentence.citation_worthy else (1 - confidence)

    @staticmethod
    def _validate_classifications(classifications: list[dict], expected_count: int) -> None:
        """Ensure the model returned one well-formed classification per sentence."""
        if len(classifications) != expected_count:
            raise ValueError(
                f"Gemini returned {len(classifications)} classifications for "
                f"{expected_count} sentences."
            )

        seen_indices: set[int] = set()
        for cls in classifications:
            idx = cls.get("sentence_index")
            if not isinstance(idx, int):
                raise ValueError(f"Invalid sentence_index in Gemini response: {cls}")
            if idx < 0 or idx >= expected_count:
                raise ValueError(
                    f"Gemini returned out-of-range sentence_index {idx} "
                    f"for batch size {expected_count}."
                )
            if idx in seen_indices:
                raise ValueError(f"Gemini returned duplicate sentence_index {idx}.")
            seen_indices.add(idx)

        missing_indices = sorted(set(range(expected_count)) - seen_indices)
        if missing_indices:
            raise ValueError(f"Gemini omitted sentence_index values: {missing_indices}")

    @staticmethod
    def _parse_classifications(response: str) -> list[dict]:
        """Parse Gemini output, tolerating fenced JSON while surfacing truncation clearly."""
        cleaned = response.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            json_match = re.search(r"\[[\s\S]*\]", cleaned)
            if not json_match:
                if "[" in cleaned and "]" not in cleaned:
                    raise ValueError(
                        "Gemini returned a truncated JSON array. "
                        "Reduce batch size or response length."
                    ) from None
                raise ValueError(f"Could not parse LLM response as JSON: {response}")

            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Gemini returned malformed or truncated JSON. "
                    "Reduce batch size or response length."
                ) from exc

        if not isinstance(parsed, list):
            raise ValueError(f"Expected a JSON array from Gemini, got: {type(parsed).__name__}")

        return parsed


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
