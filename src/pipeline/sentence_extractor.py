from model_manager import get_nlp
from entities.parsed_paper import ParsedPaper
from entities.sentence_record import SentenceRecord, CitationIntent
from utils.regex_patterns import (
    CITATION_PATTERN,
    WHITESPACE_CLEANUP_PATTERN,
    BULLET_MARKER_PATTERN,
    BARE_PUNCTUATION_PATTERN,
    HEADING_LIKE_SENTENCE_PATTERN
)

def clean_text(text: str) -> str:
    """Basic text cleanup."""
    text = WHITESPACE_CLEANUP_PATTERN.sub(" ", text).strip()
    return text

def is_noise(text: str) -> bool:
    """Filter out noise sentences (headings, bare punctuation, very short lines)."""
    text = text.strip()
    if len(text.split()) < 4:
        return True
    if BARE_PUNCTUATION_PATTERN.match(text):
        return True
    if HEADING_LIKE_SENTENCE_PATTERN.match(text):
        return True
    return False

def extract_sentences(parsed_paper: ParsedPaper) -> list[SentenceRecord]:
    """Split all sections into clean, annotated sentences."""
    nlp = get_nlp()
    records: list[SentenceRecord] = []
    
    sections_to_process = {"Abstract": parsed_paper.abstract} if parsed_paper.abstract else {}
    sections_to_process.update(parsed_paper.sections)
    
    for section_name, section_text in sections_to_process.items():
        if not section_text or not section_text.strip():
            continue
            
        doc = nlp(section_text)
        
        # Raw sentences from spacy
        raw_sents = [sent.text.strip() for sent in doc.sents]
        
        # Clean and filter
        valid_sents = []
        for s in raw_sents:
            cleaned = clean_text(BULLET_MARKER_PATTERN.sub("", s))
            if not is_noise(cleaned):
                valid_sents.append(cleaned)
                
        total_sents = len(valid_sents)
        if total_sents == 0:
            continue
            
        for i, sent_text in enumerate(valid_sents):
            has_cite = bool(CITATION_PATTERN.search(sent_text))
            
            retrieval_text = CITATION_PATTERN.sub("", sent_text)
            retrieval_text = clean_text(retrieval_text)
            
            pos = i / max(total_sents - 1, 1) if total_sents > 1 else 0.0
            
            # Default intent for already cited - typical simplification for background
            intent = CitationIntent.BACKGROUND if has_cite else None
            
            prev_sent = valid_sents[i-1] if i > 0 else None
            next_sent = valid_sents[i+1] if i < total_sents - 1 else None
            
            record = SentenceRecord(
                text=sent_text,
                section=section_name,
                position_in_section=pos,
                has_citation=has_cite,
                citation_intent=intent,
                retrieval_text=retrieval_text,
                previous_sentence=prev_sent,
                next_sentence=next_sent
            )
            records.append(record)
            
    return records
