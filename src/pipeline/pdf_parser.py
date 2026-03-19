import re
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
import requests
from pathlib import Path
import fitz

@dataclass
class ParsedPaper:
    title: str
    abstract: str
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "references": self.references,
        }

    def __str__(self) -> str:
        refs_preview = f"{len(self.references)} refs"
        return (
            f"ParsedPaper(title={self.title!r}, "
            f"abstract={self.abstract[:80]!r}..., {refs_preview})"
        )

class GrobidPDFPParser:
    def __init__(self, pdf_path: str, grobid_url: str = "http://localhost:8070"):
        self.pdf_path = pdf_path
        self.grobid_url = grobid_url.rstrip("/")

    def parse(self) -> ParsedPaper:
        url = f"{self.grobid_url}/api/processFulltextDocument"
        
        with open(self.pdf_path, "rb") as f:
            files = {"input": (self.pdf_path, f, "application/pdf")}
            response = requests.post(url, files=files)
            
        response.raise_for_status()
        
        # Parse the TEI XML response
        root = ET.fromstring(response.content)
        
        # TEI namespace
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        
        # Extract title
        title_elem = root.find(".//tei:titleStmt/tei:title", ns)
        title = title_elem.text if title_elem is not None and title_elem.text else ""
        
        # Extract abstract
        abstract_elems = root.findall(".//tei:profileDesc/tei:abstract//tei:p", ns)
        abstract = " ".join([elem.text for elem in abstract_elems if elem.text])
        
        # Extract references
        references = []
        bibl_structs = root.findall(".//tei:listBibl/tei:biblStruct", ns)
        
        for bibl in bibl_structs:
            # Try to get the title of the reference
            ref_title_elem = bibl.find(".//tei:analytic/tei:title", ns)
            if ref_title_elem is None:
                ref_title_elem = bibl.find(".//tei:monogr/tei:title", ns)
                
            if ref_title_elem is not None and ref_title_elem.text:
                references.append(ref_title_elem.text.strip())
                
        return ParsedPaper(
            title=title.strip(),
            abstract=abstract.strip(),
            references=references
        )


class NougatPDFParser:
    """Parses academic PDFs using a running Nougat API server.
    
    Start the server with: nougat_api  (listens on http://localhost:8503 by default)
    """

    def __init__(self, pdf_path: str, nougat_url: str = "http://localhost:8503"):
        self.pdf_path = pdf_path
        self.nougat_url = nougat_url.rstrip("/")

    def parse(self) -> ParsedPaper:
        url = f"{self.nougat_url}/predict/"

        with open(self.pdf_path, "rb") as f:
            files = {"file": (Path(self.pdf_path).name, f, "application/pdf")}
            response = requests.post(url, files=files)

        response.raise_for_status()

        raw = response.text
        try:
            markdown = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            markdown = raw
        self._last_markdown = markdown

        return ParsedPaper(
            title=self._extract_title(markdown),
            abstract=self._extract_abstract(markdown),
            references=self._extract_references(markdown),
        )

    def _extract_title(self, markdown: str) -> str:
        match = re.search(r"^#\s+(.+?)(?=\n#|\n\n)", markdown, re.MULTILINE | re.DOTALL)
        if not match:
            return ""
        title = match.group(1)
        title = re.sub(r"\s+", " ", title).strip()
        return title

    def _extract_abstract(self, markdown: str) -> str:
        try:
            doc = fitz.open(self.pdf_path)
            text = ""
            for i in range(min(2, len(doc))):
                text += doc[i].get_text()
            
            match = re.search(r"(?i)\bAbstract\b[\s\n]+(.*?)(?=\n\s*(?:1\.?\s+Introduction|I\.\s+Introduction|Introduction)\b)", text, re.DOTALL)
            if match:
                abstract_text = match.group(1).strip()
                abstract_text = re.sub(r"\s+", " ", abstract_text)
                if len(abstract_text) > 50:
                    return abstract_text
        except Exception as e:
            print("PyMuPDF fallback failed:", e)
            pass

        # Headed abstract: "## Abstract" or "###### Abstract"
        match = re.search(
            r"(?i)^#{1,6}\s*abstract\s*\n(.*?)(?=^#{1,6}\s[^#]|\Z)",
            markdown,
            re.MULTILINE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()

        # Bold label: "**Abstract**"
        match = re.search(
            r"(?i)\*\*abstract\*\*[:\.\s]*(.*?)(?=\n\n#{1,6}\s|\n\n\*\*\d|\Z)",
            markdown,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()

        # Plain label on its own line
        match = re.search(
            r"(?i)(?:^|\n)abstract[:\.\s]*\n+(.*?)(?=\n\n#{1,6}\s|\n\n\*\*\d|\Z)",
            markdown,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()

        # No abstract heading at all: grab unmarked body text between the
        # preamble (title + authors + possible truncation warning) and the
        # first numbered/headed section (e.g. "## 1 Introduction" or "## 2 Background").
        match = re.search(
            r"\n\+\+\+\n\n(.*?)(?=\n#{1,6}\s)",
            markdown,
            re.DOTALL,
        )
        if match and len(match.group(1).strip()) > 80:
            return match.group(1).strip()

        # Last resort: text between title block and first ## heading
        match = re.search(
            r"^#\s+.+?\n\n(?:.*?\n\n)*((?:[A-Z].{80,}(?:\n\n(?!#).{80,})*))\s*\n#{1,6}\s",
            markdown,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()

        return ""

    def _extract_references(self, markdown: str) -> list[str]:
        section_match = re.search(
            r"(?i)^#{1,6}\s*references?\s*\n(.*?)(?=^#{1,6}\s[^#]|\Z)",
            markdown,
            re.MULTILINE | re.DOTALL,
        )
        if not section_match:
            return []

        section_text = section_match.group(1)

        raw_entries = re.split(
            r"\n(?=\*\s|\[?\d+\]?[\.\s])",
            section_text.strip(),
        )

        references = []
        for entry in raw_entries:
            entry = entry.strip()
            if not entry:
                continue

            cleaned = re.sub(r"^\*\s+", "", entry)
            cleaned = re.sub(r"^\[?\d+\]?[\.\s]+", "", cleaned)
            cleaned = cleaned.strip()
            if cleaned:
                references.append(cleaned)

        return references
