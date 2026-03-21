import re
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
import requests
from pathlib import Path
import fitz

@dataclass
class ParsedPaper:
    """
    Data class representing a parsed academic paper.
    """
    title: str
    abstract: str
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        Convert the ParsedPaper into a dictionary representation.
        
        Returns:
            dict: The paper data in dictionary format containing title, abstract,
                  and references.
        """
        return {
            "title": self.title,
            "abstract": self.abstract,
            "references": self.references,
        }

    def __str__(self) -> str:
        """
        Returns a string representation of the ParsedPaper.
        """
        refs_preview = f"{len(self.references)} refs"
        abstract_preview = self.abstract[:80] + "..." if len(self.abstract) > 80 else self.abstract
        return (
            f"ParsedPaper(title={self.title!r}, "
            f"abstract={abstract_preview!r}, {refs_preview})"
        )


class GrobidPDFParser:
    """
    Parses academic PDFs using a running GROBID API server.
    """

    def __init__(self, pdf_path: str, grobid_url: str = "http://localhost:8070"):
        """
        Initialize the GrobidPDFParser.

        Args:
            pdf_path (str): Path to the PDF file to be parsed.
            grobid_url (str): The URL of the GROBID server. Defaults to "http://localhost:8070".
        """
        self.pdf_path = pdf_path
        self.grobid_url = grobid_url.rstrip("/")
        self.namespace = {"tei": "http://www.tei-c.org/ns/1.0"}

    def parse(self) -> ParsedPaper:
        """
        Process the PDF via the GROBID API and extract the paper details.

        Returns:
            ParsedPaper: The parsed paper containing title, abstract, and references.
        """
        xml_content = self._fetch_tei_xml()
        root = ET.fromstring(xml_content)
        
        return ParsedPaper(
            title=self._extract_title(root),
            abstract=self._extract_abstract(root),
            references=self._extract_references(root)
        )

    def _fetch_tei_xml(self) -> bytes:
        """
        Send the PDF to the GROBID API and retrieve the TEI XML response.

        Returns:
            bytes: The raw XML content returned by GROBID.
        """
        url = f"{self.grobid_url}/api/processFulltextDocument"
        
        with open(self.pdf_path, "rb") as f:
            files = {"input": (Path(self.pdf_path).name, f, "application/pdf")}
            response = requests.post(url, files=files)
            
        response.raise_for_status()
        return response.content

    def _extract_title(self, root: ET.Element) -> str:
        """
        Extract the paper title from the parsed TEI XML.

        Args:
            root (ET.Element): The root element of the XML.

        Returns:
            str: The extracted title, or an empty string if not found.
        """
        title_elem = root.find(".//tei:titleStmt/tei:title", self.namespace)
        if title_elem is not None and title_elem.text:
            return title_elem.text.strip()
        return ""

    def _extract_abstract(self, root: ET.Element) -> str:
        """
        Extract the paper abstract from the parsed TEI XML.

        Args:
            root (ET.Element): The root element of the XML.

        Returns:
            str: The extracted abstract.
        """
        abstract_elems = root.findall(".//tei:profileDesc/tei:abstract//tei:p", self.namespace)
        abstract = " ".join([elem.text for elem in abstract_elems if elem.text])
        return abstract.strip()

    def _extract_references(self, root: ET.Element) -> list[str]:
        """
        Extract the paper references from the parsed TEI XML.

        Args:
            root (ET.Element): The root element of the XML.

        Returns:
            list[str]: A list of extracted references.
        """
        references = []
        bibl_structs = root.findall(".//tei:listBibl/tei:biblStruct", self.namespace)
        
        for bibl in bibl_structs:
            ref_title_elem = bibl.find(".//tei:analytic/tei:title", self.namespace)
            if ref_title_elem is None:
                ref_title_elem = bibl.find(".//tei:monogr/tei:title", self.namespace)
                
            if ref_title_elem is not None and ref_title_elem.text:
                references.append(ref_title_elem.text.strip())
                
        return references


class NougatPDFParser:
    """
    Parses academic PDFs using a running Nougat API server.
    
    Start the server with: nougat_api (listens on http://localhost:8503 by default).
    """

    def __init__(self, pdf_path: str, nougat_url: str = "http://localhost:8503"):
        """
        Initialize the NougatPDFParser.

        Args:
            pdf_path (str): Path to the PDF file to be parsed.
            nougat_url (str): The URL of the Nougat server. Defaults to "http://localhost:8503".
        """
        self.pdf_path = pdf_path
        self.nougat_url = nougat_url.rstrip("/")
        self._last_markdown: str = ""

    def parse(self) -> ParsedPaper:
        """
        Process the PDF via the Nougat API and extract the paper details.

        Returns:
            ParsedPaper: The parsed paper containing title, abstract, and references.
        """
        markdown = self._fetch_markdown()
        self._last_markdown = markdown

        return ParsedPaper(
            title=self._extract_title(markdown),
            abstract=self._extract_abstract(markdown),
            references=self._extract_references(markdown),
        )

    def _fetch_markdown(self) -> str:
        """
        Send the PDF to the Nougat API and retrieve the markdown response.

        Returns:
            str: The markdown content returned by Nougat.
        """
        url = f"{self.nougat_url}/predict/"

        with open(self.pdf_path, "rb") as f:
            files = {"file": (Path(self.pdf_path).name, f, "application/pdf")}
            response = requests.post(url, files=files)

        response.raise_for_status()

        raw = response.text
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def _extract_title(self, markdown: str) -> str:
        """
        Extract the paper title from the markdown.

        Args:
            markdown (str): The raw markdown text.

        Returns:
            str: The extracted title.
        """
        match = re.search(r"^#\s+(.+?)(?=\n#|\n\n)", markdown, re.MULTILINE | re.DOTALL)
        if not match:
            return ""
        title = match.group(1)
        return re.sub(r"\s+", " ", title).strip()

    def _extract_abstract(self, markdown: str) -> str:
        """
        Extract the paper abstract using combinations of PyMuPDF text and markdown parsing.

        Args:
            markdown (str): The raw markdown text.

        Returns:
            str: The extracted abstract.
        """
        abstract = self._extract_abstract_via_pymupdf()
        if abstract:
            return abstract

        return self._extract_abstract_via_markdown(markdown)

    def _extract_abstract_via_pymupdf(self) -> str:
        """
        Attempt to extract the abstract directly from the PDF using PyMuPDF.

        Returns:
            str: The extracted abstract, or an empty string if it fails.
        """
        try:
            doc = fitz.open(self.pdf_path)
            text = ""
            for i in range(min(2, len(doc))):
                text += doc[i].get_text()
            
            match = re.search(
                r"(?i)\bAbstract\b[\s\n]+(.*?)(?=\n\s*(?:1\.?\s+Introduction|I\.\s+Introduction|Introduction)\b)", 
                text, 
                re.DOTALL
            )
            if match:
                abstract_text = match.group(1).strip()
                abstract_text = re.sub(r"\s+", " ", abstract_text)
                if len(abstract_text) > 50:
                    return abstract_text
        except Exception as e:
            print("PyMuPDF fallback failed:", e)
        
        return ""

    def _extract_abstract_via_markdown(self, markdown: str) -> str:
        """
        Extract the abstract from the markdown text using regex matching.

        Args:
            markdown (str): The raw markdown text.

        Returns:
            str: The extracted abstract.
        """
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

        # No abstract heading at all: grab unmarked body text
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
        """
        Extract the paper references from the markdown text.

        Args:
            markdown (str): The raw markdown text.

        Returns:
            list[str]: A list of extracted references.
        """
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
