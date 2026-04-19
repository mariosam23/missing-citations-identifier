import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from entities import ParsedPaper

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
        xml_content = self._fetch_tei_xml()
        root = ET.fromstring(xml_content)

        return ParsedPaper(
            title=self._extract_title(root),
            abstract=self._extract_abstract(root),
            references=self._extract_references(root),
            sections=self._extract_sections(root),
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

    def _extract_sections(self, root: ET.Element) -> dict[str, str]:
        """Extract named body sections from the TEI XML, including nested subsections."""
        ns = self.namespace
        sections: dict[str, str] = {}
        body = root.find(".//tei:body", ns)
        if body is None:
            return sections

        for div in body.findall("tei:div", ns):
            self._collect_sections(div=div, sections=sections, heading_path=[])

        return sections

    def _collect_sections(
        self,
        div: ET.Element,
        sections: dict[str, str],
        heading_path: list[str],
    ) -> None:
        """Walk nested TEI divs without dropping subsection content."""
        ns = self.namespace
        head = div.find("tei:head", ns)
        heading_text = head.text.strip() if head is not None and head.text else ""

        current_path = heading_path + [heading_text] if heading_text else heading_path.copy()
        section_name = " / ".join(current_path) if current_path else "Untitled Section"

        paragraphs = []
        for p in div.findall("tei:p", ns):
            paragraph = "".join(p.itertext()).strip()
            if paragraph:
                paragraphs.append(paragraph)

        if paragraphs:
            section_text = "\n".join(paragraphs)
            if section_name in sections:
                sections[section_name] = f"{sections[section_name]}\n{section_text}"
            else:
                sections[section_name] = section_text

        for child_div in div.findall("tei:div", ns):
            self._collect_sections(div=child_div, sections=sections, heading_path=current_path)

