import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from entities import ParsedPaper
from utils.config import config

# GROBID's TEI ``xml:id`` attribute lives in the standard XML namespace.
_XML_ID_ATTR = "{http://www.w3.org/XML/1998/namespace}id"

# Marker we inject into section text whenever GROBID emits a `<ref target="#bX">`
# in-text citation. The downstream sentence extractor parses these out and
# associates them with the sentence they fell into. Format: ``[CITE:bX]``.
CITE_MARKER_PREFIX = "[CITE:"
CITE_MARKER_SUFFIX = "]"


def cite_marker(bibkey: str) -> str:
    return f"{CITE_MARKER_PREFIX}{bibkey}{CITE_MARKER_SUFFIX}"


class GrobidPDFParser:
    """
    Parses academic PDFs using a running GROBID API server.

    In addition to extracting the title, abstract, and section text, this
    parser preserves the linkage between in-text citations and bibliography
    entries: each ``<biblStruct xml:id="bX">`` becomes a key in
    ``ParsedPaper.bibliography`` and each ``<ref target="#bX">`` in the body
    is replaced with a ``[CITE:bX]`` marker in the section text. Downstream
    sentence extraction reads those markers to attribute citations to the
    sentence they fell into.
    """

    def __init__(self, pdf_path: str, grobid_url: str | None = None):
        """
        Initialize the GrobidPDFParser.

        Args:
            pdf_path (str): Path to the PDF file to be parsed.
            grobid_url (str | None): The URL of the GROBID server. Defaults to
                ``config.GROBID_URL`` when omitted.
        """
        self.pdf_path = pdf_path
        self.grobid_url = (grobid_url or config.GROBID_URL).rstrip("/")
        self.namespace = {"tei": "http://www.tei-c.org/ns/1.0"}

    def parse(self) -> ParsedPaper:
        xml_content = self._fetch_tei_xml()
        root = ET.fromstring(xml_content)

        bibliography = self._extract_bibliography(root)
        sections = self._extract_sections(root)

        return ParsedPaper(
            title=self._extract_title(root),
            abstract=self._extract_abstract(root),
            references=list(bibliography.values()),
            sections=sections,
            bibliography=bibliography,
        )

    def _fetch_tei_xml(self) -> bytes:
        """Send the PDF to GROBID and retrieve the TEI XML response."""
        url = f"{self.grobid_url}/api/processFulltextDocument"

        with open(self.pdf_path, "rb") as f:
            files = {"input": (Path(self.pdf_path).name, f, "application/pdf")}
            # ``includeRawCitations=1`` keeps the original text of each
            # bibliography entry inside ``<note type="raw_reference">``, which
            # is the cleanest input for our downstream resolver.
            data = {"includeRawCitations": "1"}
            response = requests.post(url, files=files, data=data)

        response.raise_for_status()
        return response.content

    def _extract_title(self, root: ET.Element) -> str:
        title_elem = root.find(".//tei:titleStmt/tei:title", self.namespace)
        if title_elem is not None and title_elem.text:
            return title_elem.text.strip()
        return ""

    def _extract_abstract(self, root: ET.Element) -> str:
        abstract_elems = root.findall(".//tei:profileDesc/tei:abstract//tei:p", self.namespace)
        abstract = " ".join([self._render_paragraph(elem) for elem in abstract_elems])
        return abstract.strip()

    # ------------------------------------------------------------------
    # Bibliography extraction
    # ------------------------------------------------------------------

    def _extract_bibliography(self, root: ET.Element) -> dict[str, str]:
        """Return ``{bibkey: raw_reference_text}`` for every bibliography entry.

        ``raw_reference_text`` is rich enough to hand to a fuzzy resolver:
        title + author surnames + venue + year + DOI when GROBID provides
        them. Falls back to GROBID's ``<note type="raw_reference">`` text
        if the structured fields are sparse.
        """
        bibliography: dict[str, str] = {}
        bibl_structs = root.findall(".//tei:listBibl/tei:biblStruct", self.namespace)
        for bibl in bibl_structs:
            bibkey = bibl.get(_XML_ID_ATTR)
            if not bibkey:
                continue
            raw = self._render_bibl_struct(bibl)
            if raw:
                bibliography[bibkey] = raw
        return bibliography

    def _render_bibl_struct(self, bibl: ET.Element) -> str:
        """Render a single ``<biblStruct>`` to a human-readable reference string."""
        ns = self.namespace

        # Prefer GROBID's preserved raw citation when available.
        raw_note = bibl.find(".//tei:note[@type='raw_reference']", ns)
        if raw_note is not None and raw_note.text and raw_note.text.strip():
            return " ".join(raw_note.text.split())

        parts: list[str] = []

        # Title (analytic preferred, fall back to monogr).
        title_elem = bibl.find(".//tei:analytic/tei:title", ns)
        if title_elem is None:
            title_elem = bibl.find(".//tei:monogr/tei:title", ns)
        if title_elem is not None and title_elem.text:
            parts.append(title_elem.text.strip())

        # Authors — surnames are usually enough for OpenAlex search.
        author_names: list[str] = []
        for author in bibl.findall(".//tei:analytic/tei:author/tei:persName", ns):
            surname = author.find("tei:surname", ns)
            if surname is not None and surname.text:
                author_names.append(surname.text.strip())
        if author_names:
            parts.append(", ".join(author_names))

        # Venue.
        venue_elem = bibl.find(".//tei:monogr/tei:title", ns)
        if venue_elem is not None and venue_elem.text and venue_elem != title_elem:
            parts.append(venue_elem.text.strip())

        # Year.
        date_elem = bibl.find(".//tei:imprint/tei:date", ns)
        when = date_elem.get("when") if date_elem is not None else None
        if when:
            parts.append(when[:4])

        # DOI — kept literally so the resolver's regex can pick it up.
        doi_elem = bibl.find(".//tei:idno[@type='DOI']", ns)
        if doi_elem is not None and doi_elem.text:
            parts.append(doi_elem.text.strip())

        return ". ".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Section extraction (with citation markers preserved)
    # ------------------------------------------------------------------

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

        paragraphs: list[str] = []
        for p in div.findall("tei:p", ns):
            paragraph = self._render_paragraph(p)
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

    def _render_paragraph(self, p: ET.Element) -> str:
        """Flatten a ``<p>`` to text, replacing ``<ref target="#bX">`` with ``[CITE:bX]``.

        GROBID emits in-text citations as ``<ref type="bibr" target="#b3">[3]</ref>``.
        ``itertext()`` would walk children but lose the linkage; we walk
        manually so each citation becomes a parseable marker the sentence
        extractor can pick up.
        """
        out: list[str] = []
        if p.text:
            out.append(p.text)

        for child in p:
            tag = child.tag.split("}", 1)[-1]  # strip namespace
            if tag == "ref":
                target = child.get("target") or ""
                ref_type = child.get("type") or ""
                if ref_type == "bibr" and target.startswith("#"):
                    bibkey = target.lstrip("#")
                    out.append(cite_marker(bibkey))
                elif child.text:
                    out.append(child.text)
            else:
                # Some elements (formula, hi, etc.) carry text we want to keep.
                inner = "".join(child.itertext())
                if inner:
                    out.append(inner)

            if child.tail:
                out.append(child.tail)

        return "".join(out).strip()
