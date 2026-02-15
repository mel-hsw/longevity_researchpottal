"""PDF text extraction with section-aware parsing using PyMuPDF."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from src.schemas import ParsedDocument, Section, TextBlock

# Patterns for recognising academic section headers
SECTION_PATTERNS = [
    (r"(?i)^abstract\b", "abstract"),
    (r"(?i)^introduction\b", "introduction"),
    (r"(?i)^(background|literature\s+review)\b", "background"),
    (r"(?i)^(methods?|materials?\s+and\s+methods?|methodology|study\s+design)\b", "methods"),
    (r"(?i)^results?\b", "results"),
    (r"(?i)^discussion\b", "discussion"),
    (r"(?i)^conclusion\b", "conclusion"),
    (r"(?i)^(references|bibliography)\b", "references"),
    (r"(?i)^(supplementary|appendix|acknowledgment)\b", "supplementary"),
]


def _clean_text(text: str) -> str:
    """Normalise whitespace and strip stray artefacts."""
    text = re.sub(r"\s+", " ", text).strip()
    # Remove common PDF artefacts
    text = re.sub(r"^[>\-]+\s*", "", text)
    return text


class PDFParser:
    """Extract structured text with section boundaries from a research PDF."""

    def __init__(self, pdf_path: Path | str):
        self.pdf_path = Path(pdf_path)
        self.doc = fitz.open(str(self.pdf_path))

    # ── public API ────────────────────────────────────────────────────────────

    def parse(self, source_id: str = "") -> ParsedDocument:
        """Full parse: extract pages → detect sections → return structured doc."""
        blocks = self._extract_blocks()
        modal_size = self._modal_font_size(blocks)
        sections = self._detect_sections(blocks, modal_size)

        # Fallback: if fewer than 2 content sections, treat whole doc as "body"
        content_sections = [s for s in sections if s.name not in ("references", "supplementary")]
        if len(content_sections) < 2:
            full_text = " ".join(b.text for b in blocks)
            sections = [Section(
                name="body",
                text=_clean_text(full_text),
                start_page=0,
                end_page=self.doc.page_count - 1,
            )]

        title = self._extract_title(blocks)
        return ParsedDocument(
            source_id=source_id,
            title=title,
            sections=sections,
            total_pages=self.doc.page_count,
            raw_path=str(self.pdf_path),
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _extract_blocks(self) -> list[TextBlock]:
        """Extract text blocks with font metadata from every page."""
        blocks: list[TextBlock] = []
        for page_num in range(self.doc.page_count):
            page = self.doc[page_num]
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:  # skip images
                    continue
                for line in block.get("lines", []):
                    line_text = ""
                    sizes: list[float] = []
                    bolds: list[bool] = []
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                        sizes.append(span.get("size", 0))
                        bolds.append(bool(span.get("flags", 0) & 2**4))  # bit 4 = bold
                    line_text = line_text.strip()
                    if not line_text:
                        continue
                    blocks.append(TextBlock(
                        text=line_text,
                        font_size=max(sizes) if sizes else 0,
                        is_bold=any(bolds),
                        page_number=page_num,
                    ))
        return blocks

    @staticmethod
    def _modal_font_size(blocks: list[TextBlock]) -> float:
        """Return the most common font size (body text)."""
        if not blocks:
            return 10.0
        sizes = [round(b.font_size, 1) for b in blocks if b.font_size > 0]
        if not sizes:
            return 10.0
        return Counter(sizes).most_common(1)[0][0]

    def _match_section(self, text: str, font_size: float, is_bold: bool,
                       modal_size: float) -> str | None:
        """Check if a text block is a section header.  Return normalised name or None."""
        text_clean = text.strip().rstrip(".")
        # Must be short enough to be a header (not a paragraph)
        if len(text_clean) > 80:
            return None
        is_large = font_size > modal_size + 0.3
        for pattern, name in SECTION_PATTERNS:
            if re.match(pattern, text_clean):
                # Accept if font is larger than body OR if bold
                if is_large or is_bold:
                    return name
                # Also accept if the text is very short (standalone header line)
                if len(text_clean.split()) <= 5:
                    return name
        return None

    def _detect_sections(self, blocks: list[TextBlock],
                         modal_size: float) -> list[Section]:
        """Walk through blocks and split into sections."""
        raw_sections: list[dict] = []
        current: dict = {"name": "preamble", "lines": [], "start_page": 0, "end_page": 0}

        for block in blocks:
            sec_name = self._match_section(
                block.text, block.font_size, block.is_bold, modal_size
            )
            if sec_name:
                # Save previous section
                if current["lines"]:
                    raw_sections.append(current)
                current = {
                    "name": sec_name,
                    "lines": [],
                    "start_page": block.page_number,
                    "end_page": block.page_number,
                }
            else:
                current["lines"].append(block.text)
                current["end_page"] = block.page_number

        # Save last section
        if current["lines"]:
            raw_sections.append(current)

        # Convert to Section models
        sections: list[Section] = []
        for raw in raw_sections:
            text = _clean_text(" ".join(raw["lines"]))
            if not text:
                continue
            sections.append(Section(
                name=raw["name"],
                text=text,
                start_page=raw["start_page"],
                end_page=raw["end_page"],
            ))
        return sections

    def _extract_title(self, blocks: list[TextBlock]) -> str:
        """Heuristic: the largest-font text on the first page is the title."""
        first_page_blocks = [b for b in blocks if b.page_number == 0]
        if not first_page_blocks:
            return self.pdf_path.stem
        largest = max(first_page_blocks, key=lambda b: b.font_size)
        return largest.text.strip()[:200]

    def __del__(self):
        if hasattr(self, "doc") and self.doc:
            self.doc.close()
