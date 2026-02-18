"""Research artifact generation.

Supported artifact types
------------------------
evidence_table  — Claim | Evidence snippet | Citation | Confidence | Notes
                  Derived directly from a RAGResponse (no extra LLM calls).

Each function returns a list of row-dicts that can be rendered in the UI,
exported to CSV, or formatted as Markdown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.schemas import RAGResponse


# ── Evidence-table row ────────────────────────────────────────────────────────

@dataclass
class EvidenceRow:
    claim: str
    evidence_snippet: str
    citation: str          # "(source_id, chunk_id)"
    confidence: str        # inherited from RAGResponse
    notes: str

    def to_dict(self) -> dict[str, str]:
        return {
            "Claim": self.claim,
            "Evidence snippet": self.evidence_snippet,
            "Citation": self.citation,
            "Confidence": self.confidence,
            "Notes": self.notes,
        }


@dataclass
class EvidenceTable:
    query: str
    rows: list[EvidenceRow] = field(default_factory=list)
    overall_confidence: str = "medium"
    no_evidence: bool = False

    def to_records(self) -> list[dict[str, str]]:
        return [r.to_dict() for r in self.rows]


# ── Internal helpers ──────────────────────────────────────────────────────────

# Match a full inline citation block: (src, chunk_id) or (src, chunk_id; src, chunk_id)
_CITATION_BLOCK_RE = re.compile(r"\(([^)]+)\)")

# Match a single "source_id, chunk_id" pair inside a block (chunk_id contains __)
_PAIR_RE = re.compile(
    r"([a-z0-9_]+),\s*([a-z0-9_]+(?:__[a-zA-Z0-9_]+)+)"
)


def _parse_inline_pairs(block_content: str) -> list[tuple[str, str]]:
    """Return all (source_id, chunk_id) pairs from inside a citation block."""
    return _PAIR_RE.findall(block_content)


def _strip_citations(text: str) -> str:
    """Remove all inline citation blocks from a sentence."""
    return _CITATION_BLOCK_RE.sub("", text).strip().strip(".").strip()


def _split_sentences(text: str) -> list[str]:
    """Split answer text into individual sentences (rough heuristic)."""
    # Split on sentence-ending punctuation followed by a space and uppercase or newline
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\n])", text)
    # Also split on explicit newlines used for sub-bullets
    sentences: list[str] = []
    for part in parts:
        for line in part.split("\n"):
            line = line.strip().strip("-").strip()
            if line:
                sentences.append(line)
    return sentences


# ── Public API ────────────────────────────────────────────────────────────────

def build_evidence_table(query: str, response: RAGResponse) -> EvidenceTable:
    """Convert a RAGResponse into an EvidenceTable artifact.

    Algorithm
    ---------
    1. Build a lookup from chunk_id → Citation (for the relevant_quote).
    2. Walk every sentence in the answer.
    3. For each inline citation block found in that sentence, emit one row:
         - Claim   = the sentence text minus the citation markers
         - Evidence = the relevant_quote from the matched Citation
         - Citation = "(source_id, chunk_id)"
         - Confidence = response.confidence
         - Notes  = response.caveats (joined)
    4. Any Citation not encountered during step 3 gets a fallback row so
       nothing is silently dropped.
    """
    if response.no_evidence:
        return EvidenceTable(
            query=query,
            rows=[],
            overall_confidence="low",
            no_evidence=True,
        )

    # Build chunk_id → Citation lookup
    cit_lookup: dict[str, Any] = {c.chunk_id: c for c in response.citations}
    notes = "; ".join(response.caveats) if response.caveats else ""

    rows: list[EvidenceRow] = []
    seen_chunk_ids: set[str] = set()

    sentences = _split_sentences(response.answer)
    for sentence in sentences:
        blocks = _CITATION_BLOCK_RE.findall(sentence)
        if not blocks:
            continue

        clean_claim = _strip_citations(sentence)
        if not clean_claim:
            continue

        for block in blocks:
            pairs = _parse_inline_pairs(block)
            for source_id, chunk_id in pairs:
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)

                cit = cit_lookup.get(chunk_id)
                evidence = cit.relevant_quote if cit else ""
                citation_str = f"({source_id}, {chunk_id})"

                rows.append(
                    EvidenceRow(
                        claim=clean_claim,
                        evidence_snippet=evidence,
                        citation=citation_str,
                        confidence=response.confidence,
                        notes=notes,
                    )
                )

    # Fallback: include any citations not found via sentence scanning
    for cit in response.citations:
        if cit.chunk_id not in seen_chunk_ids:
            rows.append(
                EvidenceRow(
                    claim="(see citation)",
                    evidence_snippet=cit.relevant_quote,
                    citation=f"({cit.source_id}, {cit.chunk_id})",
                    confidence=response.confidence,
                    notes=notes,
                )
            )

    return EvidenceTable(
        query=query,
        rows=rows,
        overall_confidence=response.confidence,
        no_evidence=False,
    )
