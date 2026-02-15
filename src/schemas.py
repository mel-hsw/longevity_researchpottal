"""Pydantic data models used across the entire Phase 2 RAG pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Ingestion models ──────────────────────────────────────────────────────────

class TextBlock(BaseModel):
    text: str
    font_size: float
    is_bold: bool
    page_number: int


class Section(BaseModel):
    name: str
    text: str
    start_page: int
    end_page: int


class ParsedDocument(BaseModel):
    source_id: str
    title: str
    sections: list[Section]
    total_pages: int
    raw_path: str


class Chunk(BaseModel):
    chunk_id: str          # e.g. barry_2014__methods__003
    source_id: str
    section: str
    page_start: int
    page_end: int
    text: str              # metadata-prepended (for FAISS)
    text_raw: str          # original text (for BM25)
    token_count: int
    year: int | None = None
    source_type: str = "journal_article"


# ── RAG models ────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    source_id: str
    chunk_id: str
    relevant_quote: str = ""


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: str = "medium"       # high / medium / low
    evidence_quality: str = ""
    no_evidence: bool = False
    caveats: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_id: str
    section: str
    page_start: int
    page_end: int
    text: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    all_candidates: int = 0
    above_threshold: int = 0
    has_sufficient_evidence: bool = True
