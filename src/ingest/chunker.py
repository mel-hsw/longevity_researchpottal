"""Section-aware chunking with metadata prepending.

Design choices (from Phase 1 analysis memo):
  - 768-token target with 100-token overlap (~13 %)
  - Metadata prefix prepended so embeddings capture source context
  - Split at sentence boundaries where possible
  - Separate `text` (prefixed, for FAISS) and `text_raw` (for BM25)
"""

from __future__ import annotations

import re

import tiktoken

from src.schemas import Chunk, ParsedDocument

# Sections we skip (not useful for retrieval)
SKIP_SECTIONS = {"references", "supplementary", "preamble"}

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving trailing whitespace."""
    parts = _SENTENCE_RE.split(text)
    return [p for p in parts if p.strip()]


def chunk_document(
    parsed_doc: ParsedDocument,
    max_tokens: int = 768,
    overlap_tokens: int = 100,
    tokenizer_name: str = "cl100k_base",
) -> list[Chunk]:
    """Chunk a parsed document into retrieval-ready pieces."""
    enc = tiktoken.get_encoding(tokenizer_name)
    all_chunks: list[Chunk] = []

    for section in parsed_doc.sections:
        if section.name in SKIP_SECTIONS:
            continue

        # Metadata prefix (~25 tokens)
        prefix = (
            f"[Source: {parsed_doc.source_id} | "
            f"Section: {section.name} | "
            f"Pages: {section.start_page + 1}-{section.end_page + 1}]\n"
        )

        sentences = _split_sentences(section.text)
        if not sentences:
            continue

        # Greedy sentence-boundary chunking
        current_sents: list[str] = []
        current_tokens = 0

        def _flush(seq_num: int) -> Chunk:
            raw_text = " ".join(current_sents)
            return Chunk(
                chunk_id=f"{parsed_doc.source_id}__{section.name}__{seq_num:03d}",
                source_id=parsed_doc.source_id,
                section=section.name,
                page_start=section.start_page,
                page_end=section.end_page,
                text=prefix + raw_text,
                text_raw=raw_text,
                token_count=len(enc.encode(prefix + raw_text)),
                year=None,  # filled later from manifest
                source_type="journal_article",
            )

        seq = 0
        for sent in sentences:
            sent_tokens = len(enc.encode(sent))
            if current_tokens + sent_tokens > max_tokens and current_sents:
                all_chunks.append(_flush(seq))
                seq += 1
                # Overlap: keep last few sentences whose tokens sum ≤ overlap_tokens
                overlap_sents: list[str] = []
                overlap_tok = 0
                for s in reversed(current_sents):
                    s_tok = len(enc.encode(s))
                    if overlap_tok + s_tok > overlap_tokens:
                        break
                    overlap_sents.insert(0, s)
                    overlap_tok += s_tok
                current_sents = overlap_sents
                current_tokens = overlap_tok
            current_sents.append(sent)
            current_tokens += sent_tokens

        # Final chunk for this section
        if current_sents:
            all_chunks.append(_flush(seq))

    return all_chunks
