"""Post-retrieval and post-generation guardrails.

Includes:
  - Citation verification (Phase 1 trust behaviour)
  - Topic presence check (verify query concepts appear in context)
  - Entity check (actively strip ungrounded entities from answer)
"""

from __future__ import annotations

import re

from src.schemas import RAGResponse, RetrievalResult

# ── Stop-words for topic presence check ───────────────────────────────────────
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "about", "above",
    "after", "again", "all", "also", "and", "any", "as", "at", "because",
    "before", "between", "both", "but", "by", "could", "did", "do", "does",
    "doing", "down", "during", "each", "few", "for", "from", "further",
    "get", "got", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "it", "its", "itself", "just",
    "know", "let", "like", "make", "me", "might", "more", "most", "much",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "so", "some", "such", "take", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "us", "very", "we", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "you", "your", "yours", "yourself",
    "yourselves", "according", "across", "describe", "described", "does",
    "effect", "find", "findings", "key", "main", "role", "say", "suggest",
    "what", "associated", "evidence", "corpus",
})


# ── Citation verification ─────────────────────────────────────────────────────

def _resolve_chunk_id(raw_id: str, retrieved_ids: set[str]) -> str | None:
    """Try to match a possibly-abbreviated chunk_id to a retrieved one."""
    if raw_id in retrieved_ids:
        return raw_id
    for rid in retrieved_ids:
        if rid.endswith(f"__{raw_id}") or rid.endswith(raw_id):
            return rid
    return None


def verify_citations(
    response: RAGResponse,
    retrieval: RetrievalResult,
) -> RAGResponse:
    """Remove citations that reference chunk_ids NOT in the retrieved context."""
    retrieved_ids = {c.chunk_id for c in retrieval.chunks}

    valid = []
    removed = []
    for cite in response.citations:
        resolved = _resolve_chunk_id(cite.chunk_id, retrieved_ids)
        if resolved:
            cite.chunk_id = resolved
            valid.append(cite)
        else:
            removed.append(cite)

    if removed:
        removed_ids = [c.chunk_id for c in removed]
        response.caveats.append(
            f"Removed {len(removed)} unverified citation(s): {removed_ids}"
        )
        if response.confidence == "high":
            response.confidence = "medium"

    response.citations = valid
    return response


# ── Topic presence check ──────────────────────────────────────────────────────

def _extract_query_keywords(query: str) -> set[str]:
    """Extract meaningful keywords from the query (lowercased)."""
    # Tokenise on word boundaries, keep alphanumeric + hyphens
    tokens = re.findall(r"[A-Za-z0-9α-ωΑ-Ω][\w\-/]*", query)
    keywords = set()
    for tok in tokens:
        low = tok.lower()
        if low not in _STOP_WORDS and len(low) >= 3:
            keywords.add(low)
    return keywords


def topic_presence_check(
    query: str,
    chunks: list,
    min_keyword_hits: int = 2,
) -> bool:
    """Return True if the retrieved chunks cover the query's key concepts.

    Concatenates chunk text and checks how many query keywords appear.
    Returns False (= topic absent) when fewer than *min_keyword_hits*
    distinctive query keywords are found in the combined context.
    """
    keywords = _extract_query_keywords(query)
    if not keywords:
        return True  # can't judge — proceed

    context_lower = " ".join(c.text.lower() for c in chunks)
    hits = sum(1 for kw in keywords if kw in context_lower)

    # If fewer than min_keyword_hits (or fewer than 30 % of keywords), flag
    threshold = min(min_keyword_hits, max(1, len(keywords) // 3))
    return hits >= threshold


# ── Post-generation entity check ──────────────────────────────────────────────

# Patterns for entities that are likely to be hallucinated if not in context
_NUMBER_PATTERN = re.compile(
    r"\b\d+[\.\,]?\d*\s*[%℃°]"              # e.g. 31%, 37°C
    r"|\b\d+[\.\,]?\d*\s*(?:mg|kg|ml|min|hour|week|year|day|month)\b"   # e.g. 150 min
    r"|\b\d+[\.\,]?\d*[-–]\d+[\.\,]?\d*\b",  # e.g. 7.2-8.0
    re.IGNORECASE,
)

_GENE_PATTERN = re.compile(
    r"\b[A-Z][A-Z0-9]{1,}[-/]?[A-Z0-9α-ω]*\b"  # e.g. AMPK, PGC-1α, HbA1c, SIRT1
)

_PROPER_NOUN_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"  # e.g. Comprehensive Meta Analysis
)


def _extract_entities(text: str) -> set[str]:
    """Extract notable entities: numbers-with-units, gene names, proper nouns."""
    entities: set[str] = set()
    for m in _NUMBER_PATTERN.finditer(text):
        entities.add(m.group().strip())
    for m in _GENE_PATTERN.finditer(text):
        val = m.group().strip()
        # Skip very short or common abbreviations that aren't entities
        if len(val) >= 3 and val not in {"NOT", "AND", "BUT", "THE", "FOR",
                                          "ARE", "WAS", "HAS", "HAD", "ALL",
                                          "CAN", "MAY", "USE", "SET"}:
            entities.add(val)
    for m in _PROPER_NOUN_PATTERN.finditer(text):
        entities.add(m.group().strip())
    return entities


def entity_check(
    response: RAGResponse,
    context_text: str,
) -> RAGResponse:
    """Strip sentences containing ungrounded entities from the answer.

    Instead of merely flagging suspicious entities, this function actively
    removes sentences that contain entities (gene names, specific numbers,
    proper nouns) not found anywhere in the retrieved context.  This
    prevents specificity inflation and knowledge leakage from surviving
    into the final answer.
    """
    if response.no_evidence:
        return response

    answer_entities = _extract_entities(response.answer)
    if not answer_entities:
        return response

    context_lower = context_text.lower()
    ungrounded: set[str] = set()
    for ent in answer_entities:
        if ent not in context_text and ent.lower() not in context_lower:
            ungrounded.add(ent)

    if not ungrounded:
        return response

    # ── Active stripping: remove sentences that contain ungrounded entities ──
    stripped_sentences: list[str] = []
    kept_lines: list[str] = []

    for line in response.answer.split("\n"):
        # Check each sentence within the line
        sentences = re.split(r"(?<=[.!?])\s+", line)
        kept_in_line: list[str] = []
        for sentence in sentences:
            sentence_ungrounded = [
                ent for ent in ungrounded
                if ent in sentence or ent.lower() in sentence.lower()
            ]
            if sentence_ungrounded:
                stripped_sentences.append(sentence.strip())
            else:
                kept_in_line.append(sentence)
        if kept_in_line:
            kept_lines.append(" ".join(kept_in_line))
        elif line.strip() == "":
            kept_lines.append("")

    if stripped_sentences:
        cleaned_answer = "\n".join(kept_lines).strip()
        # Only apply stripping if we haven't gutted the answer
        if len(cleaned_answer) >= len(response.answer) * 0.3:
            response.answer = cleaned_answer
            response.caveats.append(
                f"Entity guard: removed {len(stripped_sentences)} sentence(s) "
                f"containing terms not found in context: "
                f"{sorted(ungrounded)[:5]}"
            )
        else:
            # Too much would be removed — flag instead of stripping
            response.caveats.append(
                f"Entity check: {len(ungrounded)} term(s) in the answer may not "
                f"appear in the retrieved context: {sorted(ungrounded)[:5]}"
            )

        if response.confidence == "high":
            response.confidence = "medium"

    return response
