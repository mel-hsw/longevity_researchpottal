"""LLM-based reranker: score chunk relevance with a single batch call.

After hybrid retrieval fuses BM25 + vector candidates, this reranker
uses the LLM to score each chunk's relevance to the query and re-sorts.
This filters out tangentially related chunks that matched on keywords
but lack the specific evidence the query requires.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from src.config import Config
from src.schemas import RetrievedChunk


# ── Structured output models for the reranker ────────────────────────────────

class ChunkScore(BaseModel):
    chunk_id: str
    relevance: int = Field(ge=0, le=10, description="0 = irrelevant, 10 = directly answers the query")


class RerankOutput(BaseModel):
    scores: list[ChunkScore]


RERANK_PROMPT = """\
You are a relevance judge for a research retrieval system.

Given the QUERY and a set of candidate PASSAGES, rate each passage's relevance
to the query on a scale of 0-10:
  10 = directly and specifically answers the query
   7 = contains relevant information but not a direct answer
   4 = tangentially related to the topic
   1 = mentions a keyword but is otherwise irrelevant
   0 = completely irrelevant

QUERY: {query}

PASSAGES:
{passages}

Return a relevance score for EVERY passage listed above."""


class LLMReranker:
    """Rerank retrieved chunks using an LLM relevance scorer."""

    def __init__(self, config: Config):
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=0.0,
            api_key=config.openai_api_key,
        ).with_structured_output(RerankOutput)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Score each chunk's relevance and return re-sorted list."""
        if len(chunks) <= 1:
            return chunks

        # Build passage list for the prompt
        passage_lines = []
        for c in chunks:
            # Truncate to keep prompt manageable
            preview = c.text[:400].replace("\n", " ")
            passage_lines.append(f"[{c.chunk_id}]: {preview}")
        passages_str = "\n\n".join(passage_lines)

        prompt = RERANK_PROMPT.format(query=query, passages=passages_str)

        try:
            result: RerankOutput = self.llm.invoke(prompt)
            score_map = {s.chunk_id: s.relevance for s in result.scores}
        except Exception:
            # If reranking fails, return original order
            return chunks

        # Sort by rerank score, breaking ties with original combined_score
        return sorted(
            chunks,
            key=lambda c: (score_map.get(c.chunk_id, 0), c.combined_score),
            reverse=True,
        )
