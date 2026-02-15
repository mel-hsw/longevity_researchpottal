"""LLM generation with Pydantic-enforced structured output.

Design choices (from Phase 1 analysis memo):
  - Pydantic structured output eliminates formatting noise
  - Relaxed prompt: target lengths, not rigid sentence counts
  - Explicit instructions to cite only retrieved chunk_ids
  - Critical grounding rules to suppress parametric knowledge
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.config import Config
from src.schemas import RAGResponse, RetrievalResult

SYSTEM_PROMPT = """\
You are a senior research analyst.  Your role demands the same rigour as
a peer reviewer: every claim must be traceable to the provided context.
Answer questions using ONLY the provided research context.

CRITICAL GROUNDING RULES — read these first:
- For the relevant_quote field in each citation, you MUST copy a short
  verbatim snippet (≤40 words) from the chunk that supports the claim.
  Do not paraphrase — use the source's exact wording.
- Do NOT add facts, numbers, percentages, or details from your own knowledge,
  even if you believe they are correct.  If a detail is not in the context,
  leave it out.
- If the context does not contain sufficient information to answer the
  question, set no_evidence to true and explain what is missing.
- Use a few sentences for simple questions; use multiple paragraphs for
  synthesis questions.  Focus on depth, not rigid formatting.
- Avoid vague filler (e.g., "more research is needed" or "results were
  significant") unless the source text itself uses those exact words.
  Be specific about what the sources actually say.

SCAN ALL CHUNKS before answering:
- Read EVERY chunk provided below before drafting your answer.
- When a claim appears in one chunk, check whether other chunks add nuance,
  contradictions, or supporting detail.

CITATION FORMAT:
  Each chunk header looks like: --- CHUNK <full_chunk_id> (source: <source_id>, ...) ---
  The full_chunk_id has the form: <source_id>__<section>__<number>
  Example: cells_2022__body__001

  In your citations array, use EXACTLY the full chunk_id (e.g. "cells_2022__body__001")
  and the source_id (e.g. "cells_2022").  Do NOT shorten or abbreviate them.

CITATIONRULES:
- Cite every factual claim.  In the answer text, write citations inline
  as (source_id, chunk_id), e.g. (cells_2022, cells_2022__body__001).
- ONLY cite chunk_ids that appear in the provided context below.

CONFIDENCE CALIBRATION:
- "high"  — multiple chunks explicitly and consistently support the answer;
            key claims can be traced to verbatim text in the context.
- "medium" — evidence is present but partial, from a single chunk, or the
             chunks are tangentially related to the query.
- "low"   — evidence is thin, the answer relies on inference from the
            context rather than direct statements, or the context only
            loosely relates to the question.

Context:
{context}
"""


class RAGGenerator:
    """Generate grounded answers from retrieved chunks."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.llm = ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=8000,
            api_key=self.config.openai_api_key,
        )
        self.structured_llm = self.llm.with_structured_output(RAGResponse)

    def generate(
        self,
        query: str,
        retrieval: RetrievalResult,
        prompt_version: str = "v1",
    ) -> RAGResponse:
        """Generate a cited answer for the query given retrieval results."""
        # Short-circuit when no evidence found
        if not retrieval.has_sufficient_evidence:
            return RAGResponse(
                answer=(
                    "I could not find sufficient evidence in the research "
                    "corpus to answer this question.  You may want to expand "
                    "the corpus or try rephrasing the query."
                ),
                citations=[],
                confidence="low",
                evidence_quality="No chunks met the similarity threshold.",
                no_evidence=True,
            )

        context = self._format_context(retrieval)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": query},
        ]

        response: RAGResponse = self.structured_llm.invoke(messages)
        return response

    @staticmethod
    def _format_context(retrieval: RetrievalResult) -> str:
        parts = []
        for c in retrieval.chunks:
            parts.append(
                f"--- CHUNK {c.chunk_id} (source: {c.source_id}, "
                f"section: {c.section}, score: {c.combined_score:.3f}) ---\n"
                f"{c.text}\n"
            )
        return "\n".join(parts)
