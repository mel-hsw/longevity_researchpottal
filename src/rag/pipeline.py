"""End-to-end RAG pipeline: retrieve → rerank → expand → check → generate → verify → log."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.config import Config
from src.rag.generator import RAGGenerator
from src.rag.guardrails import entity_check, topic_presence_check, verify_citations
from src.rag.reranker import LLMReranker
from src.rag.retriever import HybridRetriever
from src.schemas import RAGResponse, RetrievalResult


class QueryLogger:
    """Append query/response records to a JSONL log file."""

    def __init__(self, log_path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        query: str,
        retrieval: RetrievalResult,
        response: RAGResponse,
        prompt_version: str,
        model: str,
    ):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "prompt_version": prompt_version,
            "model": model,
            "retrieval": {
                "method": "hybrid_bm25_faiss_reranked",
                "total_candidates": retrieval.all_candidates,
                "above_threshold": retrieval.above_threshold,
                "has_sufficient_evidence": retrieval.has_sufficient_evidence,
                "returned_chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "source_id": c.source_id,
                        "section": c.section,
                        "vector_score": round(c.vector_score, 4),
                        "bm25_score": round(c.bm25_score, 4),
                        "combined_score": round(c.combined_score, 4),
                        "text_preview": c.text[:200],
                    }
                    for c in retrieval.chunks
                ],
            },
            "response": {
                "answer": response.answer,
                "citations": [c.model_dump() for c in response.citations],
                "confidence": response.confidence,
                "evidence_quality": response.evidence_quality,
                "no_evidence": response.no_evidence,
                "caveats": response.caveats,
            },
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


class RAGPipeline:
    """Wire retriever + reranker + generator + guardrails + logger together."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.retriever = HybridRetriever(self.config)
        self.reranker = LLMReranker(self.config)
        self.generator = RAGGenerator(self.config)
        self.logger = QueryLogger(self.config.log_path)
        self._last_retrieval: RetrievalResult | None = None

    def query(
        self,
        question: str,
        prompt_version: str = "v1",
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
    ) -> RAGResponse:
        """Run the full RAG pipeline for a single question.

        Flow:
          1. Hybrid retrieve (BM25 + FAISS) → rerank_candidates chunks
          2. LLM rerank → keep top final_k
          3. Expand with adjacent chunks from same source/section
          4. Topic presence check → short-circuit if topic absent
          5. Generate answer with citations
          6. Citation verification → remove hallucinated citations
          7. Entity check → strip ungrounded entities
          8. Log everything
        """
        # 1. Retrieve (returns rerank_candidates chunks)
        retrieval = self.retriever.retrieve(
            question,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )

        if not retrieval.has_sufficient_evidence:
            self._last_retrieval = retrieval
            response = self._no_evidence_response()
            self.logger.log(
                question, retrieval, response, prompt_version, self.config.llm_model
            )
            return response

        # 2. Rerank and keep top final_k
        retrieval.chunks = self.reranker.rerank(question, retrieval.chunks)
        retrieval.chunks = retrieval.chunks[: self.config.final_k]

        # 3. Expand with adjacent chunks
        retrieval = self.retriever.expand_chunks(retrieval)

        # 4. Topic presence check
        if not topic_presence_check(question, retrieval.chunks):
            retrieval.has_sufficient_evidence = False
            self._last_retrieval = retrieval
            response = self._no_evidence_response(
                reason="The retrieved passages do not appear to cover the "
                       "key concepts in your query.  The corpus may not "
                       "contain relevant information on this topic."
            )
            self.logger.log(
                question, retrieval, response, prompt_version, self.config.llm_model
            )
            return response

        self._last_retrieval = retrieval

        # 5. Generate
        response = self.generator.generate(question, retrieval, prompt_version)

        # 6. Citation verification
        response = verify_citations(response, retrieval)

        # 7. Entity check
        context_text = "\n".join(c.text for c in retrieval.chunks)
        response = entity_check(response, context_text)

        # 8. Log
        self.logger.log(
            question, retrieval, response, prompt_version, self.config.llm_model
        )
        return response

    @staticmethod
    def _no_evidence_response(reason: str | None = None) -> RAGResponse:
        return RAGResponse(
            answer=reason or (
                "I could not find sufficient evidence in the research "
                "corpus to answer this question.  You may want to expand "
                "the corpus or try rephrasing the query."
            ),
            citations=[],
            confidence="low",
            evidence_quality="Insufficient evidence in retrieved context.",
            no_evidence=True,
        )
