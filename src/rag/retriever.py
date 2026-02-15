"""Hybrid retriever: BM25 + FAISS with Reciprocal Rank Fusion.

Custom implementation (not EnsembleRetriever) to preserve per-result
scores for the similarity threshold guardrail from Phase 1.

Includes chunk expansion: after selecting the top-k chunks, adjacent
chunks from the same source/section are added to provide more complete
context to the generator.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from src.config import Config
from src.rag.embedder import get_embeddings_model
from src.schemas import RetrievalResult, RetrievedChunk


class HybridRetriever:
    """Retrieve chunks using weighted BM25 + vector search with RRF."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

        # Load FAISS
        embeddings = get_embeddings_model(self.config)
        self.vectorstore = FAISS.load_local(
            str(self.config.faiss_index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        # Load BM25 docs
        with open(self.config.bm25_index_path, "rb") as f:
            bm25_docs = pickle.load(f)
        self.bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        self.bm25_retriever.k = self.config.bm25_k

        # Load chunk index for expansion (chunk_id → chunk data)
        self._chunk_index: dict[str, dict] = {}
        if self.config.chunks_path.exists():
            with open(self.config.chunks_path) as f:
                for line in f:
                    chunk = json.loads(line)
                    self._chunk_index[chunk["chunk_id"]] = chunk

    # ── Core retrieval ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Run hybrid retrieval and return ranked chunks with scores."""
        vw = vector_weight if vector_weight is not None else self.config.vector_weight
        bw = bm25_weight if bm25_weight is not None else self.config.bm25_weight
        k = top_k if top_k is not None else self.config.rerank_candidates

        # 1. Vector search with scores
        vector_results = self.vectorstore.similarity_search_with_score(
            query, k=self.config.vector_k
        )
        # FAISS returns L2 distance; convert to similarity: 1 / (1 + dist)
        vector_scored: dict[str, tuple] = {}
        for doc, dist in vector_results:
            cid = doc.metadata["chunk_id"]
            sim = 1.0 / (1.0 + dist)
            vector_scored[cid] = (doc, sim)

        # 2. BM25 search (no native scores; use rank-based scoring)
        bm25_results = self.bm25_retriever.invoke(query)
        bm25_scored: dict[str, tuple] = {}
        for rank, doc in enumerate(bm25_results):
            cid = doc.metadata["chunk_id"]
            score = 1.0 / (rank + 1)
            bm25_scored[cid] = (doc, score)

        # 3. Reciprocal Rank Fusion
        all_ids = set(vector_scored.keys()) | set(bm25_scored.keys())
        fused: list[RetrievedChunk] = []
        for cid in all_ids:
            vec_doc, vec_score = vector_scored.get(cid, (None, 0.0))
            bm25_doc, bm25_score = bm25_scored.get(cid, (None, 0.0))
            combined = vw * vec_score + bw * bm25_score
            doc = vec_doc or bm25_doc
            fused.append(RetrievedChunk(
                chunk_id=cid,
                source_id=doc.metadata.get("source_id", ""),
                section=doc.metadata.get("section", ""),
                page_start=doc.metadata.get("page_start", 0),
                page_end=doc.metadata.get("page_end", 0),
                text=doc.page_content,
                vector_score=vec_score,
                bm25_score=bm25_score,
                combined_score=combined,
            ))
        fused.sort(key=lambda x: x.combined_score, reverse=True)

        # 4. Apply similarity threshold
        above = [c for c in fused if c.vector_score >= self.config.similarity_threshold]

        # 5. Take top-k (larger set for reranking)
        final = above[:k]

        return RetrievalResult(
            query=query,
            chunks=final,
            all_candidates=len(fused),
            above_threshold=len(above),
            has_sufficient_evidence=len(final) > 0,
        )

    # ── Chunk expansion ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_chunk_id(chunk_id: str) -> tuple[str, str, int] | None:
        """Parse 'source_id__section__NNN' into components."""
        parts = chunk_id.rsplit("__", 2)
        if len(parts) == 3:
            try:
                return parts[0], parts[1], int(parts[2])
            except ValueError:
                return None
        return None

    def expand_chunks(self, retrieval: RetrievalResult) -> RetrievalResult:
        """Add adjacent chunks from the same source/section.

        For each retrieved chunk, look up ±window adjacent chunks in the
        chunk index and add them if not already present.  Caps total chunks
        at max_chunks_after_expand.
        """
        if not self._chunk_index or self.config.chunk_expand_window <= 0:
            return retrieval

        existing_ids = {c.chunk_id for c in retrieval.chunks}
        expanded: list[RetrievedChunk] = list(retrieval.chunks)

        for chunk in retrieval.chunks:
            parsed = self._parse_chunk_id(chunk.chunk_id)
            if not parsed:
                continue
            source_id, section, num = parsed

            for offset in range(-self.config.chunk_expand_window,
                                self.config.chunk_expand_window + 1):
                if offset == 0:
                    continue
                adj_id = f"{source_id}__{section}__{num + offset:03d}"
                if adj_id in existing_ids or adj_id not in self._chunk_index:
                    continue

                adj_data = self._chunk_index[adj_id]
                expanded.append(RetrievedChunk(
                    chunk_id=adj_id,
                    source_id=adj_data.get("source_id", source_id),
                    section=adj_data.get("section", section),
                    page_start=adj_data.get("page_start", 0),
                    page_end=adj_data.get("page_end", 0),
                    text=adj_data.get("text", ""),
                    vector_score=chunk.vector_score * 0.8,  # slight discount
                    bm25_score=0.0,
                    combined_score=chunk.combined_score * 0.8,
                ))
                existing_ids.add(adj_id)

            if len(expanded) >= self.config.max_chunks_after_expand:
                break

        retrieval.chunks = expanded[:self.config.max_chunks_after_expand]
        return retrieval
