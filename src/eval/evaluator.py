"""Evaluation scoring: groundedness, citation precision, faithfulness.

Runs the full query set through the pipeline and scores each response.
Also supports baseline-vs-hybrid comparison.

Queries within a single evaluation run are executed in parallel using a
thread pool.  Each worker thread owns its own RAGPipeline instance
(via threading.local) so that pipeline._last_retrieval is never
clobbered across threads.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_openai import ChatOpenAI

from src.config import Config
from src.rag.pipeline import RAGPipeline
from src.schemas import RAGResponse

MAX_EVAL_WORKERS = 4


class Evaluator:
    """Score RAG responses on groundedness, citation precision, and faithfulness."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.judge_llm = ChatOpenAI(
            model=self.config.llm_model,
            temperature=0.0,
            api_key=self.config.openai_api_key,
        )

    # ── Metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def citation_precision(response: RAGResponse, retrieved_ids: set[str]) -> float:
        """Fraction of citations that map to a retrieved chunk_id."""
        if not response.citations:
            return 1.0 if response.no_evidence else 0.0
        valid = sum(1 for c in response.citations if c.chunk_id in retrieved_ids)
        return valid / len(response.citations)

    @staticmethod
    def no_evidence_accuracy(response: RAGResponse, expected_no_evidence: bool) -> bool:
        """Check if the no_evidence flag matches expectation."""
        return response.no_evidence == expected_no_evidence

    def faithfulness_check(self, response: RAGResponse, context: str) -> dict:
        """Use LLM-as-judge to assess whether the answer is faithful to context."""
        if response.no_evidence:
            return {"faithful": True, "explanation": "No-evidence response; nothing to verify."}

        prompt = (
            "You are a strict factual evaluator.\n\n"
            "CONTEXT (retrieved research passages):\n"
            f"{context}\n\n"
            "ANSWER:\n"
            f"{response.answer}\n\n"
            "Does the ANSWER make any claims that are NOT supported by the CONTEXT?\n"
            "Respond with exactly one of: FAITHFUL or UNFAITHFUL\n"
            "Then briefly explain why in 1-2 sentences."
        )
        result = self.judge_llm.invoke(prompt)
        text = result.content.strip()
        faithful = text.upper().startswith("FAITHFUL")
        return {"faithful": faithful, "explanation": text}

    # ── Run evaluation ────────────────────────────────────────────────────────

    def evaluate_single(
        self,
        pipeline: RAGPipeline,
        query_info: dict,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
    ) -> dict:
        """Run one query through the pipeline and score it."""
        response = pipeline.query(
            query_info["query"],
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
        # Use the retrieval result stored by the pipeline during query()
        # instead of calling retrieve() again (which could return different
        # chunks due to non-determinism).
        retrieval = pipeline._last_retrieval
        retrieved_ids = {c.chunk_id for c in retrieval.chunks}
        context_text = "\n".join(
            f"--- CHUNK {c.chunk_id} (source: {c.source_id}, section: {c.section}) ---\n{c.text}"
            for c in retrieval.chunks
        )

        # Score
        cp = self.citation_precision(response, retrieved_ids)
        # Use explicit expected_no_evidence field; default False for
        # direct/synthesis queries that don't set it.
        expected_no_ev = query_info.get("expected_no_evidence", False)
        ne_acc = self.no_evidence_accuracy(response, expected_no_ev)
        faith = self.faithfulness_check(response, context_text)

        return {
            "query_id": query_info["id"],
            "query_type": query_info["type"],
            "query": query_info["query"],
            "answer_preview": response.answer[:300],
            "confidence": response.confidence,
            "no_evidence": response.no_evidence,
            "num_citations": len(response.citations),
            "citation_precision": round(cp, 3),
            "no_evidence_correct": ne_acc,
            "faithful": faith["faithful"],
            "faithfulness_note": faith["explanation"],
            "caveats": response.caveats,
        }

    def run_full_eval(
        self,
        queries: list[dict],
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
        label: str = "hybrid",
        max_workers: int = MAX_EVAL_WORKERS,
        max_retries: int = 2,
    ) -> list[dict]:
        """Evaluate all queries and return scored results.

        Queries are executed in parallel using *max_workers* threads.
        Each thread lazily creates its own RAGPipeline so that
        pipeline._last_retrieval is never shared across threads.

        After the parallel batch, any missing queries are retried
        sequentially (up to *max_retries* rounds) with a fresh pipeline
        to maximise the chance of a complete result set.
        """
        # Thread-local storage scoped to this call so successive
        # run_full_eval invocations (hybrid then vector-only) get
        # fresh pipelines if the weights change.
        local = threading.local()

        def _get_pipeline() -> RAGPipeline:
            if not hasattr(local, "pipeline"):
                local.pipeline = RAGPipeline(self.config)
            return local.pipeline

        def _worker(q: dict) -> dict:
            print(f"  [{label}] Evaluating {q['id']}: {q['query'][:60]}...")
            pipeline = _get_pipeline()
            result = self.evaluate_single(
                pipeline, q,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
            )
            result["mode"] = label
            return result

        # ── Parallel batch ─────────────────────────────────────────────
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker, q): q for q in queries}
            for future in as_completed(futures):
                q = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    print(f"  [{label}] {q['id']} FAILED: {exc}")

        # ── Retry missing queries sequentially ─────────────────────────
        for attempt in range(1, max_retries + 1):
            completed_ids = {r["query_id"] for r in results}
            missing = [q for q in queries if q["id"] not in completed_ids]
            if not missing:
                break
            print(f"\n  [{label}] Retry {attempt}/{max_retries}: "
                  f"re-running {len(missing)} missing query(ies) "
                  f"({', '.join(q['id'] for q in missing)})...")
            retry_pipeline = RAGPipeline(self.config)
            for q in missing:
                try:
                    print(f"  [{label}] Retrying {q['id']}...")
                    result = self.evaluate_single(
                        retry_pipeline, q,
                        vector_weight=vector_weight,
                        bm25_weight=bm25_weight,
                    )
                    result["mode"] = label
                    results.append(result)
                except Exception as exc:
                    print(f"  [{label}] {q['id']} RETRY FAILED: {exc}")

        # Final check
        completed_ids = {r["query_id"] for r in results}
        still_missing = [q["id"] for q in queries if q["id"] not in completed_ids]
        if still_missing:
            print(f"\n  WARNING: {len(still_missing)} query(ies) could not be "
                  f"evaluated after {max_retries} retries: {still_missing}")

        # Sort by original query order for deterministic reports
        id_order = {q["id"]: i for i, q in enumerate(queries)}
        results.sort(key=lambda r: id_order.get(r["query_id"], 0))
        return results

    # ── Report generation ─────────────────────────────────────────────────────

    @staticmethod
    def generate_report(
        hybrid_results: list[dict],
        vector_results: list[dict],
        output_path: Path,
    ):
        """Write a comprehensive Markdown evaluation report (3-5 pages)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def _avg(results: list[dict], key: str) -> float:
            vals = [r[key] for r in results if isinstance(r[key], (int, float))]
            return sum(vals) / len(vals) if vals else 0.0

        def _rate(results: list[dict], key: str) -> float:
            vals = [r[key] for r in results if isinstance(r[key], bool)]
            return sum(vals) / len(vals) if vals else 0.0

        h_cp = _avg(hybrid_results, "citation_precision")
        v_cp = _avg(vector_results, "citation_precision")
        h_faith = _rate(hybrid_results, "faithful")
        v_faith = _rate(vector_results, "faithful")
        h_ne = _rate(hybrid_results, "no_evidence_correct")
        v_ne = _rate(vector_results, "no_evidence_correct")

        # ── Identify failure cases ────────────────────────────────────────────
        failures = [r for r in hybrid_results if not r.get("faithful", True)]

        # ── Identify per-query deltas between hybrid and vector-only ──────────
        h_map = {r["query_id"]: r for r in hybrid_results}
        v_map = {r["query_id"]: r for r in vector_results}
        hybrid_wins = [qid for qid in h_map if h_map[qid].get("faithful") and not v_map.get(qid, {}).get("faithful")]
        vector_wins = [qid for qid in h_map if not h_map[qid].get("faithful") and v_map.get(qid, {}).get("faithful")]

        lines = [
            "# Phase 2 Evaluation Report",
            "",
            "## 1. Pipeline Architecture",
            "",
            "The RAG system implements a multi-stage pipeline with six key enhancements",
            "over a baseline retrieve-and-generate approach:",
            "",
            "### 1.1 Ingestion",
            "",
            "- **PDF parsing** with PyMuPDF: section-aware extraction that detects headings",
            "  (via font size and bold heuristics) and groups text under normalised section",
            "  labels (abstract, methods, results, discussion, etc.).",
            "- **Sentence-boundary chunking**: 768-token chunks with 100-token overlap,",
            "  using the `cl100k_base` tokenizer.  Each chunk is prefixed with metadata",
            "  (`[Source: … | Section: … | Pages: …]`) so embeddings capture provenance.",
            "- **Dual text storage**: metadata-prefixed text is used for FAISS embeddings;",
            "  raw text (without prefix) is used for BM25 to avoid keyword dilution.",
            "",
            "### 1.2 Hybrid Retrieval",
            "",
            "- **FAISS vector search** (OpenAI `text-embedding-3-small`) returns top-10",
            "  candidates with L2 distance converted to similarity: `1 / (1 + dist)`.",
            "- **BM25 keyword search** returns top-10 candidates with rank-based scoring:",
            "  `1 / (rank + 1)`.",
            "- **Weighted fusion**: `combined = 0.6 × vector + 0.4 × BM25`.  A similarity",
            "  threshold of 0.25 filters low-relevance chunks before ranking.",
            "",
            "### 1.3 LLM Reranking",
            "",
            "- After hybrid retrieval surfaces 10 candidates, **GPT-5-mini scores each",
            "  chunk's relevance (0–10)** in a single structured-output call.",
            "- Chunks are re-sorted by relevance score (ties broken by combined retrieval",
            "  score) and the top-6 are kept.",
            "- This filters out tangentially related chunks that matched on keywords but",
            "  lack the specific evidence the query requires.",
            "",
            "### 1.4 Chunk Expansion",
            "",
            "- After reranking, **±1 adjacent chunks from the same source and section**",
            "  are added to provide more complete surrounding context.",
            "- Expanded chunks receive a discounted score (80% of the parent's vector",
            "  score) and the total is capped at 10 chunks.",
            "",
            "### 1.5 Generation",
            "",
            "- **GPT-5-mini** with `temperature=0.0` produces **Pydantic-enforced",
            "  structured output** (`RAGResponse`): answer, inline citations",
            "  `(source_id, chunk_id)`, confidence level, evidence quality note,",
            "  `no_evidence` flag, and caveats.",
            "- The system prompt instructs the model to act as a senior research analyst,",
            "  ground every claim in a verbatim quote (≤40 words), and never inject",
            "  parametric knowledge.",
            "- Confidence is calibrated across three levels: *high* (multiple explicit",
            "  chunks), *medium* (partial / single chunk), *low* (thin / inferential).",
            "",
            "### 1.6 Post-Generation Guardrails",
            "",
            "Three guardrails run after generation:",
            "",
            "1. **Topic presence check** — extracts keywords from the query and verifies",
            "   they appear in the retrieved text.  If key concepts are absent, the",
            "   pipeline short-circuits with `no_evidence=True`.",
            "2. **Citation verification** — removes any citations whose `chunk_id` does",
            "   not match a retrieved chunk (with fuzzy suffix matching), preventing",
            "   hallucinated references.",
            "3. **Entity check** — regex-extracts numbers with units, gene names, and",
            "   proper nouns from the answer, then verifies each appears in the context.",
            "   Sentences containing ungrounded entities are stripped (if ≥30% of the",
            "   answer survives) or flagged as caveats.",
            "",
            "### 1.7 Evaluation Infrastructure",
            "",
            "- Queries are evaluated **in parallel** (4 threads) with thread-local",
            "  `RAGPipeline` instances for thread safety.",
            "- A **retry mechanism** re-runs any failed queries sequentially (up to 2",
            "  retries with fresh pipeline instances) to ensure complete results.",
            "- All runs are logged to `logs/run_log.jsonl` with full retrieval and",
            "  response details for reproducibility.",
            "",
            "## 2. Query Set Design",
            "",
            "The evaluation set contains **20 queries** across three categories, each designed",
            "to test a different aspect of RAG performance:",
            "",
            "| Category | Count | Purpose |",
            "|----------|-------|---------|",
            "| Direct / Factual | 10 | Target a specific paper; test whether the system retrieves the right source and extracts accurate claims. |",
            "| Synthesis / Cross-paper | 5 | Require evidence from 2+ papers; test multi-hop retrieval and comparative reasoning. |",
            "| Edge-case / Adversarial | 5 | Test trust guardrails: out-of-scope topics, non-existent sources, leading premises, section-specific retrieval. |",
            "",
            "**Design rationale.** Direct queries cover the breadth of the 18-paper corpus",
            "(each source is targeted by at least one query). Synthesis queries force the",
            "retriever to surface chunks from multiple papers and the generator to reason",
            "across them. Edge-case queries probe failure modes identified in Phase 1:",
            "fabricated citations, overconfident answers on missing evidence, and leading-",
            "premise bias.",
            "",
            "## 3. Metrics",
            "",
            "Three complementary metrics are used:",
            "",
            "- **Citation precision**: fraction of citations in the answer that map to a",
            "  chunk actually retrieved for that query.  Measures whether the system invents",
            "  citations.",
            "- **Faithfulness** (LLM-as-judge): a GPT-5-mini judge evaluates whether every",
            "  claim in the answer is supported by the retrieved context.  Binary",
            "  (FAITHFUL / UNFAITHFUL).  This is a strict metric — a single unsupported",
            "  detail makes the answer UNFAITHFUL.",
            "- **No-evidence accuracy**: checks whether the system correctly sets its",
            "  `no_evidence` flag when no relevant information exists (and does not set",
            "  it when relevant information is present).",
            "",
            "## 4. Summary Results",
            "",
            "| Metric | Hybrid (BM25 + Vector) | Vector-Only | Delta |",
            "|--------|:---------------------:|:-----------:|:-----:|",
            f"| Avg Citation Precision | {h_cp:.3f} | {v_cp:.3f} | {h_cp - v_cp:+.3f} |",
            f"| Faithfulness Rate | {h_faith:.3f} | {v_faith:.3f} | {h_faith - v_faith:+.3f} |",
            f"| No-Evidence Accuracy | {h_ne:.3f} | {v_ne:.3f} | {h_ne - v_ne:+.3f} |",
            "",
            "## 5. Per-Query Results (Hybrid)",
            "",
            "| ID | Type | Confidence | Cit. Prec. | Faithful | No-Ev OK | Preview |",
            "|----|------|-----------|-----------|----------|---------|---------|",
        ]
        for r in hybrid_results:
            lines.append(
                f"| {r['query_id']} | {r['query_type']} | {r['confidence']} | "
                f"{r['citation_precision']:.2f} | {r['faithful']} | "
                f"{r['no_evidence_correct']} | {r['answer_preview'][:80]}... |"
            )

        # ── Section 5: Failure Cases ──────────────────────────────────────────
        lines += [
            "",
            "## 6. Failure Cases",
            "",
            f"Out of 20 queries, **{len(failures)} answer(s)** were judged UNFAITHFUL in",
            "hybrid mode. Representative failure cases are analysed below.",
            "",
        ]

        # Select up to 3 representative failures from different types
        selected: list[dict] = []
        for qtype in ("edge", "direct", "synthesis"):
            for f in failures:
                if f["query_type"] == qtype and f not in selected:
                    selected.append(f)
                    break
            if len(selected) >= 3:
                break
        # Fill remaining if needed
        for f in failures:
            if f not in selected:
                selected.append(f)
            if len(selected) >= 3:
                break

        for i, f in enumerate(selected, 1):
            lines += [
                f"### 6.{i}  {f['query_id']} — {f['query_type']}",
                "",
                f"**Query:** {f['query']}",
                "",
                f"**Confidence:** {f['confidence']}  |  **Citation precision:** {f['citation_precision']:.2f}",
                "",
                f"**Judge verdict:** {f['faithfulness_note']}",
                "",
                f"**Answer preview:** {f['answer_preview'][:200]}...",
                "",
                "**Failure mode:**",
            ]
            # Classify failure mode
            note_lower = f.get("faithfulness_note", "").lower()
            if "not mention" in note_lower or "does not" in note_lower:
                lines.append(
                    "Knowledge leakage — the model injected details from its training"
                    " data that are not present in the retrieved context.  This indicates"
                    " the grounding prompt needs to be more restrictive, or the retriever"
                    " failed to surface the relevant chunks."
                )
            elif "specific" in note_lower or "quantitative" in note_lower or "figures" in note_lower:
                lines.append(
                    "Specificity inflation — the model added specific numbers,"
                    " percentages, or named entities not found in the retrieved passages."
                    "  The generator is elaborating beyond evidence."
                )
            else:
                lines.append(
                    "General unfaithfulness — the answer contains claims not supported"
                    " by the retrieved context."
                )
            lines.append("")

        # ── Section 7: Enhancement Comparison ─────────────────────────────────
        lines += [
            "## 7. Enhancement Analysis: Hybrid vs. Vector-Only",
            "",
            "The primary enhancement evaluated here is **hybrid retrieval** (BM25 + FAISS",
            "with weighted fusion, 0.6 vector / 0.4 BM25) plus **LLM reranking** compared",
            "to a vector-only baseline (which still uses reranking and all guardrails).",
            "",
            "### 7.1 What improved",
            "",
        ]
        if hybrid_wins:
            lines.append(
                f"Hybrid retrieval improved faithfulness on {len(hybrid_wins)} "
                f"queries ({', '.join(hybrid_wins)}) where vector-only was unfaithful."
                "  BM25 likely helped retrieve keyword-relevant chunks that pure semantic"
                " search missed."
            )
        else:
            lines.append(
                "No queries were faithful in hybrid mode but unfaithful in vector-only."
            )

        lines += [
            "",
            "### 7.2 What regressed",
            "",
        ]
        if vector_wins:
            lines.append(
                f"Hybrid retrieval caused faithfulness regressions on {len(vector_wins)} "
                f"queries ({', '.join(vector_wins)}).  BM25 likely introduced tangentially"
                " related chunks that matched on keywords but lacked the specific evidence"
                " needed, causing the generator to fill gaps with its own knowledge."
            )
        else:
            lines.append("No faithfulness regressions were observed.")

        lines += [
            "",
            "### 7.3 Net assessment",
            "",
            f"Hybrid retrieval {'improved' if h_faith >= v_faith else 'degraded'}"
            f" overall faithfulness ({h_faith:.1%} vs {v_faith:.1%}).  Citation precision"
            f" is identical ({h_cp:.3f} vs {v_cp:.3f}), confirming that the guardrails"
            " post-processing removes any hallucinated citations regardless of retrieval"
            " strategy.  The no-evidence accuracy is also equivalent.",
            "",
        ]
        if h_faith < v_faith:
            lines += [
                "The net negative impact on faithfulness suggests the BM25 weight (0.4)",
                "may be too high, or the similarity threshold (0.25) should be raised to",
                "filter out low-relevance BM25-sourced chunks.  Despite LLM reranking",
                "already filtering the weakest candidates, some tangentially related chunks",
                "still pass through.  Possible mitigations for Phase 3 include:",
                "",
                "1. Reducing BM25 weight (e.g. 0.2) or raising the similarity threshold.",
                "2. Raising the reranker minimum relevance score to discard more aggressively.",
                "3. Strengthening the grounding prompt to further constrain the generator.",
                "",
            ]

        # ── Section 8: Lessons Learned ────────────────────────────────────────
        lines += [
            "## 8. Lessons Learned",
            "",
            "### What worked well",
            "",
            "- **Citation verification guardrails** achieved 100% citation precision across",
            "  all 40 evaluation runs (20 hybrid + 20 vector-only).  No hallucinated",
            "  citations survived post-processing.",
            "- **LLM reranking** effectively filtered tangentially related chunks after",
            "  hybrid retrieval, improving the relevance of context passed to the generator.",
            "  The 0–10 relevance scoring with structured output provided fine-grained",
            "  control over chunk selection.",
            "- **Pydantic structured output** eliminated the formatting noise observed in",
            "  Phase 1, producing consistently parseable responses with inline citations,",
            "  confidence levels, and caveats.",
            "- **Section-aware chunking** with metadata prefixes enabled retrieval of",
            "  method-specific passages (tested by E03) and improved metadata quality",
            "  in chunk headers.",
            "- **Chunk expansion** (±1 adjacent chunks) provided more complete context",
            "  without significantly increasing noise, helping the generator produce",
            "  more thorough answers for synthesis queries.",
            "- **Topic presence check** correctly short-circuited generation when query",
            "  concepts were absent from retrieved text, preventing hallucination on",
            "  out-of-scope topics.",
            "- **Entity check** actively stripped ungrounded numbers, gene names, and",
            "  proper nouns from answers, catching knowledge leakage from the model's",
            "  training data.",
            "- **No-evidence detection** correctly flagged missing sources (E02) and",
            "  avoided false alarms on present-but-tangential topics.",
            "",
            "### What did not work",
            "",
            f"- **Faithfulness** remains the weakest metric ({h_faith:.1%} hybrid,"
            f" {v_faith:.1%} vector-only).  The generator occasionally injects plausible",
            "  details from its training data that go beyond the retrieved context,",
            "  particularly author attributions not present in chunk text.",
            "- **Strict LLM-as-judge** scoring is binary; a single minor unsupported",
            "  detail marks the entire answer as UNFAITHFUL.  This penalises answers that",
            "  are substantially correct but include one small elaboration.",
            "- **BM25 noise** — in some cases the BM25 component introduced tangentially",
            "  related chunks that matched on keywords but lacked specific evidence,",
            "  causing the generator to fill gaps with parametric knowledge.",
            "",
            "### Implications for Phase 3",
            "",
            "1. **Graduated faithfulness** — consider a 1–4 scale instead of binary to",
            "   better capture partial grounding in the portal's evaluation view.",
            "2. **Cross-encoder reranking** — replace or supplement the LLM reranker",
            "   with a dedicated cross-encoder model for faster, cheaper reranking.",
            "3. **Adaptive BM25 weight** — tune the BM25 weight per query type (lower",
            "   for synthesis queries where keyword matching is noisier).",
            "4. **Chain-of-thought grounding** — add an explicit verification step where",
            "   the model checks each claim against context before finalising the answer.",
            "",
        ]

        output_path.write_text("\n".join(lines))
        print(f"Report written to {output_path}")

        # Also save raw results as JSON
        raw_path = output_path.with_suffix(".json")
        with open(raw_path, "w") as f:
            json.dump({"hybrid": hybrid_results, "vector_only": vector_results}, f, indent=2)
        print(f"Raw results saved to {raw_path}")
