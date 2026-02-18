# Personal Research Portal — Exercise & Nutrition for Longevity

Personal Research Portal (PRP) — Phase 3 deliverable for CMU AI Model Dev course.
Phases 1 → 2 → 3: Prompting → RAG → Research Portal Product.

## Quick Start (< 5 minutes)

```bash
# 1. Install dependencies (includes Streamlit + fpdf2 for Phase 3)
make setup

# 2. Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Run ingestion (parse PDFs, chunk, embed, index) — only needed once
make ingest

# 4. Launch the research portal UI
make portal

# --- or use the CLI ---
# Ask a single question
make query QUERY="What is the effect of exercise on longevity?"

# Run full evaluation (20 queries, hybrid vs vector-only)
make eval
```

## Phase 3 — Research Portal Features

| Feature | Location | Description |
|---------|----------|-------------|
| **Search / Ask** | Tab 1 | Enter a research question → grounded answer with inline citations |
| **Save threads** | Tab 1 | Persist query + chunks + answer to `outputs/threads/` |
| **History** | Tab 2 | Browse all saved research threads with full citation detail |
| **Evidence Table** | Tab 3 | Structured artifact: Claim \| Evidence \| Citation \| Confidence \| Notes |
| **Export** | Tab 3 | Download evidence tables as Markdown, CSV, or PDF |
| **Evaluation dashboard** | Tab 4 | Metrics summary + per-query faithfulness and citation precision |
| **Trust behavior** | All tabs | Missing evidence flagged explicitly + suggested next retrieval steps |

### Artifact schema (evidence table)

| Column | Source | Description |
|--------|--------|-------------|
| Claim | Answer sentence | The factual claim extracted from the answer |
| Evidence snippet | `Citation.relevant_quote` | Verbatim text (≤40 words) from the source chunk |
| Citation | `(source_id, chunk_id)` | Maps to `data_manifest.csv` and `data/processed/chunks.jsonl` |
| Confidence | `RAGResponse.confidence` | high / medium / low |
| Notes | `RAGResponse.caveats` | Caveats and guardrail warnings |

### Repo structure (Phase 3 additions)

```
src/
  app/
    __init__.py
    main.py          Streamlit portal (4 tabs)
    threads.py       Thread save / load / list
    artifacts.py     Evidence table builder
    export.py        Markdown / CSV / PDF export
outputs/
  threads/           Saved research threads (JSON)
  artifacts/         Exported evidence tables (MD, CSV)
reports/
  phase3_report.md   Final Phase 3 report (6–10 pages)
```

## Architecture

```
User query
   │
   ▼
┌───────────────────────┐
│   Hybrid Retriever    │  BM25 + FAISS (vector) with weighted fusion
│    (retriever.py)     │  0.6 × vector + 0.4 × BM25, similarity threshold 0.25
│                       │  Returns top-10 candidates above threshold
└──────────┬────────────┘
           │ top-10 candidates
           ▼
┌───────────────────────┐
│    LLM Reranker       │  GPT-5-mini scores each chunk's relevance (0–10)
│    (reranker.py)      │  Re-sorts by relevance, keeps top-6
└──────────┬────────────┘
           │ top-6 reranked
           ▼
┌───────────────────────┐
│   Chunk Expansion     │  Adds ±1 adjacent chunks from same source/section
│    (retriever.py)     │  Expanded chunks get discounted score (×0.8)
│                       │  Provides more complete context (capped at 10 chunks)
└──────────┬────────────┘
           │ up to 10 chunks
           ▼
┌───────────────────────┐
│   Topic Presence      │  Verifies query key concepts appear in chunks
│   (guardrails.py)     │  Short-circuits with no_evidence if topic absent
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│     GPT-5-mini        │  Pydantic-enforced structured output (RAGResponse)
│    (generator.py)     │  Inline citations: (source_id, chunk_id)
│                       │  Confidence: high / medium / low
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│    Guardrails         │  1. Citation verification — remove hallucinated refs
│   (guardrails.py)     │  2. Entity check — strip ungrounded numbers, genes,
│                       │     and proper nouns from the answer
└──────────┬────────────┘
           │
           ▼
  RAGResponse (answer + citations + confidence + caveats)
  + JSONL log entry saved to logs/run_log.jsonl
```

## Corpus

18 peer-reviewed papers on exercise, nutrition, and longevity.
See `data/data_manifest.csv` for full metadata.

## Ingestion Pipeline

| Stage | Implementation | Details |
|-------|---------------|---------|
| PDF parsing | PyMuPDF (`fitz`) | Section-aware extraction via font-size and bold heuristics; normalises headings to standard labels (abstract, methods, results, etc.) |
| Chunking | Sentence-boundary splitting | 768-token chunks with 100-token overlap (`cl100k_base` tokenizer); skips references and supplementary sections |
| Metadata prefixing | `[Source: X \| Section: Y \| Pages: N-M]` | Prepended to each chunk before embedding so vector search captures provenance |
| Dual text storage | `text` (prefixed) + `text_raw` (original) | Prefixed text for FAISS embeddings; raw text for BM25 to avoid keyword dilution |
| Embedding | OpenAI `text-embedding-3-small` | Stored in a FAISS index (`data/processed/faiss_index/`) |
| BM25 index | `rank-bm25` via LangChain | Pickled to `data/processed/bm25_index.pkl` |

## Enhancements (beyond baseline RAG)

| Enhancement | Implementation | Purpose |
|-------------|---------------|---------|
| Hybrid retrieval | Weighted BM25 + FAISS fusion (0.6 vector / 0.4 BM25) | Catch technical terms pure semantic search misses |
| LLM reranking | GPT-5-mini relevance scoring (0–10 scale, structured output) | Filter tangentially related chunks after retrieval |
| Chunk expansion | ±1 adjacent chunks from same source/section (discounted score) | Provide more complete surrounding context |
| Topic presence check | Keyword extraction + context matching | Prevent hallucination on absent topics by short-circuiting generation |
| Citation verification | Post-generation chunk_id matching (with fuzzy suffix match) | Remove fabricated citations; downgrade confidence if removals occur |
| Entity check | Regex extraction of numbers, gene names, proper nouns + context verification | Actively strip ungrounded entities from the answer to prevent knowledge leakage |
| Pydantic structured output | `with_structured_output()` on GPT-5-mini | Eliminate formatting noise; enforce consistent schema (answer, citations, confidence, caveats) |
| Parallel evaluation | `ThreadPoolExecutor` (4 workers) + thread-local pipelines | Speed up evaluation by running queries concurrently |
| Retry logic | Sequential re-runs of failed queries (up to 2 retries) | Ensure complete results despite transient API failures |

## Key Design Choices (from Phase 1 findings)

| Phase 1 Finding | Phase 2 Solution |
|-----------------|-----------------|
| GPT-4.1 formatting noise (>, ->) | Pydantic structured output via `with_structured_output()` |
| Claude misattributes citations across papers | Metadata prepended to each chunk: `[Source: X \| Section: Y]` |
| Rigid prompts hurt groundedness | Target lengths instead of sentence counts; confidence calibration (high/medium/low) |
| Pure semantic search misses technical terms | Hybrid BM25 + vector retrieval with weighted fusion |
| Models fabricate citations | Citation verification + entity check + topic presence check |
| Tangentially related chunks dilute context | LLM reranking (0–10) after retrieval to keep only the most relevant chunks |
| Answers inject model's training data | Entity check strips ungrounded numbers, genes, and proper nouns from answers |

## Project Structure

```
src/
  config.py            Central settings (Pydantic Settings, .env loading)
  schemas.py           All data models (RAGResponse, Citation, RetrievedChunk, etc.)
  ingest/
    pdf_parser.py      PyMuPDF extraction + section detection
    chunker.py         768-token chunks with 100-token overlap
    build_manifest.py  Generate data_manifest.csv
    run_ingest.py      Orchestrator: parse → chunk → embed → index
  rag/
    embedder.py        OpenAI text-embedding-3-small
    retriever.py       Hybrid BM25 + FAISS with weighted fusion + chunk expansion
    reranker.py        LLM-based relevance reranking (0–10 scale)
    generator.py       GPT-5-mini structured generation
    guardrails.py      Citation verification + topic check + entity check
    pipeline.py        End-to-end pipeline: retrieve → rerank → expand → check → generate → verify → log
  eval/
    queries.py         20-query evaluation set (10 direct, 5 synthesis, 5 edge-case)
    evaluator.py       Scoring (citation precision, faithfulness, no-evidence accuracy)
    run_eval.py        Run comparison: hybrid vs vector-only (parallel, with retry)
scripts/
  query.py             CLI for single queries
  generate_outputs.py  Generate sample outputs for submission
  test_unfaithful.py   Re-run specific queries for debugging
```

## Evaluation

20 queries across 3 categories:
- **10 direct**: factual questions about specific findings in individual papers
- **5 synthesis**: cross-paper comparison and analysis requiring multi-hop retrieval
- **5 edge-case**: adversarial/out-of-scope queries testing trust guardrails

Three metrics:
- **Citation precision**: fraction of citations mapping to actually retrieved chunks
- **Faithfulness** (LLM-as-judge): binary check that every claim is grounded in context
- **No-evidence accuracy**: correct identification of unanswerable queries

Evaluation runs queries in parallel (4 threads, thread-safe via thread-local pipelines)
with automatic retry for any transient failures.

Results are in `reports/evaluation_report.md` after running `make eval`.

## Configuration

All settings live in `src/config.py` (Pydantic Settings, loaded from `.env`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm_model` | `gpt-5-mini-2025-08-07` | Generation and reranking model |
| `embedding_model` | `text-embedding-3-small` | Embedding model for FAISS |
| `temperature` | `0.0` | LLM temperature |
| `vector_k` / `bm25_k` | `10` / `10` | Top-k candidates from each retriever |
| `rerank_candidates` | `10` | Candidates sent to LLM reranker |
| `final_k` | `6` | Chunks kept after reranking |
| `similarity_threshold` | `0.25` | Minimum vector similarity to keep a chunk |
| `vector_weight` / `bm25_weight` | `0.6` / `0.4` | Fusion weights |
| `chunk_expand_window` | `1` | Adjacent chunks to add (±N) |
| `max_chunks_after_expand` | `10` | Cap after expansion |
| `max_chunk_tokens` | `768` | Chunk size in tokens |
| `overlap_tokens` | `100` | Overlap between adjacent chunks |
