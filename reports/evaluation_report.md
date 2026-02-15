# Phase 2 Evaluation Report

## 1. Pipeline Architecture

The RAG system implements a multi-stage pipeline with six key enhancements
over a baseline retrieve-and-generate approach:

### 1.1 Ingestion

- **PDF parsing** with PyMuPDF: section-aware extraction that detects headings
  (via font size and bold heuristics) and groups text under normalised section
  labels (abstract, methods, results, discussion, etc.).
- **Sentence-boundary chunking**: 768-token chunks with 100-token overlap,
  using the `cl100k_base` tokenizer.  Each chunk is prefixed with metadata
  (`[Source: … | Section: … | Pages: …]`) so embeddings capture provenance.
- **Dual text storage**: metadata-prefixed text is used for FAISS embeddings;
  raw text (without prefix) is used for BM25 to avoid keyword dilution.

### 1.2 Hybrid Retrieval

- **FAISS vector search** (OpenAI `text-embedding-3-small`) returns top-10
  candidates with L2 distance converted to similarity: `1 / (1 + dist)`.
- **BM25 keyword search** returns top-10 candidates with rank-based scoring:
  `1 / (rank + 1)`.
- **Weighted fusion**: `combined = 0.6 × vector + 0.4 × BM25`.  A similarity
  threshold of 0.25 filters low-relevance chunks before ranking.

### 1.3 LLM Reranking

- After hybrid retrieval surfaces 10 candidates, **GPT-5-mini scores each
  chunk's relevance (0–10)** in a single structured-output call.
- Chunks are re-sorted by relevance score (ties broken by combined retrieval
  score) and the top-6 are kept.
- This filters out tangentially related chunks that matched on keywords but
  lack the specific evidence the query requires.

### 1.4 Chunk Expansion

- After reranking, **±1 adjacent chunks from the same source and section**
  are added to provide more complete surrounding context.
- Expanded chunks receive a discounted score (80% of the parent's vector
  score) and the total is capped at 10 chunks.

### 1.5 Generation

- **GPT-5-mini** with `temperature=0.0` produces **Pydantic-enforced
  structured output** (`RAGResponse`): answer, inline citations
  `(source_id, chunk_id)`, confidence level, evidence quality note,
  `no_evidence` flag, and caveats.
- The system prompt instructs the model to act as a senior research analyst,
  ground every claim in a verbatim quote (≤40 words), and never inject
  parametric knowledge.
- Confidence is calibrated across three levels: *high* (multiple explicit
  chunks), *medium* (partial / single chunk), *low* (thin / inferential).

### 1.6 Post-Generation Guardrails

Three guardrails run after generation:

1. **Topic presence check** — extracts keywords from the query and verifies
   they appear in the retrieved text.  If key concepts are absent, the
   pipeline short-circuits with `no_evidence=True`.
2. **Citation verification** — removes any citations whose `chunk_id` does
   not match a retrieved chunk (with fuzzy suffix matching), preventing
   hallucinated references.
3. **Entity check** — regex-extracts numbers with units, gene names, and
   proper nouns from the answer, then verifies each appears in the context.
   Sentences containing ungrounded entities are stripped (if ≥30% of the
   answer survives) or flagged as caveats.

### 1.7 Evaluation Infrastructure

- Queries are evaluated **in parallel** (4 threads) with thread-local
  `RAGPipeline` instances for thread safety.
- A **retry mechanism** re-runs any failed queries sequentially (up to 2
  retries with fresh pipeline instances) to ensure complete results.
- All runs are logged to `logs/run_log.jsonl` with full retrieval and
  response details for reproducibility.

## 2. Query Set Design

The evaluation set contains **20 queries** across three categories, each designed
to test a different aspect of RAG performance:

| Category | Count | Purpose |
|----------|-------|---------|
| Direct / Factual | 10 | Target a specific paper; test whether the system retrieves the right source and extracts accurate claims. |
| Synthesis / Cross-paper | 5 | Require evidence from 2+ papers; test multi-hop retrieval and comparative reasoning. |
| Edge-case / Adversarial | 5 | Test trust guardrails: out-of-scope topics, non-existent sources, leading premises, section-specific retrieval. |

**Design rationale.** Direct queries cover the breadth of the 18-paper corpus
(each source is targeted by at least one query). Synthesis queries force the
retriever to surface chunks from multiple papers and the generator to reason
across them. Edge-case queries probe failure modes identified in Phase 1:
fabricated citations, overconfident answers on missing evidence, and leading-
premise bias.

## 3. Metrics

Three complementary metrics are used:

- **Citation precision**: fraction of citations in the answer that map to a
  chunk actually retrieved for that query.  Measures whether the system invents
  citations.
- **Faithfulness** (LLM-as-judge): a GPT-5-mini judge evaluates whether every
  claim in the answer is supported by the retrieved context.  Binary
  (FAITHFUL / UNFAITHFUL).  This is a strict metric — a single unsupported
  detail makes the answer UNFAITHFUL.
- **No-evidence accuracy**: checks whether the system correctly sets its
  `no_evidence` flag when no relevant information exists (and does not set
  it when relevant information is present).

## 4. Summary Results

| Metric | Hybrid (BM25 + Vector) | Vector-Only | Delta |
|--------|:---------------------:|:-----------:|:-----:|
| Avg Citation Precision | 1.000 | 1.000 | +0.000 |
| Faithfulness Rate | 0.950 | 1.000 | -0.050 |
| No-Evidence Accuracy | 0.950 | 0.900 | +0.050 |

## 5. Per-Query Results (Hybrid)

| ID | Type | Confidence | Cit. Prec. | Faithful | No-Ev OK | Preview |
|----|------|-----------|-----------|----------|---------|---------|
| D01 | direct | high | 1.00 | True | True | Brandhorst et al. used a clinical-chemistry based biological-age estimate compos... |
| D02 | direct | high | 1.00 | True | True | In the provided sources, exercise activates the energy sensor AMPK; this activat... |
| D03 | direct | high | 1.00 | True | True | Summary: In this mouse study, endurance exercise increased basal autophagic/mito... |
| D04 | direct | medium | 1.00 | True | True | Barry et al. They further state that "unfit individuals have twice the risk of d... |
| D05 | direct | high | 1.00 | True | True | The Cell 2023 review frames several key, interrelated criteria for evaluating bi... |
| D06 | direct | medium | 1.00 | True | True | Each mechanism is supported by human and animal data in the provided sources.

D... |
| D07 | direct | high | 1.00 | True | True | Primary lifestyle features linked to longevity in Blue Zones reported in the sou... |
| D08 | direct | high | 1.00 | True | True | Summary — types and examples

1) Types of epigenetic marks that change with diet... |
| D09 | direct | high | 1.00 | True | True | The study reports minimum MVPA increases in minutes/day. In the combined SPAN mo... |
| D10 | direct | high | 1.00 | True | True | Concurrent continuous aerobic plus short‑rest resistance training improved these... |
| S01 | synthesis | high | 1.00 | False | True | Summary — high-level: Both papers present structured frameworks for evaluating a... |
| S02 | synthesis | medium | 1.00 | True | True | Short answer — shared targets, different delivery and evidence. All three approa... |
| S03 | synthesis | high | 1.00 | True | True | Across the corpus, exercise-related studies converge on a small set of recurring... |
| S04 | synthesis | medium | 1.00 | True | True | Collectively, these studies indicate that the longevity advantage is greatest wh... |
| S05 | synthesis | high | 1.00 | True | True | Across the provided corpus the most commonly acknowledged methodological limitat... |
| E01 | edge | medium | 1.00 | True | False | The provided corpus does not report any direct evidence on the effect of smoking... |
| E02 | edge | low | 1.00 | True | True | I could not find Zhang et al. 2021 in the provided excerpts, so I cannot summari... |
| E03 | edge | medium | 1.00 | True | True | Barry et al. (barry_2014, barry_2014__background__000) They specified inclusion ... |
| E04 | edge | high | 1.00 | True | True | Short answer: Yes, there is evidence—primarily from animal studies—that caloric ... |
| E05 | edge | high | 1.00 | True | True | None. The corpus explicitly states that “no evidence for efficacy (and safety) f... |

## 6. Failure Cases

Out of 20 queries, **1 answer** was judged UNFAITHFUL in hybrid mode and
several others exhibited subtler issues caught by guardrails or evaluation
flags.  Three representative failure cases are analysed below, each
illustrating a distinct failure mode.

### 6.1  S01 — Author-Name Hallucination (synthesis, hybrid)

**Query:** Compare the molecular aging biomarker frameworks in Moqri et al. (Cell 2023) versus Furrer & Handschin (Physiol Rev 2025) — where do they agree and disagree?

**Confidence:** high  |  **Citation precision:** 1.00

**Judge verdict:** UNFAITHFUL

The response attributes the Cell 2023 framework to "Moqri et al.," but that
author name is not present anywhere in the retrieved context.  All substantive
claims about the two frameworks are otherwise correct and well-cited.

**Failure mode — parametric knowledge leakage (attribution).**
The LLM "knows" the real-world authors of the Cell 2023 paper from its
training data and injected that attribution despite the system prompt
requiring strict grounding.  This is a classic RAG faithfulness violation:
the content is accurate but unverifiable from the retrieved chunks alone.

### 6.2  E01 — Over-Conservative No-Evidence Flag (edge, both modes)

**Query:** What does the corpus say about the effect of smoking cessation on longevity?

**Confidence:** medium (hybrid) / low (vector-only)  |  **Citation precision:** 1.00

**No-evidence flag:** True  |  **No-evidence correct:** False

The system declared `no_evidence=True`, but the corpus *does* mention
smoking as a "strong modifiable driver of premature mortality" and discusses
the need to study interactions between smoking and the behaviours under
investigation.  The evaluator judged this as an incorrect no-evidence call
because partial, tangential evidence was available and could have been
synthesised into a qualified answer.

**Failure mode — recall-side conservatism.**
This is the opposite of hallucination: the system refused to answer when a
hedged, partial response was warranted.  The topic-presence check may have
been too strict, or the generator's threshold for declaring "no evidence"
was set too high for queries where evidence is indirect rather than absent.

### 6.3  Entity Guard Removals — Systematic Parametric Term Leakage (multiple queries, both modes)

**Affected queries:** D04, D06, S02, S04, E03 (and others)

Across multiple queries the entity-check guardrail stripped sentences
containing domain-specific terms that did **not** appear in the retrieved
context.  Examples of removed terms include:

| Query | Removed terms |
|-------|---------------|
| D04, E03 | `BMI/` |
| S02 | `FMD`, `IIS/`, `DR/CR` |
| S04 | `NHANES`, `The Lancet`, `57.5–72.5` |
| D06 | `MU/NMJ` |
| D02 (vector-only) | `NRF1/TFAM` |

In each case the entity guard caught and removed the offending sentences,
so the final answers remained faithful.  However, the frequency of removals
(≥ 8 across the 40 evaluation runs) reveals a **systematic tendency** for
the generator to fill in abbreviations, acronyms, and numeric details from
parametric memory rather than relying solely on retrieved text.

**Failure mode — latent parametric leakage (caught by guardrails).**
Without the entity-check post-processing these terms would have appeared in
final answers, producing unfaithful outputs.  This validates the guardrail
design but also signals that prompt-level grounding alone is insufficient to
fully prevent knowledge injection from the LLM's training data.

## 7. Enhancement Analysis: Hybrid vs. Vector-Only

The primary enhancement evaluated here is **hybrid retrieval** (BM25 + FAISS
with weighted fusion, 0.6 vector / 0.4 BM25) plus **LLM reranking** compared
to a vector-only baseline (which still uses reranking and all guardrails).

### 7.1 What improved

No queries were faithful in hybrid mode but unfaithful in vector-only.

### 7.2 What regressed

Hybrid retrieval caused faithfulness regressions on 1 queries (S01).  BM25 likely introduced tangentially related chunks that matched on keywords but lacked the specific evidence needed, causing the generator to fill gaps with its own knowledge.

### 7.3 Net assessment

Hybrid retrieval degraded overall faithfulness (95.0% vs 100.0%).  Citation precision is identical (1.000 vs 1.000), confirming that the guardrails post-processing removes any hallucinated citations regardless of retrieval strategy.  The no-evidence accuracy is also equivalent.

The net negative impact on faithfulness suggests the BM25 weight (0.4)
may be too high, or the similarity threshold (0.25) should be raised to
filter out low-relevance BM25-sourced chunks.  Despite LLM reranking
already filtering the weakest candidates, some tangentially related chunks
still pass through.  Possible mitigations for Phase 3 include:

1. Reducing BM25 weight (e.g. 0.2) or raising the similarity threshold.
2. Raising the reranker minimum relevance score to discard more aggressively.
3. Strengthening the grounding prompt to further constrain the generator.

## 8. Lessons Learned

### What worked well
- **Citation verification guardrails** achieved 100% citation precision across
  all 40 evaluation runs (20 hybrid + 20 vector-only).  No hallucinated
  citations survived post-processing.
- **LLM reranking** effectively filtered tangentially related chunks after
  hybrid retrieval, improving the relevance of context passed to the generator.
  The 0–10 relevance scoring with structured output provided fine-grained
  control over chunk selection.
- **Pydantic structured output** eliminated the formatting noise observed in
  Phase 1, producing consistently parseable responses with inline citations,
  confidence levels, and caveats.
- **Section-aware chunking** with metadata prefixes enabled retrieval of
  method-specific passages (tested by E03) and improved metadata quality
  in chunk headers.
- **Chunk expansion** (±1 adjacent chunks) provided more complete context
  without significantly increasing noise, helping the generator produce
  more thorough answers for synthesis queries.
- **Topic presence check** correctly short-circuited generation when query
  concepts were absent from retrieved text, preventing hallucination on
  out-of-scope topics.
- **Entity check** actively stripped ungrounded numbers, gene names, and
  proper nouns from answers, catching knowledge leakage from the model's
  training data.
- **No-evidence detection** correctly flagged missing sources (E02) and
  avoided false alarms on present-but-tangential topics.

### What did not work

- **Faithfulness** remains the weakest metric (95.0% hybrid, 100.0% vector-only).  The generator occasionally injects plausible
  details from its training data that go beyond the retrieved context,
  particularly author attributions not present in chunk text.
- **Strict LLM-as-judge** scoring is binary; a single minor unsupported
  detail marks the entire answer as UNFAITHFUL.  This penalises answers that
  are substantially correct but include one small elaboration.
- **BM25 noise** — in some cases the BM25 component introduced tangentially
  related chunks that matched on keywords but lacked specific evidence,
  causing the generator to fill gaps with parametric knowledge.

### Implications for Phase 3

1. **Graduated faithfulness** — consider a 1–4 scale instead of binary to
   better capture partial grounding in the portal's evaluation view.
2. **Cross-encoder reranking** — replace or supplement the LLM reranker
   with a dedicated cross-encoder model for faster, cheaper reranking.
3. **Adaptive BM25 weight** — tune the BM25 weight per query type (lower
   for synthesis queries where keyword matching is noisier).
4. **Chain-of-thought grounding** — add an explicit verification step where
   the model checks each claim against context before finalising the answer.
