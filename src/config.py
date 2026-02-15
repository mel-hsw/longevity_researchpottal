"""Central configuration for the Phase 2 RAG pipeline."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # ── Paths ─────────────────────────────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    faiss_index_dir: Path = processed_dir / "faiss_index"
    bm25_index_path: Path = processed_dir / "bm25_index.pkl"
    chunks_path: Path = processed_dir / "chunks.jsonl"
    log_path: Path = project_root / "logs" / "run_log.jsonl"
    manifest_path: Path = data_dir / "data_manifest.csv"

    # ── Model settings ────────────────────────────────────────────────────────
    llm_model: str = "gpt-5-mini-2025-08-07"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0

    # ── Retrieval settings ────────────────────────────────────────────────────
    vector_k: int = 10
    bm25_k: int = 10
    rerank_candidates: int = 10     # how many to send to the reranker
    final_k: int = 6                # how many to keep after reranking
    similarity_threshold: float = 0.25
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    chunk_expand_window: int = 1    # add ±N adjacent chunks per retrieved chunk
    max_chunks_after_expand: int = 10  # cap total chunks after expansion

    # ── Chunking settings ─────────────────────────────────────────────────────
    max_chunk_tokens: int = 768
    overlap_tokens: int = 100

    # ── API keys (loaded from .env) ───────────────────────────────────────────
    openai_api_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
