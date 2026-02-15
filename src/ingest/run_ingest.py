"""Orchestrate the full ingestion pipeline: parse → chunk → embed → index."""

from __future__ import annotations

import json
import pickle

import pandas as pd
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm

from src.config import Config
from src.ingest.build_manifest import build_manifest
from src.ingest.chunker import chunk_document
from src.ingest.pdf_parser import PDFParser
from src.rag.embedder import get_embeddings_model
from src.schemas import Chunk


def _save_chunks(chunks: list[Chunk], path):
    """Write chunks to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
    print(f"  Saved {len(chunks)} chunks to {path}")


def _build_faiss_index(chunks: list[Chunk], config: Config):
    """Embed chunks and build a FAISS vector store."""
    embeddings = get_embeddings_model(config)
    docs = [
        Document(
            page_content=c.text,  # metadata-prepended text
            metadata={
                "chunk_id": c.chunk_id,
                "source_id": c.source_id,
                "section": c.section,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "year": c.year,
                "source_type": c.source_type,
            },
        )
        for c in chunks
    ]
    print(f"  Embedding {len(docs)} chunks (this may take a minute)...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    config.faiss_index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(config.faiss_index_dir))
    print(f"  FAISS index saved to {config.faiss_index_dir}")


def _build_bm25_index(chunks: list[Chunk], config: Config):
    """Build and pickle BM25 document list for later retrieval."""
    docs = [
        Document(
            page_content=c.text_raw,  # raw text without metadata prefix
            metadata={
                "chunk_id": c.chunk_id,
                "source_id": c.source_id,
                "section": c.section,
                "page_start": c.page_start,
                "page_end": c.page_end,
            },
        )
        for c in chunks
    ]
    config.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.bm25_index_path, "wb") as f:
        pickle.dump(docs, f)
    print(f"  BM25 docs saved to {config.bm25_index_path}")


def run_ingest(config: Config | None = None):
    """Run the full ingestion pipeline."""
    config = config or Config()

    # 1. Build or load manifest
    if not config.manifest_path.exists():
        print("Step 1: Building data manifest...")
        manifest = build_manifest(config.raw_dir, config.manifest_path)
    else:
        print("Step 1: Loading existing manifest...")
        manifest = pd.read_csv(config.manifest_path)
    print(f"  {len(manifest)} sources in manifest")

    # 2. Parse all PDFs
    print("Step 2: Parsing PDFs...")
    parsed_docs = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Parsing"):
        pdf_path = config.project_root / row["raw_path"]
        if not pdf_path.exists():
            print(f"  WARNING: {pdf_path} not found, skipping")
            continue
        parser = PDFParser(pdf_path)
        doc = parser.parse(source_id=row["source_id"])
        parsed_docs.append((doc, row))
    print(f"  Parsed {len(parsed_docs)} documents")

    # 3. Chunk all documents
    print("Step 3: Chunking documents...")
    all_chunks: list[Chunk] = []
    for doc, row in parsed_docs:
        chunks = chunk_document(
            doc,
            max_tokens=config.max_chunk_tokens,
            overlap_tokens=config.overlap_tokens,
        )
        # Fill in year from manifest
        year_val = row.get("year")
        for c in chunks:
            try:
                c.year = int(year_val) if year_val and str(year_val) != "MANUAL" else None
            except (ValueError, TypeError):
                c.year = None
        all_chunks.extend(chunks)
    print(f"  Total chunks: {len(all_chunks)}")

    # 4. Save chunks
    print("Step 4: Saving chunks...")
    _save_chunks(all_chunks, config.chunks_path)

    # 5. Build FAISS index
    print("Step 5: Building FAISS index...")
    _build_faiss_index(all_chunks, config)

    # 6. Build BM25 index
    print("Step 6: Building BM25 index...")
    _build_bm25_index(all_chunks, config)

    print("\nIngestion complete!")
    return all_chunks


if __name__ == "__main__":
    run_ingest()
