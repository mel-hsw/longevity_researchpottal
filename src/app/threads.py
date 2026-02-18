"""Research thread persistence.

A thread captures a single research session moment:
  query → retrieved chunks → generated answer

Threads are stored as JSON files under outputs/threads/.
Each file is named by its thread_id (UTC timestamp slug).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.schemas import RAGResponse, RetrievalResult

_DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "threads"


def save_thread(
    query: str,
    retrieval: RetrievalResult,
    response: RAGResponse,
    *,
    threads_dir: Path | None = None,
) -> Path:
    """Persist a research thread to disk and return the saved file path."""
    out_dir = threads_dir or _DEFAULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    thread_id = now.strftime("%Y%m%dT%H%M%SZ")

    thread: dict[str, Any] = {
        "thread_id": thread_id,
        "timestamp": now.isoformat(),
        "query": query,
        "retrieval": {
            "total_candidates": retrieval.all_candidates,
            "above_threshold": retrieval.above_threshold,
            "has_sufficient_evidence": retrieval.has_sufficient_evidence,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "source_id": c.source_id,
                    "section": c.section,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "combined_score": round(c.combined_score, 4),
                    "text": c.text,
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

    out_path = out_dir / f"{thread_id}.json"
    out_path.write_text(json.dumps(thread, indent=2))
    return out_path


def list_threads(threads_dir: Path | None = None) -> list[dict[str, Any]]:
    """Return all saved threads (metadata only), newest first."""
    out_dir = threads_dir or _DEFAULT_DIR
    if not out_dir.exists():
        return []

    threads = []
    for path in sorted(out_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
            threads.append(
                {
                    "thread_id": data.get("thread_id", path.stem),
                    "timestamp": data.get("timestamp", ""),
                    "query": data.get("query", ""),
                    "confidence": data.get("response", {}).get("confidence", ""),
                    "no_evidence": data.get("response", {}).get("no_evidence", False),
                    "citation_count": len(
                        data.get("response", {}).get("citations", [])
                    ),
                    "path": str(path),
                }
            )
        except (json.JSONDecodeError, KeyError):
            continue
    return threads


def load_thread(thread_id: str, threads_dir: Path | None = None) -> dict[str, Any]:
    """Load a single thread by its thread_id."""
    out_dir = threads_dir or _DEFAULT_DIR
    path = out_dir / f"{thread_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Thread not found: {thread_id}")
    return json.loads(path.read_text())
