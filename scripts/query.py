"""CLI entry point for single RAG queries.

Usage:
    python scripts/query.py "What is the effect of exercise on longevity?"
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.pipeline import RAGPipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/query.py 'Your question here'")
        sys.exit(1)

    question = sys.argv[1]
    pipeline = RAGPipeline()
    response = pipeline.query(question)

    print()
    print("=" * 60)
    print("QUERY:", question)
    print("=" * 60)
    print()
    print("ANSWER:")
    print(response.answer)
    print()
    print("CITATIONS:")
    if response.citations:
        for c in response.citations:
            quote = f' — "{c.relevant_quote}"' if c.relevant_quote else ""
            print(f"  - ({c.source_id}, {c.chunk_id}){quote}")
    else:
        print("  (none)")
    print()
    print(f"CONFIDENCE: {response.confidence}")
    print(f"EVIDENCE QUALITY: {response.evidence_quality}")
    if response.caveats:
        print(f"CAVEATS: {response.caveats}")
    print()
    print("[Log saved to logs/run_log.jsonl]")


if __name__ == "__main__":
    main()
