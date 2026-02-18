"""Generate sample Phase 3 artifacts from existing sample RAG outputs.

Reads outputs/sample_rag_outputs.json (no API calls needed) and produces:
  - outputs/artifacts/<query_id>_evidence_table.md
  - outputs/artifacts/<query_id>_evidence_table.csv
  - outputs/artifacts/combined_evidence_table.md  (all queries merged)

Run:
    python scripts/generate_phase3_artifacts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.app.artifacts import build_evidence_table
from src.app.export import (
    evidence_table_to_csv_bytes,
    evidence_table_to_markdown,
    evidence_table_to_markdown_bytes,
)
from src.schemas import Citation, RAGResponse


def _load_sample_outputs() -> list[dict]:
    path = _REPO / "outputs" / "sample_rag_outputs.json"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}\nRun `make outputs` first.")
    data = json.loads(path.read_text())
    return data.get("results", [])


def _to_rag_response(result: dict) -> RAGResponse:
    citations = [
        Citation(
            source_id=c["source_id"],
            chunk_id=c["chunk_id"],
            relevant_quote=c.get("relevant_quote", ""),
        )
        for c in result.get("citations", [])
    ]
    return RAGResponse(
        answer=result["answer"],
        citations=citations,
        confidence=result.get("confidence", "medium"),
        evidence_quality=result.get("evidence_quality", ""),
        no_evidence=result.get("no_evidence", False),
        caveats=result.get("caveats", []),
    )


def main() -> None:
    out_dir = _REPO / "outputs" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = _load_sample_outputs()
    print(f"Loaded {len(results)} sample results.")

    all_md_sections: list[str] = [
        "# Combined Evidence Tables — Sample RAG Outputs\n",
        "_Generated from `outputs/sample_rag_outputs.json` (Phase 3 artifact demo)._\n",
    ]

    for result in results:
        qid = result.get("query_id", "unknown")
        query = result.get("query", "")
        response = _to_rag_response(result)

        table = build_evidence_table(query, response)

        slug = f"{qid}_{query[:30].replace(' ', '_').replace('?', '').lower()}"

        # Per-query Markdown
        md_path = out_dir / f"{slug}_evidence_table.md"
        md_path.write_bytes(evidence_table_to_markdown_bytes(table))

        # Per-query CSV
        csv_path = out_dir / f"{slug}_evidence_table.csv"
        csv_path.write_bytes(evidence_table_to_csv_bytes(table))

        row_count = len(table.rows)
        print(f"  [{qid}] {row_count} rows → {md_path.name}, {csv_path.name}")

        all_md_sections.append(f"\n---\n\n## [{qid}] {query}\n")
        all_md_sections.append(evidence_table_to_markdown(table))

    combined_path = out_dir / "combined_evidence_table.md"
    combined_path.write_text("\n".join(all_md_sections))
    print(f"\nCombined artifact → {combined_path}")
    print(f"\nAll artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
