"""Generate data_manifest.csv from PDFs in data/raw/.

Extracts what it can from PDF metadata and filenames.
Fields that need manual completion are marked 'MANUAL'.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz
import pandas as pd

from src.config import Config

# Mapping from filename → source_id for non-obvious filenames
FILENAME_TO_SOURCE_ID: dict[str, str] = {
    "1-s2.0-S2468867319300720-main.pdf": "s2468867319_2019",
    "1-s2.0-S2589537025006765-main.pdf": "s2589537025_2025",
    "Barry2014.pdf": "barry_2014",
    "Brandhorst2024.pdf": "brandhorst_2024",
    "Ju2016.pdf": "ju_2016",
    "PIIS0092867423008577.pdf": "cell_2023",
    "Rattan2022.pdf": "rattan_2022",
    "Stamatakis2025.pdf": "stamatakis_2025",
    "biology-08-00040.pdf": "biology_2019",
    "cells-11-00872.pdf": "cells_2022",
    "f1000research-5-7686.pdf": "f1000_2016",
    "furrer-handschin-2025-biomarkers-of-aging-from-molecules-and-surrogates-to-physiology-and-function.pdf": "furrer_handschin_2025",
    "kwaa057.pdf": "kwaa_2020",
    "nutrients-15-04251.pdf": "nutrients_2023a",
    "nutrients-16-02878.pdf": "nutrients_2024",
    "nutrients-17-00722.pdf": "nutrients_2025",
    "s11033-025-11315-3.pdf": "mol_bio_rep_2025",
    "s12916-024-03833-x.pdf": "bmc_med_2024",
    "s13098-025-01838-x.pdf": "diabetol_metab_2025",
}


def _extract_year(filename: str, pdf_meta: dict) -> str:
    """Try to extract the publication year."""
    # From filename mapping (Author2024 pattern)
    m = re.search(r"(\d{4})", filename)
    if m:
        year = int(m.group(1))
        if 1990 <= year <= 2030:
            return str(year)
    # From PDF creation date
    creation = pdf_meta.get("creationDate", "")
    m2 = re.search(r"(\d{4})", str(creation))
    if m2:
        return m2.group(1)
    return "MANUAL"


def _extract_title(doc: fitz.Document) -> str:
    """Try to get title from PDF metadata, fallback to first-page heuristic."""
    meta_title = doc.metadata.get("title", "").strip()
    if meta_title and len(meta_title) > 5:
        return meta_title
    # Fallback: largest font on page 0
    if doc.page_count == 0:
        return "MANUAL"
    page = doc[0]
    page_dict = page.get_text("dict")
    best_text = ""
    best_size = 0.0
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("size", 0) > best_size:
                    best_size = span["size"]
                    best_text = span.get("text", "").strip()
    return best_text[:200] if best_text else "MANUAL"


def _extract_authors(doc: fitz.Document) -> str:
    """Try PDF metadata for author field."""
    author = doc.metadata.get("author", "").strip()
    return author if author else "MANUAL"


def build_manifest(raw_dir: Path | None = None, output_path: Path | None = None) -> pd.DataFrame:
    """Scan data/raw/ and generate a manifest CSV."""
    cfg = Config()
    raw_dir = raw_dir or cfg.raw_dir
    output_path = output_path or cfg.manifest_path

    rows = []
    for pdf_path in sorted(raw_dir.glob("*.pdf")):
        fname = pdf_path.name
        source_id = FILENAME_TO_SOURCE_ID.get(fname, pdf_path.stem.lower().replace(" ", "_"))

        doc = fitz.open(str(pdf_path))
        title = _extract_title(doc)
        authors = _extract_authors(doc)
        year = _extract_year(fname, doc.metadata)
        doc.close()

        rows.append({
            "source_id": source_id,
            "title": title,
            "authors": authors,
            "year": year,
            "source_type": "journal_article",
            "venue": "MANUAL",
            "url_or_doi": "MANUAL",
            "raw_path": f"data/raw/{fname}",
            "processed_path": "data/processed/chunks.jsonl",
            "tags": "MANUAL",
            "relevance_note": "MANUAL",
        })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    manual_count = df.apply(lambda row: (row == "MANUAL").sum(), axis=1).sum()
    print(f"Manifest written to {output_path}")
    print(f"  {len(df)} sources found")
    print(f"  {manual_count} fields still need manual completion (marked MANUAL)")
    return df


if __name__ == "__main__":
    build_manifest()
