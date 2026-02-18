"""Export research artifacts to Markdown, CSV, and PDF.

All functions return bytes so they can be fed directly to
st.download_button(data=...) in Streamlit.
"""

from __future__ import annotations

import csv
import io
import textwrap
from datetime import datetime, timezone

from src.app.artifacts import EvidenceTable


# ── Markdown ──────────────────────────────────────────────────────────────────

def evidence_table_to_markdown(table: EvidenceTable) -> str:
    """Render an EvidenceTable as a Markdown string."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Evidence Table",
        "",
        f"**Query:** {table.query}",
        f"**Overall confidence:** {table.overall_confidence}",
        f"**Generated:** {now}",
        "",
    ]

    if table.no_evidence:
        lines.append(
            "> No evidence found in the corpus for this query. "
            "Try rephrasing or expanding the corpus."
        )
        return "\n".join(lines)

    if not table.rows:
        lines.append("> No citations extracted from the answer.")
        return "\n".join(lines)

    # Table header
    lines += [
        "| # | Claim | Evidence Snippet | Citation | Confidence | Notes |",
        "|---|-------|-----------------|----------|------------|-------|",
    ]

    for i, row in enumerate(table.rows, 1):
        def cell(s: str) -> str:
            return s.replace("|", "\\|").replace("\n", " ").strip()

        lines.append(
            f"| {i} "
            f"| {cell(row.claim)} "
            f"| {cell(row.evidence_snippet)} "
            f"| {cell(row.citation)} "
            f"| {cell(row.confidence)} "
            f"| {cell(row.notes)} |"
        )

    return "\n".join(lines) + "\n"


def evidence_table_to_markdown_bytes(table: EvidenceTable) -> bytes:
    return evidence_table_to_markdown(table).encode("utf-8")


# ── CSV ───────────────────────────────────────────────────────────────────────

def evidence_table_to_csv_bytes(table: EvidenceTable) -> bytes:
    """Render an EvidenceTable as UTF-8 CSV bytes."""
    buf = io.StringIO()
    fieldnames = ["#", "Claim", "Evidence snippet", "Citation", "Confidence", "Notes"]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    if table.no_evidence or not table.rows:
        writer.writerow({
            "#": 1,
            "Claim": "No evidence found",
            "Evidence snippet": "",
            "Citation": "",
            "Confidence": "low",
            "Notes": "No evidence found in corpus for this query.",
        })
    else:
        for i, row in enumerate(table.rows, 1):
            writer.writerow({
                "#": i,
                "Claim": row.claim,
                "Evidence snippet": row.evidence_snippet,
                "Citation": row.citation,
                "Confidence": row.confidence,
                "Notes": row.notes,
            })

    return buf.getvalue().encode("utf-8")


# ── PDF ───────────────────────────────────────────────────────────────────────

def evidence_table_to_pdf_bytes(table: EvidenceTable) -> bytes:
    """Render an EvidenceTable as a PDF (requires fpdf2)."""
    try:
        from fpdf import FPDF  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "fpdf2 is required for PDF export. Run: pip install fpdf2"
        ) from exc

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Evidence Table", new_x="LMARGIN", new_y="NEXT")

    # Metadata
    pdf.set_font("Helvetica", "", 10)
    # Wrap long query
    wrapped_query = textwrap.shorten(table.query, width=100, placeholder="...")
    pdf.multi_cell(0, 6, f"Query: {wrapped_query}")
    pdf.cell(0, 6, f"Confidence: {table.overall_confidence}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Generated: {now}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    if table.no_evidence or not table.rows:
        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(0, 6, "No evidence found in corpus for this query.")
        return bytes(pdf.output())

    # Column widths (total = ~190 mm on A4 with 10 mm margins each side)
    col_w = {"#": 8, "Claim": 52, "Evidence": 55, "Citation": 40, "Conf": 14, "Notes": 21}

    def header_row() -> None:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(220, 220, 220)
        for label, w in col_w.items():
            pdf.cell(w, 7, label, border=1, fill=True)
        pdf.ln()

    header_row()

    pdf.set_font("Helvetica", "", 7)
    for i, row in enumerate(table.rows, 1):
        # Use multi_cell for wrapping; snapshot x/y before and after
        x_start = pdf.get_x()
        y_start = pdf.get_y()

        # Determine row height by finding the tallest cell
        def lines_needed(text: str, width: float) -> int:
            chars_per_line = max(1, int(width / 1.8))  # rough estimate at 7pt
            wrapped = textwrap.wrap(text or " ", chars_per_line)
            return max(1, len(wrapped))

        row_h = 4  # pt per line
        heights = [
            lines_needed(str(i), col_w["#"]) * row_h,
            lines_needed(row.claim, col_w["Claim"]) * row_h,
            lines_needed(row.evidence_snippet, col_w["Evidence"]) * row_h,
            lines_needed(row.citation, col_w["Citation"]) * row_h,
            lines_needed(row.confidence, col_w["Conf"]) * row_h,
            lines_needed(row.notes, col_w["Notes"]) * row_h,
        ]
        cell_h = max(heights)
        cell_h = max(cell_h, 5)

        # Check page break
        if pdf.get_y() + cell_h > pdf.page_break_trigger:
            pdf.add_page()
            header_row()
            pdf.set_font("Helvetica", "", 7)

        # Draw each cell as multi_cell, then reposition
        x = pdf.get_x()
        y = pdf.get_y()
        cells = [
            (str(i), col_w["#"]),
            (row.claim, col_w["Claim"]),
            (row.evidence_snippet, col_w["Evidence"]),
            (row.citation, col_w["Citation"]),
            (row.confidence, col_w["Conf"]),
            (row.notes, col_w["Notes"]),
        ]
        for text, w in cells:
            pdf.set_xy(x, y)
            pdf.multi_cell(w, row_h, text or "", border=1, max_line_height=row_h)
            x += w

        pdf.set_xy(x_start, y + cell_h)

    return bytes(pdf.output())
