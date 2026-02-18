"""Phase 3 — Personal Research Portal (Streamlit UI).

Tabs
----
1. Search / Ask  — query the RAG pipeline; save threads; view citations
2. History       — browse saved research threads
3. Artifacts     — generate & export evidence tables (MD / CSV / PDF)
4. Evaluation    — metrics dashboard from the Phase 2 evaluation report
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src.*` imports work when run via Streamlit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Longevity Research Portal",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
_EVAL_REPORT = _ROOT / "reports" / "evaluation_report.json"
_THREADS_DIR = _ROOT / "outputs" / "threads"
_ARTIFACTS_DIR = _ROOT / "outputs" / "artifacts"


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading RAG pipeline (first run may take ~30 s)…")
def _get_pipeline():
    from src.rag.pipeline import RAGPipeline
    return RAGPipeline()


def _pipeline_ready() -> bool:
    """Check whether the FAISS index exists before trying to load it."""
    from src.config import Config
    cfg = Config()
    return (cfg.faiss_index_dir / "index.faiss").exists()


# ── Helper: render a single RAGResponse ──────────────────────────────────────

def _render_response(query: str, response, retrieval=None) -> None:
    """Display answer, citations, and caveats in the UI."""
    conf_colour = {"high": "green", "medium": "orange", "low": "red"}.get(
        response.confidence, "grey"
    )

    if response.no_evidence:
        st.error(
            "**No evidence found in the corpus.**\n\n"
            f"{response.answer}\n\n"
            "**Suggested next steps:**\n"
            "- Try rephrasing with more specific domain terms\n"
            "- Broaden to a related concept (e.g. 'mitochondria' instead of 'ATP synthesis')\n"
            "- Check whether the paper covering this topic is in the corpus"
        )
        return

    st.markdown(f"**Confidence:** :{conf_colour}[{response.confidence}]")
    st.markdown(f"**Evidence quality:** {response.evidence_quality}")

    st.markdown("### Answer")
    st.markdown(response.answer)

    if response.citations:
        st.markdown("### Citations")
        for cit in response.citations:
            with st.expander(f"`{cit.citation if hasattr(cit, 'citation') else f'({cit.source_id}, {cit.chunk_id})'}`"):
                st.markdown(f"**Source ID:** `{cit.source_id}`")
                st.markdown(f"**Chunk ID:** `{cit.chunk_id}`")
                if cit.relevant_quote:
                    st.markdown(f"**Relevant quote:** *\"{cit.relevant_quote}\"*")

    if response.caveats:
        st.markdown("### Caveats")
        for c in response.caveats:
            st.warning(c)

    if retrieval:
        st.caption(
            f"Retrieval: {retrieval.all_candidates} candidates → "
            f"{retrieval.above_threshold} above threshold → "
            f"{len(retrieval.chunks)} chunks used"
        )


# ── Tab 1: Search / Ask ───────────────────────────────────────────────────────

def _tab_search() -> None:
    st.header("Search / Ask")

    if not _pipeline_ready():
        st.error(
            "FAISS index not found. Run ingestion first:\n```\nmake ingest\n```"
        )
        return

    pipeline = _get_pipeline()

    query = st.text_area(
        "Research question",
        placeholder="e.g. How does AMPK signaling respond to aerobic exercise?",
        height=80,
    )

    col_run, col_clear = st.columns([1, 5])
    run_clicked = col_run.button("Ask", type="primary", use_container_width=True)
    if col_clear.button("Clear", use_container_width=False):
        st.session_state.pop("last_response", None)
        st.session_state.pop("last_retrieval", None)
        st.session_state.pop("last_query", None)
        st.rerun()

    if run_clicked and query.strip():
        with st.spinner("Retrieving and generating…"):
            response = pipeline.query(query.strip())
            retrieval = pipeline._last_retrieval
        st.session_state["last_response"] = response
        st.session_state["last_retrieval"] = retrieval
        st.session_state["last_query"] = query.strip()

    response = st.session_state.get("last_response")
    retrieval = st.session_state.get("last_retrieval")
    current_query = st.session_state.get("last_query", "")

    if response:
        _render_response(current_query, response, retrieval)

        st.divider()
        col_save, col_artifact = st.columns(2)

        if col_save.button("💾 Save thread", use_container_width=True):
            if retrieval is not None:
                from src.app.threads import save_thread
                path = save_thread(current_query, retrieval, response)
                st.success(f"Thread saved → `{path.name}`")
            else:
                st.warning("No retrieval result to save.")

        if col_artifact.button("📋 Generate evidence table", use_container_width=True):
            st.session_state["artifact_query"] = current_query
            st.session_state["artifact_response"] = response
            st.info("Switch to the **Artifacts** tab to view and export the table.")


# ── Tab 2: History ────────────────────────────────────────────────────────────

def _tab_history() -> None:
    st.header("Research Threads")

    from src.app.threads import list_threads, load_thread

    threads = list_threads(_THREADS_DIR)

    if not threads:
        st.info("No threads saved yet. Run a query in **Search / Ask** and click **Save thread**.")
        return

    st.caption(f"{len(threads)} thread(s) saved")

    for meta in threads:
        label = (
            f"🕐 `{meta['thread_id']}` — {meta['query'][:80]}"
            f"{'…' if len(meta['query']) > 80 else ''}"
            f"  | conf: **{meta['confidence']}**"
            f"  | citations: {meta['citation_count']}"
            + ("  | ⚠ no evidence" if meta["no_evidence"] else "")
        )
        with st.expander(label):
            try:
                data = load_thread(meta["thread_id"], _THREADS_DIR)
            except FileNotFoundError:
                st.error("Thread file missing.")
                continue

            st.markdown(f"**Query:** {data['query']}")
            st.markdown(f"**Timestamp:** {data['timestamp']}")
            resp = data.get("response", {})
            st.markdown(f"**Confidence:** {resp.get('confidence', '—')}")

            st.markdown("**Answer:**")
            st.markdown(resp.get("answer", ""))

            citations = resp.get("citations", [])
            if citations:
                st.markdown("**Citations:**")
                for c in citations:
                    st.markdown(
                        f"- `({c['source_id']}, {c['chunk_id']})` — *\"{c.get('relevant_quote', '')}\"*"
                    )

            caveats = resp.get("caveats", [])
            if caveats:
                for cv in caveats:
                    st.warning(cv)

            ret = data.get("retrieval", {})
            st.caption(
                f"Chunks retrieved: {len(ret.get('chunks', []))} | "
                f"Above threshold: {ret.get('above_threshold', '—')}"
            )


# ── Tab 3: Artifacts ──────────────────────────────────────────────────────────

def _tab_artifacts() -> None:
    st.header("Evidence Table Generator")
    st.markdown(
        "Generate a structured evidence table — "
        "**Claim | Evidence Snippet | Citation | Confidence | Notes** — "
        "from any research query."
    )

    from src.app.artifacts import build_evidence_table
    from src.app.export import (
        evidence_table_to_csv_bytes,
        evidence_table_to_markdown_bytes,
        evidence_table_to_pdf_bytes,
    )

    # Pre-fill from Search tab if user clicked "Generate evidence table"
    prefill_query = st.session_state.get("artifact_query", "")
    prefill_response = st.session_state.get("artifact_response", None)

    query = st.text_area(
        "Research question",
        value=prefill_query,
        placeholder="e.g. What are the longevity effects of caloric restriction?",
        height=80,
    )

    use_cached = (
        prefill_response is not None
        and prefill_query == query.strip()
        and query.strip()
    )

    run_clicked = st.button("Generate evidence table", type="primary")

    if run_clicked and query.strip():
        if use_cached:
            response = prefill_response
        else:
            if not _pipeline_ready():
                st.error("FAISS index not found. Run `make ingest` first.")
                return
            pipeline = _get_pipeline()
            with st.spinner("Running RAG pipeline…"):
                response = pipeline.query(query.strip())

        table = build_evidence_table(query.strip(), response)
        st.session_state["current_table"] = table
        st.session_state["current_table_query"] = query.strip()

    table = st.session_state.get("current_table")
    if table is None:
        return

    if table.no_evidence:
        st.error(
            "No evidence found in the corpus for this query. "
            "The evidence table is empty. Try rephrasing."
        )
        return

    if not table.rows:
        st.warning("Answer contained no parseable inline citations.")
        return

    # Display table
    df = pd.DataFrame(table.to_records())
    # Rename index to #
    df.index = range(1, len(df) + 1)
    st.dataframe(df, use_container_width=True, height=350)

    st.markdown(f"**{len(table.rows)} row(s)** | Overall confidence: **{table.overall_confidence}**")

    # Export buttons
    st.markdown("### Export")
    col_md, col_csv, col_pdf = st.columns(3)

    slug = (table.query[:30].replace(" ", "_").replace("?", ""))
    fname_base = f"evidence_table_{slug}"

    col_md.download_button(
        "⬇ Markdown",
        data=evidence_table_to_markdown_bytes(table),
        file_name=f"{fname_base}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    col_csv.download_button(
        "⬇ CSV",
        data=evidence_table_to_csv_bytes(table),
        file_name=f"{fname_base}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    try:
        pdf_bytes = evidence_table_to_pdf_bytes(table)
        col_pdf.download_button(
            "⬇ PDF",
            data=pdf_bytes,
            file_name=f"{fname_base}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except ImportError:
        col_pdf.warning("PDF export requires `fpdf2`. Run `pip install fpdf2`.")

    # Also save to outputs/artifacts/
    if st.button("💾 Save artifact to repo"):
        _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        md_path = _ARTIFACTS_DIR / f"{fname_base}.md"
        csv_path = _ARTIFACTS_DIR / f"{fname_base}.csv"
        md_path.write_bytes(evidence_table_to_markdown_bytes(table))
        csv_path.write_bytes(evidence_table_to_csv_bytes(table))
        st.success(f"Saved:\n- `{md_path.name}`\n- `{csv_path.name}`")


# ── Tab 4: Evaluation ─────────────────────────────────────────────────────────

def _tab_evaluation() -> None:
    st.header("Evaluation Dashboard")

    if not _EVAL_REPORT.exists():
        st.warning(
            "Evaluation report not found at `reports/evaluation_report.json`. "
            "Run `make eval` to generate it."
        )
        _show_run_eval_option()
        return

    data = json.loads(_EVAL_REPORT.read_text())

    # Support both flat list and {hybrid: [...], vector: [...]} formats
    if isinstance(data, list):
        modes = {"results": data}
    elif isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
        modes = data
    else:
        st.error("Unrecognised evaluation report format.")
        return

    # Summary metrics per mode
    st.subheader("Summary metrics")
    summary_rows = []
    for mode, results in modes.items():
        if not isinstance(results, list):
            continue
        n = len(results)
        cite_prec = (
            sum(r.get("citation_precision", 0) for r in results) / n if n else 0
        )
        faithful = (
            sum(1 for r in results if r.get("faithful", False)) / n if n else 0
        )
        no_ev_acc = (
            sum(1 for r in results if r.get("no_evidence_correct", False)) / n if n else 0
        )
        summary_rows.append({
            "Mode": mode,
            "Queries": n,
            "Avg Citation Precision": f"{cite_prec:.3f}",
            "Faithfulness Rate": f"{faithful:.3f}",
            "No-Evidence Accuracy": f"{no_ev_acc:.3f}",
        })

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows).set_index("Mode"), use_container_width=True)

    # Per-query results
    st.subheader("Per-query results")
    mode_choice = st.selectbox("Mode", list(modes.keys())) if len(modes) > 1 else list(modes.keys())[0]
    results = modes[mode_choice] if isinstance(modes[mode_choice], list) else []

    filter_type = st.multiselect(
        "Filter by query type",
        ["direct", "synthesis", "edge_case"],
        default=["direct", "synthesis", "edge_case"],
    )

    rows = []
    for r in results:
        qt = r.get("query_type", "")
        if qt not in filter_type:
            continue
        rows.append({
            "ID": r.get("query_id", ""),
            "Type": qt,
            "Query": r.get("query", "")[:70],
            "Confidence": r.get("confidence", ""),
            "Citations": r.get("num_citations", 0),
            "Cite Precision": r.get("citation_precision", 0),
            "Faithful": "✓" if r.get("faithful") else "✗",
            "No-Ev Correct": "✓" if r.get("no_evidence_correct") else "✗",
        })

    if rows:
        df = pd.DataFrame(rows).set_index("ID")
        st.dataframe(df, use_container_width=True)

    # Representative examples
    st.subheader("Representative examples")
    col_pass, col_fail = st.columns(2)

    with col_pass:
        st.markdown("**Faithful answers**")
        faithful_examples = [r for r in results if r.get("faithful")][:3]
        for r in faithful_examples:
            with st.expander(f"[{r.get('query_id')}] {r.get('query', '')[:60]}"):
                st.markdown(f"**Query:** {r.get('query', '')}")
                st.markdown(f"**Answer preview:** {r.get('answer_preview', '')[:300]}…")
                st.markdown(f"**Citations:** {r.get('num_citations', 0)}")
                st.caption(r.get("faithfulness_note", ""))

    with col_fail:
        st.markdown("**Failure cases**")
        failure_examples = [r for r in results if not r.get("faithful")][:3]
        if not failure_examples:
            st.success("No faithfulness failures in this mode.")
        for r in failure_examples:
            with st.expander(f"[{r.get('query_id')}] {r.get('query', '')[:60]}"):
                st.markdown(f"**Query:** {r.get('query', '')}")
                st.markdown(f"**Answer preview:** {r.get('answer_preview', '')[:300]}…")
                st.caption(r.get("faithfulness_note", ""))

    _show_run_eval_option()


def _show_run_eval_option() -> None:
    st.divider()
    st.markdown("**Re-run evaluation against live pipeline:**")
    if st.button("Run evaluation set (20 queries)", type="secondary"):
        if not _pipeline_ready():
            st.error("FAISS index not found. Run `make ingest` first.")
            return
        st.warning(
            "Running evaluation makes ~20 API calls. This may take 2–5 minutes.\n"
            "Run `make eval` from the terminal for full parallel execution."
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar() -> None:
    with st.sidebar:
        st.title("🔬 Longevity Research Portal")
        st.caption("Phase 3 — Personal Research Portal")
        st.divider()

        st.markdown("**Corpus**")
        manifest_path = _ROOT / "data" / "data_manifest.csv"
        if manifest_path.exists():
            df = pd.read_csv(manifest_path)
            st.metric("Sources ingested", len(df))
            with st.expander("Browse corpus"):
                st.dataframe(
                    df[["source_id", "title", "year", "source_type"]],
                    use_container_width=True,
                    height=200,
                )
        else:
            st.info("data_manifest.csv not found.")

        st.divider()
        threads_dir = _ROOT / "outputs" / "threads"
        n_threads = len(list(threads_dir.glob("*.json"))) if threads_dir.exists() else 0
        st.metric("Saved threads", n_threads)

        artifacts_dir = _ROOT / "outputs" / "artifacts"
        n_artifacts = len(list(artifacts_dir.glob("*"))) if artifacts_dir.exists() else 0
        st.metric("Exported artifacts", n_artifacts)

        st.divider()
        st.markdown("**Quick help**")
        st.markdown(
            "1. **Search / Ask** — enter a research question\n"
            "2. Click **Save thread** to persist results\n"
            "3. Click **Generate evidence table** for a structured artifact\n"
            "4. **Artifacts** tab → download as MD / CSV / PDF\n"
            "5. **Evaluation** tab → see pipeline metrics"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 Search / Ask", "🕐 History", "📋 Artifacts", "📊 Evaluation"]
    )

    with tab1:
        _tab_search()
    with tab2:
        _tab_history()
    with tab3:
        _tab_artifacts()
    with tab4:
        _tab_evaluation()


if __name__ == "__main__":
    main()
