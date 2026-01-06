from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, List

import streamlit as st


# --------------------------------------------------------------------------------------
# Safe Imports
# --------------------------------------------------------------------------------------

def safe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


core = safe_import("core")
loaders = safe_import("loaders")
scrapers = safe_import("scrapers")
fetchers = safe_import("fetchers")
writers = safe_import("writers")
data_mod = safe_import("data")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def save_uploaded_file(uploaded) -> Path:
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return Path(tmp.name)


# --------------------------------------------------------------------------------------
# Streamlit Setup
# --------------------------------------------------------------------------------------

st.set_page_config(page_title="Foo", layout="wide")
st.title("Foo — Unified Streamlit Application")
st.caption("Integrated document ingestion, web extraction, and Markdown export")


# ======================================================================================
# SIDEBAR — CONTROLS ONLY
# ======================================================================================

with st.sidebar:
    st.header("Configuration")

    # ------------------------------------------------------------------
    # Output Directory (manual + browse workaround)
    # ------------------------------------------------------------------

    st.subheader("Output")

    output_dir = st.text_input("Output directory", "output")

    browse_file = st.file_uploader(
        "Browse to output directory (select any file inside it)",
        key="output_dir_browse",
    )

    if browse_file:
        inferred = save_uploaded_file(browse_file).parent
        output_dir = str(inferred)

    # ------------------------------------------------------------------
    # API Keys (hidden by default)
    # ------------------------------------------------------------------

    with st.expander("API Keys", expanded=False):
        openai_key = st.text_input(
            "OPENAI_API_KEY",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
        )
        groq_key = st.text_input(
            "GROQ_API_KEY",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
        )
        gemini_key = st.text_input(
            "GEMINI_API_KEY",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
        )

        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key


# ======================================================================================
# TABS
# ======================================================================================

tab_docs, tab_web, tab_write, tab_data, tab_about = st.tabs(
    ["Documents", "Web", "Write", "Data", "About"]
)

# ======================================================================================
# DOCUMENTS TAB
# ======================================================================================

with tab_docs:
    st.subheader("Document Loading")

    uploaded_docs = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md", "html", "docx", "pptx", "csv", "xlsx", "json"],
        accept_multiple_files=True,
    )

    glob_pattern = st.text_input("Or load by glob pattern (e.g., docs/*.pdf)", "")

    col1, col2 = st.columns(2)
    chunk_size = col1.number_input("Chunk size", 100, 8000, 1000, step=100)
    chunk_overlap = col2.number_input("Chunk overlap", 0, 2000, 200, step=50)

    if st.button("Load and Split", type="primary"):
        paths: List[Path] = []

        if uploaded_docs:
            paths.extend(save_uploaded_file(f) for f in uploaded_docs)

        if glob_pattern.strip() and loaders is not None:
            paths.extend(Path(p) for p in loaders.Loader().resolve_paths(glob_pattern))

        for p in paths:
            try:
                loader = loaders.Loader()
                docs = loader.load_documents(str(p))
                chunks = loader.split_documents(docs, int(chunk_size), int(chunk_overlap))

                with st.expander(f"{p.name} — {len(chunks)} chunks"):
                    st.text_area("Preview", chunks[0].page_content[:4000], height=250)
            except Exception as exc:
                st.error(f"Failed to load {p.name}")
                st.exception(exc)


# ======================================================================================
# WEB TAB
# ======================================================================================

with tab_web:
    st.subheader("Web Scraping")

    url = st.text_input("Single URL")
    batch_urls = st.text_area("Batch URLs (one per line)", height=120)

    extraction_mode = st.selectbox(
        "Extraction Mode",
        ["Full Text", "Paragraphs", "Links", "Tables"],
    )

    if st.button("Fetch", type="primary"):
        targets = (
            [u.strip() for u in batch_urls.splitlines() if u.strip()]
            if batch_urls.strip()
            else ([url.strip()] if url.strip() else [])
        )

        results: List[Any] = []

        for target in targets:
            extractor = scrapers.WebExtractor()
            if extraction_mode == "Paragraphs":
                text = "\n\n".join(extractor.scrape_paragraphs(target))
                res = core.Result(target, None, text)
            elif extraction_mode == "Links":
                links = extractor.scrape_links(target)
                text = "\n".join(f"{l.text} → {l.href}" for l in links)
                res = core.Result(target, None, text)
            elif extraction_mode == "Tables":
                res = (target, extractor.scrape_tables(target))
            else:
                res = extractor.scrape(target)

            results.append(res)

        st.session_state["foo_results"] = results


# ======================================================================================
# WRITE TAB
# ======================================================================================

with tab_write:
    st.subheader("Markdown Export")

    results = st.session_state.get("foo_results", [])
    if not results:
        st.info("No results available.")
    else:
        if st.button("Write Markdown", type="primary"):
            for i, res in enumerate(results, 1):
                if isinstance(res, tuple):
                    continue
                out = Path(output_dir) / f"result_{i}.md"
                writers.MarkdownWriter().write(res, str(out))
                st.code(str(out))


# ======================================================================================
# DATA TAB (RENAMED)
# ======================================================================================

with tab_data:
    st.subheader("SQLite Inspection")

    uploaded_db = st.file_uploader(
        "Browse for SQLite database",
        type=["db", "sqlite", "sqlite3"],
    )

    db_path = st.text_input("Or enter database path manually")

    if uploaded_db:
        db_path = str(save_uploaded_file(uploaded_db))

    if st.button("Inspect", type="primary"):
        p = Path(db_path)
        if not p.exists():
            st.error("Database not found.")
        else:
            conn = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r[0] for r in cur.fetchall()]
            conn.close()

            st.success(f"Tables found: {len(tables)}")
            st.write(tables)


# ======================================================================================
# ABOUT TAB — RUNTIME STATUS (FIXED)
# ======================================================================================

with tab_about:
    st.subheader("Runtime Status")

    def status(label: str, ok: bool) -> None:
        if ok:
            st.success(label)
        else:
            st.error(label)

    status("core.py", core is not None)
    status("loaders.py", loaders is not None)
    status("scrapers.py", scrapers is not None)
    status("fetchers.py", fetchers is not None)
    status("writers.py", writers is not None)
    status("data.py", data_mod is not None)
