"""
Workspace: Document Ingestion

Purpose:
    Streamlit UI for uploading documents and ingesting them into the Foo
    pipeline. Document loading and splitting are handled entirely by the
    corresponding Loader implementations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import streamlit as st

from langchain_core.documents import Document
from loaders import (
    PdfLoader,
    MarkdownLoader,
    CsvLoader,
    WordLoader,
)


def render(state) -> None:
    """
    Render the Document Ingestion workspace.
    """

    st.header("📄 Document Ingestion")

    st.markdown(
        """
        Upload documents to ingest them into the system.
        Each document is processed by its corresponding Loader,
        which is responsible for loading and splitting content.
        """
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md", "csv", "docx"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        return

    documents: List[Document] = []

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if suffix == ".pdf":
            loader = PdfLoader(tmp_path)
            docs = loader.load()

        elif suffix in {".txt", ".md"}:
            loader = MarkdownLoader(tmp_path)
            docs = loader.load()

        elif suffix == ".csv":
            loader = CsvLoader(tmp_path)
            docs = loader.load()

        elif suffix == ".docx":
            loader = WordLoader(tmp_path)
            docs = loader.load()

        else:
            continue

        documents.extend(docs)

    if not documents:
        st.warning("No documents were ingested.")
        return

    st.success(f"Ingested {len(documents)} document chunks.")

    with st.expander("Preview chunks"):
        for i, doc in enumerate(documents[:5], start=1):
            st.markdown(f"**Chunk {i}**")
            st.text(doc.page_content[:1000])
            st.divider()
