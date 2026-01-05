from __future__ import annotations

import streamlit as st
import pandas as pd
import tempfile
import os
from typing import Dict, Any, List

from loaders import CsvLoader, WordLoader, MarkdownLoader


def _save(upload) -> str:
    suffix = os.path.splitext(upload.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.read())
        return tmp.name


def render(session: Dict[str, Any]) -> None:
    st.header("Document Ingestion")

    source = st.selectbox("Source Type", ["CSV", "DOCX", "Markdown"])
    upload = st.file_uploader("Upload File")

    run = st.button("Load Document", type="primary")
    st.divider()

    if run:
        if not upload:
            st.warning("File required.")
            return

        path = _save(upload)

        try:
            if source == "CSV":
                docs = CsvLoader(path).load()
            elif source == "DOCX":
                docs = WordLoader(path).load()
            elif source == "Markdown":
                docs = MarkdownLoader(path).load()
            else:
                raise ValueError(source)

            session.setdefault("documents", {})
            session["documents"]["loaded"] = docs
            st.success(f"{len(docs)} document entries loaded.")
        except Exception as exc:
            st.error("Loading failed.")
            st.exception(exc)
            return

    docs = session.get("documents", {}).get("loaded")
    if not docs:
        return

    df = pd.DataFrame(
        {
            "Index": range(len(docs)),
            "Length": [len(str(d)) for d in docs],
            "Preview": [str(d)[:200] for d in docs],
        }
    )

    st.dataframe(df, use_container_width=True, hide_index=True)

    idx = st.number_input("Inspect document", 0, len(docs) - 1, 0)
    st.text_area("", str(docs[int(idx)]), height=300)
