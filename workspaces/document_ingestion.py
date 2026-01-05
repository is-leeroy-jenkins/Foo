'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                name.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="guro.py" company="Terry D. Eppler">

	     name.py
	     Copyright ©  2022  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    name.py
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

import streamlit as st
import pandas as pd
import tempfile
import os

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Foo imports
# ---------------------------------------------------------------------------
from loaders import (
    PdfLoader,
    DocxLoader,
    MarkdownLoader,
    CSVLoader,
    TextLoader,
)

from data import split_documents


# ============================================================================
# Helpers
# ============================================================================

def _save_uploaded_file(upload) -> str:
    """
    Purpose:
        Persist an uploaded file to a temporary path for loader consumption.

    Parameters:
        upload: Streamlit uploaded file

    Returns:
        str: Temporary file path
    """

    suffix = os.path.splitext(upload.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.read())
        return tmp.name


def _load_documents(
    source_type: str,
    file_path: str | None,
    text_payload: str | None,
) -> List[Any]:
    """
    Purpose:
        Dispatch document loading to the appropriate Foo loader.

    Parameters:
        source_type (str): Selected source type
        file_path (str | None): Path to uploaded file
        text_payload (str | None): Raw text payload

    Returns:
        List[Any]: Loaded document objects
    """

    if source_type == "PDF":
        return PdfLoader(file_path).load()

    if source_type == "DOCX":
        return DocxLoader(file_path).load()

    if source_type == "Markdown":
        return MarkdownLoader(file_path).load()

    if source_type == "CSV":
        return CSVLoader(file_path).load()

    if source_type == "Raw Text":
        return TextLoader(text_payload).load()

    raise ValueError(f"Unsupported source type: {source_type}")


def _normalize_chunks(chunks: List[Any]) -> pd.DataFrame:
    """
    Purpose:
        Normalize chunked documents into a tabular representation.

    Parameters:
        chunks (List[Any]): Chunked document objects

    Returns:
        pd.DataFrame
    """

    rows = []
    for idx, doc in enumerate(chunks):
        text = getattr(doc, "page_content", "")
        meta = getattr(doc, "metadata", {})

        rows.append(
            {
                "Chunk ID": idx,
                "Length": len(text),
                "Source": meta.get("source"),
                "Preview": text[:200],
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# Workspace Renderer
# ============================================================================

def render(session: Dict[str, Any]) -> None:
    """
    Purpose:
        Render the Document Ingestion & Chunking workspace.

    Parameters:
        session (Dict[str, Any]): Streamlit session state
    """

    st.header("Document Ingestion & Chunking")
    st.caption(
        "Load documents or prior text artifacts and split them into "
        "LLM-ready chunks for retrieval and agent workflows."
    )

    # ---------------------------------------------------------------------
    # Source Selection
    # ---------------------------------------------------------------------

    source_type = st.selectbox(
        "Source Type",
        [
            "PDF",
            "DOCX",
            "Markdown",
            "CSV",
            "Raw Text",
        ],
    )

    upload = None
    raw_text = None

    if source_type == "Raw Text":
        raw_text = st.text_area(
            "Paste Text",
            height=200,
            placeholder="Paste text from Web Extraction or Search results...",
        )
    else:
        upload = st.file_uploader(
            f"Upload {source_type} file",
            type=[source_type.lower()],
        )

    st.divider()

    # ---------------------------------------------------------------------
    # Chunking Controls
    # ---------------------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=4000,
            value=1000,
            step=100,
        )

    with col2:
        overlap = st.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=100,
            step=50,
        )

    execute = st.button("Load & Chunk", type="primary")
    st.divider()

    # ---------------------------------------------------------------------
    # Execution
    # ---------------------------------------------------------------------

    if execute:
        if source_type == "Raw Text" and not raw_text:
            st.warning("Please provide text to ingest.")
            return

        if source_type != "Raw Text" and not upload:
            st.warning("Please upload a file.")
            return

        with st.spinner("Loading and chunking document(s)..."):
            try:
                file_path = None
                if upload:
                    file_path = _save_uploaded_file(upload)

                documents = _load_documents(
                    source_type=source_type,
                    file_path=file_path,
                    text_payload=raw_text,
                )

                chunks = split_documents(
                    documents,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )

                session.setdefault("documents", {})
                session["documents"]["chunks"] = chunks

                st.success(f"Created {len(chunks)} chunks.")

            except Exception as exc:
                st.error("Document ingestion failed.")
                st.exception(exc)
                return

    # ---------------------------------------------------------------------
    # Chunk Table
    # ---------------------------------------------------------------------

    chunks = session.get("documents", {}).get("chunks")

    if chunks:
        st.subheader("Chunked Documents")

        df = _normalize_chunks(chunks)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        # -----------------------------------------------------------------
        # Inspector
        # -----------------------------------------------------------------

        st.divider()
        st.subheader("Chunk Inspector")

        selected_index = st.number_input(
            "Select chunk ID",
            min_value=0,
            max_value=len(chunks) - 1,
            step=1,
        )

        doc = chunks[int(selected_index)]

        st.markdown("**Metadata**")
        st.json(getattr(doc, "metadata", {}))

        st.markdown("**Chunk Text**")
        st.text_area(
            label="",
            value=getattr(doc, "page_content", ""),
            height=300,
        )

        # -----------------------------------------------------------------
        # Actions
        # -----------------------------------------------------------------

        st.divider()
        st.subheader("Actions")

        col_a, col_b = st.columns(2)

        with col_a:
            st.download_button(
                label="Download Chunk Text",
                data=getattr(doc, "page_content", ""),
                file_name=f"chunk_{selected_index}.txt",
                mime="text/plain",
            )

        with col_b:
            st.download_button(
                label="Export All Chunks (JSONL)",
                data="\n".join(
                    [
                        getattr(d, "page_content", "")
                        for d in chunks
                    ]
                ),
                file_name="chunks.jsonl",
                mime="application/json",
            )

