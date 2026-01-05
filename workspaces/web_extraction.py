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

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Foo imports
# ---------------------------------------------------------------------------
from fetchers import WebFetcher
from scrapers import WebExtractor


# ============================================================================
# Helpers
# ============================================================================

def _normalize_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Purpose:
        Normalize Foo scraper results into a tabular DataFrame.

    Parameters:
        results (List[Dict[str, Any]]): Raw scraper outputs

    Returns:
        pd.DataFrame
    """

    rows = []
    for idx, item in enumerate(results):
        rows.append(
            {
                "Index": idx,
                "Type": item.get("type"),
                "Length": len(item.get("text", "")),
                "Preview": item.get("text", "")[:200],
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# Workspace Renderer
# ============================================================================

def render(session: Dict[str, Any]) -> None:
    """
    Purpose:
        Render the Web Extraction Studio workspace.

    Parameters:
        session (Dict[str, Any]): Streamlit session state
    """

    st.header("Web Extraction Studio")
    st.caption(
        "Fetch a web page and extract structured textual content suitable for "
        "LLM prompting or retrieval pipelines."
    )

    # ---------------------------------------------------------------------
    # Input Controls
    # ---------------------------------------------------------------------

    with st.container():
        col1, col2 = st.columns([3, 1])

        with col1:
            url = st.text_input(
                "Target URL",
                placeholder="https://example.com/article",
            )

        with col2:
            fetch_button = st.button("Fetch & Extract", type="primary")

    st.divider()

    # ---------------------------------------------------------------------
    # Execution
    # ---------------------------------------------------------------------

    if fetch_button:
        if not url:
            st.warning("Please enter a valid URL.")
            return

        with st.spinner("Fetching and extracting content..."):
            try:
                fetcher = WebFetcher(url)
                html = fetcher.fetch()

                scraper = WebExtractor(html)
                results = scraper.extract()

                # Persist results in session
                session.setdefault("results", {})
                session["results"]["web_extraction"] = results

                st.success(f"Extracted {len(results)} items.")

            except Exception as exc:
                st.error("Web extraction failed.")
                st.exception(exc)
                return

    # ---------------------------------------------------------------------
    # Results Table
    # ---------------------------------------------------------------------

    results = session.get("results", {}).get("web_extraction")

    if results:
        st.subheader("Extracted Content")

        df = _normalize_results(results)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        # -----------------------------------------------------------------
        # Inspector
        # -----------------------------------------------------------------

        st.divider()
        st.subheader("Content Inspector")

        selected_index = st.number_input(
            "Select item index",
            min_value=0,
            max_value=len(results) - 1,
            step=1,
        )

        item = results[int(selected_index)]

        st.markdown("**Type**")
        st.code(item.get("type", ""), language="text")

        st.markdown("**Full Text**")
        st.text_area(
            label="",
            value=item.get("text", ""),
            height=300,
        )

        # -----------------------------------------------------------------
        # Actions
        # -----------------------------------------------------------------

        st.divider()
        st.subheader("Actions")

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("Save to Session"):
                session.setdefault("documents", {})
                session["documents"].setdefault("web", [])
                session["documents"]["web"].append(item)
                st.success("Item saved to session documents.")

        with col_b:
            st.download_button(
                label="Download Text",
                data=item.get("text", ""),
                file_name="extracted_text.txt",
                mime="text/plain",
            )

