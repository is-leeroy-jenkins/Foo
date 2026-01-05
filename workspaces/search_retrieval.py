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
from fetchers import (
    Wikipedia,
    TheNews,
    ArXiv,
    GoogleSearch,
)


# ============================================================================
# Helpers
# ============================================================================

def _normalize_results(
    provider: str,
    results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Purpose:
        Normalize heterogeneous search results into a single tabular view.

    Parameters:
        provider (str): Provider name
        results (List[Dict[str, Any]]): Raw provider results

    Returns:
        pd.DataFrame
    """

    rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(results):
        rows.append(
            {
                "Index": idx,
                "Provider": provider,
                "Title": item.get("title") or item.get("name"),
                "Source": item.get("source"),
                "Published": item.get("published")
                or item.get("date"),
                "Snippet": (item.get("snippet") or item.get("summary") or "")[:300],
                "URL": item.get("url"),
            }
        )

    return pd.DataFrame(rows)


def _run_search(
    provider: str,
    query: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Purpose:
        Dispatch search execution to the appropriate Foo fetcher.

    Parameters:
        provider (str): Selected provider
        query (str): Search query
        limit (int): Maximum results

    Returns:
        List[Dict[str, Any]]
    """

    if provider == "Wikipedia":
        return Wikipedia(query, limit=limit).fetch()

    if provider == "News":
        return TheNews(query, limit=limit).fetch()

    if provider == "ArXiv":
        return ArXiv(query, limit=limit).fetch()

    if provider == "Google CSE":
        return GoogleSearch(query, limit=limit).fetch()

    raise ValueError(f"Unsupported provider: {provider}")


# ============================================================================
# Workspace Renderer
# ============================================================================

def render(session: Dict[str, Any]) -> None:
    """
    Purpose:
        Render the Search & Retrieval workspace.

    Parameters:
        session (Dict[str, Any]): Streamlit session state
    """

    st.header("Search & Retrieval Workbench")
    st.caption(
        "Query external knowledge sources and normalize results for downstream "
        "LLM and retrieval workflows."
    )

    # ---------------------------------------------------------------------
    # Input Controls
    # ---------------------------------------------------------------------

    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            query = st.text_input(
                "Search Query",
                placeholder="e.g., budget execution anomalies FY2024",
            )

        with col2:
            provider = st.selectbox(
                "Provider",
                [
                    "Wikipedia",
                    "News",
                    "ArXiv",
                    "Google CSE",
                ],
            )

        with col3:
            limit = st.number_input(
                "Max Results",
                min_value=1,
                max_value=50,
                value=10,
            )

    execute = st.button("Run Search", type="primary")
    st.divider()

    # ---------------------------------------------------------------------
    # Execution
    # ---------------------------------------------------------------------

    if execute:
        if not query:
            st.warning("Please enter a search query.")
            return

        with st.spinner("Running search..."):
            try:
                results = _run_search(provider, query, limit)

                session.setdefault("results", {})
                session["results"]["search"] = {
                    "provider": provider,
                    "query": query,
                    "items": results,
                }

                st.success(f"Retrieved {len(results)} results.")

            except Exception as exc:
                st.error("Search execution failed.")
                st.exception(exc)
                return

    # ---------------------------------------------------------------------
    # Results Table
    # ---------------------------------------------------------------------

    search_state = session.get("results", {}).get("search")

    if search_state:
        provider = search_state["provider"]
        items = search_state["items"]

        st.subheader("Search Results")

        df = _normalize_results(provider, items)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        # -----------------------------------------------------------------
        # Inspector
        # -----------------------------------------------------------------

        st.divider()
        st.subheader("Result Inspector")

        selected_index = st.number_input(
            "Select result index",
            min_value=0,
            max_value=len(items) - 1,
            step=1,
        )

        item = items[int(selected_index)]

        st.markdown("**Title**")
        st.write(item.get("title") or item.get("name"))

        st.markdown("**URL**")
        st.write(item.get("url"))

        st.markdown("**Summary / Snippet**")
        st.text_area(
            label="",
            value=item.get("snippet") or item.get("summary") or "",
            height=200,
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
                session["documents"].setdefault("search", [])
                session["documents"]["search"].append(item)
                st.success("Result saved to session documents.")

        with col_b:
            if item.get("url"):
                st.download_button(
                    label="Download Citation",
                    data=f"{item.get('title')}\n{item.get('url')}",
                    file_name="citation.txt",
                    mime="text/plain",
                )

