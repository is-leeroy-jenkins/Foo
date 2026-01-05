from __future__ import annotations

import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from fetchers import Wikipedia, TheNews, ArXiv, GoogleSearch


def _normalize(provider: str, items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i, item in enumerate(items):
        rows.append(
            {
                "Index": i,
                "Provider": provider,
                "Title": item.get("title") or item.get("name"),
                "Source": item.get("source"),
                "Published": item.get("published"),
                "Snippet": (item.get("summary") or item.get("snippet") or "")[:300],
                "URL": item.get("url"),
            }
        )
    return pd.DataFrame(rows)


def _run(provider: str, query: str, limit: int):
    if provider == "Wikipedia":
        return Wikipedia(query, limit=limit).fetch()
    if provider == "News":
        return TheNews(query, limit=limit).fetch()
    if provider == "ArXiv":
        return ArXiv(query, limit=limit).fetch()
    if provider == "Google Search":
        return GoogleSearch(query, limit=limit).fetch()
    raise ValueError(provider)


def render(session: Dict[str, Any]) -> None:
    st.header("Search & Retrieval Workbench")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input("Query")
    with col2:
        provider = st.selectbox(
            "Provider", ["Wikipedia", "News", "ArXiv", "Google Search"]
        )
    with col3:
        limit = st.number_input("Max Results", 1, 50, 10)

    run = st.button("Run Search", type="primary")
    st.divider()

    if run:
        if not query:
            st.warning("Query required.")
            return

        try:
            items = _run(provider, query, limit)
            session.setdefault("results", {})
            session["results"]["search"] = {
                "provider": provider,
                "items": items,
            }
            st.success(f"{len(items)} results retrieved.")
        except Exception as exc:
            st.error("Search failed.")
            st.exception(exc)
            return

    state = session.get("results", {}).get("search")
    if not state:
        return

    items = state["items"]
    st.dataframe(
        _normalize(state["provider"], items),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    idx = st.number_input("Inspect result", 0, len(items) - 1, 0)
    item = items[int(idx)]

    st.markdown("**Title**")
    st.write(item.get("title") or item.get("name"))
    st.markdown("**Summary**")
    st.text_area("", item.get("summary") or "", height=200)
