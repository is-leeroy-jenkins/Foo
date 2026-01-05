from __future__ import annotations

import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from fetchers import WebFetcher
from scrapers import WebExtractor


def _normalize(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(results):
        text = r.get("text", "")
        rows.append(
            {
                "Index": i,
                "Tag": r.get("tag"),
                "Length": len(text),
                "Preview": text[:200],
            }
        )
    return pd.DataFrame(rows)


def render(session: Dict[str, Any]) -> None:
    st.header("Web Extraction Studio")

    url = st.text_input("Target URL", placeholder="https://example.com")
    run = st.button("Fetch & Extract", type="primary")
    st.divider()

    if run:
        if not url:
            st.warning("URL is required.")
            return

        try:
            html = WebFetcher(url).fetch()
            results = WebExtractor(html).extract()

            session.setdefault("results", {})
            session["results"]["web"] = results

            st.success(f"Extracted {len(results)} elements.")
        except Exception as exc:
            st.error("Extraction failed.")
            st.exception(exc)
            return

    results = session.get("results", {}).get("web")
    if not results:
        return

    st.subheader("Extracted Content")
    st.dataframe(_normalize(results), use_container_width=True, hide_index=True)

    st.divider()
    idx = st.number_input("Inspect element", 0, len(results) - 1, 0)

    item = results[int(idx)]
    st.markdown("**Tag**")
    st.code(item.get("tag", ""))
    st.markdown("**Text**")
    st.text_area("", item.get("text", ""), height=300)
