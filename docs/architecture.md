# Architecture
![](./images/foo-architecture.png)
Foo is organized as a modular Streamlit application backed by source modules that can also be used directly from Python.

```mermaid
graph TD
    A[app.py Streamlit UI] --> B[loaders.py]
    A --> C[scrapers.py]
    A --> D[fetchers.py]
    A --> E[generators.py]
    A --> F[data.py]
    B --> G[LangChain Documents]
    C --> H[core.Result]
    D --> H
    E --> I[Provider APIs]
    F --> J[SQLite / Chroma]
    H --> K[writers.py]
    L[models.py] --> A
    M[config.py] --> A
    M --> B
    M --> D
    M --> E
    M --> F
```

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `core.py` | Shared validation and response-result container. |
| `loaders.py` | File, document, cloud, notebook, and search loader wrappers. |
| `scrapers.py` | HTML extraction and lightweight page scraping. |
| `fetchers.py` | Web, public-data, science, geospatial, environmental, and retriever wrappers. |
| `generators.py` | LLM provider wrappers and generation workflows. |
| `data.py` | SQLite and vector-store data helpers. |
| `models.py` | Pydantic schemas used by application workflows. |
| `writers.py` | Markdown writer utilities for persisted results. |
| `config.py` | Environment variables, constants, mode maps, and session-state defaults. |
| `app.py` | Streamlit user interface and workflow orchestration. |
