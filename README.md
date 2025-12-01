![](https://github.com/is-leeroy-jenkins/Foo/blob/main/resources/images/foo_project.png)


A small, multi-tool framework for retrieval augmentation, SQL Querying, and Document Loading 

## 🧠 **Overview**

Foo is an extensible Python framework designed to unify:

* Conversational interaction with OpenAI-compatible LLMs
* Deterministic natural-language SQL querying
* Document ingestion and vector-based semantic retrieval
* Tool routing through a LangChain ReAct agent
* Structured error-handling via guard clauses and dialogs

- Foo’s architecture is modular by design, composed of separate tool classes for SQL, document retrieval, and future API integrations. The `Fetch` controller coordinates these tools to provide a predictable, expandable multi-modal reasoning environment.



## **🤖 Fetch**

| Category                  | Details                                                                                                                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Purpose**               | Unified orchestrator providing SQL querying, document retrieval, embeddings, and LLM chat.                                                                                                             |
| **Core Responsibilities** | Initialize LLM; load documents; build vectors; initialize SQL/doc/API tools; maintain memory; route queries; execute agent workflows.                                                                  |
| **Key Attributes**        | `model`, `temperature`, `llm`, `db_uri`, `doc_paths`, `memory`, `sql_tool`, `doc_tool`, `api_tools`, `agent`, `__tools`, `documents`, `db_toolkit`, `loader`, `tool`, `extension`, `answer`, `sources` |
| **Public Methods**        | `query_sql(question)` <br> `query_docs(question, with_sources)` <br> `query_chat(prompt)`                                                                                                              |
| **Internal Methods**      | `_init_sql_tool()` <br> `_init_doc_tool()` <br> `_init_api_tools()`                                                                                                                                    |
| **Notes**                 | Core integration engine. Uses LangChain’s `initialize_agent` with `CHAT_ZERO_SHOT_REACT_DESCRIPTION`.                                                                                                  |



## **🗃️ SQLite**

| Category                  | Details                                                                                                                                                                                                                    |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**               | Persistent storage backend for text chunks and embedding vectors using SQLite.                                                                                                                                             |
| **Core Responsibilities** | Create schema; insert embeddings; batch insert; fetch/update/delete embeddings; count and purge records.                                                                                                                   |
| **Key Attributes**        | `db_path`, `connection`, `cursor`                                                                                                                                                                                          |
| **Public Methods**        | `create()` <br> `insert(...)` <br> `insert_many(...)` <br> `fetch_all()` <br> `fetch_by_file(file)` <br> `delete_by_file(file)` <br> `update_embedding_by_id(id, vector)` <br> `count()` <br> `purge_all()` <br> `close()` |
| **Notes**                 | Stores vectors as JSON; supports multi-document ingestion; integrates with Chroma.                                                                                                                                         |


## **🧬 Chroma**

| Category                  | Details                                                                                                                                   |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**               | Wrapper around ChromaDB for vector storage and similarity search.                                                                         |
| **Core Responsibilities** | Create/load a collection; add embeddings; perform similarity queries; delete vectors; count, clear, and persist collection.               |
| **Key Attributes**        | `client`, `collection`                                                                                                                    |
| **Public Methods**        | `add(ids, texts, embeddings, metadatas)` <br> `query(text, num, where)` <br> `delete(ids)` <br> `count()` <br> `clear()` <br> `persist()` |
| **Notes**                 | Complements the SQLite backend or can operate standalone.                                                                                 |



## **📦 Loader (Base Class)**

| Category                  | Details                                                                                                             |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Purpose**               | Base loader class providing validation, file resolution, and document splitting utilities.                          |
| **Core Responsibilities** | Validate paths; expand glob patterns; split documents into overlapping chunks.                                      |
| **Key Attributes**        | `documents`, `file_path`, `pattern`, `expanded`, `candidates`, `resolved`, `loader`, `chunk_size`, `overlap_amount` |
| **Public Methods**        | `verify_exists(path)` <br> `_resolve_paths(pattern)` <br> `_split_documents(docs, chunk, overlap)`                  |
| **Notes**                 | All other loaders subclass `Loader`.                                                                                |



## **🧾 CsvLoader**

| Category                  | Details                                                                      |
| ------------------------- | ---------------------------------------------------------------------------- |
| **Purpose**               | Wraps LangChain’s `CSVLoader` to ingest CSV files.                           |
| **Core Responsibilities** | Parse CSV rows; produce Document objects; split into chunks.                 |
| **Key Attributes**        | `encoding`, `csv_args`, `source_column`                                      |
| **Public Methods**        | `load(path, encoding, csv_args, source_column)` <br> `split(chunk, overlap)` |
| **Notes**                 | Most configurable loader; supports custom parsing options.                   |



## **📄 WordLoader**

| Category                  | Details                                                  |
| ------------------------- | -------------------------------------------------------- |
| **Purpose**               | Wraps `Docx2txtLoader` to ingest `.docx` Word documents. |
| **Core Responsibilities** | Extract text; convert to Documents; split into chunks.   |
| **Key Attributes**        | `file_path`, `documents`, `loader`                       |
| **Public Methods**        | `load(path)` <br> `split(chunk, overlap)`                |
| **Notes**                 | Robust for formal documents and reports.                 |



## **📚 PdfLoader**

| Category                  | Details                                                         |
| ------------------------- | --------------------------------------------------------------- |
| **Purpose**               | Wraps `PyPDFLoader` to ingest PDF files.                        |
| **Core Responsibilities** | Extract page text; convert pages to Documents; chunk long PDFs. |
| **Key Attributes**        | `file_path`, `documents`, `loader`                              |
| **Public Methods**        | `load(path)` <br> `split(chunk, overlap)`                       |
| **Notes**                 | Ideal for large multi-page documents.                           |



## **📝 MarkLoader**

| Category                  | Details                                                               |
| ------------------------- | --------------------------------------------------------------------- |
| **Purpose**               | Wraps LangChain’s `UnstructuredMarkdownLoader` to ingest `.md` files. |
| **Core Responsibilities** | Load Markdown files, convert to Documents, split into chunks.         |
| **Key Attributes**        | `documents`, `file_path`                                              |
| **Public Methods**        | `load(path)` <br> `split(chunk, overlap)`                             |
| **Notes**                 | Ideal for GitHub-based documentation ingestion.                       |



## **🌐 HtmlLoader**

| Category                  | Details                                                                   |
| ------------------------- | ------------------------------------------------------------------------- |
| **Purpose**               | Wraps `UnstructuredHTMLLoader` to ingest HTML files from disk.            |
| **Core Responsibilities** | Extract readable text; strip markup; convert to Documents; chunk content. |
| **Key Attributes**        | `documents`, `file_path`                                                  |
| **Public Methods**        | `load(path)` <br> `split(chunk, overlap)`                                 |
| **Notes**                 | Complements `WebLoader` when HTML is local.                               |



## **🔗 WebLoader**

| Category                  | Details                                                              |
| ------------------------- | -------------------------------------------------------------------- |
| **Purpose**               | Wraps `WebBaseLoader` to load remote webpages.                       |
| **Core Responsibilities** | Fetch URL content; convert to Documents; split into chunks.          |
| **Key Attributes**        | `urls`, `documents`                                                  |
| **Public Methods**        | `load(urls)` <br> `split(chunk, overlap)`                            |
| **Notes**                 | Supports multiple URLs; ideal for ingesting live online information. |



## **📊 ExcelLoader**

| Category                  | Details                                                                         |
| ------------------------- | ------------------------------------------------------------------------------- |
| **Purpose**               | Wraps `UnstructuredExcelLoader` to ingest Excel spreadsheets.                   |
| **Core Responsibilities** | Extract sheet content; convert to Documents; chunk structured spreadsheet data. |
| **Key Attributes**        | `documents`, `file_path`, `loader`                                              |
| **Public Methods**        | `load(path, mode, headers)` <br> `split(chunk, overlap)`                        |
| **Notes**                 | Supports multiple extraction modes (elements, paged).                           |



## ⚙️ **Installation**

Install required dependencies:

```bash
pip install langchain chromadb numpy openai unstructured pypdf python-docx
```

If using OpenAI-compatible models:

```bash
pip install langchain-openai
export OPENAI_API_KEY="your-key"
```



## 🏗️ **Initialize**

```python
from Foo import Fetch

fetch = Fetch(
    db_uri="data/budget.sqlite",
    doc_paths=[
        "docs/PUBLIC_LAW.pdf",
        "docs/OMB_A11.md"
    ],
    model="gpt-4o-mini",
    temperature=0.2
)
```



## 🔍 **SQL Query**

```python
result = fetch.query_sql("List total obligations by fiscal year.")
print(result)
```



## 📚 **Document Retrieval**

```python
result = fetch.query_docs(
    "What does Section 1402 authorize?",
    with_sources=True
)
print(result)
```



## 🤖 **Free-Form Chat**

```python
reply = fetch.query_chat("Explain the difference between BA and OBL.")
print(reply)
```




## 🏗️ **Install & Import**

```python
!pip install langchain chromadb numpy openai unstructured python-docx pypdf

from Foo import Fetch
```


## 🔧 **Initialize Fetch**

```python
fetch = Fetch(
    db_uri="data/budget.sqlite",
    doc_paths=[
        "docs/APPROPRIATIONS_GUIDE.pdf",
        "docs/FINANCIAL_MANAGEMENT_POLICY.md"
    ],
    model="gpt-4o-mini",
    temperature=0.3
)
fetch
```



## 📜 **Run a SQL Query**

```python
fetch.query_sql("Select TAS, SUM(amount) from ledger group by TAS;")
```



## 📁 **Run a Document Retrieval Query**

```python
fetch.query_docs(
    "Summarize the funding limitations described in the guidance.",
    with_sources=True
)
```


## 🧠 **Conversational Query**

```python
fetch.query_chat("Explain SF-132 apportionments at a high level.")
```



## **Dependencies**

| Dependency                    | Purpose                                  | Required       | Notes                                     |
| ----------------------------- | ---------------------------------------- | -------------- | ----------------------------------------- |
| **langchain**                 | Agent framework, tools interface, memory | Yes            | Core orchestration library                |
| **chromadb**                  | Vector store for document embeddings     | Yes            | Stores and retrieves chunks               |
| **openai / langchain-openai** | Chat model support                       | Yes (LLM mode) | Must set `OPENAI_API_KEY`                 |
| **numpy**                     | Embedding/vector operations              | Yes            | Required internally by several components |
| **unstructured**              | Document parsing                         | Recommended    | Supports PDF, HTML, TXT, etc.             |
| **pypdf**                     | PDF ingestion                            | Recommended    | Used by Unstructured backend              |
| **python-docx**               | DOCX ingestion                           | Optional       | Included if DOCX used                     |
| **sqlite3**                   | SQL backend                              | Yes            | Included in Python standard library       |
| **booger**                    | Error and dialog handling                | Yes            | Needed for `Error` and `ErrorDialog`      |
| **tiktoken**                  | Token counting for LLMs                  | Optional       | Recommended for OpenAI models             |



## ⚖️ **License**

MIT License - [found here](https://github.com/is-leeroy-jenkins/Foo/blob/main/LICENSE.txt)




## 📝 **Author**

**Terry D. Eppler**
**Email:** [terryeppler@gmail.com](mailto:terryeppler@gmail.com)


