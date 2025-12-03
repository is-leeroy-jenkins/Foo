###### foo
![](https://github.com/is-leeroy-jenkins/Foo/blob/main/resources/images/foo_project.png)


A small, modular, multi-tool framework to facilitate retrieval augmentation, sql querying, and document loading for agentic workflows that is written in python.
## 📚 Table of contents

* [Features](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-features)
* [Architecture](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#%EF%B8%8F-architecture)
* [Directory structure](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#--structure)
* [Installation](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#%EF%B8%8F-installation)
* [Quick Start](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#%EF%B8%8F-quick-start)
* [Usage Examples](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-usage-examples)
* [Module overview](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-module-overview)
* [Dependencies](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#dependencies)
* [License](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#license)
* [Acknowledgments](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#acknowledgments)

## ✨ Features

* End-to-end pipeline: fetch → scrape → parse → extract → load → write.
* Pluggable components with small, well-documented base classes.
* Explicit guard clauses and type hints for safer runtime behaviour.
* Designed for unit testing and CI/CD integration.
* Minimal, production-ready surface area for integration into existing ETL/ELT systems.

## 🏛️ Architecture

Clear separation of concerns between pipeline stages. `core.py` orchestrates the stages; each stage lives in its own module with a small abstract base class and one or more implementations.

```
Fetchers -> Scrapers -> Parsers -> Extractors -> Loaders -> Writers
```

A `FooPipeline` instance chains stages into a fluent API for simple, readable data flows.

## 🧬  Structure

```
foo/
├── __init__.py
├── core.py           # Pipeline orchestration & public API
├── data.py           # Domain models, DTOs, schema helpers
├── extractors.py     # Field/entity extraction logic
├── fetchers.py       # HTTP / file / API fetchers
├── loaders.py        # Stage/transform/load utilities
├── parsers.py        # CSV / JSON / HTML parsers
├── scrapers.py       # Web scraping / DOM extraction helpers
└── writers.py        # Output writers (files, DB, APIs)
```

## 🛡️ Installation

Requires Python 3.9+.

```bash
git clone https://github.com/your-org/foo.git
cd foo
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

For development:

```bash
.venv/bin/pip install -r dev-requirements.txt
```
## Environments 

Use an isolated virtual environment for development and CI. Foo targets **Python 3.9+**. The steps below create a local `.venv` inside the project, activate it, upgrade packaging tools, and install the project requirements.

> Recommended: keep the virtual environment directory named `.venv` (add it to `.gitignore`) so IDEs like VS Code auto-detect it.

### macOS / Linux / WSL / Git Bash

```bash
# create venv
python3 -m venv .venv

# activate
source .venv/bin/activate

# upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# install runtime requirements
pip install -r requirements.txt

# (optional) install dev/test tools
pip install -r dev-requirements.txt

# (optional) install the package in editable mode for local edits
pip install -e .
```

### Windows — PowerShell

```powershell
# create venv
python -m venv .venv

# activate (PowerShell)
.\.venv\Scripts\Activate.ps1

# If execution policy blocks script runs, run as admin once:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# install requirements
pip install -r requirements.txt
pip install -r dev-requirements.txt  # optional

# optionally install editable package
pip install -e .
```

### Windows — Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Helpful tips

* Add `.venv/` to `.gitignore`:

  ```
  # virtual env
  .venv/
  ```
* Freeze installed packages after changes to requirements:

  ```bash
  pip freeze > requirements.txt
  pip freeze > dev-requirements.txt  # if you added dev packages
  ```
* Use `pip install -e .` during development so local changes are immediately available.
* If you manage multiple Python versions, use `pyenv` (macOS/Linux) or the Windows Python launcher (`py -3.9 -m venv .venv`) to ensure the right interpreter.
* For CI, prefer a reproducible lockfile (`requirements.txt` with pinned versions) or tools like `pip-tools` / `poetry` depending on your workflow.

## 🎚️ Quick start 

Minimal, end-to-end example showing the pipeline chaining pattern.

```python
from foo.core import FooPipeline

pipeline = FooPipeline()

result = (
    pipeline.fetch('https://example.com/data.json')   # HttpFetcher
           .parse('json')                             # JsonParser
           .extract()                                 # Default extractor pipeline
           .load()                                    # In-memory staging
           .write('output.json')                      # JsonWriter to disk
)

print("Pipeline completed. Output at:", result)
```

## 🤖 Usage examples

- These examples show common real-world tasks. Each snippet assumes the default implementations exist in the corresponding modules. Replace components with custom classes if required.

#### 1) Process a CSV file from disk → transform → write to SQLite

```python
from foo.core import FooPipeline
from foo.loaders import CsvLoader
from foo.writers import SqlWriter

pipeline = FooPipeline(loader=CsvLoader(), writer=SqlWriter('sqlite:///data.db'))

pipeline.fetch('/data/incoming/sales.csv', source_type='file') \
        .parse('csv') \
        .extract() \
        .load() \
        .write(table='sales')  # Persists to SQLite table 'sales'
```

- Use-case: scheduled ingestion of vendor CSVs into a local analytics DB.

#### 2) Scrape HTML page → parse → field extraction → write JSON

```python
from foo.core import FooPipeline
from foo.scrapers import BeautifulSoupScraper

pipeline = FooPipeline(scraper=BeautifulSoupScraper())

pipeline.fetch('https://example.com/products') \
        .scrape(selectors={'items': '.product'}) \
        .parse('html') \
        .extract() \
        .write('products.json')
```

- Use-case: lightweight product catalog harvesting for downstream enrichment.

#### 3) Call an internal REST API → normalize fields → post to downstream API

```python
from foo.core import FooPipeline
from foo.fetchers import ApiFetcher
from foo.writers import ApiWriter

api_fetcher = ApiFetcher(base_url='https://api.internal/v1', api_key='SECRET')
api_writer = ApiWriter(endpoint='https://downstream.example/ingest', auth_token='TOKEN')

pipeline = FooPipeline(fetcher=api_fetcher, writer=api_writer)

pipeline.fetch('/reports/daily') \
        .parse('json') \
        .extract() \
        .load() \
        .write()  # posts normalized records to downstream ingestion endpoint
```

- Use-case: internal synchronization between microservices.

#### 4) Batch-mode processing with simple CLI pattern

- Create a tiny CLI entrypoint for batch jobs:

```python
# bin/run_pipeline.py
from foo.core import FooPipeline
from foo.fetchers import FileFetcher

def main(input_path, output_path):
    pipeline = FooPipeline(fetcher=FileFetcher())
    pipeline.fetch(input_path, source_type='file') \
            .parse('csv') \
            .extract() \
            .load() \
            .write(output_path)

if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])
```

- Then schedule with cron or a job runner.

## 📡 Module overview

- A compact reference table for the main modules and their responsibilities.

| Module          |                                        Purpose | Typical classes / functions                               |
| --------------- | ---------------------------------------------: | --------------------------------------------------------- |
| `core.py`       |      Pipeline orchestrator and user-facing API | `FooPipeline`, pipeline stage registration, `throw_if`    |
| `data.py`       | Domain models, DTOs, schema validation helpers | `DataSource`, `DataRecord`, `SchemaValidator`             |
| `fetchers.py`   |       Retrieve raw payloads (HTTP, file, APIs) | `BaseFetcher`, `HttpFetcher`, `FileFetcher`, `ApiFetcher` |
| `scrapers.py`   | Extract semi-structured content from HTML/docs | `BaseScraper`, `BeautifulSoupScraper`, `RegexScraper`     |
| `parsers.py`    |      Turn raw payloads into structured objects | `JsonParser`, `CsvParser`, `HtmlParser`                   |
| `extractors.py` |          Extract and normalize fields/entities | `BaseExtractor`, `RegexExtractor`, `FieldMapper`          |
| `loaders.py`    | Transform, validate, and stage structured data | `CsvLoader`, `JsonLoader`, `SqlLoader`                    |
| `writers.py`    |            Persist outputs to files, DBs, APIs | `JsonWriter`, `CsvWriter`, `SqlWriter`, `ApiWriter`       |




## 🗃️ SQLite

| Category                  | Details                                                                                                                                                                                                                    |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**               | Persistent storage backend for text chunks and embedding vectors using SQLite.                                                                                                                                             |
| **Core Responsibilities** | Create schema; insert embeddings; batch insert; fetch/update/delete embeddings; count and purge records.                                                                                                                   |
| **Key Attributes**        | `db_path`, `connection`, `cursor`                                                                                                                                                                                          |
| **Public Methods**        | `create()` <br> `insert(...)` <br> `insert_many(...)` <br> `fetch_all()` <br> `fetch_by_file(file)` <br> `delete_by_file(file)` <br> `update_embedding_by_id(id, vector)` <br> `count()` <br> `purge_all()` <br> `close()` |
| **Notes**                 | Stores vectors as JSON; supports multi-document ingestion; integrates with Chroma.                                                                                                                                         |


## 🧬 Chroma

| Category                  | Details                                                                                                                                   |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**               | Wrapper around ChromaDB for vector storage and similarity search.                                                                         |
| **Core Responsibilities** | Create/load a collection; add embeddings; perform similarity queries; delete vectors; count, clear, and persist collection.               |
| **Key Attributes**        | `client`, `collection`                                                                                                                    |
| **Public Methods**        | `add(ids, texts, embeddings, metadatas)` <br> `query(text, num, where)` <br> `delete(ids)` <br> `count()` <br> `clear()` <br> `persist()` |
| **Notes**                 | Complements the SQLite backend or can operate standalone.                                                                                 |



## 📦 Loader (Base Class)

| Category                  | Details                                                                                                             |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Purpose**               | Base loader class providing validation, file resolution, and document splitting utilities.                          |
| **Core Responsibilities** | Validate paths; expand glob patterns; split documents into overlapping chunks.                                      |
| **Key Attributes**        | `documents`, `file_path`, `pattern`, `expanded`, `candidates`, `resolved`, `loader`, `chunk_size`, `overlap_amount` |
| **Public Methods**        | `verify_exists(path)` <br> `_resolve_paths(pattern)` <br> `_split_documents(docs, chunk, overlap)`                  |
| **Notes**                 | All other loaders subclass `Loader`.                                                                                |



## 🧾 CsvLoader

| Category                  | Details                                                                      |
| ------------------------- | ---------------------------------------------------------------------------- |
| **Purpose**               | Wraps LangChain’s `CSVLoader` to ingest CSV files.                           |
| **Core Responsibilities** | Parse CSV rows; produce Document objects; split into chunks.                 |
| **Key Attributes**        | `encoding`, `csv_args`, `source_column`                                      |
| **Public Methods**        | `load(path, encoding, csv_args, source_column)` <br> `split(chunk, overlap)` |
| **Notes**                 | Most configurable loader; supports custom parsing options.                   |



## 📄 WordLoader

| Category                  | Details                                                  |
| ------------------------- | -------------------------------------------------------- |
| **Purpose**               | Wraps `Docx2txtLoader` to ingest `.docx` Word documents. |
| **Core Responsibilities** | Extract text; convert to Documents; split into chunks.   |
| **Key Attributes**        | `file_path`, `documents`, `loader`                       |
| **Public Methods**        | `load(path)` <br> `split(chunk, overlap)`                |
| **Notes**                 | Robust for formal documents and reports.                 |



## 📚 PdfLoader

| Category                  | Details                                                         |
| ------------------------- | --------------------------------------------------------------- |
| **Purpose**               | Wraps `PyPDFLoader` to ingest PDF files.                        |
| **Core Responsibilities** | Extract page text; convert pages to Documents; chunk long PDFs. |
| **Key Attributes**        | `file_path`, `documents`, `loader`                              |
| **Public Methods**        | `load(path)` <br> `split(chunk, overlap)`                       |
| **Notes**                 | Ideal for large multi-page documents.                           |



## 📝 MarkLoader

| Category                  | Details                                                               |
| ------------------------- | --------------------------------------------------------------------- |
| **Purpose**               | Wraps LangChain’s `UnstructuredMarkdownLoader` to ingest `.md` files. |
| **Core Responsibilities** | Load Markdown files, convert to Documents, split into chunks.         |
| **Key Attributes**        | `documents`, `file_path`                                              |
| **Public Methods**        | `load(path)` <br> `split(chunk, overlap)`                             |
| **Notes**                 | Ideal for GitHub-based documentation ingestion.                       |



## 🌐 HtmlLoader

| Category                  | Details                                                                   |
| ------------------------- | ------------------------------------------------------------------------- |
| **Purpose**               | Wraps `UnstructuredHTMLLoader` to ingest HTML files from disk.            |
| **Core Responsibilities** | Extract readable text; strip markup; convert to Documents; chunk content. |
| **Key Attributes**        | `documents`, `file_path`                                                  |
| **Public Methods**        | `load(path)` <br> `split(chunk, overlap)`                                 |
| **Notes**                 | Complements `WebLoader` when HTML is local.                               |


## 🔗 WebLoader

| Category                  | Details                                                              |
| ------------------------- | -------------------------------------------------------------------- |
| **Purpose**               | Wraps `WebBaseLoader` to load remote webpages.                       |
| **Core Responsibilities** | Fetch URL content; convert to Documents; split into chunks.          |
| **Key Attributes**        | `urls`, `documents`                                                  |
| **Public Methods**        | `load(urls)` <br> `split(chunk, overlap)`                            |
| **Notes**                 | Supports multiple URLs; ideal for ingesting live online information. |



## 📊 ExcelLoader

| Category                  | Details                                                                         |
| ------------------------- | ------------------------------------------------------------------------------- |
| **Purpose**               | Wraps `UnstructuredExcelLoader` to ingest Excel spreadsheets.                   |
| **Core Responsibilities** | Extract sheet content; convert to Documents; chunk structured spreadsheet data. |
| **Key Attributes**        | `documents`, `file_path`, `loader`                                              |
| **Public Methods**        | `load(path, mode, headers)` <br> `split(chunk, overlap)`                        |
| **Notes**                 | Supports multiple extraction modes (elements, paged).                           |


## 🚀 Fetch (Fetcher)

| Category                  | Details                                                                                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**               | Retrieve raw payloads from diverse sources (HTTP APIs, local files, S3, internal services).                                                                                      |
| **Core Responsibilities** | Perform network or file I/O; normalize source metadata; implement retries/backoff; return raw bytes / text / streams to the pipeline.                                            |
| **Key Attributes**        | `session` (requests.Session) <br> `base_url` <br> `headers` <br> `auth` <br> `timeout` <br> `retries` <br> `backoff_strategy` <br> `source_type` <br> `logger`                   |
| **Public Methods**        | `fetch(source, *, source_type=None, params=None, headers=None)` <br> `stream(source, chunk_size=...)`                                                                            |
| **Internal Methods**      | `_request(url, params, headers)` <br> `_read_file(path)` <br> `_handle_response(resp)` <br> `_normalize_source(source, source_type)`                                             |
| **Notes**                 | Implements `BaseFetcher` interface. Designed to be swap-in extensible (S3Fetcher, ApiFetcher, FileFetcher). Supports streaming, idempotent fetches, and pluggable auth handlers. |


## 🔬 Scrape (Scraper)

| Category                  | Details                                                                                                                                                                    |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**               | Extract semi-structured information from HTML or document sources (DOM traversal, selector-based extraction, lightweight rendering support).                               |
| **Core Responsibilities** | Load HTML into a parser (BeautifulSoup / lxml); run selector queries; normalize nodes into serializable records; optionally follow pagination links or embedded resources. |
| **Key Attributes**        | `parser` (e.g., `'lxml'`) <br> `selectors` (dict) <br> `timeout` <br> `max_pages` <br> `user_agent` <br> `render_js` (bool) <br> `logger`                                  |
| **Public Methods**        | `scrape(content, selectors=None)` <br> `paginate(start_url, selectors, max_pages=...)`                                                                                     |
| **Internal Methods**      | `_select_nodes(soup, selector)` <br> `_node_to_record(node)` <br> `_follow_next(link)`                                                                                     |
| **Notes**                 | Meant for light-to-moderate scraping tasks (not a headless-browsing crawler). Site-specific scrapers inherit `BaseScraper` to encapsulate DOM rules and rate limits.       |



## 🔍 Parse (Parser)

| Category                  | Details                                                                                                                                                                  |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Purpose**               | Convert raw payloads (bytes / text / streams) into structured representations: JSON objects, tabular rows, DOM trees, or typed records.                                  |
| **Core Responsibilities** | Detect format when necessary; parse JSON/CSV/HTML/text; optionally coerce types; provide consistent structured output for extraction stage.                              |
| **Key Attributes**        | `format` (json/csv/html/text) <br> `encoding` <br> `csv_dialect` <br> `schema` (optional) <br> `chunk_size` <br> `logger`                                                |
| **Public Methods**        | `parse(raw, format=None)` <br> `iter_parse(stream, format=None)`                                                                                                         |
| **Internal Methods**      | `_parse_json(text)` <br> `_parse_csv(text)` <br> `_parse_html(text)` <br> `_coerce_types(record)`                                                                        |
| **Notes**                 | Parsers are lightweight and deterministic. Prefer returning iterators for large payloads. Parsers should surface clear parse errors (line/offset) for upstream handling. |



## 🎛️ Extract (Extractor)

| Category                  | Details                                                                                                                                                                |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**               | Normalize and extract fields/entities from parsed structures (apply field mappings, regex/NER, value normalization, enrichment hooks).                                 |
| **Core Responsibilities** | Apply mapping rules; run regex or rule-based extractors; validate and clean values; produce canonical records ready for loading.                                       |
| **Key Attributes**        | `rules` (mapping / extraction rules) <br> `stopwords` / `vocab` <br> `normalizers` <br> `validator` <br> `logger`                                                      |
| **Public Methods**        | `extract(parsed_iterable)` <br> `extract_one(parsed)` <br> `bulk_extract(records)`                                                                                     |
| **Internal Methods**      | `_apply_rules(record)` <br> `_run_regex(field, pattern)` <br> `_normalize(value)` <br> `_validate(record)`                                                             |
| **Notes**                 | Designed for domain-specific transformations. A `FieldMapper`/`RegexExtractor` implements the base contract. Keep extractors pure where possible to ease unit testing. |



## 🏗️ Load (Loader)

| Category                  | Details                                                                                                                                                                                          |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Purpose**               | Stage and transform canonical records into an in-memory table or an intermediate persistence layer (DataFrame, temp table, batch buffer) preparing for final write.                              |
| **Core Responsibilities** | Validate against schema; perform light transformations (type casts, dedupe); batch records for efficient writes; provide transactional hooks.                                                    |
| **Key Attributes**        | `schema` <br> `batch_size` <br> `buffer` <br> `engine` / `connection_string` <br> `transformers` <br> `logger`                                                                                   |
| **Public Methods**        | `load(records)` <br> `flush()` <br> `upsert(records, table=...)`                                                                                                                                 |
| **Internal Methods**      | `_validate_record(record)` <br> `_transform_record(record)` <br> `_write_batch(batch)` <br> `_begin_tx()` / `_commit_tx()`                                                                       |
| **Notes**                 | `Loaders` are the boundary between ephemeral processing and durable persistence. Implementations include `CsvLoader`, `JsonLoader`, and `SqlLoader` — all support batch/transactional semantics. |



## 📝 Write (Writer)

| Category                  | Details                                                                                                                                                               |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**               | Persist staged data to final sinks: files (JSON/CSV), relational DBs, or remote APIs; manage format-specific serialization and final consistency checks.              |
| **Core Responsibilities** | Serialize records; open/close resources; ensure atomic writes where possible; expose idempotent write modes; return destination metadata or URLs.                     |
| **Key Attributes**        | `target` (path / table / endpoint) <br> `mode` (append/replace/upsert) <br> `encoding` <br> `engine` / `connection` <br> `timeout` <br> `logger`                      |
| **Public Methods**        | `write(records, *, destination=None)` <br> `commit()` <br> `abort()`                                                                                                  |
| **Internal Methods**      | `_serialize(record)` <br> `_open_sink()` <br> `_close_sink()` <br> `_ensure_atomic(tmp, final)`                                                                       |
| **Notes**                 | Writers should support safe write patterns (write-to-temp + rename) and idempotency tokens for HTTP sinks. `ApiWriter` must respect backpressure and retry semantics. |



## 🧠 Core (Pipeline)

| Category                  | Details                                                                                                                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Purpose**               | High-level fluent orchestrator that composes fetchers, scrapers, parsers, extractors, loaders, and writers into a single pipeline.                                                                     |
| **Core Responsibilities** | Hold component references; provide chaining API (`fetch().parse().extract().load().write()`); validate stage wiring; propagate context/config; handle lifecycle (init/close).                          |
| **Key Attributes**        | `fetcher`, `scraper`, `parser`, `extractor`, `loader`, `writer` <br> `config` <br> `logger` <br> `metrics` <br> `error_policy`                                                                         |
| **Public Methods**        | `fetch(source, **kwargs)` <br> `scrape(**kwargs)` <br> `parse(format=None)` <br> `extract()` <br> `load()` <br> `write(destination=None)` <br> `run()`                                                 |
| **Internal Methods**      | `_resolve_component(name)` <br> `_ensure_stage(stage_name)` <br> `_capture_metrics(stage, duration, status)` <br> `_handle_error(exc, stage)`                                                          |
| **Notes**                 | `FooPipeline` is intentionally thin — it delegates behavior to components but enforces the fluent contract and common policies (time outs, logging, retry). Ideal place to implement global telemetry. |




# Dependencies

Core runtime dependencies expected by the repository. Pin exact versions in your `requirements.txt` for reproducible installs.

| Package           | Minimum version | Purpose                                       |
| ----------------- | --------------: | --------------------------------------------- |
| `python`          |           `3.9` | Language runtime                              |
| `requests`        |        `>=2.28` | HTTP fetching (HttpFetcher)                   |
| `beautifulsoup4`  |        `>=4.12` | HTML scraping/parsing                         |
| `lxml`            |         `>=4.9` | Fast HTML/XML parsing for BeautifulSoup       |
| `pandas`          |         `>=1.5` | CSV/DF processing (optional, used by loaders) |
| `sqlalchemy`      |         `>=1.4` | DB writers/loaders (SqlWriter/SqlLoader)      |
| `psycopg2-binary` |         `>=2.9` | Postgres driver (if using Postgres)           |
| `python-dotenv`   |        `>=0.21` | Manage environment variables (optional)       |

**Development / testing (suggested)**

| Package      | Minimum version | Purpose              |
| ------------ | --------------: | -------------------- |
| `pytest`     |         `>=7.2` | Unit tests           |
| `pytest-cov` |         `>=4.0` | Test coverage        |
| `black`      |        `>=24.3` | Code formatting      |
| `mypy`       |         `>=1.3` | Static typing checks |
| `pre-commit` |         `>=3.4` | Pre-commit hooks     |

# License

MIT License. See [LICENSE](https://github.com/is-leeroy-jenkins/Foo/blob/main/LICENSE.txt) for full text.

# Acknowledgments

Foo applies pragmatic, traditional data-engineering patterns to create a compact, maintainable toolkit. The design reflects practices used in production ETL/ELT systems: small interfaces, explicit validation, and a predictable pipeline flow that reduces accidental coupling during maintenance. Contributions welcome via issues and pull requests.




