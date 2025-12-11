###### foo
[]()

*A Modular Python Framework for Retrieval-Augmented Pipelines and Agentic Workflows*

<p align="left">
  <a href="https://github.com/is-leeroy-jenkins/Foo/blob/main/LICENSE"><img src="https://img.shields.io/github/license/is-leeroy-jenkins/Foo?logo=opensourceinitiative&label=License" alt="License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python" alt="Python 3.9+"></a>
</p>

---

## 📚 Table of Contents

* [Features](#features)
* [Architecture](#architecture)
* [Directory Structure](#directory-structure)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Usage Examples](#usage-examples)
* [Loaders](#loaders)
* [Fetchers](#fetchers)
* [Scrapers](#scrapers)
* [Dependencies](#dependencies)
* [Technical Notes](#technical-notes)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## ✨ Features

* Modular, pluggable pipeline for document, web, and data retrieval and processing.
* Robust, extensible loaders and fetchers for all common document and web data formats.
* Clean separation of fetch, scrape, load, convert, and write stages.
* Integrates with OpenAI, LangChain, ChromaDB, and advanced document stores.
* Strong type safety and error handling.
* Simple, testable, and extensible codebase.

---

## 🏛️ Architecture

```
Fetcher → Scraper → Loader → Converter → Writer
```

Each stage is a pluggable, testable class. The core orchestrator is the `Fetch` pipeline.

---

## 🗂️ Directory Structure

```
foo/
├── __init__.py
├── core.py
├── data.py
├── fetchers.py
├── loaders.py
├── scrapers.py
├── config.py
├── requirements.txt
```

---

## 🛡️ Installation

```bash
git clone https://github.com/is-leeroy-jenkins/Foo.git
cd Foo
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

---

## 🚀 Quick Start

```python
from foo.core import Fetch
fetcher = Fetch(model='gpt-4o', db_uri='sqlite:///foo.sqlite', doc_paths=['docs/*.pdf'])
response = fetcher.query_docs("Summarize the uploaded PDFs.")
print(response)
```

---

## 🔍 Usage Examples

**Fetch Web Page Paragraphs:**

```python
from foo.scrapers import WebExtractor
extractor = WebExtractor()
paragraphs = extractor.scrape_paragraphs("https://example.com")
print(paragraphs)
```

**Load and Chunk a PDF:**

```python
from foo.loaders import PdfLoader
loader = PdfLoader()
doc = loader.load('docs/report.pdf')
chunks = loader.split(doc)
print(chunks)
```

---

## 📄 Loaders

### 🛰️ Loader

Abstract base class for all loaders; provides document loading and splitting interface.

* `load(path)` – Loads the document from the specified path.
* `split(doc, chunk=1000, overlap=100)` – Splits a loaded document into overlapping text chunks.

---

### 🛰️ CsvLoader

Loads and splits CSV files for tabular data ingestion.

* `load(path)` – Loads and parses a CSV file.
* `split(doc, chunk=1000, overlap=100)` – Splits CSV content for batch processing.

---

### 🛰️ PdfLoader

Loads PDF files, supporting robust text extraction and chunking.

* `load(path)` – Loads and extracts text from a PDF document.
* `split(doc, chunk=1000, overlap=100)` – Splits PDF text into manageable chunks.

---

### 🛰️ DocxLoader

Loads and extracts content from DOCX (Word) documents.

* `load(path)` – Loads and parses a DOCX file.
* `split(doc, chunk=1000, overlap=100)` – Splits DOCX text for analysis.

---

### 🛰️ HtmlLoader

Loads and parses local HTML documents.

* `load(path)` – Loads HTML content from a file.
* `split(doc, chunk=1000, overlap=100)` – Splits HTML body text into chunks.

---

### 🛰️ PptxLoader

Loads and extracts text from PowerPoint (`.pptx`) files.

* `load(path)` – Loads slide contents from a PowerPoint file.
* `split(doc, chunk=1000, overlap=100)` – Splits slide text for downstream use.

---

### 🛰️ ExcelLoader

Loads and processes Excel spreadsheets (XLS/XLSX).

* `load(path)` – Loads and reads an Excel file.
* `split(doc, chunk=1000, overlap=100)` – Splits spreadsheet content for batch processing.

---

### 🛰️ TextLoader

Loads plain text files, supporting chunked analysis.

* `load(path)` – Loads the content of a text file.
* `split(doc, chunk=1000, overlap=100)` – Splits text file content into chunks.

---

### 🛰️ JsonLoader

Loads structured data from JSON files.

* `load(path)` – Loads and parses JSON data.
* `split(doc, chunk=1000, overlap=100)` – Splits JSON-encoded text as appropriate.

---

### 🛰️ MarkdownLoader

Loads and splits Markdown (`.md`) documents.

* `load(path)` – Loads a Markdown file’s content.
* `split(doc, chunk=1000, overlap=100)` – Splits Markdown into logical text chunks.

---

### 🛰️ XmlLoader

Loads and parses XML documents.

* `load(path)` – Loads and parses XML content.
* `split(doc, chunk=1000, overlap=100)` – Splits XML text nodes for further use.

---

### 🛰️ ImageLoader

Loads and processes image files for downstream tasks (e.g., OCR, embeddings).

* `load(path)` – Loads an image file.
* `split(doc, chunk=1000, overlap=100)` – Optionally splits or processes image regions.

---

### 🛰️ YouTubeLoader

Loads YouTube video transcripts and metadata.

* `load(path)` – Retrieves transcript/caption text for a given video ID or URL.
* `split(doc, chunk=1000, overlap=100)` – Splits transcript into chunks.

---

### 🛰️ UnstructuredLoader

Flexible loader for mixed-format or “messy” documents.

* `load(path)` – Loads and attempts to parse various unstructured document formats.
* `split(doc, chunk=1000, overlap=100)` – Splits extracted text for processing.

---

## 🛰️ Fetchers

### 🛰️ Fetcher

Abstract base class for all fetchers, defining the core fetch interface.

* `fetch(url, **kwargs)` – Performs a data retrieval request to a specified endpoint.

---

### 🛰️ WebFetcher

Fetches HTML content using `requests` and provides rich methods for extracting text and elements from web pages.

* `fetch(url, time=10)` – Performs an HTTP GET and returns a structured Result.
* `html_to_text(html)` – Converts raw HTML to compact plain text.
* `scrape_paragraphs(uri)` – Extracts all `<p>` text blocks from a page.
* `scrape_lists(uri)` – Extracts all `<li>` text from lists.
* `scrape_tables(uri)` – Flattens all table cell contents.
* `scrape_articles(uri)` – Extracts content from `<article>` tags.
* `scrape_headings(uri)` – Extracts headings (`<h1>`–`<h6>`).
* `scrape_divisions(uri)` – Extracts text from `<div>` elements.
* `scrape_sections(uri)` – Extracts text from `<section>` elements.
* `scrape_blockquotes(uri)` – Extracts `<blockquote>` text.
* `scrape_hyperlinks(uri)` – Extracts all hyperlinks (`<a href>`).
* `scrape_images(uri)` – Extracts image sources (`<img src>`).
* `create_schema(function, tool, description, parameters, required)` – Dynamically builds an OpenAI Tool API schema for function calling.

---

### 🛰️ WebCrawler

JavaScript-capable crawler using `crawl4ai` or Playwright, for dynamic content.

* `fetch(url, depth=1, **kwargs)` – Recursively crawls and fetches HTML from linked pages.

---

### 🛰️ StarMap

Fetches celestial map images using coordinates from StarMap.org.

* `fetch_by_coordinates(ra, dec)` – Generates a star map based on right ascension and declination.

---

### 🛰️ ArxivFetcher

Loads arXiv papers via the `ArxivRetriever`, returning results as document objects.

* `fetch(query, **kwargs)` – Retrieves papers matching the specified query.

---

### 🛰️ GoogleDriveFetcher

Loads files from Google Drive using LangChain retrievers.

* `fetch(query, **kwargs)` – Retrieves documents or file metadata from Google Drive.

---

### 🛰️ WikipediaFetcher

Retrieves Wikipedia articles with full metadata support.

* `fetch(query, **kwargs)` – Retrieves article text and metadata for a search term.

---

### 🛰️ NewsFetcher

Fetches news articles using Thenewsapi.com.

* `fetch(query, **kwargs)` – Retrieves news articles based on keyword and category.

---

### 🛰️ GoogleSearch

Uses Google Custom Search API for web search.

* `fetch(query, **kwargs)` – Executes a web search and returns the top results.

---

### 🛰️ GoogleMaps

Integrates with Google Maps for geocoding, address validation, and directions.

* `geocode(address)` – Returns geocoordinates for a given address.
* `directions(origin, destination)` – Retrieves navigation routes.
* `validate(address)` – Validates a given address.

---

### 🛰️ GoogleWeather

Retrieves weather data using Google Weather API.

* `fetch(location)` – Returns weather info for a location.
* `resolve_location(query)` – Performs geocoding to determine a location from a query.

---

### 🛰️ NavalObservatory

Fetches astronomical and time data from the U.S. Naval Observatory.

* `fetch_julian_date()` – Returns current Julian date.
* `fetch_sidereal_time()` – Returns local sidereal time.

---

### 🛰️ SatelliteCenter

Interfaces with NASA SSCWeb for satellite and ground station data.

* `fetch_orbits(satellite, start, end)` – Retrieves orbital tracks for a satellite.
* `fetch_ground_stations()` – Lists ground station metadata.

---

### 🛰️ EarthObservatory

Connects to NASA EONET for global natural event data.

* `fetch_events(count)` – Returns recent global events (fires, storms, volcanoes, etc).
* `fetch_categories()` – Returns the event categories.

---

### 🛰️ GlobalImagery

Pulls satellite imagery from NASA GIBS WMS.

* `fetch_imagery(bbox, date)` – Returns satellite map tiles or images.

---

### 🛰️ NearbyObjects

Retrieves near-Earth object (NEO) and fireball data from JPL’s CNEOS/SSD APIs.

* `fetch_neos(start, end)` – Returns near-Earth object data for date range.
* `fetch_fireballs(start, end)` – Returns fireball events for date range.

---

## 🛰️ Scrapers

### 🛰️ Extractor

Abstract base for HTML → plain-text extraction.

* `raw_html` – Raw HTML content to be extracted.
* `extract` – Extraction method to convert HTML to text.

---

### 🛰️ WebExtractor

Concrete, synchronous extractor using `requests` and BeautifulSoup for HTML→text extraction.

* `fetch(url, time=10)` – Performs HTTP GET and returns a canonicalized Result.
* `html_to_text(html)` – Converts HTML to compact plain text (scripts/styles removed).
* `scrape_paragraphs(uri)` – Extracts all `<p>` blocks from a page.
* `scrape_lists(uri)` – Extracts `<li>` text from lists.
* `scrape_tables(uri)` – Extracts cell contents from all `<table>` structures.
* `scrape_articles(uri)` – Extracts consolidated text from `<article>` elements.
* `scrape_headings(uri)` – Extracts headings `<h1>`–`<h6>`.
* `scrape_divisions(uri)` – Extracts cleaned text from `<div>` blocks.
* `scrape_sections(uri)` – Extracts readable text from `<section>` elements.
* `scrape_blockquotes(uri)` – Extracts text from `<blockquote>` elements.
* `scrape_hyperlinks(uri)` – Extracts all hyperlink hrefs.
* `scrape_images(uri)` – Extracts image references from `<img src="...">`.
* `create_schema(function, tool, description, parameters, required)` – Builds dynamic OpenAI Tool API schema.

---

## 📦 Dependencies

| Package           | Purpose/Description          | Link                                                    |
| ----------------- | ---------------------------- | ------------------------------------------------------- |
| beautifulsoup4    | HTML/XML parsing             | [PyPI](https://pypi.org/project/beautifulsoup4/)        |
| requests          | HTTP client                  | [PyPI](https://pypi.org/project/requests/)              |
| playwright        | Headless browser automation  | [PyPI](https://pypi.org/project/playwright/)            |
| langchain         | LLM & RAG framework          | [LangChain](https://python.langchain.com/)              |
| chromadb          | Vector DB for embeddings     | [PyPI](https://pypi.org/project/chromadb/)              |
| pandas            | Data analysis                | [PyPI](https://pypi.org/project/pandas/)                |
| numpy             | Numeric computing            | [PyPI](https://pypi.org/project/numpy/)                 |
| matplotlib        | Visualization                | [PyPI](https://pypi.org/project/matplotlib/)            |
| owslib            | Geospatial Web Services      | [PyPI](https://pypi.org/project/OWSLib/)                |
| astroquery        | Astronomy data               | [PyPI](https://pypi.org/project/astroquery/)            |
| unstructured      | Document parsing             | [Docs](https://unstructured-io.github.io/unstructured/) |
| pytube            | YouTube video download       | [PyPI](https://pypi.org/project/pytube/)                |
| docx2txt          | DOCX text extraction         | [PyPI](https://pypi.org/project/docx2txt/)              |
| pillow            | Image processing             | [PyPI](https://pypi.org/project/Pillow/)                |
| python-pptx       | PowerPoint processing        | [PyPI](https://pypi.org/project/python-pptx/)           |
| PyMuPDF (fitz)    | PDF parsing                  | [PyPI](https://pypi.org/project/PyMuPDF/)               |
| scikit-learn      | Machine learning             | [PyPI](https://pypi.org/project/scikit-learn/)          |
| tiktoken          | OpenAI tokenization          | [PyPI](https://pypi.org/project/tiktoken/)              |
| pyyaml            | YAML file parsing            | [PyPI](https://pypi.org/project/PyYAML/)                |
| tabulate          | Tabular text/markdown output | [PyPI](https://pypi.org/project/tabulate/)              |
| python-dotenv     | Manage .env files            | [PyPI](https://pypi.org/project/python-dotenv/)         |
| typing_extensions | Type hinting support         | [PyPI](https://pypi.org/project/typing-extensions/)     |

---

## ⚙️ Technical Notes

* Pluggable, modular pipeline—add new fetchers/loaders by subclassing.
* Type-safety and error handling by design.
* Compatible with CI/CD and production data environments.

---

## 📝 License

MIT License
Copyright © 2022–2025 Terry D. Eppler

---

## 🙏 Acknowledgments

* Project lead: Terry D. Eppler ([terryeppler@gmail.com](mailto:terryeppler@gmail.com))
* Inspired by open-source Python, ML, and LLM communities.

