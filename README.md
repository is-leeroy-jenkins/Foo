###### foo
![](https://github.com/is-leeroy-jenkins/Foo/blob/main/resources/images/foo_project.png)


Foo is designed to give users explicit, hands-on control over how data is retrieved, loaded, extracted, queried, and explored from a variety of sources. Through a tab-based interface, users can load content from URLs, raw text, and local files; selectively scrape structured elements from web pages; fetch information from external services; interact with conversational analysis tools; and perform location-based workflows such as geocoding and weather lookups. 

Foo is modular by design, allowing individual capabilities—loaders, scrapers, fetchers, mappers, and databases to operate independently while remaining composable within a single, cohesive workspace.

## 🎥 Demo

![](https://github.com/is-leeroy-jenkins/Foo/blob/main/resources/images/foo-demo.gif)


### 🕸️ Web Application

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://fooo-py.streamlit.app/)


## 🔑 API KEY SETUP

- [OpenAI](https://github.com/is-leeroy-jenkins/Buddy/blob/main/resources/setup/openai.md)
- [Geolocation](https://developers.google.com/maps/documentation/geolocation/get-api-key)
- [Google Maps](https://developers.google.com/maps/documentation/embed/get-api-key)
- [Gov Data](https://www.govinfo.gov/api-signup)
- [NASA](https://api.nasa.gov/)
- [The News API](https://www.thenewsapi.com/register)
- 

## 📚 Table of Contents

* [Features](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-features)
* [Architecture](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#%EF%B8%8F-installation)
* [Directory Structure](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-directory-structure)
* [Installation](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-installation)
* [Quick Start](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-quick-start)
* [Usage Examples](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-example-usage)
* [Loaders](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-loader)
* [Fetchers](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-fetcher)
* [Scrapers](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-scrapers)
* [Dependencies](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-dependencies)
* [Technical Notes](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#%EF%B8%8F-technical-notes)
* [License](https://github.com/is-leeroy-jenkins/Foo/blob/main/LICENSE.txt)

## ✨ Features

* Modular, pluggable pipeline for document, web, and data retrieval and processing.
* Robust, extensible loaders and fetchers for all common document and web data formats.
* Powerful HTML/text scraping and cleaning.
* Integrates with OpenAI, LangChain, ChromaDB, and advanced document stores.
* Strong type safety and error handling.
* Simple, extensible codebase.

## 🏛️ Architecture

```
📄 Fetcher → 🕸️ Scraper → 📤 Loader  
```

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

## 🛡️ Installation

```bash

git clone https://github.com/is-leeroy-jenkins/Foo.git
cd Foo
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```


---

## ▶️ Running the Streamlit App

```bash

streamlit run app.py
```

The application runs locally and does not require a database or background services.

---

## 🚀 Quick Start

```python
from foo.core import Fetch
fetcher = Fetch(model='gpt-4o', db_uri='sqlite:///foo.sqlite', doc_paths=['docs/*.pdf'])
response = fetcher.query_docs("Summarize the uploaded PDFs.")
print(response)
```



## 🔍 Example Usage 

**Scrape Web Page Paragraphs:**

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



## 🗂️ Loader

*Abstract base class for all loaders; provides document loading and splitting interface.*

* `load(path)` – Loads the document from the specified path.

* `split(doc, chunk=1000, overlap=100)` – Splits a loaded document into overlapping text chunks.

## 🧾 CsvLoader

*Loads and splits CSV files for tabular data ingestion.*

* `load(path)` – Loads and parses a CSV file.

* `split(doc, chunk=1000, overlap=100)` – Splits CSV content for batch processing.

## 📄 PdfLoader

*Loads PDF files, supporting robust text extraction and chunking.*

* `load(path)` – Loads and extracts text from a PDF document.

* `split(doc, chunk=1000, overlap=100)` – Splits PDF text into manageable chunks.


## 📝 DocxLoader

*Loads and extracts content from DOCX (Word) documents.*

* `load(path)` – Loads and parses a DOCX file.

* `split(doc, chunk=1000, overlap=100)` – Splits DOCX text for analysis.


## 🌐 HtmlLoader

*Loads and parses local HTML documents.*

* `load(path)` – Loads HTML content from a file.

* `split(doc, chunk=1000, overlap=100)` – Splits HTML body text into chunks.


## 📊 PptxLoader

*Loads and extracts text from PowerPoint (`.pptx`) files.*

* `load(path)` – Loads slide contents from a PowerPoint file.

* `split(doc, chunk=1000, overlap=100)` – Splits slide text for downstream use.


## 📈 ExcelLoader

*Loads and processes Excel spreadsheets (XLS/XLSX).*

* `load(path)` – Loads and reads an Excel file.

* `split(doc, chunk=1000, overlap=100)` – Splits spreadsheet content for batch processing.


## 📜 TextLoader

*Loads plain text files, supporting chunked analysis.*

* `load(path)` – Loads the content of a text file.

* `split(doc, chunk=1000, overlap=100)` – Splits text file content into chunks.


## 🗃️ JsonLoader

*Loads structured data from JSON files.*

* `load(path)` – Loads and parses JSON data.

* `split(doc, chunk=1000, overlap=100)` – Splits JSON-encoded text as appropriate.


## 📝 MarkdownLoader

*Loads and splits Markdown (`.md`) documents.*

* `load(path)` – Loads a Markdown file’s content.

* `split(doc, chunk=1000, overlap=100)` – Splits Markdown into logical text chunks.



## 🗂️ XmlLoader

*Loads and parses XML documents.*

* `load(path)` – Loads and parses XML content.

* `split(doc, chunk=1000, overlap=100)` – Splits XML text nodes for further use.

## 🖼️ ImageLoader

*Loads and processes image files for downstream tasks (e.g., OCR, embeddings).*

* `load(path)` – Loads an image file.

* `split(doc, chunk=1000, overlap=100)` – Optionally splits or processes image regions.


## 📺 YouTubeLoader

*Loads YouTube video transcripts and metadata.*

* `load(path)` – Retrieves transcript/caption text for a given video ID or URL.

* `split(doc, chunk=1000, overlap=100)` – Splits transcript into chunks.


## 💾 UnstructuredLoader

*Flexible loader for mixed-format or “messy” documents.*

* `load(path)` – Loads and attempts to parse various unstructured document formats

* `split(doc, chunk=1000, overlap=100)` – Splits extracted text for processing.


## 🤖 Fetcher

*Abstract base class for all fetchers, defining the core fetch interface.*

* `fetch(url, **kwargs)` – Performs a data retrieval request to a specified endpoint.


## 🌍 WebFetcher

*Fetches HTML content using `requests` and provides rich methods for extracting text and elements from web pages.*

* `fetch(url, time=10)` – Performs an HTTP GET and returns a structured Result.

* `html_to_text(html)` – Converts raw HTML to compact plain text.


## 🕸️ WebCrawler

*JavaScript-capable crawler using `crawl4ai` or Playwright, for dynamic content.*

* `fetch(url, depth=1, **kwargs)` – Recursively crawls and fetches HTML from linked pages.


## 🌌 StarMap

*Fetches celestial map images using coordinates from StarMap.org.*

* `fetch_by_coordinates(ra, dec)` – Generates a star map based on right ascension and declination.


## 📚 ArxivFetcher

*Loads arXiv papers via the `ArxivRetriever`, returning results as document objects.*

* `fetch(query, **kwargs)` – Retrieves papers matching the specified query.


## 🗂️ GoogleDriveFetcher

*Loads files from Google Drive using LangChain retrievers.*

* `fetch(query, **kwargs)` – Retrieves documents or file metadata from Google Drive.


## 📖 WikipediaFetcher

*Retrieves Wikipedia articles with full metadata support.*

* `fetch(query, **kwargs)` – Retrieves article text and metadata for a search term.



## 📰 NewsFetcher

*Fetches news articles using Thenewsapi.com.*

* `fetch(query, **kwargs)` – Retrieves news articles based on keyword and category.


## 🔎 GoogleSearch

*Uses Google Custom Search API for web search.*

* `fetch(query, **kwargs)` – Executes a web search and returns the top results.


## 🗺️ GoogleMaps

*Integrates with Google Maps for geocoding, address validation, and directions.*

* `geocode(address)` – Returns geocoordinates for a given address.

* `directions(origin, destination)` – Retrieves navigation routes.

* `validate(address)` – Validates a given address.


## ☁️ GoogleWeather

*Retrieves weather data using Google Weather API.*

* `fetch(location)` – Returns weather info for a location.

* `resolve_location(query)` – Performs geocoding to determine a location from a query.


## 🕰️ NavalObservatory

*Fetches astronomical and time data from the U.S. Naval Observatory.*

* `fetch_julian_date()` – Returns current Julian date.

* `fetch_sidereal_time()` – Returns local sidereal time.


## 🛰️ SatelliteCenter

*Interfaces with NASA SSCWeb for satellite and ground station data.*

* `fetch_orbits(satellite, start, end)` – Retrieves orbital tracks for a satellite.

* `fetch_ground_stations()` – Lists ground station metadata.


## 🌋 EarthObservatory

*Connects to NASA EONET for global natural event data.*

* `fetch_events(count)` – Returns recent global events (fires, storms, volcanoes, etc).

* `fetch_categories()` – Returns the event categories.


## 🗾 GlobalImagery

*Pulls satellite imagery from NASA GIBS WMS.*

* `fetch_imagery(bbox, date)` – Returns satellite map tiles or images.


## ☄️ NearbyObjects

*Retrieves near-Earth object (NEO) and fireball data from JPL’s CNEOS/SSD APIs.*

* `fetch_neos(start, end)` – Returns near-Earth object data for date range.

* `fetch_fireballs(start, end)` – Returns fireball events for date range.


## 🕸️ Scrapers

## 🧩 Extractor

*Abstract base for HTML → plain-text extraction.*

* `__init__(self, raw_html: str = '')` — Initialize with optional raw HTML to extract.

* `extract(self)` — Abstract method for extracting readable text from HTML. Must be implemented by subclasses.



## 🕷️ WebExtractor

*Concrete, synchronous extractor using `requests` and BeautifulSoup for HTML → text extraction.*

* `__init__(self, raw_html: str = '')` — Initialize the extractor, optionally with raw HTML.

* `fetch(self, url: str, time: int = 10)` — Performs an HTTP GET, returns a canonicalized Result.

* `html_to_text(self, html: str)` — Converts HTML to plain, readable text, removing scripts/styles.

* `scrape_paragraphs(self, uri: str)` — Extracts all `<p>` blocks from the page at the given URI.

* `scrape_lists(self, uri: str)` — Extracts all `<li>` items from lists.

* `scrape_tables(self, uri: str)` — Extracts and flattens all table cell contents.

* `scrape_articles(self, uri: str)` — Extracts text from `<article>` elements.

* `scrape_headings(self, uri: str)` — Extracts all headings (`<h1>`–`<h6>`).

* `scrape_divisions(self, uri: str)` — Extracts text from `<div>` elements.

* `scrape_sections(self, uri: str)` — Extracts text from `<section>` elements.

* `scrape_blockquotes(self, uri: str)` — Extracts text from `<blockquote>` elements.

* `scrape_hyperlinks(self, uri: str)` — Extracts all hyperlinks (from `<a href>`).

* `scrape_images(self, uri: str)` — Extracts image sources (from `<img src>`).

* `create_schema(self, function, tool, description, parameters, required)` — Builds an OpenAI Tool API schema dynamically for a function.



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



## ⚙️ Technical Notes

* Pluggable, modular pipeline—add new fetchers/loaders by subclassing.

* Type-safety and error handling by design.

* Compatible with CI/CD and production data environments.



## 📝 License

- MIT License [here](https://github.com/is-leeroy-jenkins/Foo/blob/main/LICENSE.txt)

- Copyright © 2022–2025 Terry D. Eppler



## 🙏 Acknowledgments

* Project lead: Terry D. Eppler ([terryeppler@gmail.com](mailto:terryeppler@gmail.com))

* Inspired by open-source Python, ML, and LLM communities.
