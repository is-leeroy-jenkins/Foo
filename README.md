##### _
![](https://github.com/is-leeroy-jenkins/Foo/blob/main/resources/images/foo_project.png)


*A Modular Python Framework for Retrieval-Augmented Pipelines and Agentic Workflows*

<p align="left">
  <a href="https://github.com/is-leeroy-jenkins/Foo/blob/main/LICENSE"><img src="https://img.shields.io/github/license/is-leeroy-jenkins/Foo?logo=opensourceinitiative&label=License" alt="License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python" alt="Python 3.9+"></a>
</p>



## 📚 Table of Contents

* [Features](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-features)
* [Architecture](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#%EF%B8%8F-architecture)
* [Directory Structure](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#%EF%B8%8F-directory-structure)
* [Installation](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#%EF%B8%8F-installation)
* [Quick Start](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-quick-start)
* [Usage Examples](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-usage-examples)
* [Fetchers](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#%EF%B8%8F-fetchers)
* [Loaders](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-loaders)
* [Dependencies](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-dependencies)
* [Module/Class Summary](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#%EF%B8%8F-moduleclass-summary)
* [Technical Notes](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#%EF%B8%8F-technical-notes)
* [License](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-license)
* [Acknowledgments](https://github.com/is-leeroy-jenkins/Foo?tab=readme-ov-file#-acknowledgments)



## ✨ Features

* Modular, pluggable pipeline for document, web, and data retrieval and processing.
* Production-ready fetchers for Federal APIs, science sources, Google, LLM, and more.
* Robust, extensible loaders for all common document formats and storage providers.
* Integrates with OpenAI, LangChain, ChromaDB, and advanced document stores.
* Strong code contracts (type hints, guard clauses, error handling).
* Clean separation of fetch, extract, load, convert, and write stages.



## 🏛️ Architecture

```
Fetcher → Extractor → Loader → Converter → Writer
```

Each stage is a pluggable, testable class. All orchestration is handled by the `Fetch` pipeline.



## 🗂️ Directory Structure

```
foo/
├── __init__.py
├── core.py
├── data.py
├── fetchers.py
├── loaders.py
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



## 🚀 Quick Start

```python
from foo.core import Fetch
fetcher = Fetch(model='gpt-4o', db_uri='sqlite:///foo.sqlite', doc_paths=['docs/*.pdf'])
response = fetcher.query_docs("Summarize the uploaded PDFs.")
print(response)
```



## 🔍 Usage Examples

**Fetch Congressional Bills:**

```python
from foo.fetchers import Congress
bills = Congress().fetch_bills(congress=118)
print(bills)
```

**Load and Chunk a PDF:**

```python
from foo.loaders import PdfLoader
loader = PdfLoader()
docs = loader.load('docs/report.pdf')
for chunk in loader.split(docs, chunk=800, overlap=100):
    print(chunk)
```

**Extract Article Content:**

```python
from foo.extractors import ArticleExtractor
extractor = ArticleExtractor()
main_text = extractor.extract(html="<article>Some story here.</article>")
print(main_text)
```

**Convert DOCX to CSV:**

```python
from foo.converters import Converter
csv_data = Converter.to_csv('docs/mydoc.docx')
print(csv_data)
```


## 📄 Loaders

### 🛰️ Loader 
- `method name` - description of method 
- `method name` - decriptions of metho


### 🛰️ CsvLoader
- `method name` - description of method 
- `method name` - decriptions of method


## 🛰️ Fetchers

### 📦 Fetcher
- `method name` - description of method 
- `method name` - decriptions of method

### 📝 Congress
- `method name` - description of method 
- `method name` - decriptions of method


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



## 🗂️ Module/Class Summary

* **core.py:** Fetch (pipeline orchestrator), FooPipeline
* **fetchers.py:** *see table above*
* **loaders.py:** *see table above*
* **data.py:** Result, Schema, Document
* **config.py:** Config




## 📝 License

MIT License
Copyright © 2022–2025 Terry D. Eppler



## 🙏 Acknowledgments

* Project lead: Terry D. Eppler ([terryeppler@gmail.com](mailto:terryeppler@gmail.com))
* Inspired by open-source Python, ML, and LLM communities.

