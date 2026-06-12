# Fetching and Scraping

Foo separates lightweight scraping helpers from broader fetcher classes.

- `scrapers.py` contains HTML extraction utilities.
- `fetchers.py` contains web, archive, geospatial, environmental, astronomical, public-data, and retriever wrappers.

```python
from fetchers import WebFetcher

fetcher = WebFetcher()
result = fetcher.fetch("https://example.com")
text = fetcher.html_to_text(result.text)
```
