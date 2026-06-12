# Loading Data

The loader layer wraps LangChain and related file loaders for text, CSV, PDF, Excel, Word, Markdown, HTML, JSON, PowerPoint, XML, notebook, cloud, and search-oriented sources.

Use the API reference for exact constructor and method signatures.

```python
from loaders import TextLoader

loader = TextLoader()
documents = loader.load("sample.txt")
chunks = loader.split_tokens(size=1000, amount=200)
```
