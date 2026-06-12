# Generators

The generator layer wraps supported model providers and normalizes prompt submission, request options, response extraction, and provider-specific settings.

Provider API keys are read from environment-backed configuration values in `config.py`.

```python
from generators import Grok

generator = Grok()
response = generator.generate_text("Summarize this dataset.")
```
