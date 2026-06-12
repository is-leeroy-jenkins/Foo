![](../images/foo-project.png)


#### Foo Documentation

Foo is a Python framework for loading, scraping, fetching, generating, and managing data for machine-learning and analysis workflows.

## Documentation contents

- **User Guide** explains the primary workflows at a practical level.
- **Architecture** describes how the application modules fit together.
- **API Reference** is generated from Google-style Python docstrings through `mkdocstrings`.
- **GitHub Pages** explains how to publish the generated site from the repository.

## Local build

```powershell
python -m pip install -r requirements-docs.txt
mkdocs serve
```

Open the local URL printed by MkDocs, usually `http://127.0.0.1:8000/`.
