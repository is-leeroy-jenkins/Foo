# Development

## Install documentation dependencies

```powershell
python -m pip install -r requirements-docs.txt
```

## Serve documentation locally

```powershell
mkdocs serve
```

## Build static site

```powershell
mkdocs build --strict
```

## Docstring rules

- Use Google-style docstrings.
- Include a meaningful `Purpose:` section.
- Use `Args:` for real public parameters only.
- Use `Attributes:` for class attributes.
- Use `Returns:` only when a meaningful value is returned.
- Do not add `Returns:` sections to constructors.
- Do not use `Returns: None`.
- Preserve signatures and executable behavior when documentation is regenerated.
