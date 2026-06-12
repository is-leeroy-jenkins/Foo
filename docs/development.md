# Development

This page describes how to maintain, extend, document, and validate Foo as a Python and Streamlit
application.

Foo is organized as a modular application rather than a single monolithic script. The Streamlit UI
in `app.py` coordinates workflow selection and user interaction, while the supporting modules handle
loading, fetching, scraping, generation, persistence, structured models, and output writing.

## 🧭 Development Goals

Foo development should prioritize stability, traceability, and maintainability.

The main goals are:

* Preserve existing runtime behavior unless a change is intentional.
* Keep each source module responsible for one clear area of functionality.
* Keep the Streamlit UI focused on orchestration and display.
* Keep provider-specific logic inside provider wrapper classes.
* Keep persistence logic inside the data-management layer.
* Keep output serialization inside writer classes.
* Maintain clean Google-style docstrings for MkDocs and mkdocstrings.
* Keep error logging consistent across all handled exception paths.

Development work should improve the application without blurring module boundaries.

## 🧱 Source Layout

The core source files are organized by responsibility.

| File            | Development Responsibility                                                                  |
| --------------- | ------------------------------------------------------------------------------------------- |
| `app.py`        | Streamlit UI, session state, mode selection, layout, controls, and orchestration.           |
| `config.py`     | Application constants, paths, API key references, mode maps, and session defaults.          |
| `core.py`       | Shared result and validation primitives.                                                    |
| `loaders.py`    | Document loading and source ingestion classes.                                              |
| `fetchers.py`   | Web, API, public-data, science, weather, geospatial, and government-data retrieval classes. |
| `scrapers.py`   | HTML extraction and web scraping helpers.                                                   |
| `generators.py` | LLM provider wrappers and generation request handling.                                      |
| `data.py`       | SQLite and Chroma persistence utilities.                                                    |
| `models.py`     | Pydantic models for structured application objects.                                         |
| `writers.py`    | Markdown and output writing classes.                                                        |

Documentation files live under:

```text
docs/
```

Documentation assets live under:

```text
docs/images/
```

The MkDocs configuration lives at the repository root:

```text
mkdocs.yml
```

## ⚙️ Local Environment

Create and activate a virtual environment before installing dependencies.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade packaging tools:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

Install application dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install documentation dependencies:

```powershell
python -m pip install -r requirements-docs.txt
```

If mkdocstrings reports that signature formatting requires Black or Ruff, install one of them:

```powershell
python -m pip install black
```

or:

```powershell
python -m pip install ruff
```

## 🖥️ Running the Application

Foo is a Streamlit application.

Run it from the repository root:

```powershell
streamlit run app.py
```

Streamlit will normally open the application in a browser automatically. If not, use the local URL
printed in the terminal.

The Streamlit interface depends on values in `config.py`, local files, optional API keys, and
installed provider libraries. Missing optional dependencies or API credentials may affect specific
modes without necessarily preventing the whole application from starting.

## 🧪 Basic Validation

Before committing changes, run a basic syntax validation pass.

```powershell
python -m compileall .
```

For a targeted file:

```powershell
python -m py_compile loaders.py
python -m py_compile fetchers.py
python -m py_compile generators.py
```

A file must compile before documentation or runtime behavior can be trusted.

## 📚 Documentation Build

Build the documentation locally before pushing documentation changes:

```powershell
mkdocs build
```

For stricter validation:

```powershell
mkdocs build --strict
```

Use `--strict` when preparing changes for GitHub Pages because warnings that appear locally are
usually the same warnings that will fail or degrade the published site.

Run the local documentation server:

```powershell
mkdocs serve
```

Then open the local URL printed by MkDocs, usually:

```text
http://127.0.0.1:8000/
```

## 🧾 Docstring Standards

Foo API documentation is generated from source docstrings. Public modules, classes, functions,
methods, and properties should use Google-style docstrings.

Use these section names:

```text
Purpose:
Args:
Returns:
```

Use `Args:` only for real parameters in the function or method signature. Do not document `self`.

Use `Returns:` only when the function or method returns a meaningful value.

Do not add `Returns:` sections to constructors.

Do not add generic return entries like:

```text
Returns:
    None: This method does not return a value.
```

Do not use malformed or old-style headings such as:

```text
Parameters:
-----------
Returns:
--------
Purpose:
_________
Parametes:
```

Those patterns can produce Griffe warnings during `mkdocs build`.

## 🧩 Example Docstring Pattern

Use this pattern for functions that validate input and return a value:

```python
def normalize_name(name: str) -> str:
	"""Normalize a name value.

	Purpose:
		Converts a user-supplied name into a normalized string suitable for display,
		lookup, or storage. The function trims leading and trailing whitespace and
		rejects empty values before returning the normalized result.

	Args:
		name (str): Name value to normalize.

	Returns:
		Normalized name value.

	Raises:
		ValueError: Raised when the supplied name is missing or empty.
	"""
```

Use this pattern for procedures that mutate state or render UI but do not return a meaningful value:

```python
def render_controls() -> None:
	"""Render Streamlit controls.

	Purpose:
		Renders the Streamlit controls used to collect runtime options from the user.
		The function updates Streamlit state and writes UI elements to the page.
	"""
```

Use this pattern for constructors:

```python
def __init__(self) -> None:
	"""Initialize the loader.

	Purpose:
		Initializes default loader state, including file path, document collection,
		chunking configuration, and compatibility fields used by later method calls.
	"""
```

## 🔐 Error Logging Pattern

Foo uses the `boogr` error wrapper and logger pattern in handled exception paths.

When a method already catches exceptions and wraps them with `Error`, it should also write the
structured error before re-raising it.

Preferred pattern:

```python
except Exception as e:
	exception = Error(e)
	exception.module = "loaders"
	exception.cause = "TextLoader"
	exception.method = "load( self, filepath: str ) -> List[ Document ] | None"
	Logger().write(exception)
	raise exception
```

The metadata should be stable and reviewer-safe.

Do not put live values in `exception.method`.

Avoid:

```python
exception.method = f"load({filepath})"
```

Prefer:

```python
exception.method = "load( self, filepath: str ) -> List[ Document ] | None"
```

## 🛡️ Error Metadata Rules

When adding or updating logging, follow these rules:

* Use the source module name for `exception.module`.
* Use the class name or functional component for `exception.cause`.
* Use a stable method or function signature for `exception.method`.
* Do not include API keys.
* Do not include tokens.
* Do not include full file paths unless explicitly intended.
* Do not include file contents.
* Do not include raw document text.
* Do not include OCR output or scraped page bodies.
* Preserve the existing fallback or re-raise behavior.

Logging should improve diagnostics without exposing sensitive data.

## 🧪 Return Annotation Rules

All public functions, methods, and properties should have return annotations.

Good examples:

```python
def __dir__(self) -> list[str]:
```

```python
@property
def mode_options(self) -> list[str]:
```

```python
def fetch(self, question: str) -> list[Document] | None:
```

```python
def to_dict(self) -> dict[str, object]:
```

For methods that do not return a meaningful value:

```python
def close(self) -> None:
```

If the method returns a value, document that value in the `Returns:` section. If the method only
mutates state, renders UI, writes files, logs errors, or closes resources, do not add a `Returns:`
section unless the return value is meaningful.

## 📦 Dependency Management

Application dependencies belong in:

```text
requirements.txt
```

Documentation dependencies belong in:

```text
requirements-docs.txt
```

Keep these separate. Runtime users should not need documentation tooling unless they are building
the docs.

When adding a dependency, verify:

* Why it is needed.
* Which module imports it.
* Whether it is required at runtime or only for documentation.
* Whether it has platform-specific installation concerns.
* Whether it should be optional.
* Whether it affects GitHub Actions or deployment.

## 🧰 Module Development Guidance

### `app.py`

Keep `app.py` focused on Streamlit orchestration.

Appropriate responsibilities:

* Page configuration.
* Sidebar controls.
* Mode selection.
* Session-state initialization.
* UI rendering.
* Calling loader, fetcher, scraper, generator, data, and writer classes.

Avoid putting large provider-specific logic directly in `app.py`. Move that behavior into the
appropriate module.

### `config.py`

Keep configuration centralized.

Appropriate responsibilities:

* API key environment lookups.
* Application constants.
* UI labels and mode maps.
* Session-state defaults.
* Paths to local assets and databases.
* Documentation or application-level constants.

Avoid hard-coding provider credentials or user-specific absolute paths.

### `core.py`

Keep `core.py` small and stable.

Appropriate responsibilities:

* Shared validation helpers.
* Shared result containers.
* Small cross-module primitives.

Avoid adding provider-specific logic to `core.py`.

### `loaders.py`

Keep loading classes source-specific.

A loader should:

* Validate required inputs.
* Resolve file paths or source identifiers.
* Load source content.
* Return document objects or a documented result.
* Preserve metadata where useful.
* Log wrapped exceptions consistently.

Avoid mixing unrelated API retrieval logic into loaders. API retrieval belongs in `fetchers.py`.

### `fetchers.py`

Keep fetcher classes focused on retrieval.

A fetcher should:

* Validate request inputs.
* Build provider or service parameters.
* Make the retrieval call.
* Normalize the result.
* Return a documented object.
* Log wrapped exceptions consistently.

Avoid UI rendering in fetchers. Display belongs in `app.py`.

### `scrapers.py`

Keep scraping classes focused on HTML extraction.

A scraper should:

* Fetch or receive HTML.
* Parse HTML.
* Extract readable content or structured elements.
* Return cleaned text or lists of extracted values.
* Log wrapped exceptions consistently.

Avoid storing scraped content directly unless the workflow explicitly calls the data layer.

### `generators.py`

Keep generation classes provider-focused.

A generator should:

* Validate prompt and provider settings.
* Build requests.
* Call the configured provider.
* Extract response text or structured output.
* Log wrapped exceptions consistently.

Avoid embedding UI-specific state in generator classes.

### `data.py`

Keep persistence logic in the data layer.

Data classes should:

* Manage connections.
* Create tables or collections.
* Insert, update, delete, or retrieve records.
* Validate identifiers.
* Keep SQL behavior explicit and safe.
* Log wrapped exceptions consistently.

Avoid putting Streamlit rendering or provider calls directly in `data.py`.

### `models.py`

Keep models declarative.

Pydantic models should:

* Define structured fields.
* Use clear type annotations.
* Include class docstrings that explain model purpose.
* Avoid operational side effects.

### `writers.py`

Keep writer classes focused on output.

Writers should:

* Validate required inputs.
* Resolve output paths.
* Create parent directories when needed.
* Write the output format.
* Return the written path when meaningful.
* Log wrapped exceptions consistently.

## 🧭 Adding a New Loader

When adding a new loader:

1. Add the class to `loaders.py`.
2. Inherit from `Loader` when the common loader behavior applies.
3. Add typed attributes if the class stores runtime state.
4. Add a constructor with initialized defaults.
5. Add a `__dir__` method if consistent with the surrounding loader classes.
6. Add a public `load(...)` method or equivalent operation.
7. Validate mandatory arguments.
8. Return a documented result type.
9. Wrap and log exceptions where the class already uses handled exception paths.
10. Update `app.py` if the loader should be exposed in the UI.
11. Update `docs/loading.md`.
12. Confirm the API page renders correctly.

## 🌐 Adding a New Fetcher

When adding a new fetcher:

1. Add the class to `fetchers.py`.
2. Inherit from `Fetcher` when the common fetcher behavior applies.
3. Store API keys or configuration from `config.py`.
4. Validate required arguments.
5. Keep service-specific parameters inside the fetcher class.
6. Normalize returned data.
7. Return a typed and documented result.
8. Use the standard error logging pattern.
9. Update `app.py` if the fetcher should be selectable from the UI.
10. Update `docs/fetching-scraping.md`.
11. Confirm the API page renders correctly.

## 🤖 Adding a New Generator

When adding a new AI provider or generation mode:

1. Add the provider wrapper to `generators.py`.
2. Keep provider credentials in `config.py`.
3. Add model-option properties where appropriate.
4. Validate prompt and model inputs.
5. Keep request construction inside helper methods.
6. Keep response extraction inside helper methods.
7. Preserve streaming or tool behavior if the provider supports it.
8. Use stable error metadata.
9. Update the Streamlit UI only after the provider wrapper works.
10. Update `docs/generation.md`.
11. Confirm the API page renders without Griffe warnings.

## 🗄️ Adding a Data Operation

When adding a data operation:

1. Add the operation to `data.py`.
2. Annotate arguments and return values.
3. Validate table names, column names, and required values.
4. Use parameterized SQL for values.
5. Avoid accepting destructive SQL from user input.
6. Preserve transaction behavior.
7. Log wrapped exceptions consistently.
8. Update `docs/data-management.md`.
9. Confirm the API page renders correctly.

## 🧾 Adding a Writer

When adding a writer:

1. Add the writer class to `writers.py`.
2. Validate required inputs.
3. Resolve output paths safely.
4. Create parent directories if needed.
5. Return the generated output path when meaningful.
6. Use the standard logging pattern.
7. Update documentation if the output format is user-facing.
8. Confirm the API page renders correctly.

## 🧪 Pre-Commit Checklist

Before committing changes, run through this checklist:

* Source files compile.
* The Streamlit app starts.
* Changed workflows run manually.
* New or changed public functions have docstrings.
* Public return values are annotated.
* Meaningful return values are documented.
* Constructors do not include `Returns:` sections.
* Existing exception handlers preserve behavior.
* Wrapped exceptions are logged before being re-raised.
* No credentials or raw user content are logged.
* MkDocs builds locally.
* Griffe warnings are resolved.
* Markdown links are valid.
* Referenced images exist under `docs/images/`.
* GitHub Pages workflow files are still valid.

## 📚 Documentation Maintenance

When changing source code, update documentation in the same change set.

| Change Type               | Documentation Update                                  |
| ------------------------- | ----------------------------------------------------- |
| New loader                | Update `loading.md` and `api/loaders.md`.             |
| New fetcher               | Update `fetching-scraping.md` and `api/fetchers.md`.  |
| New generator             | Update `generation.md` and `api/generators.md`.       |
| New persistence method    | Update `data-management.md` and `api/data.md`.        |
| New writer                | Update output documentation and `api/writers.md`.     |
| New model                 | Update model documentation and `api/models.md`.       |
| New architecture behavior | Update `architecture.md`.                             |
| New dependency            | Update `requirements.txt` or `requirements-docs.txt`. |

Documentation should explain how the feature is used, where it lives, and how it fits into the
application.

## 🚀 GitHub Pages Readiness

Before publishing to GitHub Pages:

1. Build locally with MkDocs.
2. Fix warnings.
3. Confirm `site_url` is correct in `mkdocs.yml`.
4. Confirm `repo_url` is correct in `mkdocs.yml`.
5. Confirm all pages in `docs/` are represented in navigation if they should be public.
6. Confirm images resolve from the published site path.
7. Confirm the GitHub Pages workflow builds from the correct branch.
8. Confirm repository Pages settings use GitHub Actions as the source.

## ✅ Summary

Foo development should be deliberate and source-grounded. Keep UI orchestration, loading, fetching,
scraping, generation, persistence, models, and writing responsibilities separated. Preserve runtime
behavior when improving documentation. Maintain Google-style docstrings so the API reference stays
clean. Run local validation before publishing or committing changes.
