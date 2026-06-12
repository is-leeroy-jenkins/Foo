# Data API

The `data.py` module provides Foo’s local data persistence layer. It includes a base database
configuration class, a SQLite implementation for relational storage, and a Chroma wrapper for
vector-oriented storage.

Use this module when a workflow needs to create tables, insert records, retrieve rows, update
records, delete records, import Excel data, or manage vector-store entries.

## 🧭 Purpose

The Data API separates persistence behavior from the rest of the application.

The module supports three persistence concerns:

| Concern                | Class    | Purpose                                                                |
| ---------------------- | -------- | ---------------------------------------------------------------------- |
| Database configuration | `DB`     | Resolves provider, driver, path, and connection-string information.    |
| Relational storage     | `SQLite` | Manages local SQLite database operations.                              |
| Vector storage         | `Chroma` | Manages Chroma collection operations for semantic-retrieval workflows. |

This keeps database and vector-store logic out of the Streamlit UI, loaders, fetchers, scrapers,
generators, and writers.

## 🧱 Module Objects

The module exposes the following primary objects:

| Object     | Type     | Description                                                                      |
| ---------- | -------- | -------------------------------------------------------------------------------- |
| `throw_if` | Function | Validates required values before a database or vector-store operation continues. |
| `DB`       | Class    | Base class for provider, driver, path, and connection-string configuration.      |
| `SQLite`   | Class    | Local SQLite database implementation.                                            |
| `Chroma`   | Class    | Chroma vector-store implementation.                                              |

The module has its own `throw_if(name, value)` helper. Its behavior mirrors Foo’s general
required-value validation pattern: reject `None` and reject empty strings after trimming whitespace.

## 🧰 Required Value Validation

Use `throw_if(name, value)` when a database operation requires a non-empty argument.

Example:

```python
from data import throw_if

throw_if("table", table_name)
throw_if("columns", column_names)
```

This helper only checks for missing or blank values. It does not validate SQL syntax, table-name
safety, column-name safety, file existence, embedding dimensions, or vector-store collection
semantics. Those checks should be handled by the calling method or workflow.

## 🗄️ DB Base Class

`DB` is the base configuration class for database-oriented implementations.

It initializes common database state, including:

| Attribute      | Description                            |
| -------------- | -------------------------------------- |
| `provider`     | Database provider name.                |
| `source`       | Source or local data source reference. |
| `table_name`   | Active table name.                     |
| `column_names` | Active column-name list.               |
| `path`         | Data or database path.                 |
| `driver`       | Provider-specific driver value.        |

`DB` also exposes three properties:

| Property            | Return Type | Description                                              |
| ------------------- | ----------- | -------------------------------------------------------- |
| `driver_info`       | `str`       | Returns provider-specific driver information.            |
| `data_path`         | `str`       | Returns the configured data path.                        |
| `connection_string` | `str`       | Returns a connection string for the configured provider. |

The base class should remain configuration-focused. It should not perform Streamlit rendering, file
loading, web fetching, or AI-provider calls.

## 🔌 DB Usage Pattern

A database-specific implementation can inherit from `DB` to reuse provider and connection
configuration.

Conceptual pattern:

```python
from data import DB

database = DB()

print(database.driver_info)
print(database.data_path)
print(database.connection_string)
```

In normal Foo workflows, use `SQLite` or `Chroma` directly rather than instantiating `DB` alone.

## 🧮 SQLite Class

`SQLite` inherits from `DB` and provides relational database operations backed by SQLite.

The class manages:

* Local database connection.
* Cursor state.
* SQL statement state.
* Table names.
* Column names.
* Placeholder values.
* Query parameters.
* Table discovery.
* Database lifecycle operations.

Use `SQLite` for structured data, metadata, tabular rows, source inventories, extracted records, and
other relational persistence tasks.

## 🧾 SQLite Methods

`SQLite` exposes the following public methods:

| Method                                      | Return Type           | Purpose                                                              |
| ------------------------------------------- | --------------------- | -------------------------------------------------------------------- |
| `create()`                                  | `None`                | Creates required local tables.                                       |
| `create_table(sql)`                         | `None`                | Creates a table using a supplied SQL statement.                      |
| `insert(table, columns, values)`            | `None`                | Inserts one record into a table.                                     |
| `insert_many(source_file, chunks, vectors)` | `None`                | Inserts multiple chunk/vector records associated with a source file. |
| `fetch_all(table)`                          | `List[Tuple] \| None` | Retrieves all rows from a table.                                     |
| `fetch_one(table, where, params)`           | `Tuple \| None`       | Retrieves one row matching a condition.                              |
| `update(table, pairs, where, params)`       | `None`                | Updates rows matching a condition.                                   |
| `delete(table, where, params)`              | `None`                | Deletes rows matching a condition.                                   |
| `import_excel(path)`                        | `None`                | Imports Excel workbook data into SQLite.                             |
| `close()`                                   | `None`                | Closes the active SQLite connection.                                 |

The exact SQL behavior, parameter order, and state mutations are documented in the generated API
reference below.

## 📚 SQLite Storage Use Cases

SQLite is appropriate for relational and table-oriented data.

Good use cases include:

* Source metadata.
* File inventories.
* Loader results.
* Fetch result metadata.
* Scraped URL inventories.
* Extracted table rows.
* Prompt or generation metadata.
* Processing history.
* Configuration snapshots.
* User-selected workflow records.
* Audit-friendly structured data.

SQLite is not the best place for large raw document archives, binary blobs, full image data, or
high-dimensional semantic search. Use normal file storage or Chroma-backed workflows for those use
cases.

## 🧪 Example: Create the Local Database

```python
from data import SQLite

database = SQLite()
database.create()
database.close()
```

This pattern initializes the SQLite layer and then closes the connection after the operation
completes.

## 🧪 Example: Create a Table

```python
from data import SQLite

database = SQLite()

database.create_table(
	"""
	CREATE TABLE IF NOT EXISTS sources (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT NOT NULL,
		uri TEXT NOT NULL
	)
	"""
)

database.close()
```

Use explicit SQL when creating tables. Keep schema changes deliberate and reviewable.

## 🧪 Example: Insert a Record

```python
from data import SQLite

database = SQLite()

database.insert(
	table="sources",
	columns=["name", "uri"],
	values=["Example", "https://example.com"],
)

database.close()
```

Use normalized scalar values for SQLite storage. Store large text bodies, binary content, or
embeddings separately when appropriate.

## 🧪 Example: Fetch All Rows

```python
from data import SQLite

database = SQLite()

rows = database.fetch_all("sources")

for row in rows or []:
	print(row)

database.close()
```

`fetch_all(...)` returns rows as tuples when data is available. Code that consumes the result should
handle `None` defensively.

## 🧪 Example: Fetch One Row

```python
from data import SQLite

database = SQLite()

row = database.fetch_one(
	table="sources",
	where="name = ?",
	params=("Example",),
)

print(row)

database.close()
```

Use parameterized values through `params` rather than formatting user-supplied values directly into
SQL.

## 🧪 Example: Update Rows

```python
from data import SQLite

database = SQLite()

database.update(
	table="sources",
	pairs={"uri": "https://www.example.com"},
	where="name = ?",
	params=("Example",),
)

database.close()
```

Keep update conditions explicit so broad accidental updates are avoided.

## 🧪 Example: Delete Rows

```python
from data import SQLite

database = SQLite()

database.delete(
	table="sources",
	where="name = ?",
	params=("Example",),
)

database.close()
```

Deletion should be reserved for deliberate cleanup workflows. Prefer reviewable conditions over
broad table-level deletion.

## 🧪 Example: Import Excel Data

```python
from data import SQLite

database = SQLite()
database.import_excel("data/input.xlsx")
database.close()
```

Excel import is useful when a workflow starts with workbook-based structured data and needs to move
it into local SQLite storage.

## 🧬 Chroma Class

`Chroma` provides vector-oriented storage support.

Use Chroma for semantic-retrieval workflows where text chunks, embeddings, and metadata need to be
stored and queried by similarity.

The class constructor accepts:

| Parameter    | Description                                      |
| ------------ | ------------------------------------------------ |
| `path`       | Path used by the Chroma storage implementation.  |
| `collection` | Collection name used to organize vector records. |

The class exposes methods for adding records, querying records, deleting records, counting records,
clearing a collection, and persisting local state.

## 🧾 Chroma Methods

`Chroma` exposes the following public methods:

| Method                                   | Return Type         | Purpose                                                          |
| ---------------------------------------- | ------------------- | ---------------------------------------------------------------- |
| `add(ids, texts, embeddings, metadatas)` | `None`              | Adds text, embedding, and metadata records to the collection.    |
| `query(text, num, where)`                | `List[str] \| None` | Queries the collection and returns matching text results.        |
| `delete(ids)`                            | `None`              | Deletes vector records by identifier.                            |
| `count()`                                | `int \| None`       | Returns the number of records in the collection.                 |
| `clear()`                                | `None`              | Clears collection records.                                       |
| `persist()`                              | `None`              | Persists collection state where supported by the backing client. |

The Chroma class is intended for retrieval-oriented storage, not ordinary relational data.

## 📚 Chroma Storage Use Cases

Chroma is appropriate for semantic-search and retrieval workflows.

Good use cases include:

* Embedded document chunks.
* Chunk metadata.
* Retrieval collections.
* Searchable knowledge stores.
* RAG context storage.
* Semantic search over loaded or scraped content.
* Similarity search across processed documents.

Chroma should not replace SQLite for ordinary structured tables. A common pattern is to use both:

```text
SQLite
- Source table
- Document metadata
- Processing history
- Audit-friendly rows

Chroma
- Text chunks
- Embeddings
- Semantic-retrieval collections
```

## 🧪 Example: Initialize a Chroma Collection

```python
from data import Chroma

store = Chroma(
	path="data/chroma",
	collection="foo_documents",
)
```

The configured path and collection identify where vector-store content is managed.

## 🧪 Example: Add Vector Records

```python
from data import Chroma

store = Chroma(
	path="data/chroma",
	collection="foo_documents",
)

store.add(
	ids=["doc-001"],
	texts=["Example document text"],
	embeddings=[[0.1, 0.2, 0.3]],
	metadatas=[{"source": "example"}],
)
```

The embedding vectors should match the dimensionality expected by the embedding model and Chroma
collection.

## 🧪 Example: Query a Chroma Collection

```python
from data import Chroma

store = Chroma(
	path="data/chroma",
	collection="foo_documents",
)

results = store.query(
	text="example search text",
	num=5,
	where=None,
)

for result in results or []:
	print(result)
```

Use `num` to control how many matches are requested. Use `where` for metadata filtering when the
workflow supports it.

## 🧪 Example: Count and Clear Records

```python
from data import Chroma

store = Chroma(
	path="data/chroma",
	collection="foo_documents",
)

count = store.count()
print(count)

store.clear()
store.persist()
```

Clearing a collection should be deliberate. Use it for reset workflows, testing, or maintenance
operations.

## 🔗 Relationship to Loaders

Loaders produce document objects and text content that can be persisted.

Typical relationship:

```text
Loader
  |
  v
Documents / text / metadata
  |
  +--> SQLite for source metadata
  |
  +--> Chroma for chunk embeddings
```

Use SQLite when the output is structured. Use Chroma when the output is intended for semantic
retrieval.

## 🔗 Relationship to Fetchers and Scrapers

Fetchers and scrapers produce retrieved data, page metadata, cleaned text, links, tables, and
extracted elements.

These outputs can be stored as:

| Output             | Suggested Storage     |
| ------------------ | --------------------- |
| URL                | SQLite                |
| Status code        | SQLite                |
| Page title         | SQLite                |
| Extracted links    | SQLite                |
| Extracted tables   | SQLite                |
| Clean article text | Chroma or output file |
| Large page body    | Chroma or output file |
| Chunk embeddings   | Chroma                |

This keeps structured metadata searchable through SQLite while preserving semantic search
capabilities through Chroma.

## 🔗 Relationship to Generators

Generator outputs can be stored for traceability and reuse.

Potential SQLite records include:

* Prompt identifier.
* Provider name.
* Model name.
* Generation timestamp.
* Output summary.
* Token or usage metadata when available.
* Workflow mode.
* Source reference.

Potential Chroma records include:

* Generated summaries.
* Generated notes.
* Generated knowledge-base entries.
* Embedded generated output for retrieval.

Do not store secrets, API keys, tokens, or sensitive user content unless the workflow explicitly
requires and protects it.

## 🔗 Relationship to Writers

Writers can export data retrieved from SQLite or Chroma.

Examples:

* Export a table of fetched URLs to Markdown.
* Export scraped page summaries.
* Export retrieved semantic-search results.
* Export generation history.
* Export database inspection reports.

The data layer should retrieve the records. The writer layer should format and serialize them.

## 🔐 Error Handling

The data module uses the Foo structured error logging pattern where exception handlers wrap errors
with `boogr.Error`.

The expected pattern is:

```python
except Exception as e:
	exception = Error(e)
	exception.module = "data"
	exception.cause = "SQLite"
	exception.method = "fetch_all( self, table: str ) -> List[ Tuple ] | None"
	Logger().write(exception)
	raise exception
```

Use stable method signatures in `exception.method`.

Do not include:

* Raw SQL values containing user data.
* API keys.
* Tokens.
* Full file contents.
* Raw document bodies.
* Full scraped pages.
* Sensitive source material.

The logging goal is diagnostic traceability, not content capture.

## 🛡️ SQL Safety Guidance

SQLite workflows should avoid unsafe SQL construction.

Follow these practices:

* Validate table names.
* Validate column names.
* Use parameterized values for user-provided data.
* Avoid concatenating raw user input directly into SQL statements.
* Keep destructive operations explicit.
* Avoid broad `DELETE` or `UPDATE` statements without a condition.
* Keep schema migrations reviewable.
* Close connections after work completes.

Table and column identifiers cannot be parameterized in the same way as values, so identifier
validation is important when identifiers come from user-facing controls.

## 🧪 Testing Guidance

When testing `data.py`, verify:

* The module compiles.
* `DB` properties return expected strings.
* `SQLite.create()` initializes expected local tables.
* `SQLite.create_table(...)` creates the requested table.
* `SQLite.insert(...)` writes a row.
* `SQLite.fetch_all(...)` returns inserted rows.
* `SQLite.fetch_one(...)` returns the expected row.
* `SQLite.update(...)` changes only intended rows.
* `SQLite.delete(...)` removes only intended rows.
* `SQLite.import_excel(...)` handles valid Excel files.
* `SQLite.close()` closes the connection cleanly.
* `Chroma.add(...)` accepts matching IDs, text, embeddings, and metadata.
* `Chroma.query(...)` returns documented results.
* `Chroma.count()` returns a count or documented fallback.
* `Chroma.clear()` and `Chroma.persist()` behave as expected.
* Existing exception handlers log wrapped errors before re-raising.

## ⚠️ Maintenance Notes

When changing `data.py`, keep persistence behavior stable unless the change is intentional.

Avoid:

* Moving UI rendering into `data.py`.
* Moving provider API calls into `data.py`.
* Mixing fetcher logic with database logic.
* Mixing writer serialization with database logic.
* Logging sensitive values.
* Returning undocumented result shapes.
* Adding unannotated public methods.
* Adding constructor `Returns:` docstring sections.

Maintain:

* Type annotations.
* Google-style docstrings.
* Explicit argument validation.
* Stable error metadata.
* Existing fallback behavior.
* Existing transaction semantics.
* Clear separation between SQLite and Chroma responsibilities.

## 🧾 API Documentation

The generated API reference for this module is rendered below.
