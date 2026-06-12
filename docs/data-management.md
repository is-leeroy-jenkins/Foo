# Data Management

Foo includes a data-management layer for local persistence, table inspection, table maintenance, and
vector-oriented storage workflows. This layer supports the application by separating database
operations from the Streamlit user interface, document loaders, fetchers, scrapers, generators, and
output writers.

The main data-management module is `data.py`. It defines shared database configuration behavior, a
SQLite implementation, and Chroma-oriented vector storage support.

## 🗄️ Data Management Overview

Foo’s data-management layer provides the persistence foundation for workflows that need to store,
inspect, or reuse processed information.

The data layer is responsible for:

* Establishing database configuration.
* Building provider-specific connection information.
* Managing SQLite database connections.
* Creating tables used by Foo workflows.
* Importing structured data into local tables.
* Fetching rows from local tables.
* Updating and deleting table records.
* Listing available database tables.
* Supporting vector-oriented storage through Chroma.
* Providing a clear separation between persistence logic and UI logic.

In the broader Foo architecture, data management sits between the processing layers and durable
local storage.

```text
Loaders / Fetchers / Scrapers / Generators
                  |
                  v
              data.py
                  |
        +---------+----------+
        |                    |
        v                    v
   SQLite database      Chroma storage
```

## 🧱 Source Module

The data-management implementation is concentrated in:

```text
data.py
```

The module contains three primary classes:

| Class    | Purpose                                                                                                |
| -------- | ------------------------------------------------------------------------------------------------------ |
| `DB`     | Base database configuration class that resolves provider, driver, path, and connection-string details. |
| `SQLite` | SQLite database implementation that manages local database operations.                                 |
| `Chroma` | Chroma-oriented storage class for vector database workflows.                                           |

The module also defines its own `throw_if(name, value)` guard for required-value validation.

## ⚙️ DB Base Class

The `DB` class provides shared database configuration behavior.

It tracks common database attributes such as:

* `provider`
* `source`
* `table_name`
* `column_names`
* `path`
* `driver`

The class exposes properties for provider-specific configuration:

| Property            | Purpose                                                 |
| ------------------- | ------------------------------------------------------- |
| `driver_info`       | Resolves the driver string for supported providers.     |
| `data_path`         | Resolves the local data path for the selected provider. |
| `connection_string` | Builds a connection string for the configured provider. |

The base class is useful because database-specific classes can inherit common provider and path
behavior instead of duplicating configuration logic.

## 🧮 SQLite Class

The `SQLite` class inherits from `DB` and provides local SQLite database operations.

It initializes a connection to the local Foo SQLite database and maintains runtime state for
database operations. The class tracks:

* Database path.
* SQLite connection.
* SQLite cursor.
* Current file path.
* Current table name.
* SQL statement state.
* Placeholder state.
* Column names.
* Query parameters.
* Available tables.

The class supports common database operations, including:

| Operation           | Purpose                                                       |
| ------------------- | ------------------------------------------------------------- |
| `create()`          | Creates required local tables when they do not already exist. |
| `create_table(...)` | Creates a table from supplied column definitions.             |
| `insert(...)`       | Inserts records into a target table.                          |
| `fetch_all(...)`    | Retrieves all rows from a table or query.                     |
| `fetch_one(...)`    | Retrieves one matching row.                                   |
| `update(...)`       | Updates records that match a condition.                       |
| `delete(...)`       | Deletes records that match a condition.                       |
| `import_excel(...)` | Imports Excel data into local SQLite storage.                 |
| `get_tables()`      | Lists available database tables.                              |
| `close()`           | Closes the active database connection.                        |

This class is the primary local relational storage interface for Foo.

## 🧬 Chroma Class

The `Chroma` class supports vector-oriented storage workflows.

Chroma is used for workflows where document chunks, embeddings, and retrieval-oriented data need to
be persisted or queried separately from ordinary relational tables.

Typical Chroma use cases include:

* Storing embedded document chunks.
* Supporting retrieval-augmented generation workflows.
* Managing vector collections.
* Persisting semantic-search data.
* Keeping retrieval data separate from ordinary SQLite tables.

The exact use of Chroma depends on the active Foo workflow and the embedding or retrieval strategy
selected by the application.

## 🧭 How Data Management Fits the Application

Foo uses the data layer as a supporting service rather than as a standalone user-facing workflow.

The data layer can be used after content is produced by:

* A loader that ingests documents.
* A fetcher that retrieves public or web data.
* A scraper that extracts HTML content.
* A generator that produces AI-assisted output.
* A processor that creates structured rows, chunks, embeddings, or metadata.

The normal flow is:

```text
Source content
    |
    v
Load / fetch / scrape / generate
    |
    v
Normalize or process result
    |
    v
Persist in SQLite or Chroma
    |
    v
Inspect, retrieve, export, or reuse
```

## 📚 SQLite Workflow

A typical SQLite workflow in Foo follows this pattern:

1. Create or open the local SQLite database.
2. Create the required table if it does not already exist.
3. Insert rows into the table.
4. Query or inspect the table.
5. Update or delete rows when necessary.
6. Close the connection when the operation is complete.

Conceptually:

```python
from data import SQLite

database = SQLite()
database.create()

# Use SQLite methods to insert, fetch, update, or delete records.

database.close()
```

The exact method arguments depend on the operation being performed. Refer to the API reference for
method signatures and return types.

## 🧾 Table-Oriented Storage

SQLite is best suited for structured data.

Good candidates for SQLite storage include:

* Source metadata.
* File inventory records.
* Scraped page summaries.
* API response metadata.
* Prompt records.
* Processing history.
* Tabular datasets.
* Normalized public-data rows.
* User-selected configuration snapshots.
* Generated-output metadata.

SQLite is not ideal for large binary files, raw document archives, or high-dimensional embedding
search. Those should be handled by file storage or vector storage workflows instead.

## 🔎 Vector-Oriented Storage

Chroma is best suited for semantic retrieval workflows.

Good candidates for Chroma storage include:

* Document chunks.
* Embedding vectors.
* Chunk metadata.
* Retrieval collections.
* Semantic search indexes.
* Retrieval-augmented generation context.

A common pattern is to store structured metadata in SQLite and vector-search data in Chroma.

```text
SQLite
- Table names
- Source identifiers
- File metadata
- Processing metadata
- Audit-friendly structured rows

Chroma
- Text chunks
- Embeddings
- Collection metadata
- Semantic retrieval records
```

This split keeps relational data and vector data aligned without forcing one storage engine to do
both jobs.

## 🔐 Error Handling and Logging

The data layer uses the Foo error-handling pattern based on `boogr`.

When database operations fail inside a handled exception path, the error is wrapped with structured
metadata and written to the configured exception log.

The standard pattern is:

```python
except Exception as e:
    exception = Error(e)
    exception.module = "data"
    exception.cause = "<ClassName>"
    exception.method = "<stable method signature>"
    Logger().write(exception)
    raise exception
```

This pattern helps keep local diagnostic information consistent across the project.

The error metadata should remain stable and reviewer-safe:

* Do not include raw user data in `exception.method`.
* Do not include full file contents in error metadata.
* Do not include live API keys, tokens, or credentials.
* Use the module name, class name, and stable method signature.

## 🧪 Data Quality Considerations

Data-management workflows should protect the consistency of local state.

Recommended practices:

* Validate table names before executing table-level operations.
* Validate column names before building SQL statements.
* Prefer parameterized SQL for values.
* Avoid dynamically concatenating user-provided SQL.
* Keep schema changes explicit.
* Commit only after successful write operations.
* Close database resources when work is complete.
* Preserve existing fallback behavior when adding logging or documentation improvements.

For documentation and maintenance, database methods should clearly state:

* What table or collection they operate on.
* What arguments are required.
* What value is returned.
* What exceptions may be raised or wrapped.
* Whether the method mutates state or returns data.

## 🛡️ Safe Query Guidance

When a workflow exposes SQL-like functionality to users, the application should restrict execution
to safe read-only statements unless mutation is explicitly intended.

Read-only inspection queries should be separated from write operations such as:

* `INSERT`
* `UPDATE`
* `DELETE`
* `DROP`
* `ALTER`
* `CREATE`

This separation protects the local database from accidental destructive operations and keeps the
data-management interface predictable.

## 🧰 Practical Usage Patterns

### Inspect Available Tables

Use table inspection when the UI needs to let a user choose from existing database tables.

```python
from data import SQLite

database = SQLite()

# Use the table-listing method exposed by the SQLite class.
# Then display the returned table names in the UI.
```

### Create a Local Table

Use table creation when a workflow produces structured records that should be stored locally.

```python
from data import SQLite

database = SQLite()

# Create a table before inserting rows.
# Use explicit column names and SQLite-compatible column types.
```

### Insert Processed Records

Use insert workflows after data has been loaded, fetched, scraped, or generated.

```python
from data import SQLite

database = SQLite()

# Insert processed rows into the target table.
# Prefer normalized scalar values for table storage.
```

### Store Retrieval Data

Use Chroma-oriented workflows when the content is intended for semantic search or
retrieval-augmented generation.

```python
from data import Chroma

store = Chroma()

# Use Chroma methods to manage collections, documents, metadata, and embeddings.
```

## 🧩 Relationship to Other Modules

The data-management layer is used by or supports several other Foo modules.

| Module          | Relationship to Data Management                                   |
| --------------- | ----------------------------------------------------------------- |
| `app.py`        | Presents database controls and results through Streamlit.         |
| `loaders.py`    | Produces documents or chunks that can be persisted.               |
| `fetchers.py`   | Produces retrieved data that can be stored or indexed.            |
| `scrapers.py`   | Produces extracted text and structured content that can be saved. |
| `generators.py` | Produces outputs and metadata that can be persisted.              |
| `models.py`     | Defines structured objects that may be serialized or stored.      |
| `writers.py`    | Exports stored or processed content to output files.              |

## 📖 API Reference

For full method signatures, return annotations, and class-level documentation, use the API
reference:

* [Data API](api/data.md)
* [Core API](api/core.md)
* [Models API](api/models.md)

The API pages are generated from the source docstrings using MkDocs and mkdocstrings.

## ✅ Maintenance Checklist

When updating `data.py`, verify the following before committing:

* Database methods retain their existing runtime behavior.
* Public methods have Google-style docstrings.
* Methods with meaningful return values have accurate `Returns:` sections.
* Methods returning lists, dictionaries, cursors, connections, or storage objects are annotated and
  documented.
* Existing `try/except` blocks log wrapped exceptions before re-raising.
* Error metadata uses stable method signatures.
* SQL statements are safe for the operation being performed.
* The documentation build completes without Griffe warnings.
* The relevant API page renders correctly.

## 🧭 Summary

Foo’s data-management layer provides the local persistence foundation for the rest of the
application. SQLite supports relational and table-oriented workflows, while Chroma supports
vector-oriented retrieval workflows. Keeping this layer separate from the UI, loaders, fetchers,
scrapers, generators, and writers makes the application easier to maintain, document, and extend.
