![](https://github.com/is-leeroy-jenkins/Foo/blob/main/resources/images/foo_project.png)


A small, multi-tool framework for retrieval augmentation, SQL Querying, and Document Loading 

## 🧠 **Overview**

Foo is an extensible Python framework designed to unify:

* Conversational interaction with OpenAI-compatible LLMs
* Deterministic natural-language SQL querying
* Document ingestion and vector-based semantic retrieval
* Tool routing through a LangChain ReAct agent
* Structured error-handling via guard clauses and dialogs

- Foo’s architecture is modular by design, composed of separate tool classes for SQL, document retrieval, and future API integrations. The `Fetch` controller coordinates these tools to provide a predictable, expandable multi-modal reasoning environment.



## 🧮 **Class Architecture**

Foo includes the following classes:

| Class                   | Purpose                                                                                |
| ----------------------- | -------------------------------------------------------------------------------------- |
| **Fetch**               | Main orchestrator; initializes LLM, SQL, doc tools, embeddings, memory, and the agent. |
| **SqlQueryTool**        | Deterministic SQLite execution engine for SQL-only question answering.                 |
| **DocumentQueryTool**   | Retrieval-augmented QA engine using ChromaDB embeddings.                               |
| **ApiTool** *(stub)*    | Future-ready API integration layer.                                                    |
| **Error / ErrorDialog** | Structured error handling from the `booger` library.                                   |
| **throw_if**            | Guard clause utility for parameter validation.                                         |



##  📚 **Class Diagram**
<svg width="920" height="700" xmlns="http://www.w3.org/2000/svg">
  <style>
    .box { fill:#f4f4f4; stroke:#333; stroke-width:1.5; }
    text { font-family:Arial, sans-serif; font-size:14px; }
    .title { font-weight:bold; }
    .section { font-style:italic; }
  </style>

  <!-- Fetch -->
  <rect class="box" x="40" y="40" width="360" height="220" rx="6" />
  <text x="50" y="65" class="title">Fetch</text>

  <text x="50" y="95" class="section">Fields</text>
  <text x="60" y="115">db_uri</text>
  <text x="60" y="135">doc_paths</text>
  <text x="60" y="155">model</text>
  <text x="60" y="175">sql_tool / doc_tool / api_tool</text>

  <text x="50" y="205" class="section">Methods</text>
  <text x="60" y="225">query_sql()</text>
  <text x="60" y="245">query_docs()</text>
  <text x="60" y="265">query_chat()</text>

  <!-- SqlQueryTool -->
  <rect class="box" x="40" y="300" width="260" height="140" rx="6" />
  <text x="50" y="325" class="title">SqlQueryTool</text>
  <text x="50" y="355" class="section">Methods</text>
  <text x="60" y="375">run()</text>
  <text x="60" y="395">_execute_sql()</text>

  <!-- DocumentQueryTool -->
  <rect class="box" x="340" y="300" width="260" height="160" rx="6" />
  <text x="350" y="325" class="title">DocumentQueryTool</text>
  <text x="350" y="355" class="section">Methods</text>
  <text x="360" y="375">run()</text>
  <text x="360" y="395">_load_documents()</text>
  <text x="360" y="415">_embed_and_store()</text>
  <text x="360" y="435">_retrieve()</text>

  <!-- ApiTool -->
  <rect class="box" x="640" y="300" width="220" height="100" rx="6" />
  <text x="650" y="325" class="title">ApiTool</text>
  <text x="650" y="355" class="section">Methods</text>
  <text x="660" y="375">run()</text>

  <!-- Error -->
  <rect class="box" x="640" y="40" width="240" height="120" rx="6" />
  <text x="650" y="65" class="title">Error</text>
  <text x="650" y="95" class="section">Fields</text>
  <text x="660" y="115">message, module, cause, method</text>

  <!-- ErrorDialog -->
  <rect class="box" x="640" y="180" width="240" height="100" rx="6" />
  <text x="650" y="205" class="title">ErrorDialog</text>
  <text x="650" y="235" class="section">Methods</text>
  <text x="660" y="255">show()</text>

  <!-- throw_if -->
  <rect class="box" x="360" y="520" width="220" height="120" rx="6" />
  <text x="370" y="545" class="title">throw_if (utility)</text>
  <text x="370" y="575" class="section">Methods</text>
  <text x="380" y="595">empty()</text>
  <text x="380" y="615">null()</text>
  <text x="380" y="635">negative()</text>
</svg>


## 🧰 **System Diagram**

<svg width="760" height="520" xmlns="http://www.w3.org/2000/svg">
  <style>
    .box { fill:#e8e8e8; stroke:#333; stroke-width:1.5; }
    .arrow { stroke:#000; stroke-width:1.5; marker-end:url(#arrow); }
    text { font-family:Arial, sans-serif; font-size:14px; }
  </style>

  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="5" refY="3"
      orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#000" />
    </marker>
  </defs>

  <rect class="box" x="310" y="20" width="140" height="50" rx="6" />
  <text x="330" y="50">User Input</text>

  <rect class="box" x="300" y="120" width="160" height="60" rx="6" />
  <text x="305" y="155">DetermineQueryType</text>

  <rect class="box" x="40" y="240" width="160" height="60" rx="6" />
  <text x="75" y="275">SqlQueryTool</text>

  <rect class="box" x="240" y="240" width="160" height="60" rx="6" />
  <text x="255" y="275">DocumentQueryTool</text>

  <rect class="box" x="440" y="240" width="160" height="60" rx="6" />
  <text x="485" y="275">ChatLLM</text>

  <rect class="box" x="640" y="240" width="160" height="60" rx="6" />
  <text x="665" y="275">ReActAgent</text>

  <rect class="box" x="300" y="400" width="160" height="60" rx="6" />
  <text x="330" y="435">Final Answer</text>

  <line class="arrow" x1="380" y1="70" x2="380" y2="120" />
  <line class="arrow" x1="300" y1="150" x2="200" y2="240" />
  <line class="arrow" x1="380" y1="180" x2="320" y2="240" />
  <line class="arrow" x1="440" y1="180" x2="520" y2="240" />
  <line class="arrow" x1="460" y1="150" x2="720" y2="240" />

  <line class="arrow" x1="120" y1="300" x2="380" y2="400" />
  <line class="arrow" x1="320" y1="300" x2="380" y2="400" />
  <line class="arrow" x1="520" y1="300" x2="380" y2="400" />
  <line class="arrow" x1="720" y1="300" x2="460" y2="400" />
</svg>



## ⚙️ **Installation**

Install required dependencies:

```bash
pip install langchain chromadb numpy openai unstructured pypdf python-docx
```

If using OpenAI-compatible models:

```bash
pip install langchain-openai
export OPENAI_API_KEY="your-key"
```



## 🏗️ **Initialize**

```python
from Foo import Fetch

fetch = Fetch(
    db_uri="data/budget.sqlite",
    doc_paths=[
        "docs/PUBLIC_LAW.pdf",
        "docs/OMB_A11.md"
    ],
    model="gpt-4o-mini",
    temperature=0.2
)
```



## 🔍 **SQL Query**

```python
result = fetch.query_sql("List total obligations by fiscal year.")
print(result)
```



## 📚 **Document Retrieval**

```python
result = fetch.query_docs(
    "What does Section 1402 authorize?",
    with_sources=True
)
print(result)
```



## 🤖 **Free-Form Chat**

```python
reply = fetch.query_chat("Explain the difference between BA and OBL.")
print(reply)
```




## 🏗️ **Install & Import**

```python
!pip install langchain chromadb numpy openai unstructured python-docx pypdf

from Foo import Fetch
```


## 🔧 **Initialize Fetch**

```python
fetch = Fetch(
    db_uri="data/budget.sqlite",
    doc_paths=[
        "docs/APPROPRIATIONS_GUIDE.pdf",
        "docs/FINANCIAL_MANAGEMENT_POLICY.md"
    ],
    model="gpt-4o-mini",
    temperature=0.3
)
fetch
```



## 📜 **Run a SQL Query**

```python
fetch.query_sql("Select TAS, SUM(amount) from ledger group by TAS;")
```



## 📁 **Run a Document Retrieval Query**

```python
fetch.query_docs(
    "Summarize the funding limitations described in the guidance.",
    with_sources=True
)
```


## 🧠 **Conversational Query**

```python
fetch.query_chat("Explain SF-132 apportionments at a high level.")
```



## **Public API**

| Class                   | Method                        | Description                                   |
| ----------------------- | ----------------------------- | --------------------------------------------- |
| **Fetch**               | `query_sql()`                 | SQL-only question answering.                  |
|                         | `query_docs()`                | Vector-based retrieval QA.                    |
|                         | `query_chat()`                | Conversational LLM interaction.               |
|                         | `_init_sql_tool()`            | Configure SQLite tool.                        |
|                         | `_init_doc_tool()`            | Configure ChromaDB doc tool.                  |
|                         | `_init_api_tools()`           | Placeholder for future API tools.             |
| **SqlQueryTool**        | `run(query)`                  | Executes SQL against SQLite safely.           |
| **DocumentQueryTool**   | `run(question, with_sources)` | Performs RAG retrieval and answer generation. |
| **ApiTool**             | `run(args)`                   | Stub for future service integrations.         |
| **Error / ErrorDialog** | `show()`                      | Structured exception rendering.               |



## **Dependencies**

| Dependency                    | Purpose                                  | Required       | Notes                                     |
| ----------------------------- | ---------------------------------------- | -------------- | ----------------------------------------- |
| **langchain**                 | Agent framework, tools interface, memory | Yes            | Core orchestration library                |
| **chromadb**                  | Vector store for document embeddings     | Yes            | Stores and retrieves chunks               |
| **openai / langchain-openai** | Chat model support                       | Yes (LLM mode) | Must set `OPENAI_API_KEY`                 |
| **numpy**                     | Embedding/vector operations              | Yes            | Required internally by several components |
| **unstructured**              | Document parsing                         | Recommended    | Supports PDF, HTML, TXT, etc.             |
| **pypdf**                     | PDF ingestion                            | Recommended    | Used by Unstructured backend              |
| **python-docx**               | DOCX ingestion                           | Optional       | Included if DOCX used                     |
| **sqlite3**                   | SQL backend                              | Yes            | Included in Python standard library       |
| **booger**                    | Error and dialog handling                | Yes            | Needed for `Error` and `ErrorDialog`      |
| **tiktoken**                  | Token counting for LLMs                  | Optional       | Recommended for OpenAI models             |



## ⚖️ **License**

MIT License - [found here](https://github.com/is-leeroy-jenkins/Foo/blob/main/LICENSE.txt)




## 📝 **Author**

**Terry D. Eppler**
**Email:** [terryeppler@gmail.com](mailto:terryeppler@gmail.com)


