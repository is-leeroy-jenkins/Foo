# Processors API

## Purpose

The `processors` module provides the text-processing, natural-language processing, document parsing, and geometry-aware PDF extraction services used by the Foo application after document ingestion and before analysis, tokenization, embedding, or persistence.

The module supports:

* Text normalization and cleanup
* HTML, XML, Markdown, image-reference, and encoding removal
* Sentence, paragraph, page, word, and token segmentation
* Vocabulary and frequency-distribution generation
* TF-IDF vector creation
* Sentence-transformer encoding and semantic search
* NLTK tokenization, stemming, lemmatization, part-of-speech tagging, and named-entity recognition
* Microsoft Word document processing
* Geometry-aware PDF text extraction
* Repeated PDF header and footer detection
* PDF artifact cleanup, spacing repair, and hyphenation reconstruction
* PDF table extraction and export to CSV or Excel
* Centralized validation and exception logging through `boogr.Error` and `boogr.Logger`

## Module Constants

The module defines reusable character and pattern collections used by the processing classes:

| Constant | Purpose |
|---|---|
| `DELIMITERS` | Sentence and clause delimiters used during text segmentation and cleanup. |
| `SYMBOLS` | Configured symbols that can be removed while preserving selected sentence delimiters. |
| `ASCII_LETTERS` | Set containing uppercase and lowercase ASCII letters. |
| `DIGITS` | Set containing decimal digits. |
| `PUNCTUATION` | Set containing characters from `string.punctuation`. |
| `WHITESPACE` | Set containing supported whitespace characters. |
| `CONTROL_CHARACTERS` | Set containing ASCII control characters and the delete character. |
| `NUMERALS` | Regular-expression pattern used to identify Roman numerals. |

## Processing Architecture

The processing hierarchy is organized around the `Processor` base class.

```text
Processor
├── TextParser
├── NltkParser
├── WordParser
└── PdfParser
```

`Processor` initializes shared runtime state used by the specialized parser classes, including token collections, page buffers, chunk collections, vocabulary stores, encoding objects, lemmatizers, stemmers, and frequency distributions.

## Text Processing

The `TextParser` class provides the general-purpose text preprocessing functionality used by Foo.

### Cleanup Operations

| Method | Purpose |
|---|---|
| `collapse_whitespace()` | Replaces contiguous whitespace with one ordinary space. |
| `remove_punctuation()` | Removes non-terminal punctuation while preserving sentence-ending punctuation. |
| `reduce_repeats()` | Reduces excessive punctuation and symbol repetitions caused by OCR or extraction. |
| `normalize_text()` | Converts alphabetic content to lowercase. |
| `remove_errors()` | Repairs mojibake and removes invalid extraction characters. |
| `remove_fragments()` | Removes isolated extraction debris and unsupported private-use glyphs. |
| `remove_symbols()` | Removes configured symbols while preserving selected sentence delimiters. |
| `remove_html()` | Extracts visible text from HTML content. |
| `remove_xml()` | Extracts visible text from XML content. |
| `remove_markdown()` | Removes Markdown syntax while preserving visible content. |
| `remove_stopwords()` | Removes standalone English stopwords. |
| `remove_encodings()` | Decodes HTML entities and normalizes Unicode artifacts. |
| `remove_headers()` | Removes repeated headers and footers from fixed-length logical pages. |
| `remove_numbers()` | Removes decimal digit sequences. |
| `remove_numerals()` | Removes qualifying uppercase Roman numerals. |
| `remove_images()` | Removes Markdown, HTML, data-URI, and standalone image references. |

### Segmentation Operations

| Method | Purpose |
|---|---|
| `split_sentences()` | Segments text into sentences using NLTK. |
| `split_pages()` | Splits text into form-feed-delimited or fixed-line logical pages. |
| `split_paragraphs()` | Splits text files at blank-line paragraph boundaries. |
| `chunk_files()` | Splits files into sentence-based chunks. |
| `chunk_data()` | Builds fixed-size word chunks from a text file. |
| `chunk_datasets()` | Cleans and chunks multiple source files into Excel datasets. |
| `convert_jsonl()` | Converts text files into chunked JSONL-style output. |

### Tokenization and Vectorization

| Method | Purpose |
|---|---|
| `tiktokenize()` | Encodes text into TikToken token identifiers. |
| `create_frequency_distribution()` | Creates a token-frequency DataFrame. |
| `create_vocabulary()` | Creates a vocabulary Series from token frequencies. |
| `create_wordbag()` | Creates a bag-of-words DataFrame. |
| `create_vectors()` | Creates TF-IDF vectors while preserving source token order. |
| `encode_sentences()` | Encodes supplied tokens with a sentence-transformer model. |
| `semantic_search()` | Returns the most similar tokens for a semantic query. |

### Text Processing Example

```python
from processors import TextParser


parser = TextParser()

source_text = """
Foo   processes extracted text, removes encoding artifacts,
and prepares content for downstream analysis.
"""

collapsed_text = parser.collapse_whitespace( source_text )
normalized_text = parser.normalize_text( collapsed_text )
cleaned_text = parser.remove_encodings( normalized_text )
sentences = parser.split_sentences( cleaned_text )
```

## NLTK Processing

The `NltkParser` class provides NLTK-backed lexical and grammatical processing.

### Supported Operations

| Method | Purpose |
|---|---|
| `initialize_resources()` | Verifies and downloads required NLTK resources. |
| `word_tokenizer()` | Tokenizes lowercase text into words. |
| `sentence_tokenizer()` | Tokenizes lowercase text into sentences. |
| `word_stemmer()` | Applies Porter stemming to word tokens. |
| `word_lemmatizer()` | Applies WordNet lemmatization to word tokens. |
| `pos_tagger()` | Assigns part-of-speech tags to tokens. |
| `named_entity_recognition()` | Extracts named entities and their NLTK labels. |
| `chunk_words()` | Groups word tokens into fixed-size chunks. |
| `chunk_sentences()` | Groups sentence tokens into fixed-size chunks. |

### NLTK Example

```python
from processors import NltkParser


parser = NltkParser()

text = "The Environmental Protection Agency manages federal environmental programs."

tokens = parser.word_tokenizer( text )
tagged_tokens = parser.pos_tagger( text )
named_entities = parser.named_entity_recognition( text )
```

## Word Document Processing

The `WordParser` class provides Word document extraction and lexical analysis.

The class maintains document text, paragraphs, sentences, cleaned sentences, vocabulary, and frequency-distribution state.

### Supported Operations

| Method | Purpose |
|---|---|
| `extract_text()` | Extracts text for the selected document page. |
| `split_sentences()` | Segments extracted page text into sentences. |
| `clean_sentences()` | Removes unsupported characters and normalizes sentence text. |
| `create_vocabulary()` | Creates a stopword-filtered vocabulary. |
| `compute_frequency_distribution()` | Calculates word frequencies across cleaned sentences. |
| `summarize()` | Prints document, sentence, vocabulary, and frequency statistics. |

## PDF Processing

The `PdfParser` class provides geometry-aware PDF extraction and cleanup.

Rather than relying solely on linear PDF text extraction, the parser evaluates the location of each text block on the page. Blocks are classified as header, body, or footer content according to configurable page-height ratios.

### Geometry-Aware Extraction

`extract_pages()` reads PDF text blocks and records:

* Page number
* Block index
* Horizontal and vertical coordinates
* Block midpoint
* Header, body, or footer zone
* Extracted text
* Drop status

The resulting page dictionaries can be processed by `remove_repeats()` and reconstructed by `rebuild_pages()`.

### Header and Footer Removal

`remove_repeats()` normalizes header and footer candidates and marks repeated content for removal. The method also recognizes common page-number formats, including:

* Numeric page labels
* Roman-numeral page labels
* `Page n`
* `Page n of n`
* `p. n`

### PDF Cleanup Operations

| Method | Purpose |
|---|---|
| `geometric_extract()` | Runs geometry-aware page extraction and rebuilds the resulting text. |
| `extract_pages()` | Extracts positioned text blocks from PDF pages. |
| `remove_repeats()` | Marks repeated headers, footers, and page labels for removal. |
| `clean_artifacts()` | Removes parser markers, image markers, file paths, and PDF structural debris. |
| `repair_spacing()` | Repairs malformed punctuation and letter-spaced text. |
| `rejoin_hyphenation()` | Rejoins line-break and embedded word hyphenation. |
| `rebuild_pages()` | Reconstructs cleaned page text from retained blocks. |
| `extract_lines()` | Returns nonempty lines from geometry-aware extraction. |
| `extract_text()` | Returns rebuilt text from the selected PDF pages. |
| `extract_tables()` | Extracts PDF tables into pandas DataFrames. |

### Export Operations

| Method | Purpose |
|---|---|
| `export_csv()` | Writes each extracted table to a separate CSV file. |
| `export_text()` | Writes extracted lines to a UTF-8 text file. |
| `export_excel()` | Writes extracted tables to separate Excel worksheets. |

### Geometry-Aware PDF Example

```python
from processors import PdfParser


parser = PdfParser(
	headers=False,
	size=10,
	tables=True,
)

pages = parser.extract_pages(
	path="documents/report.pdf",
	header_ratio=0.08,
	footer_ratio=0.08,
)

cleaned_pages = parser.remove_repeats(
	pages=pages,
	minimum_repeats=3,
)

text = parser.rebuild_pages(
	pages=cleaned_pages,
	preserve_page_breaks=True,
)

text = parser.clean_artifacts( text )
text = parser.repair_spacing( text )
text = parser.rejoin_hyphenation( text )
```

### Table Extraction Example

```python
from processors import PdfParser


parser = PdfParser()

df_tables = parser.extract_tables(
	path="documents/report.pdf",
)

if df_tables:
	parser.export_excel(
		tables=df_tables,
		path="output/report_tables.xlsx",
	)
```

## Error Handling

Processing methods validate required values through `throw_if()`.

Wrapped failures are converted to the project `Error` type, populated with module, class, and method metadata, written through `Logger`, and re-raised.

```python
except Exception as e:
	exception = Error( e )
	exception.module = 'processors'
	exception.cause = 'TextParser'
	exception.method = 'remove_html( self, text: str ) -> str'
	Logger( ).write( exception )
	raise exception
```

## API Reference

::: processors
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false
      show_signature_annotations: true
      separate_signature: true
      members_order: source
      heading_level: 2
      filters:
        - "!^_"