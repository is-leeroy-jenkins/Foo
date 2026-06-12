'''
  ******************************************************************************************
      Assembly:                Foo
      Filename:                laoders.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="loaders.py" company="Terry D. Eppler">

	     loaders.py
	     Copyright ©  2025  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    loaders.py — document loading wrappers for the Foo application.

    Purpose:
        Provides loader classes that convert files, web resources, cloud objects, notebooks,
        email messages, and public-data sources into LangChain Document objects. The module
        centralizes validation, chunking, and loader-specific state so Foo can feed consistent
        document payloads into retrieval, embedding, analysis, and generation workflows.
  </summary>
  ******************************************************************************************
'''
import arxiv
import docx2txt

from boogr import Error, Logger
import config as cfg
import glob
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import (
	CSVLoader,
	Docx2txtLoader,
	PyPDFLoader,
	JSONLoader,
	GithubFileLoader,
	UnstructuredExcelLoader,
	RecursiveUrlLoader,
	WebBaseLoader,
	YoutubeLoader,
	ArxivLoader,
	WikipediaLoader,
	UnstructuredEmailLoader,
	SharePointLoader,
	GoogleDriveLoader,
	UnstructuredPowerPointLoader,
	OutlookMessageLoader,
	OneDriveLoader,
	UnstructuredXMLLoader,
	PubMedLoader,
	OpenCityDataLoader,
	NotebookLoader,
	S3FileLoader,
)

from langchain_google_community import (GCSFileLoader, SpeechToTextLoader)
from langchain_community.document_loaders import S3DirectoryLoader
from langchain_google_community import GCSDirectoryLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_core.document_loaders.base import BaseLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
import os
from pathlib import Path
import re
from typing import Optional, List, Dict, Any
import wikipedia
from lxml import etree

def throw_if( name: str, value: object ) -> None:
	"""Validate a required argument.
	
	Purpose:
		Validates that a required argument contains a usable value before the surrounding
		workflow continues. This guard centralizes early validation so loaders and helper
		routines fail with consistent, readable error messages.
	
	Args:
		name (str): Name value used by the operation.
		value (object): Value value used by the operation.
	
	Raises:
		ValueError: Raised when a required argument or option is missing or invalid."""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be None.' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty.' )

class Loader( ):
	"""Loader component.
	
	Purpose:
		Provides shared path validation, document-loading support, and document-splitting
		behavior for concrete loader wrappers. The base class centralizes common runtime state
		so specialized loaders can return LangChain Document objects through a consistent
		contract.
	
	Attributes:
		documents (Optional[List[Document]]): Documents value maintained by the Loader runtime state.
		file_path (Optional[str]): File path value maintained by the Loader runtime state.
		pattern (Optional[str]): Pattern value maintained by the Loader runtime state.
		expanded (Optional[List[str]]): Expanded value maintained by the Loader runtime state.
		candidates (Optional[List[str]]): Candidates value maintained by the Loader runtime state.
		resolved (Optional[List[str]]): Resolved value maintained by the Loader runtime state.
		loader (Optional[BaseLoader]): Loader value maintained by the Loader runtime state.
		splitter (Optional[RecursiveCharacterTextSplitter | CharacterTextSplitter]): Splitter value maintained by the Loader runtime state.
		chunk_size (Optional[int]): Chunk size value maintained by the Loader runtime state.
		overlap_amount (Optional[int]): Overlap amount value maintained by the Loader runtime state."""
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	pattern: Optional[ str ]
	expanded: Optional[ List[ str ] ]
	candidates: Optional[ List[ str ] ]
	resolved: Optional[ List[ str ] ]
	loader: Optional[ BaseLoader ]
	splitter: Optional[ RecursiveCharacterTextSplitter | CharacterTextSplitter ]
	chunk_size: Optional[ int ]
	overlap_amount: Optional[ int ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the Loader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		self.documents = [ ]
		self.candidates = [ ]
		self.resolved = [ ]
		self.expanded = [ ]
		self.file_path = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def verify_exists( self, path: str ) -> str | None:
		"""Verify exists.
		
		Purpose:
			Validates that a supplied filesystem path points to an existing file before a loader
			attempts to parse it. The method stores the validated path for later use and raises a
			structured Foo error when validation fails.
		
		Args:
			path (str): Filesystem path or object identifier used by the loader.
		
		Returns:
			String value produced by the operation.
		
		Raises:
			FileNotFoundError: Raised when a required source file cannot be found.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'path', path )
			self.file_path = path
			if not os.path.isfile( self.file_path ):
				raise FileNotFoundError( f'File not found: {self.file_path}' )
			else:
				self.file_path = path
			return self.file_path
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'Loader'
			exception.method = '_ensure_existing_file( self, path: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def resolve_paths( self, pattern: str ) -> List[ str ] | None:
		"""Resolve paths.
		
		Purpose:
			Resolves a file path or glob pattern into validated filesystem paths. The method expands
			matching files, removes duplicates through a sorted set, and raises a structured Foo
			error when no files can be resolved.
		
		Args:
			pattern (str): Filesystem path or glob pattern to resolve.
		
		Returns:
			List produced by the operation.
		
		Raises:
			FileNotFoundError: Raised when a required source file cannot be found.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'pattern', pattern )
			self.candidates.append( pattern )
			for p in self.candidates:
				if os.path.isfile( p ):
					self.resolved.append( p )
				else:
					for m in glob.glob( p ):
						if os.path.isfile( m ):
							self.resolved.append( m )
			
			if not self.resolved:
				raise FileNotFoundError( f'No files matched or existed for input: {pattern}' )
			return sorted( set( self.resolved ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'Loader'
			exception.method = 'resolve_paths( self, pattern: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def load_documents( self, path: str, encoding: Optional[ str ],
			csv_args: Optional[ Dict[ str, Any ] ],
			source_column: Optional[ str ] ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads a file into LangChain Document objects through the configured base loader path.
			The method records loader configuration and returns parsed documents for downstream
			splitting, retrieval, or generation workflows.
		
		Args:
			path (str): Filesystem path or object identifier used by the loader.
			encoding (Optional[str]): Optional text encoding used while reading the source file.
			csv_args (Optional[Dict[str, Any]]): Optional CSV parsing arguments passed to the underlying loader.
			source_column (Optional[str]): Optional source column used for metadata attribution.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			self.file_path = self.verify_exists( path )
			self.encoding = encoding
			self.csv_args = csv_args
			self.source_column = source_column
			self.loader = BaseLoader( file_path=self.file_path, encoding=self.encoding,
				csv_args=self.csv_args, source_column=self.source_column )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'CSV'
			exception.method = 'loader( )'
			Logger( ).write( exception )
			raise exception
	
	def split_documents( self, docs: List[ Document ], chunk: int = 1000, overlap: int = 200 ) -> \
			List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits existing LangChain Document objects into smaller chunks using a token-aware
			recursive text splitter. The method stores chunk settings and returns the re-chunked
			document list for retrieval and embedding workflows.
		
		Args:
			docs (List[Document]): LangChain documents to split.
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'docs', docs )
			self.documents = docs
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
				model_name='gpt-4o',
				chunk_size=self.chunk_size, overlap=self.overlap_amount )
			return self.splitter.split_documents( documents=self.documents )
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'Loader'
			exception.method = ('split_documents( self, **kwargs ) -> List[ Document ]')
			Logger( ).write( exception )
			raise exception

class TextLoader( Loader ):
	"""TextLoader component.
	
	Purpose:
		Loads plain-text files into LangChain Document objects and provides token-aware or
		character-aware splitting utilities for downstream retrieval, embedding, and generation
		workflows.
	
	Attributes:
		file_path (Optional[str]): File path value maintained by the TextLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the TextLoader runtime state.
		splitter (Optional[RecursiveCharacterTextSplitter | CharacterTextSplitter]): Splitter value maintained by the TextLoader runtime state.
		raw_text (Optional[str]): Raw text value maintained by the TextLoader runtime state.
		separator (Optional[str]): Separator value maintained by the TextLoader runtime state.
		length_function (Optional[object]): Length function value maintained by the TextLoader runtime state."""
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	splitter: Optional[ RecursiveCharacterTextSplitter | CharacterTextSplitter ]
	raw_text: Optional[ str ]
	separator: Optional[ str ]
	length_function: Optional[ object ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the TextLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.file_path = None
		self.raw_text = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.separator = "\n\n"
		self.length_function = len
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the TextLoader component. The ordered
			list supports interactive inspection, documentation surfaces, and UI code that displays
			available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'raw_text',
				'separator',
				'length_function',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'split_tokens',
				'split_chars',
		]
	
	def load( self, filepath: str ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the TextLoader workflow. The
			method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			filepath (str): Filesystem path to the file being loaded.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'filepath', filepath )
			self.file_path = self.verify_exists( filepath )
			
			with open( self.file_path, mode='r', encoding='utf-8', errors='ignore' ) as handle:
				self.raw_text = handle.read( )
			
			self.documents = [
					Document(
						page_content=self.raw_text if isinstance( self.raw_text, str ) else '',
						metadata={
								'source': os.path.basename( self.file_path ),
								'loader': 'TextLoader',
								'path': self.file_path,
						}
					)
			]
			
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'TextLoader'
			exception.method = 'load( self, filepath: str ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split_tokens( self, size: int = 1000, amount: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the TextLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			size (int): Maximum chunk size used by the splitter.
			amount (int): Overlap amount used by the splitter.
		
		Returns:
			List produced by the operation.
		
		Raises:
			ValueError: Raised when a required argument or option is missing or invalid.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if not isinstance( self.raw_text, str ) or not self.raw_text:
				raise ValueError( 'No text loaded!' )
			
			self.chunk_size = size
			self.overlap_amount = amount
			self.splitter = CharacterTextSplitter.from_tiktoken_encoder(
				encoding_name='cl100k_base',
				chunk_size=self.chunk_size,
				chunk_overlap=self.overlap_amount
			)
			
			self.documents = self.splitter.create_documents( texts=[ self.raw_text ] )
			
			for document in self.documents:
				if not isinstance( getattr( document, 'metadata', None ), dict ):
					document.metadata = { }
				
				document.metadata.setdefault( 'source',
					os.path.basename( self.file_path ) if self.file_path else '' )
				document.metadata[ 'loader' ] = 'TextLoader'
				document.metadata[ 'split_mode' ] = 'tokens'
			
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'TextLoader'
			exception.method = 'split_tokens( self, size: int=1000, amount: int=200 ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split_chars( self, size: int = 1000, amount: int = 200,
			seps: str = "\n\n" ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the TextLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			size (int): Maximum chunk size used by the splitter.
			amount (int): Overlap amount used by the splitter.
			seps (str): Seps value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			ValueError: Raised when a required argument or option is missing or invalid.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if not isinstance( self.raw_text, str ) or not self.raw_text:
				raise ValueError( 'No text loaded!' )
			
			self.chunk_size = size
			self.overlap_amount = amount
			self.separator = seps
			self.splitter = CharacterTextSplitter(
				separator=self.separator,
				chunk_size=self.chunk_size,
				chunk_overlap=self.overlap_amount,
				length_function=self.length_function
			)
			
			self.documents = self.splitter.create_documents( texts=[ self.raw_text ] )
			
			for document in self.documents:
				if not isinstance( getattr( document, 'metadata', None ), dict ):
					document.metadata = { }
				
				document.metadata.setdefault( 'source',
					os.path.basename( self.file_path ) if self.file_path else '' )
				document.metadata[ 'loader' ] = 'TextLoader'
				document.metadata[ 'split_mode' ] = 'chars'
			
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'TextLoader'
			exception.method = (
					'split_chars( self, size: int=1000, amount: int=200, '
					'seps: str="\\n\\n" ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class CsvLoader( Loader ):
	"""CsvLoader component.
	
	Purpose:
		Loads comma-separated or delimiter-separated files into LangChain Document objects. The
		loader wraps the LangChain CSV loader while preserving Foo-specific path validation,
		column handling, and split behavior.
	
	Attributes:
		loader (Optional[CSVLoader]): Loader value maintained by the CsvLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the CsvLoader runtime state.
		splitter (Optional[RecursiveCharacterTextSplitter]): Splitter value maintained by the CsvLoader runtime state.
		file_path (Optional[str]): File path value maintained by the CsvLoader runtime state.
		quote_char (Optional[str]): Quote char value maintained by the CsvLoader runtime state.
		csv_args (Optional[Dict[str, Any]]): Csv args value maintained by the CsvLoader runtime state.
		columns (Optional[List[str]]): Columns value maintained by the CsvLoader runtime state."""
	loader: Optional[ CSVLoader ]
	documents: Optional[ List[ Document ] ]
	splitter: Optional[ RecursiveCharacterTextSplitter ]
	file_path: Optional[ str ]
	quote_char: Optional[ str ]
	csv_args: Optional[ Dict[ str, Any ] ]
	columns: Optional[ List[ str ] ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the CsvLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.file_path = None
		self.columns = None
		self.csv_args = None
		self.documents = None
		self.quote_char = '"'
		self.pattern = ','
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the CsvLoader component. The ordered
			list supports interactive inspection, documentation surfaces, and UI code that displays
			available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'delimiter',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'split',
				'csv_args',
				'columns',
		]
	
	def load( self, filepath: str, columns: Optional[ List[ str ] ] = None,
			delimiter: str = ',', quotechar: str = '"' ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the CsvLoader workflow. The
			method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			filepath (str): Filesystem path to the file being loaded.
			columns (Optional[List[str]]): Optional columns used by the CSV loader.
			delimiter (str): Delimiter used by the CSV loader.
			quotechar (str): Quote character used by the CSV loader.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'filepath', filepath )
			
			self.file_path = self.verify_exists( filepath )
			self.columns = columns
			self.pattern = delimiter if isinstance( delimiter, str ) and delimiter else ','
			self.quote_char = quotechar if isinstance( quotechar, str ) and quotechar else '"'
			self.csv_args = { 'delimiter': self.pattern, 'quotechar': self.quote_char, }
			
			if isinstance( self.columns, list ) and self.columns:
				self.csv_args[ 'fieldnames' ] = self.columns
			
			self.loader = CSVLoader( file_path=self.file_path, csv_args=self.csv_args,
				content_columns=self.columns, )
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'CsvLoader'
			exception.method = (
					'load( self, filepath: str, columns: Optional[ List[ str ] ]=None, '
					'delimiter: str=",", quotechar: str=\'"\' ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, size: int = 1000, amount: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the CsvLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			size (int): Maximum chunk size used by the splitter.
			amount (int): Overlap amount used by the splitter.
		
		Returns:
			List produced by the operation.
		
		Raises:
			ValueError: Raised when a required argument or option is missing or invalid.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = size
			self.overlap_amount = amount
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'CsvLoader'
			exception.method = (
					'split( self, size: int=1000, amount: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class XmlLoader( Loader ):
	"""XmlLoader component.
	
	Purpose:
		Loads XML files into LangChain Document objects and exposes XML tree inspection helpers.
		The loader supports both document loading and XPath-style element retrieval for XML-
		oriented workflows.
	
	Attributes:
		file_path (Optional[str]): File path value maintained by the XmlLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the XmlLoader runtime state.
		loader (Optional[UnstructuredXMLLoader]): Loader value maintained by the XmlLoader runtime state.
		splitter (Optional[RecursiveCharacterTextSplitter]): Splitter value maintained by the XmlLoader runtime state.
		chunk_size (Optional[int]): Chunk size value maintained by the XmlLoader runtime state.
		overlap_amount (Optional[int]): Overlap amount value maintained by the XmlLoader runtime state.
		xml_tree (Optional[etree._ElementTree]): Xml tree value maintained by the XmlLoader runtime state.
		xml_root (Optional[etree._Element]): Xml root value maintained by the XmlLoader runtime state.
		xml_namespaces (Optional[Dict[str, str]]): Xml namespaces value maintained by the XmlLoader runtime state."""
	
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	loader: Optional[ UnstructuredXMLLoader ]
	splitter: Optional[ RecursiveCharacterTextSplitter ]
	chunk_size: Optional[ int ]
	overlap_amount: Optional[ int ]
	xml_tree: Optional[ etree._ElementTree ]
	xml_root: Optional[ etree._Element ]
	xml_namespaces: Optional[ Dict[ str, str ] ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the XmlLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.loader = None
		self.splitter = None
		self.chunk_size = None
		self.overlap_amount = None
		self.xml_tree = None
		self.xml_root = None
		self.xml_namespaces = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the XmlLoader component. The ordered
			list supports interactive inspection, documentation surfaces, and UI code that displays
			available attributes and methods.
		
		Returns:
			List produced by the operation."""
		return [
				"loader",
				"documents",
				"splitter",
				"file_path",
				"expanded",
				"candidates",
				"resolved",
				"chunk_size",
				"overlap_amount",
				"xml_tree",
				"xml_root",
				"xml_namespaces",
				"verify_exists",
				"resolve_paths",
				"split_documents",
				"load",
				"split",
				"load_tree",
				"get_elements",
		]
	
	def load( self, filepath: str ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the XmlLoader workflow. The
			method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			filepath (str): Filesystem path to the file being loaded.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			self.file_path = self.verify_exists( filepath )
			self.loader = UnstructuredXMLLoader( file_path=self.file_path, mode="elements" )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = "chonky"
			exception.cause = "XmlLoader"
			exception.method = "load(self, filepath: str)"
			Logger( ).write( exception )
			raise exception
	
	def split( self, size: int = 1000, amount: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the XmlLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			size (int): Maximum chunk size used by the splitter.
			amount (int): Overlap amount used by the splitter.
		
		Returns:
			List produced by the operation.
		
		Raises:
			ValueError: Raised when a required argument or option is missing or invalid.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if self.documents is None:
				raise ValueError( "No documents loaded via load()." )
			self.chunk_size = size
			self.overlap_amount = amount
			split_docs = self.split_documents( docs=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			
			return split_docs
		except Exception as e:
			exception = Error( e )
			exception.module = "chonky"
			exception.cause = "XmlLoader"
			exception.method = "split(self, size: int = 1000, amount: int = 200)"
			Logger( ).write( exception )
			raise exception
	
	def load_tree( self, filepath: str ) -> etree._ElementTree | None:
		"""Load documents.
		
		Purpose:
			Loads source content for the XmlLoader workflow using the specialized Load tree path.
			The method validates input settings, delegates to the underlying loader implementation,
			and returns LangChain Document objects.
		
		Args:
			filepath (str): Filesystem path to the file being loaded.
		
		Returns:
			Value produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			self.file_path = self.verify_exists( filepath )
			parser = etree.XMLParser( recover=True, remove_comments=True, remove_blank_text=True )
			self.xml_tree = etree.parse( self.file_path, parser )
			self.xml_root = self.xml_tree.getroot( )
			self.xml_namespaces = {
					prefix if prefix is not None else "default": uri
					for prefix, uri in (self.xml_root.nsmap or { }).items( )
			}
			
			return self.xml_tree
		except Exception as e:
			exception = Error( e )
			exception.module = "chonky"
			exception.cause = "XmlLoader"
			exception.method = "load_tree(self, filepath: str)"
			Logger( ).write( exception )
			raise exception
	
	def get_elements( self, xpath: str ) -> List[ etree._Element ] | None:
		"""Get elements.
		
		Purpose:
			Retrieves XML elements from the loaded XML tree using the supplied XPath expression. The
			method stores the XPath value and returns matching elements for callers that need direct
			XML inspection.
		
		Args:
			xpath (str): XPath expression used to select XML elements.
		
		Returns:
			List produced by the operation.
		
		Raises:
			ValueError: Raised when a required argument or option is missing or invalid.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if self.xml_root is None:
				raise ValueError( "XML tree not loaded. Call load_tree() first." )
			elements = self.xml_root.xpath( xpath, namespaces=self.xml_namespaces )
			return list( elements )
		except Exception as e:
			exception = Error( e )
			exception.module = "chonky"
			exception.cause = "XmlLoader"
			exception.method = "get_elements(self, xpath: str)"
			Logger( ).write( exception )
			raise exception

class WebLoader( Loader ):
	"""WebLoader component.
	
	Purpose:
		Loads web pages into LangChain Document objects through URL-based loading workflows. The
		loader supports direct page loading, recursive URL traversal, and chunking for retrieval
		pipelines.
	
	Attributes:
		loader (Optional[RecursiveUrlLoader | WebBaseLoader]): Loader value maintained by the WebLoader runtime state.
		url (Optional[str]): Url value maintained by the WebLoader runtime state.
		web_paths (Optional[str | List[str]]): Web paths value maintained by the WebLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the WebLoader runtime state.
		file_path (Optional[str]): File path value maintained by the WebLoader runtime state.
		max_depth (Optional[int]): Max depth value maintained by the WebLoader runtime state.
		timeout (Optional[int]): Timeout value maintained by the WebLoader runtime state.
		ignore (Optional[bool]): Ignore value maintained by the WebLoader runtime state.
		with_progress (Optional[bool]): With progress value maintained by the WebLoader runtime state.
		recursive (Optional[bool]): Recursive value maintained by the WebLoader runtime state.
		prevent_outside (Optional[bool]): Prevent outside value maintained by the WebLoader runtime state."""
	loader: Optional[ RecursiveUrlLoader | WebBaseLoader ]
	url: Optional[ str ]
	web_paths: Optional[ str | List[ str ] ]
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	max_depth: Optional[ int ]
	timeout: Optional[ int ]
	ignore: Optional[ bool ]
	with_progress: Optional[ bool ]
	recursive: Optional[ bool ]
	prevent_outside: Optional[ bool ]
	
	def __init__( self, recursive: bool = False, max_depth: int = 2,
			prevent_outside: bool = True, timeout: int = 10,
			ignore: bool = True, progress: bool = True ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the WebLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract.
		
		Args:
			recursive (bool): Whether recursive loading is enabled.
			max_depth (int): Maximum traversal depth for recursive loading.
			prevent_outside (bool): Whether recursive loading is restricted to the source domain.
			timeout (int): Timeout value used by network-backed loading operations.
			ignore (bool): Optional ignore pattern or ignore setting passed to the loader.
			progress (bool): Whether progress reporting is enabled where supported."""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.url = None
		self.web_paths = None
		self.max_depth = max_depth
		self.timeout = timeout
		self.ignore = ignore
		self.with_progress = progress
		self.recursive = recursive
		self.prevent_outside = prevent_outside
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the WebLoader component. The ordered
			list supports interactive inspection, documentation surfaces, and UI code that displays
			available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'url',
				'web_paths',
				'max_depth',
				'timeout',
				'ignore',
				'with_progress',
				'recursive',
				'prevent_outside',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'load_pages',
				'split',
		]
	
	def load( self, urls: str | List[ str ], depth: int = 2, timeout: int = 10,
			ignore: bool = True, progress: bool = True,
			prevent_outside: bool = True ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the WebLoader workflow. The
			method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			urls (str | List[str]): One or more URLs used by the web-oriented loader.
			depth (int): Depth value used by the operation.
			timeout (int): Timeout value used by network-backed loading operations.
			ignore (bool): Optional ignore pattern or ignore setting passed to the loader.
			progress (bool): Whether progress reporting is enabled where supported.
			prevent_outside (bool): Whether recursive loading is restricted to the source domain.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if self.recursive:
				return self.load_recursive(
					urls=urls,
					depth=depth,
					timeout=timeout,
					ignore=ignore,
					prevent_outside=prevent_outside
				)
			
			return self.load_pages(
				urls=urls,
				timeout=timeout,
				ignore=ignore,
				progress=progress
			)
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'WebLoader'
			exception.method = (
					'load( self, urls: str | List[ str ], depth: int=2, '
					'timeout: int=10, ignore: bool=True, progress: bool=True, '
					'prevent_outside: bool=True ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def load_pages( self, urls: str | List[ str ], timeout: int = 10,
			ignore: bool = True, progress: bool = True ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content for the WebLoader workflow using the specialized Load pages path.
			The method validates input settings, delegates to the underlying loader implementation,
			and returns LangChain Document objects.
		
		Args:
			urls (str | List[str]): One or more URLs used by the web-oriented loader.
			timeout (int): Timeout value used by network-backed loading operations.
			ignore (bool): Optional ignore pattern or ignore setting passed to the loader.
			progress (bool): Whether progress reporting is enabled where supported.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'urls', urls )
			
			self.web_paths = [ urls ] if isinstance( urls, str ) else list( urls )
			self.timeout = timeout
			self.ignore = ignore
			self.with_progress = progress
			
			self.loader = WebBaseLoader(
				web_paths=self.web_paths,
				show_progress=self.with_progress,
				continue_on_failure=self.ignore,
				requests_kwargs={ 'timeout': self.timeout }
			)
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'WebLoader'
			exception.method = (
					'load_pages( self, urls: str | List[ str ], timeout: int=10, '
					'ignore: bool=True, progress: bool=True ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def load_recursive( self, urls: str | List[ str ], depth: int = 2,
			timeout: int = 10, ignore: bool = True,
			prevent_outside: bool = True ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content for the WebLoader workflow using the specialized Load recursive
			path. The method validates input settings, delegates to the underlying loader
			implementation, and returns LangChain Document objects.
		
		Args:
			urls (str | List[str]): One or more URLs used by the web-oriented loader.
			depth (int): Depth value used by the operation.
			timeout (int): Timeout value used by network-backed loading operations.
			ignore (bool): Optional ignore pattern or ignore setting passed to the loader.
			prevent_outside (bool): Whether recursive loading is restricted to the source domain.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'urls', urls )
			
			self.url = urls[ 0 ] if isinstance( urls, list ) else urls
			self.max_depth = depth
			self.timeout = timeout
			self.ignore = ignore
			self.prevent_outside = prevent_outside
			
			self.loader = RecursiveUrlLoader(
				url=self.url,
				max_depth=self.max_depth,
				timeout=self.timeout,
				continue_on_failure=self.ignore,
				prevent_outside=self.prevent_outside
			)
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'WebLoader'
			exception.method = (
					'load_recursive( self, urls: str | List[ str ], depth: int=2, '
					'timeout: int=10, ignore: bool=True, prevent_outside: bool=True ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the WebLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			ValueError: Raised when a required argument or option is missing or invalid.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			return self.split_documents(
				docs=self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'WebLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class PdfLoader( Loader ):
	"""PdfLoader component.
	
	Purpose:
		Loads PDF files into LangChain Document objects using configurable extraction options.
		The loader normalizes extraction settings, manages image and table-related options, and
		supports chunking loaded content.
	
	Attributes:
		loader (Optional[PyPDFLoader]): Loader value maintained by the PdfLoader runtime state.
		file_path (Optional[str]): File path value maintained by the PdfLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the PdfLoader runtime state.
		mode (Optional[str]): Mode value maintained by the PdfLoader runtime state.
		extraction (Optional[str]): Extraction value maintained by the PdfLoader runtime state.
		include_images (Optional[bool]): Include images value maintained by the PdfLoader runtime state.
		image_format (Optional[str]): Image format value maintained by the PdfLoader runtime state.
		custom_delimiter (Optional[str]): Custom delimiter value maintained by the PdfLoader runtime state.
		image_parser (Optional[RapidOCRBlobParser]): Image parser value maintained by the PdfLoader runtime state."""
	loader: Optional[ PyPDFLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	extraction: Optional[ str ]
	include_images: Optional[ bool ]
	image_format: Optional[ str ]
	custom_delimiter: Optional[ str ]
	image_parser: Optional[ RapidOCRBlobParser ]
	
	def __init__( self, size: int = 1000, overlap: int = 150,
			has_tables: bool = True, include: bool = True ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the PdfLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract.
		
		Args:
			size (int): Maximum chunk size used by the splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
			has_tables (bool): Has tables value used by the operation.
			include (bool): Include value used by the operation."""
		super( ).__init__( )
		self.enable_tables = has_tables
		self.include_images = include
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = size
		self.overlap_amount = overlap
		self.loader = None
		self.mode = None
		self.extraction = None
		self.image_format = None
		self.custom_delimiter = None
		self.image_parser = None
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the PdfLoader component. The ordered
			list supports interactive inspection, documentation surfaces, and UI code that displays
			available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'mode',
				'extraction',
				'include_images',
				'image_format',
				'custom_delimiter',
				'image_parser',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'split',
				'mode_options',
				'extraction_options',
				'image_options',
		]
	
	@property
	def mode_options( self ):
		"""Return supported options.
		
		Purpose:
			Returns the supported option values exposed by the PdfLoader component. The values are
			used by UI selectors and validation logic to keep runtime choices aligned with the
			loader implementation.
		
		Returns:
			Value produced by the operation."""
		return [ 'page', 'single' ]
	
	@property
	def extraction_options( self ):
		"""Return supported options.
		
		Purpose:
			Returns the supported option values exposed by the PdfLoader component. The values are
			used by UI selectors and validation logic to keep runtime choices aligned with the
			loader implementation.
		
		Returns:
			Value produced by the operation."""
		return [ 'plain', 'layout' ]
	
	@property
	def image_options( self ):
		"""Return supported options.
		
		Purpose:
			Returns the supported option values exposed by the PdfLoader component. The values are
			used by UI selectors and validation logic to keep runtime choices aligned with the
			loader implementation.
		
		Returns:
			Value produced by the operation."""
		return [ 'html-img', 'markdown-img', 'text-img' ]
	
	def _normalize_mode( self, mode: str ) -> str:
		"""Normalize an option value.
		
		Purpose:
			Normalizes a user-supplied option for the PdfLoader workflow into a supported canonical
			value. The method accepts display-oriented or optional input and returns the value
			expected by the underlying loader implementation.
		
		Args:
			mode (str): Loader mode or parsing mode requested by the caller.
		
		Returns:
			String value produced by the operation."""
		value = mode.strip( ).lower( ) if isinstance( mode, str ) else 'single'
		
		if value == 'elements':
			return 'page'
		
		if value not in self.mode_options:
			return 'single'
		
		return value
	
	def _normalize_extraction( self, extract: str ) -> str:
		"""Normalize an option value.
		
		Purpose:
			Normalizes a user-supplied option for the PdfLoader workflow into a supported canonical
			value. The method accepts display-oriented or optional input and returns the value
			expected by the underlying loader implementation.
		
		Args:
			extract (str): Extract value used by the operation.
		
		Returns:
			String value produced by the operation."""
		value = extract.strip( ).lower( ) if isinstance( extract, str ) else 'plain'
		
		if value == 'ocr':
			return 'layout'
		
		if value not in self.extraction_options:
			return 'plain'
		
		return value
	
	def _normalize_image_format( self, format: str ) -> str:
		"""Normalize an option value.
		
		Purpose:
			Normalizes a user-supplied option for the PdfLoader workflow into a supported canonical
			value. The method accepts display-oriented or optional input and returns the value
			expected by the underlying loader implementation.
		
		Args:
			format (str): Format value used by the operation.
		
		Returns:
			String value produced by the operation."""
		value = format.strip( ).lower( ) if isinstance( format, str ) else 'markdown-img'
		
		if value == 'text':
			return 'markdown-img'
		
		if value not in self.image_options:
			return 'markdown-img'
		
		return value
	
	def load( self, filepath: str, mode: str = 'single', extract: str = 'plain',
			include: bool = False, format: str = 'markdown-img' ) -> List[ Document ]:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the PdfLoader workflow. The
			method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			filepath (str): Filesystem path to the file being loaded.
			mode (str): Loader mode or parsing mode requested by the caller.
			extract (str): Extract value used by the operation.
			include (bool): Include value used by the operation.
			format (str): Format value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'path', filepath )
			
			self.file_path = self.verify_exists( filepath )
			self.mode = self._normalize_mode( mode )
			self.extraction = self._normalize_extraction( extract )
			self.include_images = include
			self.image_format = self._normalize_image_format( format )
			
			if self.include_images:
				self.image_parser = RapidOCRBlobParser( )
				self.loader = PyPDFLoader(
					file_path=self.file_path,
					mode=self.mode,
					extraction_mode=self.extraction,
					extract_images=self.include_images,
					images_inner_format=self.image_format,
					images_parser=self.image_parser
				)
			else:
				self.loader = PyPDFLoader(
					file_path=self.file_path,
					mode=self.mode,
					extraction_mode=self.extraction
				)
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'PdfLoader'
			exception.method = (
					'load( self, filepath: str, mode: str="single", '
					'extract: str="plain", include: bool=False, '
					'format: str="markdown-img" ) -> List[ Document ]'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the PdfLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			ValueError: Raised when a required argument or option is missing or invalid.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'PdfLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class ExcelLoader( Loader ):
	"""ExcelLoader component.
	
	Purpose:
		Loads Excel workbooks into LangChain Document objects and supports configurable
		extraction modes. The loader wraps spreadsheet loading while preserving common Foo path
		validation and splitting behavior.
	
	Attributes:
		loader (Optional[UnstructuredExcelLoader]): Loader value maintained by the ExcelLoader runtime state.
		file_path (Optional[str]): File path value maintained by the ExcelLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the ExcelLoader runtime state.
		mode (Optional[str]): Mode value maintained by the ExcelLoader runtime state.
		has_headers (Optional[bool]): Has headers value maintained by the ExcelLoader runtime state."""
	loader: Optional[ UnstructuredExcelLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	has_headers: Optional[ bool ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the ExcelLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
		self.has_headers = True
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the ExcelLoader component. The
			ordered list supports interactive inspection, documentation surfaces, and UI code that
			displays available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'mode',
				'has_headers',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'split',
				'mode_options',
		]
	
	@property
	def mode_options( self ):
		"""Return supported options.
		
		Purpose:
			Returns the supported option values exposed by the ExcelLoader component. The values are
			used by UI selectors and validation logic to keep runtime choices aligned with the
			loader implementation.
		
		Returns:
			Value produced by the operation."""
		return [ 'single', 'elements' ]
	
	def _normalize_mode( self, mode: str ) -> str:
		"""Normalize an option value.
		
		Purpose:
			Normalizes a user-supplied option for the ExcelLoader workflow into a supported
			canonical value. The method accepts display-oriented or optional input and returns the
			value expected by the underlying loader implementation.
		
		Args:
			mode (str): Loader mode or parsing mode requested by the caller.
		
		Returns:
			String value produced by the operation."""
		value = mode.strip( ).lower( ) if isinstance( mode, str ) else 'single'
		
		if value in [ 'page', 'paged' ]:
			return 'elements'
		
		if value not in self.mode_options:
			return 'single'
		
		return value
	
	def load( self, path: str, mode: str = 'single',
			has_headers: bool = True ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the ExcelLoader workflow. The
			method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			path (str): Filesystem path or object identifier used by the loader.
			mode (str): Loader mode or parsing mode requested by the caller.
			has_headers (bool): Has headers value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'path', path )
			
			self.file_path = self.verify_exists( path )
			self.mode = self._normalize_mode( mode )
			self.has_headers = has_headers
			
			self.loader = UnstructuredExcelLoader(
				file_path=self.file_path,
				mode=self.mode
			)
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'ExcelLoader'
			exception.method = (
					'load( self, path: str, mode: str="single", '
					'has_headers: bool=True ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the ExcelLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			ValueError: Raised when a required argument or option is missing or invalid.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'ExcelLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class WordLoader( Loader ):
	"""WordLoader component.
	
	Purpose:
		Loads Microsoft Word documents into LangChain Document objects. The loader provides a
		document-specific wrapper around Word parsing and shared chunking behavior.
	
	Attributes:
		loader (Optional[Docx2txtLoader]): Loader value maintained by the WordLoader runtime state.
		file_path (Optional[str]): File path value maintained by the WordLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the WordLoader runtime state."""
	loader: Optional[ Docx2txtLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the WordLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.documents = None
		self.file_path = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the WordLoader component. The ordered
			list supports interactive inspection, documentation surfaces, and UI code that displays
			available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, path: str ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the WordLoader workflow. The
			method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			path (str): Filesystem path or object identifier used by the loader.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = Docx2txtLoader( self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'WordLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the WordLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			ValueError: Raised when a required argument or option is missing or invalid.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			_splits = self.split_documents( docs=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return _splits
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'WordLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class MarkdownLoader( Loader ):
	"""MarkdownLoader component.
	
	Purpose:
		Loads Markdown files into LangChain Document objects with configurable parsing modes.
		The loader prepares Markdown content for analysis, retrieval, and generation workflows.
	
	Attributes:
		loader (Optional[UnstructuredMarkdownLoader]): Loader value maintained by the MarkdownLoader runtime state.
		file_path (str | None): File path value maintained by the MarkdownLoader runtime state.
		documents (List[Document] | None): Documents value maintained by the MarkdownLoader runtime state.
		mode (Optional[str]): Mode value maintained by the MarkdownLoader runtime state."""
	loader: Optional[ UnstructuredMarkdownLoader ]
	file_path: str | None
	documents: List[ Document ] | None
	mode: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the MarkdownLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the MarkdownLoader component. The
			ordered list supports interactive inspection, documentation surfaces, and UI code that
			displays available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'mode',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'split',
				'mode_options',
		]
	
	@property
	def mode_options( self ):
		"""Return supported options.
		
		Purpose:
			Returns the supported option values exposed by the MarkdownLoader component. The values
			are used by UI selectors and validation logic to keep runtime choices aligned with the
			loader implementation.
		
		Returns:
			Value produced by the operation."""
		return [ 'single', 'elements' ]
	
	def _normalize_mode( self, mode: str ) -> str:
		"""Normalize an option value.
		
		Purpose:
			Normalizes a user-supplied option for the MarkdownLoader workflow into a supported
			canonical value. The method accepts display-oriented or optional input and returns the
			value expected by the underlying loader implementation.
		
		Args:
			mode (str): Loader mode or parsing mode requested by the caller.
		
		Returns:
			String value produced by the operation."""
		value = mode.strip( ).lower( ) if isinstance( mode, str ) else 'single'
		
		if value in [ 'page', 'paged' ]:
			return 'elements'
		
		if value not in self.mode_options:
			return 'single'
		
		return value
	
	def load( self, path: str, mode: str = 'single' ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the MarkdownLoader workflow.
			The method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			path (str): Filesystem path or object identifier used by the loader.
			mode (str): Loader mode or parsing mode requested by the caller.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.mode = self._normalize_mode( mode )
			self.loader = UnstructuredMarkdownLoader(
				file_path=self.file_path,
				mode=self.mode
			)
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'MarkdownLoader'
			exception.method = (
					'load( self, path: str, mode: str="single" ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the MarkdownLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			ValueError: Raised when a required argument or option is missing or invalid.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			_documents = self.split_documents(
				docs=self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return _documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'MarkdownLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class HtmlLoader( Loader ):
	"""HtmlLoader component.
	
	Purpose:
		Loads local HTML files into LangChain Document objects. The loader provides a file-based
		HTML ingestion path separate from URL-based web crawling.
	
	Attributes:
		loader (Optional[UnstructuredHTMLLoader]): Loader value maintained by the HtmlLoader runtime state.
		file_path (str | None): File path value maintained by the HtmlLoader runtime state.
		documents (List[Document] | None): Documents value maintained by the HtmlLoader runtime state."""
	loader: Optional[ UnstructuredHTMLLoader ]
	file_path: str | None
	documents: List[ Document ] | None
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the HtmlLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the HtmlLoader component. The ordered
			list supports interactive inspection, documentation surfaces, and UI code that displays
			available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, path: str ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the HtmlLoader workflow. The
			method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			path (str): Filesystem path or object identifier used by the loader.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = UnstructuredHTMLLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'HTML'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the HtmlLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			ValueError: Raised when a required argument or option is missing or invalid.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'HtmlLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class JsonLoader( Loader ):
	"""JsonLoader component.
	
	Purpose:
		Loads JSON files into LangChain Document objects using configurable schema and content-
		key settings. The loader supports JSON lines and text-mode options used by LangChain
		JSON loading.
	
	Attributes:
		loader (Optional[JSONLoader]): Loader value maintained by the JsonLoader runtime state.
		file_path (Optional[str]): File path value maintained by the JsonLoader runtime state.
		jq_schema (Optional[str]): Jq schema value maintained by the JsonLoader runtime state.
		content_key (Optional[str]): Content key value maintained by the JsonLoader runtime state.
		text_content (Optional[bool]): Text content value maintained by the JsonLoader runtime state.
		json_lines (Optional[bool]): Json lines value maintained by the JsonLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the JsonLoader runtime state."""
	loader: Optional[ JSONLoader ]
	file_path: Optional[ str ]
	jq_schema: Optional[ str ]
	content_key: Optional[ str ]
	text_content: Optional[ bool ]
	json_lines: Optional[ bool ]
	documents: Optional[ List[ Document ] ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the JsonLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.jq_schema = '.'
		self.content_key = None
		self.text_content = True
		self.json_lines = False
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the JsonLoader component. The ordered
			list supports interactive inspection, documentation surfaces, and UI code that displays
			available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'jq_schema',
				'content_key',
				'text_content',
				'json_lines',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'split',
		]
	
	def load( self, filepath: str, jq_schema: str = '.',
			content_key: Optional[ str ] = None, is_text: bool = True,
			is_lines: bool = False ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the JsonLoader workflow. The
			method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			filepath (str): Filesystem path to the file being loaded.
			jq_schema (str): Jq schema value used by the operation.
			content_key (Optional[str]): Content key value used by the operation.
			is_text (bool): Is text value used by the operation.
			is_lines (bool): Is lines value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'filepath', filepath )
			self.file_path = self.verify_exists( filepath )
			self.jq_schema = jq_schema if isinstance( jq_schema,
				str ) and jq_schema.strip( ) else '.'
			self.content_key = (content_key.strip( )
			                    if isinstance( content_key,
				str ) and content_key.strip( ) else None)
			self.text_content = bool( is_text )
			self.json_lines = bool( is_lines )
			kwargs = {
					'file_path': self.file_path,
					'jq_schema': self.jq_schema,
					'text_content': self.text_content,
					'json_lines': self.json_lines,
			}
			
			if self.content_key:
				kwargs[ 'content_key' ] = self.content_key
			
			self.loader = JSONLoader( **kwargs )
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'JsonLoader'
			exception.method = (
					'load( self, filepath: str, jq_schema: str=".", '
					'content_key: Optional[ str ]=None, is_text: bool=True, '
					'is_lines: bool=False ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the JsonLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			ValueError: Raised when a required argument or option is missing or invalid.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				docs=self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'JsonLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class ArXivLoader( Loader ):
	"""ArXivLoader component.
	
	Purpose:
		Loads arXiv search results into LangChain Document objects. The loader wraps arXiv
		retrieval and prepares scholarly metadata and summaries for downstream analysis.
	
	Attributes:
		loader (Optional[ArxivLoader]): Loader value maintained by the ArXivLoader runtime state.
		file_path (Optional[str]): File path value maintained by the ArXivLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the ArXivLoader runtime state.
		max_documents (Optional[int]): Max documents value maintained by the ArXivLoader runtime state.
		max_characters (Optional[int]): Max characters value maintained by the ArXivLoader runtime state.
		include_metadata (Optional[bool]): Include metadata value maintained by the ArXivLoader runtime state.
		query (Optional[str]): Query value maintained by the ArXivLoader runtime state."""
	loader: Optional[ ArxivLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	max_documents: Optional[ int ]
	max_characters: Optional[ int ]
	include_metadata: Optional[ bool ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the ArXivLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.max_documents = None
		self.max_characters = None
		self.include_metadata = False
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the ArXivLoader component. The
			ordered list supports interactive inspection, documentation surfaces, and UI code that
			displays available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'max_documents',
		         'max_characters',
		         'include_metadata',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, query: str, max_chars: int = 1000 ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the ArXivLoader workflow. The
			method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			query (str): Search query or lookup value submitted to the loader.
			max_chars (int): Max chars value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'query', query )
			self.query = query
			self.max_characters = max_chars
			self.loader = ArxivLoader( query=self.query, doc_content_chars_max=self.max_characters )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'ArxivLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the ArXivLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			ValueError: Raised when a required argument or option is missing or invalid.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'ArxivLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class WikiLoader( Loader ):
	"""WikiLoader component.
	
	Purpose:
		Loads Wikipedia results into LangChain Document objects. The loader wraps Wikipedia
		retrieval settings such as document count, character limits, and metadata inclusion.
	
	Attributes:
		loader (Optional[WikipediaLoader]): Loader value maintained by the WikiLoader runtime state.
		file_path (Optional[str]): File path value maintained by the WikiLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the WikiLoader runtime state.
		query (Optional[str]): Query value maintained by the WikiLoader runtime state.
		max_documents (Optional[int]): Max documents value maintained by the WikiLoader runtime state.
		max_characters (Optional[int]): Max characters value maintained by the WikiLoader runtime state.
		include_all (Optional[bool]): Include all value maintained by the WikiLoader runtime state."""
	loader: Optional[ WikipediaLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	max_documents: Optional[ int ]
	max_characters: Optional[ int ]
	include_all: Optional[ bool ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the WikiLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.max_documents = None
		self.max_characters = None
		self.include_all = False
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the WikiLoader component. The ordered
			list supports interactive inspection, documentation surfaces, and UI code that displays
			available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'max_documents',
				'max_characters',
				'include_all',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'split',
		]
	
	def load( self, query: str, max_docs: int = 25, max_chars: int = 4000,
			include_all: bool = False ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the WikiLoader workflow. The
			method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			query (str): Search query or lookup value submitted to the loader.
			max_docs (int): Max docs value used by the operation.
			max_chars (int): Max chars value used by the operation.
			include_all (bool): Include all value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'query', query )
			
			self.query = query
			self.max_documents = max_docs
			self.max_characters = max_chars
			self.include_all = include_all
			
			self.loader = WikipediaLoader(
				query=self.query,
				load_max_docs=self.max_documents,
				load_all_available_meta=self.include_all,
				doc_content_chars_max=self.max_characters
			)
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'WikiLoader'
			exception.method = (
					'load( self, query: str, max_docs: int=25, max_chars: int=4000, '
					'include_all: bool=False ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the WikiLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			ValueError: Raised when a required argument or option is missing or invalid.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'WikiLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class GithubLoader( Loader ):
	"""GithubLoader component.
	
	Purpose:
		Loads files from GitHub repositories into LangChain Document objects. The loader
		supports repository, branch, file type, and access-token settings used by the GitHub
		file loader.
	
	Attributes:
		loader (Optional[GithubFileLoader]): Loader value maintained by the GithubLoader runtime state.
		file_path (Optional[str]): File path value maintained by the GithubLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the GithubLoader runtime state.
		repo (Optional[str]): Repo value maintained by the GithubLoader runtime state.
		branch (Optional[str]): Branch value maintained by the GithubLoader runtime state.
		access_token (Optional[str]): Access token value maintained by the GithubLoader runtime state.
		github_url (Optional[str]): Github url value maintained by the GithubLoader runtime state.
		file_filter (Optional[object]): File filter value maintained by the GithubLoader runtime state.
		pattern (Optional[str]): Pattern value maintained by the GithubLoader runtime state."""
	loader: Optional[ GithubFileLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	repo: Optional[ str ]
	branch: Optional[ str ]
	access_token: Optional[ str ]
	github_url: Optional[ str ]
	file_filter: Optional[ object ]
	pattern: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the GithubLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.github_url = None
		self.repo = None
		self.branch = None
		self.access_token = None
		self.file_filter = None
		self.pattern = None
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the GithubLoader component. The
			ordered list supports interactive inspection, documentation surfaces, and UI code that
			displays available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'repo',
				'branch',
				'access_token',
				'github_url',
				'file_filter',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'split',
		]
	
	def load( self, url: str, repo: str, branch: str, filetype: str = '.md',
			access_token: Optional[ str ] = None ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the GithubLoader workflow. The
			method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			url (str): URL used by the web-oriented loader.
			repo (str): Repo value used by the operation.
			branch (str): Branch value used by the operation.
			filetype (str): Filetype value used by the operation.
			access_token (Optional[str]): Access token value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'url', url )
			throw_if( 'repo', repo )
			throw_if( 'branch', branch )
			
			self.github_url = url
			self.repo = repo
			self.branch = branch
			self.access_token = access_token.strip( ) if isinstance( access_token,
				str ) and access_token.strip( ) else None
			self.pattern = filetype.strip( ) if isinstance( filetype,
				str ) and filetype.strip( ) else '.md'
			self.file_filter = lambda file_path: file_path.endswith( self.pattern )
			
			kwargs = {
					'repo': self.repo,
					'branch': self.branch,
					'github_api_url': self.github_url,
					'file_filter': self.file_filter,
			}
			
			if self.access_token:
				kwargs[ 'access_token' ] = self.access_token
			
			self.loader = GithubFileLoader( **kwargs )
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'GithubLoader'
			exception.method = (
					'load( self, url: str, repo: str, branch: str, '
					'filetype: str=".md", access_token: Optional[ str ]=None ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the GithubLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			ValueError: Raised when a required argument or option is missing or invalid.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'GithubLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class PowerPointLoader( Loader ):
	"""PowerPointLoader component.
	
	Purpose:
		Loads PowerPoint presentation files into LangChain Document objects. The loader supports
		single-file and multiple-file ingestion and chunking for slide content.
	
	Attributes:
		loader (Optional[UnstructuredPowerPointLoader]): Loader value maintained by the PowerPointLoader runtime state.
		file_path (Optional[str]): File path value maintained by the PowerPointLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the PowerPointLoader runtime state.
		mode (Optional[str]): Mode value maintained by the PowerPointLoader runtime state.
		query (Optional[str]): Query value maintained by the PowerPointLoader runtime state."""
	loader: Optional[ UnstructuredPowerPointLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the PowerPointLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the PowerPointLoader component. The
			ordered list supports interactive inspection, documentation surfaces, and UI code that
			displays available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'query',
				'mode',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'load_multiple',
				'split',
		]
	
	def _normalize_mode( self, mode: str ) -> str:
		"""Normalize an option value.
		
		Purpose:
			Normalizes a user-supplied option for the PowerPointLoader workflow into a supported
			canonical value. The method accepts display-oriented or optional input and returns the
			value expected by the underlying loader implementation.
		
		Args:
			mode (str): Loader mode or parsing mode requested by the caller.
		
		Returns:
			String value produced by the operation."""
		value = mode.strip( ).lower( ) if isinstance( mode, str ) else 'single'
		
		if value == 'multiple':
			return 'elements'
		
		if value not in [ 'single', 'elements' ]:
			return 'single'
		
		return value
	
	def load( self, path: str, mode: str = 'single' ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the PowerPointLoader workflow.
			The method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			path (str): Filesystem path or object identifier used by the loader.
			mode (str): Loader mode or parsing mode requested by the caller.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'path', path )
			
			self.file_path = self.verify_exists( path )
			self.mode = self._normalize_mode( mode )
			self.loader = UnstructuredPowerPointLoader(
				file_path=self.file_path,
				mode=self.mode
			)
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'PowerPointLoader'
			exception.method = (
					'load( self, path: str, mode: str="single" ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def load_multiple( self, path: str ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content for the PowerPointLoader workflow using the specialized Load
			multiple path. The method validates input settings, delegates to the underlying loader
			implementation, and returns LangChain Document objects.
		
		Args:
			path (str): Filesystem path or object identifier used by the loader.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			return self.load( path, mode='elements' )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'PowerPointLoader'
			exception.method = 'load_multiple( self, path: str ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the PowerPointLoader workflow into smaller chunks.
			The method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			ValueError: Raised when a required argument or option is missing or invalid.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'PowerPointLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None' )
			Logger( ).write( exception )
			raise exception

class OutlookLoader( Loader ):
	"""OutlookLoader component.
	
	Purpose:
		Loads Outlook message files into LangChain Document objects. The loader prepares email
		content for later retrieval, inspection, or generation workflows.
	
	Attributes:
		loader (Optional[OutlookMessageLoader]): Loader value maintained by the OutlookLoader runtime state.
		file_path (Optional[str]): File path value maintained by the OutlookLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the OutlookLoader runtime state.
		query (Optional[str]): Query value maintained by the OutlookLoader runtime state.
		max_documents (Optional[int]): Max documents value maintained by the OutlookLoader runtime state.
		max_characters (Optional[int]): Max characters value maintained by the OutlookLoader runtime state.
		query (Optional[str]): Query value maintained by the OutlookLoader runtime state."""
	loader: Optional[ OutlookMessageLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	max_documents: Optional[ int ]
	max_characters: Optional[ int ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the OutlookLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.max_documents = 2
		self.max_characters = 1000
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the OutlookLoader component. The
			ordered list supports interactive inspection, documentation surfaces, and UI code that
			displays available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'max_charactes',
		         'max_documents',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, path: str ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the OutlookLoader workflow. The
			method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			path (str): Filesystem path or object identifier used by the loader.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = OutlookMessageLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'OutlookLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the OutlookLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'OutlookLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class WebCrawler( Loader ):
	"""WebCrawler component.
	
	Purpose:
		Loads web content with recursive crawling support and shared chunking behavior. The
		crawler stores URL traversal settings and returns LangChain Document objects for fetched
		pages.
	
	Attributes:
		loader (Optional[RecursiveUrlLoader | WebBaseLoader]): Loader value maintained by the WebCrawler runtime state.
		url (Optional[str]): Url value maintained by the WebCrawler runtime state.
		web_paths (Optional[str | List[str]]): Web paths value maintained by the WebCrawler runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the WebCrawler runtime state.
		file_path (Optional[str]): File path value maintained by the WebCrawler runtime state.
		max_depth (Optional[int]): Max depth value maintained by the WebCrawler runtime state.
		timeout (Optional[int]): Timeout value maintained by the WebCrawler runtime state.
		ignore (Optional[bool]): Ignore value maintained by the WebCrawler runtime state.
		with_progress (Optional[bool]): With progress value maintained by the WebCrawler runtime state.
		recursive (Optional[bool]): Recursive value maintained by the WebCrawler runtime state.
		prevent_outside (Optional[bool]): Prevent outside value maintained by the WebCrawler runtime state."""
	loader: Optional[ RecursiveUrlLoader | WebBaseLoader ]
	url: Optional[ str ]
	web_paths: Optional[ str | List[ str ] ]
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	max_depth: Optional[ int ]
	timeout: Optional[ int ]
	ignore: Optional[ bool ]
	with_progress: Optional[ bool ]
	recursive: Optional[ bool ]
	prevent_outside: Optional[ bool ]
	
	def __init__( self, url: str, recursive: bool = False, max_depth: int = 2,
			prevent_outside: bool = True, timeout: int = 10,
			ignore: bool = True, progress: bool = True ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the WebCrawler object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract.
		
		Args:
			url (str): URL used by the web-oriented loader.
			recursive (bool): Whether recursive loading is enabled.
			max_depth (int): Maximum traversal depth for recursive loading.
			prevent_outside (bool): Whether recursive loading is restricted to the source domain.
			timeout (int): Timeout value used by network-backed loading operations.
			ignore (bool): Optional ignore pattern or ignore setting passed to the loader.
			progress (bool): Whether progress reporting is enabled where supported."""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.chunk_size = None
		self.overlap_amount = None
		self.url = url
		self.web_paths = None
		self.max_depth = max_depth
		self.timeout = timeout
		self.ignore = ignore
		self.with_progress = progress
		self.recursive = recursive
		self.prevent_outside = prevent_outside
		self.loader = RecursiveUrlLoader( url=self.url, max_depth=self.max_depth,
			timeout=self.timeout, continue_on_failure=self.ignore,
			prevent_outside=self.prevent_outside )
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the WebCrawler component. The ordered
			list supports interactive inspection, documentation surfaces, and UI code that displays
			available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [
				'loader',
				'documents',
				'splitter',
				'pattern',
				'file_path',
				'expanded',
				'candidates',
				'resolved',
				'chunk_size',
				'overlap_amount',
				'url',
				'web_paths',
				'max_depth',
				'timeout',
				'ignore',
				'with_progress',
				'recursive',
				'prevent_outside',
				'verify_exists',
				'resolve_paths',
				'split_documents',
				'load',
				'load_pages',
				'split',
		]
	
	def load( self, urls: str | List[ str ], depth: int = 2, timeout: int = 10,
			ignore: bool = True, progress: bool = True,
			prevent_outside: bool = True ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the WebCrawler workflow. The
			method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			urls (str | List[str]): One or more URLs used by the web-oriented loader.
			depth (int): Depth value used by the operation.
			timeout (int): Timeout value used by network-backed loading operations.
			ignore (bool): Optional ignore pattern or ignore setting passed to the loader.
			progress (bool): Whether progress reporting is enabled where supported.
			prevent_outside (bool): Whether recursive loading is restricted to the source domain.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if self.recursive:
				return self.load_recursive(
					urls=urls,
					depth=depth,
					timeout=timeout,
					ignore=ignore,
					prevent_outside=prevent_outside
				)
			
			return self.load_pages(
				urls=urls,
				timeout=timeout,
				ignore=ignore,
				progress=progress
			)
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'WebCrawler'
			exception.method = (
					'load( self, urls: str | List[ str ], depth: int=2, '
					'timeout: int=10, ignore: bool=True, progress: bool=True, '
					'prevent_outside: bool=True ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def load_pages( self, urls: str | List[ str ], timeout: int = 10,
			ignore: bool = True, progress: bool = True ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content for the WebCrawler workflow using the specialized Load pages path.
			The method validates input settings, delegates to the underlying loader implementation,
			and returns LangChain Document objects.
		
		Args:
			urls (str | List[str]): One or more URLs used by the web-oriented loader.
			timeout (int): Timeout value used by network-backed loading operations.
			ignore (bool): Optional ignore pattern or ignore setting passed to the loader.
			progress (bool): Whether progress reporting is enabled where supported.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'urls', urls )
			
			self.web_paths = [ urls ] if isinstance( urls, str ) else list( urls )
			self.timeout = timeout
			self.ignore = ignore
			self.with_progress = progress
			
			self.loader = WebBaseLoader(
				web_paths=self.web_paths,
				show_progress=self.with_progress,
				continue_on_failure=self.ignore,
				requests_kwargs={ 'timeout': self.timeout }
			)
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'WebCrawler'
			exception.method = (
					'load_pages( self, urls: str | List[ str ], timeout: int=10, '
					'ignore: bool=True, progress: bool=True ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def load_recursive( self, urls: str | List[ str ], depth: int = 2,
			timeout: int = 10, ignore: bool = True,
			prevent_outside: bool = True ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content for the WebCrawler workflow using the specialized Load recursive
			path. The method validates input settings, delegates to the underlying loader
			implementation, and returns LangChain Document objects.
		
		Args:
			urls (str | List[str]): One or more URLs used by the web-oriented loader.
			depth (int): Depth value used by the operation.
			timeout (int): Timeout value used by network-backed loading operations.
			ignore (bool): Optional ignore pattern or ignore setting passed to the loader.
			prevent_outside (bool): Whether recursive loading is restricted to the source domain.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'urls', urls )
			
			self.url = urls[ 0 ] if isinstance( urls, list ) else urls
			self.max_depth = depth
			self.timeout = timeout
			self.ignore = ignore
			self.prevent_outside = prevent_outside
			
			self.loader = RecursiveUrlLoader(
				url=self.url,
				max_depth=self.max_depth,
				timeout=self.timeout,
				continue_on_failure=self.ignore,
				prevent_outside=self.prevent_outside
			)
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'WebCrawler'
			exception.method = (
					'load_recursive( self, urls: str | List[ str ], depth: int=2, '
					'timeout: int=10, ignore: bool=True, prevent_outside: bool=True ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the WebCrawler workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			ValueError: Raised when a required argument or option is missing or invalid.
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			
			self.chunk_size = chunk
			self.overlap_amount = overlap
			return self.split_documents(
				docs=self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'WebCrawler'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class SpfxLoader( Loader ):
	"""SpfxLoader component.
	
	Purpose:
		Loads SharePoint document-library content into LangChain Document objects. The loader
		supports library-level and folder-level loading for SharePoint-backed document
		workflows.
	
	Attributes:
		loader (Optional[SharePointLoader]): Loader value maintained by the SpfxLoader runtime state.
		file_path (Optional[str]): File path value maintained by the SpfxLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the SpfxLoader runtime state.
		library_id (Optional[str]): Library id value maintained by the SpfxLoader runtime state.
		subsite_id (Optional[str]): Subsite id value maintained by the SpfxLoader runtime state.
		folder_id (Optional[str]): Folder id value maintained by the SpfxLoader runtime state.
		object_ids (Optional[List[str]]): Object ids value maintained by the SpfxLoader runtime state.
		query (Optional[str]): Query value maintained by the SpfxLoader runtime state.
		with_token (Optional[bool]): With token value maintained by the SpfxLoader runtime state.
		is_recursive (Optional[bool]): Is recursive value maintained by the SpfxLoader runtime state."""
	loader: Optional[ SharePointLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	library_id: Optional[ str ]
	subsite_id: Optional[ str ]
	folder_id: Optional[ str ]
	object_ids: Optional[ List[ str ] ]
	query: Optional[ str ]
	with_token: Optional[ bool ]
	is_recursive: Optional[ bool ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the SpfxLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.folder_id = None
		self.library_id = None
		self.subsite_id = None
		self.object_ids = [ ]
		self.with_token = None
		self.is_recursive = None
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the SpfxLoader component. The ordered
			list supports interactive inspection, documentation surfaces, and UI code that displays
			available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'folder_id',
		         'library_id',
		         'subsite_id',
		         'object_id',
		         'with_token',
		         'is_recursive',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, library_id: str ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the SpfxLoader workflow. The
			method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			library_id (str): Library id value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'library_id', library_id )
			self.library_id = library_id
			self.is_recursive = True
			self.with_token = True
			self.loader = SharePointLoader( document_library_id=self.library_id,
				recursive=self.is_recursive, auth_with_token=self.with_token )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'SpfxLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def load_folder( self, library_id: str, folder_id: str ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content for the SpfxLoader workflow using the specialized Load folder path.
			The method validates input settings, delegates to the underlying loader implementation,
			and returns LangChain Document objects.
		
		Args:
			library_id (str): Library id value used by the operation.
			folder_id (str): Folder id value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'library_id', library_id )
			throw_if( 'folder_id', folder_id )
			self.library_id = library_id
			self.folder_id = folder_id
			self.loader = SharePointLoader( document_library_id=self.library_id,
				folder_id=self.folder_id )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'SpfxLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the SpfxLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'SpfxLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class OneDriveDocLoader( Loader ):
	"""OneDriveDocLoader component.
	
	Purpose:
		Loads OneDrive documents and folders into LangChain Document objects. The loader wraps
		OneDrive file and folder access while preserving common split behavior.
	
	Attributes:
		loader (Optional[OneDriveLoader]): Loader value maintained by the OneDriveDocLoader runtime state.
		file_path (Optional[str]): File path value maintained by the OneDriveDocLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the OneDriveDocLoader runtime state.
		client_id (Optional[str]): Client id value maintained by the OneDriveDocLoader runtime state.
		drive_id (Optional[str]): Drive id value maintained by the OneDriveDocLoader runtime state.
		client_secret (Optional[str]): Client secret value maintained by the OneDriveDocLoader runtime state."""
	loader: Optional[ OneDriveLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	client_id: Optional[ str ]
	drive_id: Optional[ str ]
	client_secret: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the OneDriveDocLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.drive_id = None
		self.client_id = None
		self.client_secret = None
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the OneDriveDocLoader component. The
			ordered list supports interactive inspection, documentation surfaces, and UI code that
			displays available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'query',
		         'drive_id',
		         'client_id',
		         'client_secret',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'load_folder',
		         'split', ]
	
	@property
	def file_options( self ) -> List[ str ] | None:
		"""Return supported options.
		
		Purpose:
			Returns the supported option values exposed by the OneDriveDocLoader component. The
			values are used by UI selectors and validation logic to keep runtime choices aligned
			with the loader implementation.
		
		Returns:
			Value produced by the operation."""
		return [ 'pdf', 'doc', 'docx', 'txt' ]
	
	def load( self, id: str ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the OneDriveDocLoader workflow.
			The method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			id (str): Id value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'id', id )
			self.drive_id = id
			self.loader = OneDriveLoader( drive_id=self.drive_id )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'WikiLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def load_folder( self, id: str, path: str ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content for the OneDriveDocLoader workflow using the specialized Load
			folder path. The method validates input settings, delegates to the underlying loader
			implementation, and returns LangChain Document objects.
		
		Args:
			id (str): Id value used by the operation.
			path (str): Filesystem path or object identifier used by the loader.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'id', id )
			self.drive_id = id
			self.file_path = path
			self.loader = OneDriveLoader( drive_id=self.drive_id, folder_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'WikiLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the OneDriveDocLoader workflow into smaller chunks.
			The method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'WikiLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class GoogleLoader( Loader ):
	"""GoogleLoader component.
	
	Purpose:
		Loads Google Cloud Storage files and folders into LangChain Document objects. The loader
		centralizes project, bucket, and recursion options for Google-backed ingestion.
	
	Attributes:
		loader (Optional[GoogleDriveLoader]): Loader value maintained by the GoogleLoader runtime state.
		file_path (Optional[str]): File path value maintained by the GoogleLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the GoogleLoader runtime state.
		query (Optional[str]): Query value maintained by the GoogleLoader runtime state.
		file_id (Optional[str]): File id value maintained by the GoogleLoader runtime state.
		folder_id (Optional[str]): Folder id value maintained by the GoogleLoader runtime state.
		query (Optional[str]): Query value maintained by the GoogleLoader runtime state.
		is_recursive (Optional[bool]): Is recursive value maintained by the GoogleLoader runtime state."""
	loader: Optional[ GoogleDriveLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	file_id: Optional[ str ]
	folder_id: Optional[ str ]
	query: Optional[ str ]
	is_recursive: Optional[ bool ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the GoogleLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.file_id = None
		self.folder_id = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.is_recursive = None
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the GoogleLoader component. The
			ordered list supports interactive inspection, documentation surfaces, and UI code that
			displays available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'query',
		         'folder_id',
		         'file_id',
		         'is_recursive',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'load_folder',
		         'split', ]
	
	@property
	def file_options( self ):
		"""Return supported options.
		
		Purpose:
			Returns the supported option values exposed by the GoogleLoader component. The values
			are used by UI selectors and validation logic to keep runtime choices aligned with the
			loader implementation.
		
		Returns:
			Value produced by the operation."""
		return [ 'document',
		         'sheet',
		         'pdf' ]
	
	def load_file( self, file_id: str, recursive: bool = False ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content for the GoogleLoader workflow using the specialized Load file path.
			The method validates input settings, delegates to the underlying loader implementation,
			and returns LangChain Document objects.
		
		Args:
			file_id (str): File id value used by the operation.
			recursive (bool): Whether recursive loading is enabled.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'file_id', file_id )
			throw_if( 'recursive', recursive )
			self.file_id = file_id
			self.is_recursive = recursive
			self.loader = GoogleDriveLoader( file_ids=[ self.file_id ],
				recursive=self.is_recursive )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'GoogleDriveLoader'
			exception.method = 'load_File( self, file_id: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def load_folder( self, folder_id: str, recursive: bool = False ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content for the GoogleLoader workflow using the specialized Load folder
			path. The method validates input settings, delegates to the underlying loader
			implementation, and returns LangChain Document objects.
		
		Args:
			folder_id (str): Folder id value used by the operation.
			recursive (bool): Whether recursive loading is enabled.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'folder_id', folder_id )
			self.folder_id = folder_id
			self.is_recursive = recursive
			self.loader = GoogleDriveLoader( folder_id=self.folder_id, recursive=self.is_recursive )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'GoogleDriveLoader'
			exception.method = 'load_folder( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the GoogleLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'GoogleDriveLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class EmailLoader( Loader ):
	"""EmailLoader component.
	
	Purpose:
		Loads email files into LangChain Document objects and optionally includes attachments.
		The loader supports email parsing options used for retrieval and document analysis
		workflows.
	
	Attributes:
		loader (Optional[UnstructuredEmailLoader]): Loader value maintained by the EmailLoader runtime state.
		file_path (Optional[str]): File path value maintained by the EmailLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the EmailLoader runtime state.
		has_attachments (Optional[bool]): Has attachments value maintained by the EmailLoader runtime state.
		mode (Optional[str]): Mode value maintained by the EmailLoader runtime state."""
	loader: Optional[ UnstructuredEmailLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	has_attachments: Optional[ bool ]
	mode: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the EmailLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ):
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the EmailLoader component. The
			ordered list supports interactive inspection, documentation surfaces, and UI code that
			displays available attributes and methods.
		
		Returns:
			Value produced by the operation."""
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'has_attachments',
		         'mode',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, path: str, mode: str = 'single', attachments: bool = True ) -> List[ Document ]:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the EmailLoader workflow. The
			method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			path (str): Filesystem path or object identifier used by the loader.
			mode (str): Loader mode or parsing mode requested by the caller.
			attachments (bool): Attachments value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.mode = mode
			self.has_attachments = attachments
			self.loader = UnstructuredEmailLoader( file_path=self.file_path, mode=self.mode,
				process_attachments=self.has_attachments )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'EmailLoader'
			exception.method = ('load( self, path: str, mode: str=elements, '
			                    'include_headers: bool=True ) -> List[ Document ]')
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the EmailLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'foo'
			exception.cause = 'EmailLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class PubMedSearchLoader( Loader ):
	"""PubMedSearchLoader component.
	
	Purpose:
		Loads PubMed search results into LangChain Document objects. The loader prepares
		biomedical publication results for downstream text processing and retrieval.
	
	Attributes:
		loader (Optional[PubMedLoader]): Loader value maintained by the PubMedSearchLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the PubMedSearchLoader runtime state.
		query (Optional[str]): Query value maintained by the PubMedSearchLoader runtime state.
		max_docs (Optional[int]): Max docs value maintained by the PubMedSearchLoader runtime state."""
	loader: Optional[ PubMedLoader ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	max_docs: Optional[ int ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the PubMedSearchLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.query = None
		self.max_docs = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the PubMedSearchLoader component. The
			ordered list supports interactive inspection, documentation surfaces, and UI code that
			displays available attributes and methods.
		
		Returns:
			List produced by the operation."""
		return [
				'loader',
				'documents',
				'query',
				'max_docs',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load( self, query: str, max_docs: int = 5 ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the PubMedSearchLoader
			workflow. The method validates required inputs, configures the underlying loader,
			captures runtime state, and returns parsed documents using the established Foo loader
			contract.
		
		Args:
			query (str): Search query or lookup value submitted to the loader.
			max_docs (int): Max docs value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'query', query )
			self.query = query
			self.max_docs = max_docs
			self.loader = PubMedLoader( query=self.query, load_max_docs=self.max_docs )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'PubMedSearchLoader'
			exception.method = (
					'load( self, query: str, max_docs: int=5 ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the PubMedSearchLoader workflow into smaller chunks.
			The method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'PubMedSearchLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class OpenCityLoader( Loader ):
	"""OpenCityLoader component.
	
	Purpose:
		Loads Open City Data records into LangChain Document objects. The loader uses city and
		dataset identifiers to retrieve civic data for analysis workflows.
	
	Attributes:
		loader (Optional[OpenCityDataLoader]): Loader value maintained by the OpenCityLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the OpenCityLoader runtime state.
		city_id (Optional[str]): City id value maintained by the OpenCityLoader runtime state.
		dataset_id (Optional[str]): Dataset id value maintained by the OpenCityLoader runtime state.
		limit (Optional[int]): Limit value maintained by the OpenCityLoader runtime state."""
	loader: Optional[ OpenCityDataLoader ]
	documents: Optional[ List[ Document ] ]
	city_id: Optional[ str ]
	dataset_id: Optional[ str ]
	limit: Optional[ int ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the OpenCityLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.city_id = None
		self.dataset_id = None
		self.limit = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the OpenCityLoader component. The
			ordered list supports interactive inspection, documentation surfaces, and UI code that
			displays available attributes and methods.
		
		Returns:
			List produced by the operation."""
		return [
				'loader',
				'documents',
				'city_id',
				'dataset_id',
				'limit',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load( self, city_id: str, dataset_id: str, limit: int = 100 ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the OpenCityLoader workflow.
			The method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			city_id (str): City id value used by the operation.
			dataset_id (str): Dataset id value used by the operation.
			limit (int): Limit value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'city_id', city_id )
			throw_if( 'dataset_id', dataset_id )
			self.city_id = city_id
			self.dataset_id = dataset_id
			self.limit = limit
			self.loader = OpenCityDataLoader(
				city_id=self.city_id,
				dataset_id=self.dataset_id,
				limit=self.limit
			)
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'OpenCityLoader'
			exception.method = (
					'load( self, city_id: str, dataset_id: str, limit: int=100 ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the OpenCityLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'OpenCityLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class JupyterNotebookLoader( Loader ):
	"""JupyterNotebookLoader component.
	
	Purpose:
		Loads Jupyter notebook files into LangChain Document objects with configurable output
		and traceback handling. The loader prepares notebook source and optional outputs for
		retrieval workflows.
	
	Attributes:
		loader (Optional[NotebookLoader]): Loader value maintained by the JupyterNotebookLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the JupyterNotebookLoader runtime state.
		file_path (Optional[str]): File path value maintained by the JupyterNotebookLoader runtime state.
		include_outputs (Optional[bool]): Include outputs value maintained by the JupyterNotebookLoader runtime state.
		max_output_length (Optional[int]): Max output length value maintained by the JupyterNotebookLoader runtime state.
		remove_newline (Optional[bool]): Remove newline value maintained by the JupyterNotebookLoader runtime state.
		traceback (Optional[bool]): Traceback value maintained by the JupyterNotebookLoader runtime state."""
	loader: Optional[ NotebookLoader ]
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	include_outputs: Optional[ bool ]
	max_output_length: Optional[ int ]
	remove_newline: Optional[ bool ]
	traceback: Optional[ bool ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the JupyterNotebookLoader object with default configuration, runtime state,
			and compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.file_path = None
		self.include_outputs = None
		self.max_output_length = None
		self.remove_newline = None
		self.traceback = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the JupyterNotebookLoader component.
			The ordered list supports interactive inspection, documentation surfaces, and UI code
			that displays available attributes and methods.
		
		Returns:
			List produced by the operation."""
		return [
				'loader',
				'documents',
				'file_path',
				'include_outputs',
				'max_output_length',
				'remove_newline',
				'traceback',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load( self, path: str, include_outputs: bool = False, max_output_length: int = 10,
			remove_newline: bool = False, traceback: bool = False ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the JupyterNotebookLoader
			workflow. The method validates required inputs, configures the underlying loader,
			captures runtime state, and returns parsed documents using the established Foo loader
			contract.
		
		Args:
			path (str): Filesystem path or object identifier used by the loader.
			include_outputs (bool): Include outputs value used by the operation.
			max_output_length (int): Max output length value used by the operation.
			remove_newline (bool): Remove newline value used by the operation.
			traceback (bool): Traceback value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.include_outputs = include_outputs
			self.max_output_length = max_output_length
			self.remove_newline = remove_newline
			self.traceback = traceback
			
			self.loader = NotebookLoader(
				self.file_path,
				include_outputs=self.include_outputs,
				max_output_length=self.max_output_length,
				remove_newline=self.remove_newline,
				traceback=self.traceback
			)
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'JupyterNotebookLoader'
			exception.method = (
					'load( self, path: str, include_outputs: bool=False, '
					'max_output_length: int=10, remove_newline: bool=False, '
					'traceback: bool=False ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the JupyterNotebookLoader workflow into smaller
			chunks. The method records chunk size and overlap settings, delegates to the configured
			text splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'JupyterNotebookLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class GoogleCloudFileLoader( Loader ):
	"""GoogleCloudFileLoader component.
	
	Purpose:
		Loads a single Google Cloud Storage blob into LangChain Document objects. The loader
		captures project, bucket, and blob settings for file-level cloud ingestion.
	
	Attributes:
		loader (Optional[GCSFileLoader]): Loader value maintained by the GoogleCloudFileLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the GoogleCloudFileLoader runtime state.
		project_name (Optional[str]): Project name value maintained by the GoogleCloudFileLoader runtime state.
		bucket (Optional[str]): Bucket value maintained by the GoogleCloudFileLoader runtime state.
		blob (Optional[str]): Blob value maintained by the GoogleCloudFileLoader runtime state."""
	loader: Optional[ GCSFileLoader ]
	documents: Optional[ List[ Document ] ]
	project_name: Optional[ str ]
	bucket: Optional[ str ]
	blob: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the GoogleCloudFileLoader object with default configuration, runtime state,
			and compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.project_name = None
		self.bucket = None
		self.blob = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the GoogleCloudFileLoader component.
			The ordered list supports interactive inspection, documentation surfaces, and UI code
			that displays available attributes and methods.
		
		Returns:
			List produced by the operation."""
		return [
				'loader',
				'documents',
				'project_name',
				'bucket',
				'blob',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load( self, project_name: str, bucket: str, blob: str ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the GoogleCloudFileLoader
			workflow. The method validates required inputs, configures the underlying loader,
			captures runtime state, and returns parsed documents using the established Foo loader
			contract.
		
		Args:
			project_name (str): Project name value used by the operation.
			bucket (str): Bucket value used by the operation.
			blob (str): Blob value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'project_name', project_name )
			throw_if( 'bucket', bucket )
			throw_if( 'blob', blob )
			self.project_name = project_name
			self.bucket = bucket
			self.blob = blob
			self.loader = GCSFileLoader( project_name=self.project_name, bucket=self.bucket,
				blob=self.blob )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleCloudStorageFileLoader'
			exception.method = (
					'load( self, project_name: str, bucket: str, blob: str ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the GoogleCloudFileLoader workflow into smaller
			chunks. The method records chunk size and overlap settings, delegates to the configured
			text splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleCloudStorageFileLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class AwsFileLoader( Loader ):
	"""AwsFileLoader component.
	
	Purpose:
		Loads a single Amazon S3 object into LangChain Document objects. The loader accepts AWS
		credential and region settings while preserving the standard loader contract.
	
	Attributes:
		loader (Optional[S3FileLoader]): Loader value maintained by the AwsFileLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the AwsFileLoader runtime state.
		bucket (Optional[str]): Bucket value maintained by the AwsFileLoader runtime state.
		key (Optional[str]): Key value maintained by the AwsFileLoader runtime state.
		aws_access_key_id (Optional[str]): Aws access key id value maintained by the AwsFileLoader runtime state.
		aws_secret_access_key (Optional[str]): Aws secret access key value maintained by the AwsFileLoader runtime state.
		aws_session_token (Optional[str]): Aws session token value maintained by the AwsFileLoader runtime state.
		region_name (Optional[str]): Region name value maintained by the AwsFileLoader runtime state."""
	loader: Optional[ S3FileLoader ]
	documents: Optional[ List[ Document ] ]
	bucket: Optional[ str ]
	key: Optional[ str ]
	aws_access_key_id: Optional[ str ]
	aws_secret_access_key: Optional[ str ]
	aws_session_token: Optional[ str ]
	region_name: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the AwsFileLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.bucket = None
		self.key = None
		self.aws_access_key_id = None
		self.aws_secret_access_key = None
		self.aws_session_token = None
		self.region_name = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the AwsFileLoader component. The
			ordered list supports interactive inspection, documentation surfaces, and UI code that
			displays available attributes and methods.
		
		Returns:
			List produced by the operation."""
		return [
				'loader',
				'documents',
				'bucket',
				'key',
				'aws_access_key_id',
				'aws_secret_access_key',
				'aws_session_token',
				'region_name',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load( self, bucket: str, key: str, aws_access_key_id: Optional[ str ] = None,
			aws_secret_access_key: Optional[ str ] = None, aws_session_token: Optional[
				str ] = None,
			region_name: Optional[ str ] = None ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the AwsFileLoader workflow. The
			method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			bucket (str): Bucket value used by the operation.
			key (str): Key value used by the operation.
			aws_access_key_id (Optional[str]): Aws access key id value used by the operation.
			aws_secret_access_key (Optional[str]): Aws secret access key value used by the operation.
			aws_session_token (Optional[str]): Aws session token value used by the operation.
			region_name (Optional[str]): Region name value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'bucket', bucket )
			throw_if( 'key', key )
			
			self.bucket = bucket
			self.key = key
			self.aws_access_key_id = aws_access_key_id
			self.aws_secret_access_key = aws_secret_access_key
			self.aws_session_token = aws_session_token
			self.region_name = region_name
			
			kwargs: Dict[ str, Any ] = { }
			if self.aws_access_key_id:
				kwargs[ 'aws_access_key_id' ] = self.aws_access_key_id
			if self.aws_secret_access_key:
				kwargs[ 'aws_secret_access_key' ] = self.aws_secret_access_key
			if self.aws_session_token:
				kwargs[ 'aws_session_token' ] = self.aws_session_token
			if self.region_name:
				kwargs[ 'region_name' ] = self.region_name
			
			self.loader = S3FileLoader(
				self.bucket,
				self.key,
				**kwargs
			)
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'AwsFileLoader'
			exception.method = (
					'load( self, bucket: str, key: str, '
					'aws_access_key_id: Optional[ str ]=None, '
					'aws_secret_access_key: Optional[ str ]=None, '
					'aws_session_token: Optional[ str ]=None, '
					'region_name: Optional[ str ]=None ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the AwsFileLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'AwsFileLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class GoogleSpeechToTextLoader( Loader ):
	"""GoogleSpeechToTextLoader component.
	
	Purpose:
		Loads speech-to-text output from Google Cloud audio transcription into LangChain
		Document objects. The loader captures project, file, and transcription configuration
		settings.
	
	Attributes:
		loader (Optional[SpeechToTextLoader]): Loader value maintained by the GoogleSpeechToTextLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the GoogleSpeechToTextLoader runtime state.
		project_id (Optional[str]): Project id value maintained by the GoogleSpeechToTextLoader runtime state.
		file_path (Optional[str]): File path value maintained by the GoogleSpeechToTextLoader runtime state.
		config (Optional[Dict[str, Any]]): Config value maintained by the GoogleSpeechToTextLoader runtime state."""
	loader: Optional[ SpeechToTextLoader ]
	documents: Optional[ List[ Document ] ]
	project_id: Optional[ str ]
	file_path: Optional[ str ]
	config: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the GoogleSpeechToTextLoader object with default configuration, runtime
			state, and compatibility fields required by later method calls. The constructor performs
			local state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.project_id = None
		self.file_path = None
		self.config = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the GoogleSpeechToTextLoader
			component. The ordered list supports interactive inspection, documentation surfaces, and
			UI code that displays available attributes and methods.
		
		Returns:
			List produced by the operation."""
		return [
				'loader',
				'documents',
				'project_id',
				'file_path',
				'config',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load( self, project_id: str, file_path: str,
			config: Optional[ Dict[ str, Any ] ] = None ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the GoogleSpeechToTextLoader
			workflow. The method validates required inputs, configures the underlying loader,
			captures runtime state, and returns parsed documents using the established Foo loader
			contract.
		
		Args:
			project_id (str): Project id value used by the operation.
			file_path (str): File path value used by the operation.
			config (Optional[Dict[str, Any]]): Config value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'project_id', project_id )
			throw_if( 'file_path', file_path )
			
			self.project_id = project_id
			self.file_path = file_path
			self.config = config
			
			if self.config:
				self.loader = SpeechToTextLoader(
					project_id=self.project_id,
					file_path=self.file_path,
					config=self.config
				)
			else:
				self.loader = SpeechToTextLoader(
					project_id=self.project_id,
					file_path=self.file_path
				)
			
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleSpeechToTextAudioLoader'
			exception.method = (
					'load( self, project_id: str, file_path: str, '
					'config: Optional[ Dict[ str, Any ] ]=None ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the GoogleSpeechToTextLoader workflow into smaller
			chunks. The method records chunk size and overlap settings, delegates to the configured
			text splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleSpeechToTextAudioLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class GoogleBucketLoader( Loader ):
	"""GoogleBucketLoader component.
	
	Purpose:
		Loads Google Cloud Storage bucket content into LangChain Document objects. The loader
		supports prefix-based loading and optional continue-on-failure behavior.
	
	Attributes:
		loader (Optional[GCSDirectoryLoader]): Loader value maintained by the GoogleBucketLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the GoogleBucketLoader runtime state.
		project_name (Optional[str]): Project name value maintained by the GoogleBucketLoader runtime state.
		bucket (Optional[str]): Bucket value maintained by the GoogleBucketLoader runtime state.
		prefix (Optional[str]): Prefix value maintained by the GoogleBucketLoader runtime state.
		continue_on_failure (Optional[bool]): Continue on failure value maintained by the GoogleBucketLoader runtime state."""
	loader: Optional[ GCSDirectoryLoader ]
	documents: Optional[ List[ Document ] ]
	project_name: Optional[ str ]
	bucket: Optional[ str ]
	prefix: Optional[ str ]
	continue_on_failure: Optional[ bool ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the GoogleBucketLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.project_name = None
		self.bucket = None
		self.prefix = None
		self.continue_on_failure = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the GoogleBucketLoader component. The
			ordered list supports interactive inspection, documentation surfaces, and UI code that
			displays available attributes and methods.
		
		Returns:
			List produced by the operation."""
		return [
				'loader',
				'documents',
				'project_name',
				'bucket',
				'prefix',
				'continue_on_failure',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load( self, project_name: str, bucket: str, prefix: Optional[ str ] = None,
			continue_on_failure: bool = False ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the GoogleBucketLoader
			workflow. The method validates required inputs, configures the underlying loader,
			captures runtime state, and returns parsed documents using the established Foo loader
			contract.
		
		Args:
			project_name (str): Project name value used by the operation.
			bucket (str): Bucket value used by the operation.
			prefix (Optional[str]): Prefix value used by the operation.
			continue_on_failure (bool): Continue on failure value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'project_name', project_name )
			throw_if( 'bucket', bucket )
			self.project_name = project_name
			self.bucket = bucket
			self.prefix = prefix
			self.continue_on_failure = continue_on_failure
			kwargs: Dict[ str, Any ] = {
					'project_name': self.project_name,
					'bucket': self.bucket,
					'continue_on_failure': self.continue_on_failure,
			}
			
			if self.prefix:
				kwargs[ 'prefix' ] = self.prefix
			
			self.loader = GCSDirectoryLoader( **kwargs )
			self.documents = self.loader.load( )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleBucketLoader'
			exception.method = (
					'load( self, project_name: str, bucket: str, '
					'prefix: Optional[ str ]=None, continue_on_failure: bool=False ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the GoogleBucketLoader workflow into smaller chunks.
			The method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount
			)
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleBucketLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class AwsBucketLoader( Loader ):
	"""AwsBucketLoader component.
	
	Purpose:
		Loads Amazon S3 bucket content into LangChain Document objects. The loader supports
		prefix, credential, region, and endpoint settings for S3-backed ingestion.
	
	Attributes:
		loader (Optional[S3DirectoryLoader]): Loader value maintained by the AwsBucketLoader runtime state.
		documents (Optional[List[Document]]): Documents value maintained by the AwsBucketLoader runtime state.
		bucket (Optional[str]): Bucket value maintained by the AwsBucketLoader runtime state.
		prefix (Optional[str]): Prefix value maintained by the AwsBucketLoader runtime state.
		aws_access_key_id (Optional[str]): Aws access key id value maintained by the AwsBucketLoader runtime state.
		aws_secret_access_key (Optional[str]): Aws secret access key value maintained by the AwsBucketLoader runtime state.
		aws_session_token (Optional[str]): Aws session token value maintained by the AwsBucketLoader runtime state.
		region_name (Optional[str]): Region name value maintained by the AwsBucketLoader runtime state.
		endpoint_url (Optional[str]): Endpoint url value maintained by the AwsBucketLoader runtime state."""
	loader: Optional[ S3DirectoryLoader ]
	documents: Optional[ List[ Document ] ]
	bucket: Optional[ str ]
	prefix: Optional[ str ]
	aws_access_key_id: Optional[ str ]
	aws_secret_access_key: Optional[ str ]
	aws_session_token: Optional[ str ]
	region_name: Optional[ str ]
	endpoint_url: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the AwsBucketLoader object with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor performs local
			state preparation without changing the public loader contract."""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.bucket = None
		self.prefix = None
		self.aws_access_key_id = None
		self.aws_secret_access_key = None
		self.aws_session_token = None
		self.region_name = None
		self.endpoint_url = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.
		
		Purpose:
			Returns a stable list of public members exposed by the AwsBucketLoader component. The
			ordered list supports interactive inspection, documentation surfaces, and UI code that
			displays available attributes and methods.
		
		Returns:
			List produced by the operation."""
		return [
				'loader',
				'documents',
				'bucket',
				'prefix',
				'aws_access_key_id',
				'aws_secret_access_key',
				'aws_session_token',
				'region_name',
				'endpoint_url',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load( self, bucket: str, prefix: Optional[ str ] = None, aws_access_key_id: Optional[ str ] = None,
			aws_secret_access_key: Optional[ str ] = None, aws_session_token: Optional[ str ] = None,
			region_name: Optional[ str ] = None,
			endpoint_url: Optional[ str ] = None ) -> List[ Document ] | None:
		"""Load documents.
		
		Purpose:
			Loads source content into LangChain Document objects for the AwsBucketLoader workflow.
			The method validates required inputs, configures the underlying loader, captures runtime
			state, and returns parsed documents using the established Foo loader contract.
		
		Args:
			bucket (str): Bucket value used by the operation.
			prefix (Optional[str]): Prefix value used by the operation.
			aws_access_key_id (Optional[str]): Aws access key id value used by the operation.
			aws_secret_access_key (Optional[str]): Aws secret access key value used by the operation.
			aws_session_token (Optional[str]): Aws session token value used by the operation.
			region_name (Optional[str]): Region name value used by the operation.
			endpoint_url (Optional[str]): Endpoint url value used by the operation.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'bucket', bucket )
			self.bucket = bucket
			self.prefix = prefix
			self.aws_access_key_id = aws_access_key_id
			self.aws_secret_access_key = aws_secret_access_key
			self.aws_session_token = aws_session_token
			self.region_name = region_name
			self.endpoint_url = endpoint_url
			
			kwargs: Dict[ str, Any ] = { }
			if self.prefix:
				kwargs[ 'prefix' ] = self.prefix
			if self.aws_access_key_id:
				kwargs[ 'aws_access_key_id' ] = self.aws_access_key_id
			if self.aws_secret_access_key:
				kwargs[ 'aws_secret_access_key' ] = self.aws_secret_access_key
			if self.aws_session_token:
				kwargs[ 'aws_session_token' ] = self.aws_session_token
			if self.region_name:
				kwargs[ 'region_name' ] = self.region_name
			if self.endpoint_url:
				kwargs[ 'endpoint_url' ] = self.endpoint_url
			
			self.loader = S3DirectoryLoader(
				self.bucket,
				**kwargs
			)
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'AmazonBucketLoader'
			exception.method = 'load( self, *args ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Split documents.
		
		Purpose:
			Splits documents already loaded by the AwsBucketLoader workflow into smaller chunks. The
			method records chunk size and overlap settings, delegates to the configured text
			splitter, and returns chunked LangChain Document objects.
		
		Args:
			chunk (int): Maximum chunk size for the text splitter.
			overlap (int): Number of overlapping characters or tokens between chunks.
		
		Returns:
			List produced by the operation.
		
		Raises:
			Error: Re-raised after the source exception is wrapped with structured Foo metadata."""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'AmazonBucketLoader'
			exception.method = 'split( self, *args ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception