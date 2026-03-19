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
	     Copyright ©  2022  Terry Eppler

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
    loaders.py
  </summary>
  ******************************************************************************************
'''
import glob
import os
from typing import Optional, List, Dict, Any
from langchain_community.document_loaders import (
	ArxivLoader,
	Docx2txtLoader,
	OutlookMessageLoader,
	PyPDFLoader,
	TextLoader as TextDocLoader,
	UnstructuredExcelLoader,
	UnstructuredEmailLoader,
	UnstructuredMarkdownLoader,
	UnstructuredPowerPointLoader,
	UnstructuredHTMLLoader,
	WikipediaLoader,
	WebBaseLoader,
	JSONLoader,
	GithubFileLoader,
	UnstructuredXMLLoader,
	RecursiveUrlLoader,
	PubMedLoader,
	OpenCityDataLoader,
	NotebookLoader,
	S3FileLoader
)

from langchain_google_community import ( GCSFileLoader, SpeechToTextLoader )
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.onedrive import OneDriveLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_community.document_loaders.sharepoint import SharePointLoader
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_google_community import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import S3DirectoryLoader
from langchain_google_community import GCSDirectoryLoader

from lxml import etree

import config as cfg
from boogr import Error

def throw_if( name: str, value: Any ) -> None:
	'''

		Purpose:
		-----------
		Simple guard which raises ValueError when `value` is falsy (None, empty).

		Parameters:
		-----------
		name (str): Variable name used in the raised message.
		value (Any): Value to validate.

		Returns:
		-----------
		None: Raises ValueError when `value` is falsy.

	'''
	if value is None:
		raise ValueError( f"Argument '{name}' cannot be empty!" )

class Loader( ):
	'''

		Purpose:
		--------
		Base class providing shared utilities for concrete loader wrappers.
		Encapsulates file validation, path resolution, and document splitting.

		Attributes:
		----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		-------
		_ensure_existing_file( self, path: str ) -> str
		_resolve_paths( self, pattern: str ) -> List[ str ]
		_split_documents( self, docs: List[ Document ], chunk: int=1000, overlap: int=200 ) ->
		List[ Document ]

	'''
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	pattern: Optional[ str ]
	expanded: Optional[ List[ str ] ]
	candidates: Optional[ List[ str ] ]
	resolved: Optional[ List[ str ] ]
	loader: Optional[ BaseLoader ]
	splitter: Optional[ RecursiveCharacterTextSplitter ]
	chunk_size: Optional[ int ]
	overlap_amount: Optional[ int ]
	
	def __init__( self ) -> None:
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
		'''

			Purpose:
			--------
			Ensure the given file path exists.

			Parameters:
			-----------
			path (str): Path to a file on disk.

			Returns:
			--------
			str: The validated file path.

		'''
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
			exception.module = 'loaders'
			exception.cause = 'Loader'
			exception.method = 'verify_exists( self, path: str ) -> str'
			raise exception
			
	def resolve_paths( self, pattern: str ) -> List[ str ] | None:
		'''

			Purpose:
			--------
			Normalize a string glob pattern or a list of paths to a list of real file paths.

			Parameters:
			-----------
			pattern (str | List[str]): Path pattern or list of file paths.

			Returns:
			--------
			List[str]: Validated list of file paths.

		'''
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
			exception.module = 'loaders'
			exception.cause = 'Loader'
			exception.method = 'resolve_paths( self, pattern: str ) -> List[ str ]'
			raise exception
			
	def load_documents( self, path: str, encoding: Optional[ str ], csv_args: Optional[ Dict[ str, Any ] ],
			source_column: Optional[ str ] ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load files into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the CSV file.
			encoding (Optional[str]): File encoding (e.g., 'utf-8') if known.
			source_column (Optional[str]): Column name used for source attribution.

			Returns:
			--------
			List[Document]: List of LangChain Document objects parsed from the CSV.

		'''
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
			exception.module = 'loaders'
			exception.cause = 'CSV'
			exception.method = 'load_documents( self, **kwargs )'
			raise exception
			
	def split_documents( self, docs: List[ Document ], chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split long Document objects into smaller chunks for better token management.

			Parameters:
			-----------
			docs (List[Document]): Input LangChain Document objects.
			chunk_size (int): Max characters in each chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Re-chunked list of Document objects.

		'''
		try:
			throw_if( 'docs', docs )
			self.documents = docs
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder( model_name='gpt-4o',
				chunk_size=self.chunk_size, overlap=self.overlap_amount )
			return self.splitter.split_documents( documents=self.documents )
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'Loader'
			exception.method = ('split_documents( self, docs: List[ Document ], chunk: int=1000, '
			                    'overlap: int=200 ) -> List[ Document ]')
			raise exception

class TextLoader( Loader ):
	'''

		Purpose:
		--------
		Provides LangChain's TextLoader functionality to parse plain-text files
		into Document objects.

		Attributes:
		-----------
		documents - List[ Document ]
		file_path - str
		pattern - str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int
		encoding - str

		Methods:
		--------
		verify_exists( self, path: str ) -> str,
		resolve_paths( self, pattern: str ) -> List[ str ],
		split_documents( self, docs: List[ Document ] ) -> List[ Document ]
		load( path: str, encoding: Optional[ str ]=None ) -> List[ Document ]
		split( ) -> List[ Document ]

	'''
	loader: Optional[ TextDocLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	encoding: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.encoding = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.

		'''
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
		         'encoding',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, path: str, encoding: Optional[ str ] = None ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load a plain-text file into LangChain Document objects.

			Parameters:
			-----------
			path (str): File path to the text file.
			encoding (Optional[str]): Optional text encoding such as 'utf-8'.

			Returns:
			--------
			List[Document]: List of parsed Document objects from the text file.

		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.encoding = encoding
			
			if self.encoding:
				self.loader = TextDocLoader(
					file_path=self.file_path,
					encoding=self.encoding )
			else:
				self.loader = TextDocLoader( file_path=self.file_path )
			
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'TextLoader'
			exception.method = (
					'load( self, path: str, encoding: Optional[ str ]=None ) '
					'-> List[ Document ]'
			)
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded text documents into manageable chunks for downstream LLM
			processing.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents(
				self.documents,
				chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'TextLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) '
					'-> List[ Document ]'
			)
			raise exception

class CsvLoader( Loader ):
	'''

		Purpose:
		--------
		Wrap LangChain's CSVLoader to load CSV files into LangChain Document objects.

	'''
	loader: Optional[ CSVLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	encoding: Optional[ str ]
	csv_args: Optional[ Dict[ str, Any ] ]
	source_column: Optional[ str ]
	delimiter: Optional[ str ]
	quotechar: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.loader = None
		self.file_path = None
		self.documents = [ ]
		self.encoding = None
		self.csv_args = None
		self.source_column = None
		self.delimiter = None
		self.quotechar = None
	
	def __dir__( self ) -> List[ str ]:
		return [
				'loader',
				'file_path',
				'documents',
				'encoding',
				'csv_args',
				'source_column',
				'delimiter',
				'quotechar',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load(
			self,
			path: str,
			encoding: Optional[ str ] = 'utf-8',
			source_column: Optional[ str ] = None,
			delimiter: str = ',',
			quotechar: str = '"' ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load a CSV file into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the CSV file.
			encoding (Optional[str]): File encoding.
			source_column (Optional[str]): Optional source column name.
			delimiter (str): CSV delimiter.
			quotechar (str): CSV quote character.

			Returns:
			--------
			List[Document] | None: Loaded Document objects.

		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.encoding = encoding
			self.source_column = source_column
			self.delimiter = delimiter
			self.quotechar = quotechar
			self.csv_args = {
					'delimiter': self.delimiter,
					'quotechar': self.quotechar,
			}
			self.loader = CSVLoader(
				file_path=self.file_path,
				source_column=self.source_column,
				csv_args=self.csv_args,
				encoding=self.encoding
			)
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'CsvLoader'
			exception.method = (
					'load( self, path: str, encoding: Optional[ str ]="utf-8", '
					'source_column: Optional[ str ]=None, delimiter: str=",", '
					'quotechar: str=\'"\' ) -> List[ Document ] | None'
			)
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded CSV documents into smaller chunks.

			Parameters:
			-----------
			chunk (int): Maximum number of characters per chunk.
			overlap (int): Number of overlapping characters between chunks.

			Returns:
			--------
			List[Document] | None: Chunked Document objects.

		'''
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
			exception.module = 'Foo'
			exception.cause = 'CsvLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) '
					'-> List[ Document ] | None'
			)
			raise exception

class WebLoader( Loader ):
	'''

		Purpose:
		--------
		Functionality to load all text from HTML webpages into
		a document format that we can use downstream.

		Attributes:
		----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int
		url - str
		loader - WebBaseLoader

		Methods:
		--------
		verify_exists( self, path: str ) -> str,
		resolve_paths( self, pattern: str ) -> List[ str ],
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ]
		load( self, urls: str | List[ str ] ) -> List[ Document ]
		split( self ) -> List[ Document ]

	'''
	loader: Optional[ RecursiveUrlLoader | WebBaseLoader ]
	url: Optional[ str ]
	web_paths: Optional[ str | List[ str ] ]
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	max_depth: Optional[ int ]
	tiemout: Optional[ int ]
	ignore: Optional[ bool ]
	with_progress: Optional[ bool ]
	recursive: Optional[ bool ]
	prevent_outside: Optional[ bool ]
	
	def __init__( self, recursive: bool = False, max_depth: int = 2,
			prevent_outside: bool = True, timeout: int = 10,
			ignore: bool = True, progress: bool = True ) -> None:
		'''

			Purpose:
			--------
			Initialize a WebLoader instance for either single-page loading
			or recursive crawling.

			Parameters:
			-----------
			recursive (bool): Indicates whether the loader should crawl
				recursively from a starting URL.
			max_depth (int): Maximum recursive crawl depth to use when
				recursive mode is enabled.
			prevent_outside (bool): Indicates whether recursively loaded
				documents should be restricted to the starting domain.
			timeout (int): Maximum request timeout in seconds.
			ignore (bool): Indicates whether fetch failures should be ignored.
			progress (bool): Indicates whether WebBaseLoader should display
				progress while loading multiple pages.

			Returns:
			--------
			None.

		'''
		super( ).__init__( )
		self.max_depth = max_depth
		self.tiemout = timeout
		self.url = None
		self.documents = None
		self.file_path = None
		self.web_paths = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.ignore = ignore
		self.with_progress = progress
		self.recursive = recursive
		self.prevent_outside = prevent_outside
	
	def __dir__( self ):
		'''

			Purpose:
			--------
			Return a list of all available members.

			Parameters:
			-----------
			None.

			Returns:
			--------
			List[ str ]: A list of attribute and method names available on
				the class.

		'''
		return [ 'loader',
		         'documents',
		         'splitter',
		         'pattern',
		         'file_path',
		         'expanded',
		         'max_depth',
		         'timeout',
		         'candidates',
		         'resolved',
		         'chunk_size',
		         'overlap_amount',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'load_pages',
		         'load_recursive',
		         'split',
		         'urls',
		         'recursive',
		         'prevent_outside', ]
	
	def _same_domain_only( self, docs: List[ Document ], source_url: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Filter recursively crawled documents so only documents from the
			same network location as the starting URL are retained.

			Parameters:
			-----------
			docs (List[ Document ]): Documents returned by the recursive
				web loader.
			source_url (str): Starting URL used to seed the crawl.

			Returns:
			--------
			List[ Document ] | None: Documents restricted to the same
				domain as the starting URL.

		'''
		try:
			throw_if( 'docs', docs )
			throw_if( 'source_url', source_url )
			
			from urllib.parse import urlparse
			
			_origin = urlparse( source_url ).netloc.lower( )
			_results = [ ]
			
			for d in docs:
				if not hasattr( d, 'metadata' ) or not isinstance( d.metadata, dict ):
					continue
				
				_source = d.metadata.get( 'source' ) or d.metadata.get( 'url' )
				if not isinstance( _source, str ) or not _source.strip( ):
					continue
				
				_netloc = urlparse( _source ).netloc.lower( )
				if _netloc == _origin:
					_results.append( d )
			
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'WebLoader'
			exception.method = ('_same_domain_only( self, docs: List[ Document ], '
			                    'source_url: str ) -> List[ Document ]')
			raise exception
	
	def load( self, urls: str | List[ str ] ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load either one or more web pages or recursively crawl from a
			starting URL, depending on the loader configuration.

			Parameters:
			-----------
			urls (str | List[ str ]): A single URL string or a list of URL
				strings to load.

			Returns:
			--------
			List[ Document ] | None: Parsed Document objects from fetched
				HTML content.

		'''
		try:
			throw_if( 'urls', urls )
			
			if self.recursive:
				if isinstance( urls, list ):
					if not urls:
						raise ValueError( 'No URLs were provided!' )
					self.url = urls[ 0 ]
				else:
					self.url = urls
				
				self.documents = self.load_recursive(
					url=self.url,
					depth=self.max_depth,
					max_time=self.tiemout,
					ignore=self.ignore
				)
				return self.documents
			else:
				if isinstance( urls, str ):
					self.web_paths = [ urls ]
				else:
					self.web_paths = urls
				
				self.documents = self.load_pages(
					urls=self.web_paths,
					depth=self.max_depth,
					timeout=self.tiemout,
					ignore=self.ignore,
					progress=self.with_progress
				)
				return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'WebLoader'
			exception.method = 'load( self, urls: str | List[ str ] ) -> List[ Document ]'
			raise exception
	
	def load_recursive( self, url: str, depth: int = 2, max_time: int = 10,
			ignore: bool = True ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Recursively crawl a starting URL and convert fetched HTML pages
			into Document objects.

			Parameters:
			-----------
			url (str): Starting URL for the crawl.
			depth (int): Maximum recursive crawl depth.
			max_time (int): Maximum request timeout in seconds.
			ignore (bool): Indicates whether fetch failures should be
				ignored.

			Returns:
			--------
			List[ Document ] | None: Parsed Document objects from fetched
				HTML content.

		'''
		try:
			throw_if( 'url', url )
			self.url = url
			self.max_depth = depth
			self.tiemout = max_time
			self.ignore = ignore
			self.loader = RecursiveUrlLoader(
				self.url,
				max_depth=self.max_depth,
				timeout=self.tiemout,
				continue_on_failure=self.ignore
			)
			self.documents = self.loader.load( )
			
			if self.prevent_outside:
				self.documents = self._same_domain_only(
					docs=self.documents,
					source_url=self.url
				)
			
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'WebLoader'
			exception.method = ('load_recursive( self, url: str, depth: int=2, '
			                    'max_time: int=10, ignore: bool=True ) -> List[ Document ]')
			raise exception
	
	def load_pages( self, urls: List[ str ], depth: int = 2, timeout: int = 10,
			ignore: bool = True, progress: bool = True ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load one or more web pages and convert them into Document objects.

			Parameters:
			-----------
			urls (List[ str ]): One or more URL strings to load.
			depth (int): Preserved for API compatibility with the recursive
				loader configuration.
			timeout (int): Maximum request timeout in seconds.
			ignore (bool): Indicates whether fetch failures should be
				ignored.
			progress (bool): Indicates whether WebBaseLoader should display
				progress while loading.

			Returns:
			--------
			List[ Document ] | None: Parsed Document objects from fetched
				HTML content.

		'''
		try:
			throw_if( 'urls', urls )
			self.web_paths = urls
			self.max_depth = depth
			self.tiemout = timeout
			self.ignore = ignore
			self.with_progress = progress
			self.loader = WebBaseLoader(
				web_paths=self.web_paths,
				show_progress=self.with_progress,
				continue_on_failure=self.ignore
			)
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'WebLoader'
			exception.method = ('load_pages( self, urls: List[ str ], depth: int=2, '
			                    'timeout: int=10, ignore: bool=True, '
			                    'progress: bool=True ) -> List[ Document ]')
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded web documents into smaller chunks for better
			LLM processing.

			Parameters:
			-----------
			chunk (int): Maximum number of characters per chunk.
			overlap (int): Number of overlapping characters between chunks.

			Returns:
			--------
			List[ Document ] | None: Chunked Document objects.

		'''
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
			exception.module = 'loaders'
			exception.cause = 'WebLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			raise exception
			
class PdfReader( Loader ):
	'''

		Purpose:
		--------
		Wrap LangChain's PyPDFLoader to extract and chunk PDF documents.
		
		Attributes:
		----------
		loader - SharePointLoader
		file_path - str
		documents - List[ Documents }
		library_id - str
		mode - str
		folder_id - str
		object_ids - List[ str ]
		query - str
		with_token - bool
		is_recursive - bool
		
		Methods:
		--------
		verify_exists( self, path: str ) -> str,
		resolve_paths( self, pattern: str ) -> List[ str ],
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ]
		load( path: str, mode: str ) -> List[ Document ]
		split( ) -> List[ Document ]

	'''
	loader: Optional[ PyPDFLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
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
		         'mode',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, path: str, mode: str='single' ) -> List[ Document ] | None:
		'''


			Purpose:
			--------
			Load a PDF file and convert its contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): File path to a PDF document.

			Returns:
			--------
			List[Document]: List of parsed Document objects from the PDF.


		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.mode = mode
			self.loader = PyPDFLoader( file_path=self.file_path, mode=self.mode )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'PdfLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		'''
			Purpose:
			--------
			Split loaded PDF documents into smaller chunks.
	
			Parameters:
			-----------
			chunk (int): Maximum number of characters per chunk.
			overlap (int): Number of overlapping characters between chunks.
	
			Returns:
			--------
			List[Document] | None: Chunked Document objects.
	
		'''
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
			exception.module = 'Foo'
			exception.cause = 'PdfReader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) '
					'-> List[ Document ] | None'
			)
			raise exception
			
class PdfLoader( PdfReader ):
	"""
	
		PdfLoader
	
		Public, SDK-oriented PDF loader with:
			- Page-aware metadata
			- Two-stage chunking
			- Configurable chunk profiles
			- Table isolation
			- Optional OCR fallback
			
	"""
	loader: Optional[ PyPDFLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	extraction: Optional[ str ]
	include_images: Optional[ bool ]
	image_format: Optional[ str ]
	custom_delimiter: Optional[ str ]
	image_parser: Optional[ RapidOCRBlobParser ]
	
	
	def __init__( self, size: int=1000, overlap: int=150,
			has_tables: bool=True, include: bool=True ) -> None:
		"""
		
			Purpose:
			---------
			Initialize the PdfLoader.
	
			Parameters:
				path:
					Path to the PDF file.
				size:
					Target chunk size (characters).
				overlap:
					Overlap between chunks.
				has_tables:
					Enable table detection and isolation.
				use_ocr:
					Enable OCR fallback for image-only PDFs.
		"""
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
		self.image_format = None
		self.custom_delimiter = None

	@property
	def mode_options( self ):
		'''
			
			Returns:
			--------
			A List[ str ] of mode options
		
		'''
		return [ 'page', 'single' ]

	@property
	def extraction_options( self ):
		'''
			
			Returns:
			--------
			A List[ str ] of mode options
		
		'''
		return [ 'plain', 'layout' ]

	@property
	def image_options( self ):
		'''
			
			Returns:
			--------
			A List[ str ] of mode options
		
		'''
		return [ 'html-img', 'markdown-img', 'text-img' ]
	
	def load( self, path: str, mode: str = 'single', extract: str = 'plain',
			include: bool = False, format: str = 'markdown-img' ) -> List[ Document ]:
		"""
		
			Purpose:
			---------
			Loads PDF document into Langchain document objects. Attempts image
			extraction only when explicitly requested, and falls back to text-only
			parsing if the image path fails.
		
			Parameters:
			-----------
			path:
				Path to the PDF file.
			mode:
				PDF loading mode passed to PyPDFLoader.
			extract:
				Extraction mode passed to PyPDFLoader.
			include:
				When True, attempt image extraction. Defaults to False for stability.
			format:
				Image output format used when image extraction is enabled.
		
			Returns:
			--------
			List[Document]
		"""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.mode = mode
			self.extraction = extract
			self.include_images = include
			self.image_format = format
			
			if self.include_images:
				try:
					self.image_parser = RapidOCRBlobParser( )
					self.loader = PyPDFLoader(
						file_path=self.file_path,
						mode=self.mode,
						extraction_mode=self.extraction,
						extract_images=self.include_images,
						images_inner_format=self.image_format,
						images_parser=self.image_parser )
					self.documents = self.loader.load( )
					return self.documents
				except Exception:
					self.loader = PyPDFLoader(
						file_path=self.file_path,
						mode=self.mode,
						extraction_mode=self.extraction )
					self.documents = self.loader.load( )
					return self.documents
			else:
				self.loader = PyPDFLoader(
					file_path=self.file_path,
					mode=self.mode,
					extraction_mode=self.extraction )
				self.documents = self.loader.load( )
				return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'PdfLoader'
			exception.method = 'load( self, path: str, mode: str=single, extract: str=plain ) -> List[ Document ]'
			raise exception

class ExcelLoader( Loader ):
	'''


		Purpose:
		--------
		Provides LangChain's UnstructuredExcelLoader functionality
		to parse Excel spreadsheets into documents.

		Attibutes:
		----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str,
		resolve_paths( self, pattern: str ) -> List[ str ],
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ]
		load( path: str, mode: str ) -> List[ Document ]
		split( ) -> List[ Document ]


	'''
	loader: Optional[ UnstructuredExcelLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	has_headers: Optional[ bool ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
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
	
	@property
	def mode_options( self ):
		'''
			
			Returns:
			-------
			List[ str ] of loading mode options
			
		'''
		return [ 'single', 'page' ]
	
	def load( self, path: str, mode: str = 'elements', has_headers: bool = True ) -> List[
		                                                                                 Document ] | None:
		'''


			Purpose:
			--------
			Load and convert Excel data into LangChain Document objects.

			Parameters:
			-----------
			path (str): File path to the Excel spreadsheet.
			mode (str): Extraction mode, either 'elements' or 'paged'.
			headers (bool): Whether to include column headers in parsing.

			Returns:
			--------
			List[Document]: List of parsed Document objects from Excel content.


		'''
		try:
			throw_if( 'path', path )
			self.mode = mode
			self.has_headers = has_headers
			self.file_path = self.verify_exists( path )
			self.loader = UnstructuredExcelLoader( file_path=self.file_path, mode=self.mode )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'ExcelLoader'
			exception.method = 'load( self, **kwargs ) -> List[ Document ]'
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		'''


			Purpose:
			--------
			Split loaded Excel documents into manageable chunks.

			Parameters:
			-----------
			chunk_size (int): Maximum characters per chunk.
			chunk_overlap (int): Characters overlapping between chunks.

			Returns:
			--------
			List[Document]: Chunked and cleaned list of Document objects.


		'''
		try:
			if self.documents is None:
				raise ValueError( 'No documents loaded!' )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=overlap )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'ExcelLoader'
			exception.method = 'split( self,  **kwargs  ) -> List[ Document ]'
			raise exception

class WordLoader( Loader ):
	'''


		Purpose:
		--------
		Provides LangChain's Docx2txtLoader functionality to
		convert docx files into Document objects.

		Attributes:
		----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str,
		resolve_paths( self, pattern: str ) -> List[ str ],
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ]
		load( path: str, mode: str ) -> List[ Document ]
		split( ) -> List[ Document ]


	'''
	loader: Optional[ Docx2txtLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.documents = None
		self.file_path = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
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
		'''


			Purpose:
			--------
			Load the contents of a Word .docx file into LangChain Document objects.

			Parameters:
			-----------
			path (str): File path to the .docx document.

			Returns:
			--------
			List[Document]: Parsed Document list from Word file.


		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = Docx2txtLoader( self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'WordLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		'''


			Purpose:
			--------
			Split Word documents into text chunks suitable for LLM processing.

			Parameters:
			-----------
			chunk_size (int): Maximum characters per chunk.
			chunk_overlap (int): Overlap between chunks in characters.

			Returns:
			--------
			List[Document]: Chunked list of Document objects.


		'''
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
			exception.module = 'loaders'
			exception.cause = 'WordLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			raise exception
			
class MarkdownLoader( Loader ):
	'''


		Purpose:
		--------
		Wrap LangChain's UnstructuredMarkdownLoader to parse Markdown files into Document objects.
		
		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int
		
		Methods:
		--------
		verify_exists( self, path: str ) -> str,
		resolve_paths( self, pattern: str ) -> List[ str ],
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ]
		load( path: str, mode: str ) -> List[ Document ]
		split( ) -> List[ Document ]


	'''
	loader: Optional[ UnstructuredMarkdownLoader ]
	file_path: str | None
	documents: List[ Document ] | None
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
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
		'''

			Purpose:
			--------
			Load a Markdown (.md) file into LangChain Document objects.

			Parameters:
			-----------
			path (str): File path to the Markdown file.

			Returns:
			--------
			List[Document]: List of parsed Document objects from the Markdown file.

		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = UnstructuredMarkdownLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'MarkdownLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ] '
			raise exception
			
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Markdown content into text chunks for LLM consumption.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Number of characters that overlap between chunks.

			Returns:
			--------
			List[Document]: Split Document chunks from the original Markdown content.

		'''
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			_documents = self.split_documents( docs=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return _documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'MarkdownLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			raise exception

class HtmlLoader( Loader ):
	'''

		Purpose:
		--------
		Provides the UnstructuredHTMLLoader's functionality to parse HTML files into Document objects.

		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str,
		resolve_paths( self, pattern: str ) -> List[ str ],
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ]
		load( path: str, mode: str ) -> List[ Document ]
		split( ) -> List[ Document ]

	'''
	loader: Optional[ UnstructuredHTMLLoader ]
	file_path: str | None
	documents: List[ Document ] | None
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
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
		'''

			Purpose:
			--------
			Load an HTML file and convert its contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = UnstructuredHTMLLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'HTML'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded HTML documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
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
			exception.module = 'loaders'
			exception.cause = 'HtmlLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			raise exception
	
class ArXivLoader( Loader ):
	'''

		Purpose:
		--------
		Provides the Arxiv loading functionality
		to parse video research papers into Document objects.
		
		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int
		
		Methods:
		--------
		verify_exists( self, path: str ) -> str;
		resolve_paths( self, pattern: str ) -> List[ str ];
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ];
		load( path: str, mode: str ) -> List[ Document ];
		split( ) -> List[ Document ];

	'''
	loader: Optional[ ArxivLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	max_documents: Optional[ int ]
	max_characters: Optional[ int ]
	include_metadata: Optional[ bool ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.max_documents = 2
		self.max_characters = 1000
		self.include_metadata = False
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
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
	
	def load( self, question: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load an video file and convert its contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'question', question )
			self.query = question
			self.loader = ArxivLoader( query=self.query,
				doc_content_chars_max=self.max_characters )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'ArxivLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			raise exception
			
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Youtube Transcript documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
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
			exception.cause = 'ArxivLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			raise exception
			
class WikiLoader( Loader ):
	'''

		Purpose:
		--------
		Provides the Arxiv loading functionality
		to parse video research papers into Document objects.
		
		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int
		
		Methods:
		--------
		verify_exists( self, path: str ) -> str;
		resolve_paths( self, pattern: str ) -> List[ str ];
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ];
		load( path: str, mode: str ) -> List[ Document ];
		split( ) -> List[ Document ];

	'''
	loader: Optional[ WikipediaLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	max_documents: Optional[ int ]
	max_characters: Optional[ int ]
	include_all: Optional[ bool ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.max_documents = 25
		self.max_characters = 4000
		self.include_all
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
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
		         'include_all',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, question: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load an wikipedia and convert its contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'question', question )
			self.query = question
			self.loader = WikipediaLoader( query=self.query, max_documents=self.max_documents,
				load_all_available_meta=self.include_all, doc_content_chars_max=self.max_characters )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'WikiLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			raise exception
			
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Wikipedia search documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
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
			exception.cause = 'WikiLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			raise exception
			
class OutlookLoader( Loader ):
	'''

		Purpose:
		--------
		Provides the Arxiv loading functionality
		to parse video research papers into Document objects.
		
		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int
		
		Methods:
		--------
		verify_exists( self, path: str ) -> str;
		resolve_paths( self, pattern: str ) -> List[ str ];
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ];
		load( path: str, mode: str ) -> List[ Document ];
		split( ) -> List[ Document ];

	'''
	loader: Optional[ OutlookMessageLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	max_documents: Optional[ int ]
	max_characters: Optional[ int ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
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
		'''

			Returns:
			--------
			A list of all available members.


		'''
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
		'''

			Purpose:
			--------
			Load Outlook Message from a path converting contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = OutlookMessageLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'OutlookLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			raise exception
			
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Wikipedia search documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
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
			exception.cause = 'OutlookLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			raise exception
			
class SpfxLoader( Loader ):
	'''

		Purpose:
		--------
		Provides the Sharepoint loading functionality
		to parse video research papers into Document objects.
		
		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int
		loader - SharePointLoader
		library_id - str
		subsite_id - str
		folder_id - str
		object_ids - List[ str ]
		query - str
		with_token - bool
		is_recursive - bool
		
		Methods:
		--------
		verify_exists( self, path: str ) -> str;
		resolve_paths( self, pattern: str ) -> List[ str ];
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ];
		load( path: str, mode: str ) -> List[ Document ];
		split( ) -> List[ Document ];

	'''
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
		'''

			Returns:
			--------
			A list of all available members.


		'''
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
		'''

			Purpose:
			--------
			Load Sharepoint files and convert their contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
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
			exception.module = 'loaders'
			exception.cause = 'SpfxLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			raise exception
			
	
	def load_folder( self, library_id: str, folder_id: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load Sharepoint files and convert their contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'library_id', library_id )
			throw_if( 'folder_id', folder_id)
			self.library_id = library_id
			self.folder_id = folder_id
			self.loader = SharePointLoader( document_library_id=self.library_id, folder_id=self.folder_id )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'SpfxLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			raise exception
			
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Sharepoint file documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
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
			exception.cause = 'SpfxLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			raise exception
			
class PowerPointLoader( Loader ):
	'''

		Purpose:
		--------
		Provides PowerPoint loading functionality
		to parse ppt files into Document objects.
		
		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int
		
		Methods:
		--------
		verify_exists( self, path: str ) -> str;
		resolve_paths( self, pattern: str ) -> List[ str ];
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ];
		load( path: str, mode: str ) -> List[ Document ];
		split( ) -> List[ Document ];

	'''
	loader: Optional[ UnstructuredPowerPointLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
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
		         'mode',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	
	def load( self, path: str, mode: str='single' ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load PowerPoint slides and convert their content into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.mode = mode
			self.loader = UnstructuredPowerPointLoader( file_path=self.file_path, mode=self.mode  )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'PowerPointLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			raise exception
			
	
	def load_multiple( self, path: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load PowerPoint slides and convert their content into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.mode = 'multiple'
			self.loader = UnstructuredPowerPointLoader( file_path=self.file_path, mode=self.mode )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'PowerPointLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			raise exception
			
			
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Wikipedia search documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
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
			exception.cause = 'PowerPointLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			raise exception
			
class OneDriveDocLoader( Loader ):
	'''

		Purpose:
		--------
		Provides OneDrvie loading functionality
		to parse contents into Document objects.
		
		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int
		
		Methods:
		--------
		verify_exists( self, path: str ) -> str;
		resolve_paths( self, pattern: str ) -> List[ str ];
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ];
		load( path: str, mode: str ) -> List[ Document ];
		split( ) -> List[ Document ];

	'''
	loader: Optional[ OneDriveLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	client_id: Optional[ str ]
	drive_id: Optional[ str ]
	client_secret: Optional[ str ]
	
	def __init__( self ) -> None:
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
		'''

			Returns:
			--------
			A list of all available members.


		'''
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
	
	def load( self, id: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load an onedrive file and convert its contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'id', id )
			self.drive_id = id
			self.loader = OneDriveLoader( drive_id=self.drive_id )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'WikiLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			raise exception
			

	def load_folder( self, id: str, path: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load an onedrive file and convert its contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'id', id )
			self.drive_id = id
			self.file_path = path
			self.loader = OneDriveLoader( drive_id=self.drive_id, folder_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'WikiLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			raise exception
			
			
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Wikipedia search documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
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
			exception.cause = 'WikiLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			raise exception
			
class GoogleCloudFileLoader( Loader ):
	'''

		Purpose:
		--------
		Provides Google Drive loading functionality
		to parse contents into Document objects.
		
		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int
		
		Methods:
		--------
		verify_exists( self, path: str ) -> str;
		resolve_paths( self, pattern: str ) -> List[ str ];
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ];
		load( path: str, mode: str ) -> List[ Document ];
		split( ) -> List[ Document ];

	'''
	loader: Optional[ GCSFileLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	document_id: Optional[ str ]
	folder_id: Optional[ str ]
	query: Optional[ str ]
	is_recursive: Optional[ bool ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.document_id = None
		self.folder_id = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.is_recursive = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
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
		         'document_Id',
		         'is_recursive',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'load_folder',
		         'split', ]
	
	def load( self, id: str, recursive: bool=False ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load an google drive file by id and convert its contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'id', id )
			throw_if( 'recursive', recursive )
			self.document_id = id
			self.is_recursive = recursive
			self.loader = GCSFileLoader( document_ids=[ self.document_id ],
				recursive=self.is_recursive )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleDriveLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			raise exception
			

	def load_folder( self, id: str, recursive: bool=True ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load an google drive file and convert its contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'id', id )
			self.folder_id = id
			self.is_recursive = recursive
			self.loader = GCSFileLoader( folder_id=self.folder_id, recursive=self.is_recursive )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleDriveLoader'
			exception.method = 'load_folder( self, path: str ) -> List[ Document ]'
			raise exception
			
			
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded google drive documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
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
			exception.cause = 'GoogleDriveLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			raise exception
			
class EmailLoader( Loader ):
	'''


		Purpose:
		--------
		Provides LangChain's UnstructuredEmailLoader functionality
		to parse email documents (*.eml) into documents.

		Attibutes:
		----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str,
		resolve_paths( self, pattern: str ) -> List[ str ],
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ]
		load( path: str, mode: str ) -> List[ Document ]
		split( ) -> List[ Document ]


	'''
	loader: Optional[ UnstructuredEmailLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	has_attachments: Optional[ bool ]
	mode: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
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
	
	def load( self, path: str, mode: str='single', attachments: bool=True ) -> List[ Document ] | None:
		'''


			Purpose:
			--------
			Load and convert Email data (*.eml) into LangChain Document objects.

			Parameters:
			-----------
			path (str): File path to the Excel spreadsheet.
			mode (str): Extraction mode, either 'elements' or 'paged'.
			include_headers (bool): Whether to include column headers in parsing.

			Returns:
			--------
			List[Document]: List of parsed Document objects from Email content.


		'''
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
			exception.module = 'loaders'
			exception.cause = 'EmailLoader'
			exception.method = ('load( self, path: str, mode: str=elements, '
			                    'include_headers: bool=True ) -> List[ Document ]')
			raise exception
			
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''


			Purpose:
			--------
			Split loaded Email documents into manageable chunks.

			Parameters:
			-----------
			chunk_size (int): Maximum characters per chunk.
			chunk_overlap (int): Characters overlapping between chunks.

			Returns:
			--------
			List[Document]: Chunked and cleaned list of Document objects.


		'''
		try:
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'EmailLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			raise exception

class JsonLoader( Loader ):
	'''

		Purpose:
		--------
		Provides the UnstructuredHTMLLoader's functionality to parse HTML files into Document objects.

		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str,
		resolve_paths( self, pattern: str ) -> List[ str ],
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ]
		load( path: str, mode: str ) -> List[ Document ]
		split( ) -> List[ Document ]

	'''
	loader: Optional[ JSONLoader ]
	file_path: str | None
	jq: Optional[ str ]
	is_text: Optional[ bool ]
	is_lines: Optional[ bool ]
	documents: List[ Document ] | None
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.is_text = None
		self.is_lines = None
		self.jq = '.messages[].content'
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
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
	
	def load( self, filepath: str, is_text: bool = True, is_lines: bool = False ) -> List[
		                                                                                 Document ] | None:
		'''

			Purpose:
			--------
			Load an HTML file and convert its contents into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'filepath', filepath )
			self.file_path = self.verify_exists( filepath )
			self.is_text = is_text
			self.is_lines = is_lines
			self.loader = JSONLoader( file_path=self.file_path, jq_schema=self.jq,
				text_content=self.is_text, json_lines=self.is_lines )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'JsonLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded HTML documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
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
			exception.module = 'loaders'
			exception.cause = 'JsonLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			raise exception

class GithubLoader( Loader ):
	'''

		Purpose:
		--------
		Provides the functionality to laod github files in to langchain documents

		Attributes:
		-----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int

		Methods:
		--------
		verify_exists( self, path: str ) -> str;
		resolve_paths( self, pattern: str ) -> List[ str ];
		split_documents( self, docs: List[ Document ]  ) -> List[ Document ];
		load( path: str, mode: str ) -> List[ Document ];
		split( ) -> List[ Document ];

	'''
	loader: Optional[ GithubFileLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	max_documents: Optional[ int ]
	max_characters: Optional[ int ]
	include_all: Optional[ bool ]
	query: Optional[ str ]
	repo: Optional[ str ]
	branch: Optional[ str ]
	access_token: Optional[ str ]
	github_url: Optional[ str ]
	file_filter: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.max_documents = None
		self.max_characters = None
		self.include_all = None
		self.github_url = None
		self.repo = None
		self.branch = None
		self.file_filter = None
	
	def __dir__( self ):
		'''

			Returns:
			--------
			A list of all available members.


		'''
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
		         'include_all',
		         'repo',
		         'branch',
		         'file_filter',
		         'verify_exists',
		         'resolve_paths',
		         'split_documents',
		         'load',
		         'split', ]
	
	def load( self, url: str, repo: str, branch: str, filetype: str = '.md' ) -> List[
		                                                                             Document ] | None:
		'''

			Purpose:
			--------
			Load filtered contents of Github repo/branch into LangChain Document objects.

			Parameters:
			-----------
			url (str):
			repo (str):
			branch (str):
			filetype (str):

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'url', url )
			self.github_url = url
			self.repo = repo
			self.branch = branch
			self.pattern = filetype
			self.file_filter = lambda file_path: file_path.endswith( self.pattern )
			self.loader = GithubFileLoader( repo=self.repo, branch=self.branch,
				github_api_url=self.github_url, file_filter=self.file_filter )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GithubLoader'
			exception.method = 'load( self, **kwargs  ) -> List[ Document ]'
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Wikipedia search documents into manageable text chunks.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.

			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.

		'''
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
			exception.module = 'loaders'
			exception.cause = 'GithubLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			raise exception

class XmlLoader( Loader ):
	"""
	Purpose:
	--------
	Load XML files using two explicit and independent paths:

	1. Unstructured semantic loading via LangChain's UnstructuredXMLLoader,
	   producing LangChain Document objects suitable for RAG and embeddings.

	2. Structured XML loading via lxml, producing an ElementTree suitable for
	   XPath queries and schema-aware processing.

	Attributes:
	----------
	file_path : Optional[str]
		Path to the XML file.

	documents : Optional[List[Document]]
		Documents produced by UnstructuredXMLLoader.

	loader : Optional[UnstructuredXMLLoader]
		Active unstructured loader instance.

	splitter : Optional[RecursiveCharacterTextSplitter]
		Text splitter for document chunking.

	chunk_size : Optional[int]
		Chunk size for splitting documents.

	overlap_amount : Optional[int]
		Overlap size for splitting documents.

	xml_tree : Optional[etree._ElementTree]
		Parsed XML tree produced by lxml.

	xml_root : Optional[etree._Element]
		Root element of the parsed XML tree.

	xml_namespaces : Optional[Dict[str, str]]
		Namespace mapping extracted from the XML root.

	Public Methods:
	---------------
	load
	split
	load_tree
	get_elements
	"""
	
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
		"""
		
			Returns:
			--------
			
			List[str]
				List of available members.
				
		"""
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
		"""
			
			Purpose:
			--------
			Load an XML file using LangChain's UnstructuredXMLLoader to produce
			semantic Document objects.
	
			Parameters:
			-----------
			filepath : str
				Path to the XML file.
	
			Returns:
			--------
			List[Document] | None
				Parsed LangChain Document objects.
				
		"""
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
			raise exception
	
	def split( self, size: int = 1000, amount: int = 200 ) -> List[ Document ] | None:
		"""
			
			Purpose:
			--------
			Split loaded unstructured Documents into smaller chunks.
	
			Parameters:
			-----------
			size : int
				Maximum number of characters per chunk.
	
			amount : int
				Number of overlapping characters between chunks.
	
			Returns:
			--------
			List[Document] | None
				Split Document chunks.
			
		"""
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
			raise exception
	
	def load_tree( self, filepath: str ) -> etree._ElementTree | None:
		"""
			
			Purpose:
			--------
			Parse an XML file into a structured lxml ElementTree and store it
			on the instance.
	
			Parameters:
			-----------
			filepath : str
				Path to the XML file.
	
			Returns:
			--------
			etree._ElementTree | None
				Parsed XML tree.
			
		"""
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
			raise exception
	
	def get_elements( self, xpath: str ) -> List[ etree._Element ] | None:
		"""
		
			Purpose:
			--------
			Retrieve XML elements using an XPath expression against the
			loaded XML tree.
	
			Parameters:
			-----------
			xpath : str
				XPath expression.
	
			Returns:
			--------
			List[etree._Element] | None
				Matching XML elements.
			
		"""
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
			raise exception

class PubMedSearchLoader( Loader ):
	'''

		Purpose:
		--------
		Provides PubMed loading functionality for biomedical literature search results.

	'''
	loader: Optional[ PubMedLoader ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	max_docs: Optional[ int ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.query = None
		self.max_docs = None
	
	def __dir__( self ) -> List[ str ]:
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
		'''

			Purpose:
			--------
			Load PubMed search results into LangChain Document objects.

			Parameters:
			-----------
			query (str): PubMed search query.
			max_docs (int): Maximum number of records to load.

			Returns:
			--------
			List[Document] | None: Loaded PubMed documents.

		'''
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
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
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
			raise exception

class OpenCityLoader( Loader ):
	'''

		Purpose:
		--------
		Provides Open City Data loading functionality backed by Socrata.

	'''
	loader: Optional[ OpenCityDataLoader ]
	documents: Optional[ List[ Document ] ]
	city_id: Optional[ str ]
	dataset_id: Optional[ str ]
	limit: Optional[ int ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.city_id = None
		self.dataset_id = None
		self.limit = None
	
	def __dir__( self ) -> List[ str ]:
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
		'''

			Purpose:
			--------
			Load records from an Open City Data dataset into LangChain Document objects.

			Parameters:
			-----------
			city_id (str): City domain identifier such as 'data.sfgov.org'.
			dataset_id (str): Dataset identifier such as 'vw6y-z8j6'.
			limit (int): Maximum number of records to load.

			Returns:
			--------
			List[Document] | None: Loaded city data records.

		'''
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
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
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
			raise exception

class JupyterNotebookLoader( Loader ):
	'''

		Purpose:
		--------
		Provides Jupyter Notebook loading functionality for .ipynb files.

	'''
	loader: Optional[ NotebookLoader ]
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	include_outputs: Optional[ bool ]
	max_output_length: Optional[ int ]
	remove_newline: Optional[ bool ]
	traceback: Optional[ bool ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.file_path = None
		self.include_outputs = None
		self.max_output_length = None
		self.remove_newline = None
		self.traceback = None
	
	def __dir__( self ) -> List[ str ]:
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
		'''

			Purpose:
			--------
			Load a Jupyter notebook into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the .ipynb notebook file.
			include_outputs (bool): Include cell outputs when True.
			max_output_length (int): Max output characters to include.
			remove_newline (bool): Remove newline characters when True.
			traceback (bool): Include full traceback output when True.

			Returns:
			--------
			List[Document] | None: Loaded notebook document(s).

		'''
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
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
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
			raise exception

class GoogleCloudStorageFileLoader( Loader ):
	'''

		Purpose:
		--------
		Provides Google Cloud Storage file loading functionality.

	'''
	loader: Optional[ GCSFileLoader ]
	documents: Optional[ List[ Document ] ]
	project_name: Optional[ str ]
	bucket: Optional[ str ]
	blob: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.project_name = None
		self.bucket = None
		self.blob = None
	
	def __dir__( self ) -> List[ str ]:
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
		'''

			Purpose:
			--------
			Load a single Google Cloud Storage object into LangChain Document objects.

			Parameters:
			-----------
			project_name (str): Google Cloud project name or ID.
			bucket (str): GCS bucket name.
			blob (str): GCS object name.

			Returns:
			--------
			List[Document] | None: Loaded document(s).

		'''
		try:
			throw_if( 'project_name', project_name )
			throw_if( 'bucket', bucket )
			throw_if( 'blob', blob )
			self.project_name = project_name
			self.bucket = bucket
			self.blob = blob
			self.loader = GCSFileLoader(
				project_name=self.project_name,
				bucket=self.bucket,
				blob=self.blob
			)
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
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
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
			exception.cause = 'GoogleCloudStorageFileLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None'
			)
			raise exception

class AwsFileLoader( Loader ):
	'''

		Purpose:
		--------
		Provides AWS S3 file loading functionality.

	'''
	loader: Optional[ S3FileLoader ]
	documents: Optional[ List[ Document ] ]
	bucket: Optional[ str ]
	key: Optional[ str ]
	aws_access_key_id: Optional[ str ]
	aws_secret_access_key: Optional[ str ]
	aws_session_token: Optional[ str ]
	region_name: Optional[ str ]
	
	def __init__( self ) -> None:
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
	
	def load(
			self,
			bucket: str,
			key: str,
			aws_access_key_id: Optional[ str ] = None,
			aws_secret_access_key: Optional[ str ] = None,
			aws_session_token: Optional[ str ] = None,
			region_name: Optional[ str ] = None ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load a single AWS S3 object into LangChain Document objects.

			Parameters:
			-----------
			bucket (str): S3 bucket name.
			key (str): S3 object key.
			aws_access_key_id (Optional[str]): Optional AWS access key.
			aws_secret_access_key (Optional[str]): Optional AWS secret key.
			aws_session_token (Optional[str]): Optional AWS session token.
			region_name (Optional[str]): Optional AWS region.

			Returns:
			--------
			List[Document] | None: Loaded document(s).

		'''
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
			
			self.loader = S3FileLoader( self.bucket, self.key, **kwargs )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'AwsFileLoader'
			exception.method = (
					'load( self, bucket: str, key: str, aws_access_key_id: Optional[ str ]=None, '
					'aws_secret_access_key: Optional[ str ]=None, '
					'aws_session_token: Optional[ str ]=None, region_name: Optional[ str ]=None ) '
					'-> List[ Document ] | None'
			)
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
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
					'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None'
			)
			raise exception

class GoogleSpeechToTextAudioLoader( Loader ):
	'''

		Purpose:
		--------
		Provides Google Speech-to-Text loading functionality for audio transcription.

	'''
	loader: Optional[ SpeechToTextLoader ]
	documents: Optional[ List[ Document ] ]
	project_id: Optional[ str ]
	file_path: Optional[ str ]
	config: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.project_id = None
		self.file_path = None
		self.config = None
	
	def __dir__( self ) -> List[ str ]:
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
	
	def load(
			self,
			project_id: str,
			file_path: str,
			config: Optional[ Dict[ str, Any ] ] = None ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Transcribe audio with Google Cloud Speech-to-Text and load the transcript
			into LangChain Document objects.

			Parameters:
			-----------
			project_id (str): Google Cloud project ID.
			file_path (str): Local path or gs:// URI for the audio file.
			config (Optional[Dict[str, Any]]): Optional recognition config.

			Returns:
			--------
			List[Document] | None: Loaded transcript document(s).

		'''
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
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
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
					'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None'
			)
			raise exception

class GoogleBucketLoader( Loader ):
	'''

		Purpose:
		--------
		Provides Google Cloud Storage bucket loading functionality using
		LangChain's GCSDirectoryLoader.

	'''
	loader: Optional[ GCSDirectoryLoader ]
	documents: Optional[ List[ Document ] ]
	project_name: Optional[ str ]
	bucket: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.project_name = None
		self.bucket = None
	
	def __dir__( self ) -> List[ str ]:
		return [
				'loader',
				'documents',
				'project_name',
				'bucket',
				'chunk_size',
				'overlap_amount',
				'load',
				'split',
				'split_documents',
		]
	
	def load( self, project_name: str, bucket: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load all supported objects from a Google Cloud Storage bucket into
			LangChain Document objects.

			Parameters:
			-----------
			project_name (str): Google Cloud project name or project ID.
			bucket (str): Google Cloud Storage bucket name.

			Returns:
			--------
			List[Document] | None: Loaded bucket documents.

		'''
		try:
			throw_if( 'project_name', project_name )
			throw_if( 'bucket', bucket )
			self.project_name = project_name
			self.bucket = bucket
			self.loader = GCSDirectoryLoader(
				project_name=self.project_name,
				bucket=self.bucket
			)
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'loaders'
			exception.cause = 'GoogleBucketLoader'
			exception.method = (
					'load( self, project_name: str, bucket: str ) '
					'-> List[ Document ] | None'
			)
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded Google Cloud Storage bucket documents into smaller
			chunks for downstream processing.

			Parameters:
			-----------
			chunk (int): Maximum number of characters per chunk.
			overlap (int): Number of overlapping characters between chunks.

			Returns:
			--------
			List[Document] | None: Chunked Document objects.

		'''
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
			raise exception

class AmazonBucketLoader( Loader ):
	'''

		Purpose:
		--------
		Provides AWS S3 bucket/directory loading functionality using
		LangChain's S3DirectoryLoader.

	'''
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
	
	def load(
			self,
			bucket: str,
			prefix: Optional[ str ] = None,
			aws_access_key_id: Optional[ str ] = None,
			aws_secret_access_key: Optional[ str ] = None,
			aws_session_token: Optional[ str ] = None,
			region_name: Optional[ str ] = None,
			endpoint_url: Optional[ str ] = None ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load all supported objects from an AWS S3 bucket or bucket prefix
			into LangChain Document objects.

			Parameters:
			-----------
			bucket (str): AWS S3 bucket name.
			prefix (Optional[str]): Optional key prefix used to restrict loaded
				objects to a virtual folder or subtree.
			aws_access_key_id (Optional[str]): Optional AWS access key ID.
			aws_secret_access_key (Optional[str]): Optional AWS secret access key.
			aws_session_token (Optional[str]): Optional AWS session token.
			region_name (Optional[str]): Optional AWS region name.
			endpoint_url (Optional[str]): Optional custom S3-compatible endpoint.

			Returns:
			--------
			List[Document] | None: Loaded bucket documents.

		'''
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
			exception.method = (
					'load( self, bucket: str, prefix: Optional[ str ]=None, '
					'aws_access_key_id: Optional[ str ]=None, '
					'aws_secret_access_key: Optional[ str ]=None, '
					'aws_session_token: Optional[ str ]=None, '
					'region_name: Optional[ str ]=None, '
					'endpoint_url: Optional[ str ]=None ) -> List[ Document ] | None'
			)
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded AWS S3 bucket documents into smaller chunks for
			downstream processing.

			Parameters:
			-----------
			chunk (int): Maximum number of characters per chunk.
			overlap (int): Number of overlapping characters between chunks.

			Returns:
			--------
			List[Document] | None: Chunked Document objects.

		'''
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
			exception.cause = 'AmazonBucketLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) '
					'-> List[ Document ] | None'
			)
			raise exception