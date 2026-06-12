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
	"""Perform the throw if operation.

	Purpose:
		Executes the throw if operation using the existing Foo implementation. The method preserves
		original runtime behavior while exposing documentation compatible with MkDocs.

	Args:
		name (str): Value used by the operation.
		value (object): Value used by the operation.
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be None.' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty.' )

class Loader( ):
	"""Represent the Loader component.

	Purpose:
		Provides the Loader object used by Foo workflows. This class keeps its runtime state and
		public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
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
			Initializes the Loader instance with the default runtime state and configuration required by
			later method calls. The constructor preserves the original initialization behavior.
		"""
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
		"""Perform the verify exists operation.

		Purpose:
			Executes the verify exists operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			path (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'Loader'
			exception.method = '_ensure_existing_file( self, path: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def resolve_paths( self, pattern: str ) -> List[ str ] | None:
		"""Perform the resolve paths operation.

		Purpose:
			Executes the resolve paths operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			pattern (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'Loader'
			exception.method = 'resolve_paths( self, pattern: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def load_documents( self, path: str, encoding: Optional[ str ],
			csv_args: Optional[ Dict[ str, Any ] ],
			source_column: Optional[ str ] ) -> List[ Document ] | None:
		"""Perform the load documents operation.

		Purpose:
			Executes the load documents operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			path (str): Value used by the operation.
			encoding (Optional[str]): Value used by the operation.
			csv_args (Optional[Dict[str, Any]]): Value used by the operation.
			source_column (Optional[str]): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'CSV'
			exception.method = 'loader( )'
			Logger( ).write( exception )
			raise exception
	
	def split_documents( self, docs: List[ Document ], chunk: int = 1000, overlap: int = 200 ) -> \
			List[ Document ] | None:
		"""Perform the split documents operation.

		Purpose:
			Executes the split documents operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			docs (List[Document]): Value used by the operation.
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'Loader'
			exception.method = ('split_documents( self, **kwargs ) -> List[ Document ]')
			Logger( ).write( exception )
			raise exception

class TextLoader( Loader ):
	"""Represent the TextLoader component.

	Purpose:
		Provides the TextLoader object used by Foo workflows. This class keeps its runtime state and
		public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	splitter: Optional[ RecursiveCharacterTextSplitter | CharacterTextSplitter ]
	raw_text: Optional[ str ]
	separator: Optional[ str ]
	length_function: Optional[ object ]
	
	def __init__( self ) -> None:
		"""Initialize instance.

		Purpose:
			Initializes the TextLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
		super( ).__init__( )
		self.file_path = None
		self.raw_text = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.separator = "\n\n"
		self.length_function = len
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			filepath (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'TextLoader'
			exception.method = 'load( self, filepath: str ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split_tokens( self, size: int = 1000, amount: int = 200 ) -> List[ Document ] | None:
		"""Perform the split tokens operation.

		Purpose:
			Executes the split tokens operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			size (int): Value used by the operation.
			amount (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'TextLoader'
			exception.method = 'split_tokens( self, size: int=1000, amount: int=200 ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split_chars( self, size: int = 1000, amount: int = 200,
			seps: str = "\n\n" ) -> List[ Document ] | None:
		"""Perform the split chars operation.

		Purpose:
			Executes the split chars operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			size (int): Value used by the operation.
			amount (int): Value used by the operation.
			seps (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'TextLoader'
			exception.method = (
					'split_chars( self, size: int=1000, amount: int=200, '
					'seps: str="\\n\\n" ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class CsvLoader( Loader ):
	"""Represent the CsvLoader component.

	Purpose:
		Provides the CsvLoader object used by Foo workflows. This class keeps its runtime state and
		public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
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
			Initializes the CsvLoader instance with the default runtime state and configuration required
			by later method calls. The constructor preserves the original initialization behavior.
		"""
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
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			filepath (str): Value used by the operation.
			columns (Optional[List[str]]): Value used by the operation.
			delimiter (str): Value used by the operation.
			quotechar (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'CsvLoader'
			exception.method = (
					'load( self, filepath: str, columns: Optional[ List[ str ] ]=None, '
					'delimiter: str=",", quotechar: str=\'"\' ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, size: int = 1000, amount: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			size (int): Value used by the operation.
			amount (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'CsvLoader'
			exception.method = (
					'split( self, size: int=1000, amount: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class XmlLoader( Loader ):
	"""Represent the XmlLoader component.

	Purpose:
		Provides the XmlLoader object used by Foo workflows. This class keeps its runtime state and
		public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
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
		"""Initialize instance.

		Purpose:
			Initializes the XmlLoader instance with the default runtime state and configuration required
			by later method calls. The constructor preserves the original initialization behavior.
		"""
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
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			filepath (str): Value used by the operation.

		Returns:
			Result produced by the operation.
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
			Logger( ).write( exception )
			raise exception
	
	def split( self, size: int = 1000, amount: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			size (int): Value used by the operation.
			amount (int): Value used by the operation.

		Returns:
			Result produced by the operation.
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
			Logger( ).write( exception )
			raise exception
	
	def load_tree( self, filepath: str ) -> etree._ElementTree | None:
		"""Perform the load tree operation.

		Purpose:
			Executes the load tree operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			filepath (str): Value used by the operation.

		Returns:
			Result produced by the operation.
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
			Logger( ).write( exception )
			raise exception
	
	def get_elements( self, xpath: str ) -> List[ etree._Element ] | None:
		"""Perform the get elements operation.

		Purpose:
			Executes the get elements operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			xpath (str): Value used by the operation.

		Returns:
			Result produced by the operation.
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
			Logger( ).write( exception )
			raise exception

class WebLoader( Loader ):
	"""Represent the WebLoader component.

	Purpose:
		Provides the WebLoader object used by Foo workflows. This class keeps its runtime state and
		public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
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
			Initializes the WebLoader instance with the default runtime state and configuration required
			by later method calls. The constructor preserves the original initialization behavior.

		Args:
			recursive (bool): Value used by the operation.
			max_depth (int): Value used by the operation.
			prevent_outside (bool): Value used by the operation.
			timeout (int): Value used by the operation.
			ignore (bool): Value used by the operation.
			progress (bool): Value used by the operation.
		"""
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
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			urls (str | List[str]): Value used by the operation.
			depth (int): Value used by the operation.
			timeout (int): Value used by the operation.
			ignore (bool): Value used by the operation.
			progress (bool): Value used by the operation.
			prevent_outside (bool): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
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
		"""Perform the load pages operation.

		Purpose:
			Executes the load pages operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			urls (str | List[str]): Value used by the operation.
			timeout (int): Value used by the operation.
			ignore (bool): Value used by the operation.
			progress (bool): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
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
		"""Perform the load recursive operation.

		Purpose:
			Executes the load recursive operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			urls (str | List[str]): Value used by the operation.
			depth (int): Value used by the operation.
			timeout (int): Value used by the operation.
			ignore (bool): Value used by the operation.
			prevent_outside (bool): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'WebLoader'
			exception.method = (
					'load_recursive( self, urls: str | List[ str ], depth: int=2, '
					'timeout: int=10, ignore: bool=True, prevent_outside: bool=True ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'WebLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class PdfLoader( Loader ):
	"""Represent the PdfLoader component.

	Purpose:
		Provides the PdfLoader object used by Foo workflows. This class keeps its runtime state and
		public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
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
	
	def __init__( self, size: int = 1000, overlap: int = 150,
			has_tables: bool = True, include: bool = True ) -> None:
		"""Initialize instance.

		Purpose:
			Initializes the PdfLoader instance with the default runtime state and configuration required
			by later method calls. The constructor preserves the original initialization behavior.

		Args:
			size (int): Value used by the operation.
			overlap (int): Value used by the operation.
			has_tables (bool): Value used by the operation.
			include (bool): Value used by the operation.
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
		self.extraction = None
		self.image_format = None
		self.custom_delimiter = None
		self.image_parser = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
	def mode_options( self ) -> List[ str ]:
		"""Return mode options.

		Purpose:
			Returns configured option values exposed by this component for selection, validation, or
			display.

		Returns:
			Result produced by the operation.
		"""
		return [ 'page', 'single' ]
	
	@property
	def extraction_options( self ) -> List[ str ]:
		"""Return extraction options.

		Purpose:
			Returns configured option values exposed by this component for selection, validation, or
			display.

		Returns:
			Result produced by the operation.
		"""
		return [ 'plain', 'layout' ]
	
	@property
	def image_options( self ) -> List[ str ]:
		"""Return image options.

		Purpose:
			Returns configured option values exposed by this component for selection, validation, or
			display.

		Returns:
			Result produced by the operation.
		"""
		return [ 'html-img', 'markdown-img', 'text-img' ]
	
	def _normalize_mode( self, mode: str ) -> str:
		"""Perform the  normalize mode operation.

		Purpose:
			Executes the  normalize mode operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			mode (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		value = mode.strip( ).lower( ) if isinstance( mode, str ) else 'single'
		
		if value == 'elements':
			return 'page'
		
		if value not in self.mode_options:
			return 'single'
		
		return value
	
	def _normalize_extraction( self, extract: str ) -> str:
		"""Perform the  normalize extraction operation.

		Purpose:
			Executes the  normalize extraction operation using the existing Foo implementation. The
			method preserves original runtime behavior while exposing documentation compatible with
			MkDocs.

		Args:
			extract (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		value = extract.strip( ).lower( ) if isinstance( extract, str ) else 'plain'
		
		if value == 'ocr':
			return 'layout'
		
		if value not in self.extraction_options:
			return 'plain'
		
		return value
	
	def _normalize_image_format( self, format: str ) -> str:
		"""Perform the  normalize image format operation.

		Purpose:
			Executes the  normalize image format operation using the existing Foo implementation. The
			method preserves original runtime behavior while exposing documentation compatible with
			MkDocs.

		Args:
			format (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		value = format.strip( ).lower( ) if isinstance( format, str ) else 'markdown-img'
		
		if value == 'text':
			return 'markdown-img'
		
		if value not in self.image_options:
			return 'markdown-img'
		
		return value
	
	def load( self, filepath: str, mode: str = 'single', extract: str = 'plain',
			include: bool = False, format: str = 'markdown-img' ) -> List[ Document ]:
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			filepath (str): Value used by the operation.
			mode (str): Value used by the operation.
			extract (str): Value used by the operation.
			include (bool): Value used by the operation.
			format (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'PdfLoader'
			exception.method = (
					'load( self, filepath: str, mode: str="single", '
					'extract: str="plain", include: bool=False, '
					'format: str="markdown-img" ) -> List[ Document ]'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'PdfLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class ExcelLoader( Loader ):
	"""Represent the ExcelLoader component.

	Purpose:
		Provides the ExcelLoader object used by Foo workflows. This class keeps its runtime state
		and public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
	loader: Optional[ UnstructuredExcelLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	has_headers: Optional[ bool ]
	
	def __init__( self ) -> None:
		"""Initialize instance.

		Purpose:
			Initializes the ExcelLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
		self.has_headers = True
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
	def mode_options( self ) -> List[ str ]:
		"""Return mode options.

		Purpose:
			Returns configured option values exposed by this component for selection, validation, or
			display.

		Returns:
			Result produced by the operation.
		"""
		return [ 'single', 'elements' ]
	
	def _normalize_mode( self, mode: str ) -> str:
		"""Perform the  normalize mode operation.

		Purpose:
			Executes the  normalize mode operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			mode (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		value = mode.strip( ).lower( ) if isinstance( mode, str ) else 'single'
		
		if value in [ 'page', 'paged' ]:
			return 'elements'
		
		if value not in self.mode_options:
			return 'single'
		
		return value
	
	def load( self, path: str, mode: str = 'single',
			has_headers: bool = True ) -> List[ Document ] | None:
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			path (str): Value used by the operation.
			mode (str): Value used by the operation.
			has_headers (bool): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'ExcelLoader'
			exception.method = (
					'load( self, path: str, mode: str="single", '
					'has_headers: bool=True ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'ExcelLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class WordLoader( Loader ):
	"""Represent the WordLoader component.

	Purpose:
		Provides the WordLoader object used by Foo workflows. This class keeps its runtime state and
		public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
	loader: Optional[ Docx2txtLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	
	def __init__( self ) -> None:
		"""Initialize instance.

		Purpose:
			Initializes the WordLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
		super( ).__init__( )
		self.documents = None
		self.file_path = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			path (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = Docx2txtLoader( self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WordLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'WordLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class MarkdownLoader( Loader ):
	"""Represent the MarkdownLoader component.

	Purpose:
		Provides the MarkdownLoader object used by Foo workflows. This class keeps its runtime state
		and public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
	loader: Optional[ UnstructuredMarkdownLoader ]
	file_path: str | None
	documents: List[ Document ] | None
	mode: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize instance.

		Purpose:
			Initializes the MarkdownLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
	def mode_options( self ) -> List[ str ]:
		"""Return mode options.

		Purpose:
			Returns configured option values exposed by this component for selection, validation, or
			display.

		Returns:
			Result produced by the operation.
		"""
		return [ 'single', 'elements' ]
	
	def _normalize_mode( self, mode: str ) -> str:
		"""Perform the  normalize mode operation.

		Purpose:
			Executes the  normalize mode operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			mode (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		value = mode.strip( ).lower( ) if isinstance( mode, str ) else 'single'
		
		if value in [ 'page', 'paged' ]:
			return 'elements'
		
		if value not in self.mode_options:
			return 'single'
		
		return value
	
	def load( self, path: str, mode: str = 'single' ) -> List[ Document ] | None:
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			path (str): Value used by the operation.
			mode (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'MarkdownLoader'
			exception.method = (
					'load( self, path: str, mode: str="single" ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'MarkdownLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class HtmlLoader( Loader ):
	"""Represent the HtmlLoader component.

	Purpose:
		Provides the HtmlLoader object used by Foo workflows. This class keeps its runtime state and
		public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
	loader: Optional[ UnstructuredHTMLLoader ]
	file_path: str | None
	documents: List[ Document ] | None
	
	def __init__( self ) -> None:
		"""Initialize instance.

		Purpose:
			Initializes the HtmlLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			path (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = UnstructuredHTMLLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'HTML'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'HtmlLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class JsonLoader( Loader ):
	"""Represent the JsonLoader component.

	Purpose:
		Provides the JsonLoader object used by Foo workflows. This class keeps its runtime state and
		public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
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
			Initializes the JsonLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
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
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			filepath (str): Value used by the operation.
			jq_schema (str): Value used by the operation.
			content_key (Optional[str]): Value used by the operation.
			is_text (bool): Value used by the operation.
			is_lines (bool): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'JsonLoader'
			exception.method = (
					'load( self, filepath: str, jq_schema: str=".", '
					'content_key: Optional[ str ]=None, is_text: bool=True, '
					'is_lines: bool=False ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'JsonLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class ArXivLoader( Loader ):
	"""Represent the ArXivLoader component.

	Purpose:
		Provides the ArXivLoader object used by Foo workflows. This class keeps its runtime state
		and public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
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
			Initializes the ArXivLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
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
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			query (str): Value used by the operation.
			max_chars (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		try:
			throw_if( 'query', query )
			self.query = query
			self.max_characters = max_chars
			self.loader = ArxivLoader( query=self.query, doc_content_chars_max=self.max_characters )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'ArxivLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'ArxivLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class WikiLoader( Loader ):
	"""Represent the WikiLoader component.

	Purpose:
		Provides the WikiLoader object used by Foo workflows. This class keeps its runtime state and
		public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
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
			Initializes the WikiLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
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
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			query (str): Value used by the operation.
			max_docs (int): Value used by the operation.
			max_chars (int): Value used by the operation.
			include_all (bool): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = (
					'load( self, query: str, max_docs: int=25, max_chars: int=4000, '
					'include_all: bool=False ) -> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class GithubLoader( Loader ):
	"""Represent the GithubLoader component.

	Purpose:
		Provides the GithubLoader object used by Foo workflows. This class keeps its runtime state
		and public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
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
			Initializes the GithubLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
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
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			url (str): Value used by the operation.
			repo (str): Value used by the operation.
			branch (str): Value used by the operation.
			filetype (str): Value used by the operation.
			access_token (Optional[str]): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'GithubLoader'
			exception.method = (
					'load( self, url: str, repo: str, branch: str, '
					'filetype: str=".md", access_token: Optional[ str ]=None ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'GithubLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class PowerPointLoader( Loader ):
	"""Represent the PowerPointLoader component.

	Purpose:
		Provides the PowerPointLoader object used by Foo workflows. This class keeps its runtime
		state and public interface available for loading, fetching, generation, scraping, or
		supporting operations without altering the executable behavior of the original
		implementation.
	"""
	loader: Optional[ UnstructuredPowerPointLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize instance.

		Purpose:
			Initializes the PowerPointLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the  normalize mode operation.

		Purpose:
			Executes the  normalize mode operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			mode (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		value = mode.strip( ).lower( ) if isinstance( mode, str ) else 'single'
		
		if value == 'multiple':
			return 'elements'
		
		if value not in [ 'single', 'elements' ]:
			return 'single'
		
		return value
	
	def load( self, path: str, mode: str = 'single' ) -> List[ Document ] | None:
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			path (str): Value used by the operation.
			mode (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'PowerPointLoader'
			exception.method = (
					'load( self, path: str, mode: str="single" ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def load_multiple( self, path: str ) -> List[ Document ] | None:
		"""Perform the load multiple operation.

		Purpose:
			Executes the load multiple operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			path (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		try:
			return self.load( path, mode='elements' )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'PowerPointLoader'
			exception.method = 'load_multiple( self, path: str ) -> List[ Document ] | None'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'PowerPointLoader'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class OutlookLoader( Loader ):
	"""Represent the OutlookLoader component.

	Purpose:
		Provides the OutlookLoader object used by Foo workflows. This class keeps its runtime state
		and public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
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
			Initializes the OutlookLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.max_documents = 2
		self.max_characters = 1000
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			path (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = OutlookMessageLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'OutlookLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'OutlookLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class WebCrawler( Loader ):
	"""Represent the WebCrawler component.

	Purpose:
		Provides the WebCrawler object used by Foo workflows. This class keeps its runtime state and
		public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
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
			Initializes the WebCrawler instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.

		Args:
			url (str): Value used by the operation.
			recursive (bool): Value used by the operation.
			max_depth (int): Value used by the operation.
			prevent_outside (bool): Value used by the operation.
			timeout (int): Value used by the operation.
			ignore (bool): Value used by the operation.
			progress (bool): Value used by the operation.
		"""
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
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			urls (str | List[str]): Value used by the operation.
			depth (int): Value used by the operation.
			timeout (int): Value used by the operation.
			ignore (bool): Value used by the operation.
			progress (bool): Value used by the operation.
			prevent_outside (bool): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
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
		"""Perform the load pages operation.

		Purpose:
			Executes the load pages operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			urls (str | List[str]): Value used by the operation.
			timeout (int): Value used by the operation.
			ignore (bool): Value used by the operation.
			progress (bool): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
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
		"""Perform the load recursive operation.

		Purpose:
			Executes the load recursive operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			urls (str | List[str]): Value used by the operation.
			depth (int): Value used by the operation.
			timeout (int): Value used by the operation.
			ignore (bool): Value used by the operation.
			prevent_outside (bool): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'WebCrawler'
			exception.method = (
					'load_recursive( self, urls: str | List[ str ], depth: int=2, '
					'timeout: int=10, ignore: bool=True, prevent_outside: bool=True ) '
					'-> List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'WebCrawler'
			exception.method = (
					'split( self, chunk: int=1000, overlap: int=200 ) -> '
					'List[ Document ] | None'
			)
			Logger( ).write( exception )
			raise exception

class SpfxLoader( Loader ):
	"""Represent the SpfxLoader component.

	Purpose:
		Provides the SpfxLoader object used by Foo workflows. This class keeps its runtime state and
		public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
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
			Initializes the SpfxLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
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
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			library_id (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'SpfxLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def load_folder( self, library_id: str, folder_id: str ) -> List[ Document ] | None:
		"""Perform the load folder operation.

		Purpose:
			Executes the load folder operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			library_id (str): Value used by the operation.
			folder_id (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'SpfxLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'SpfxLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class OneDriveDocLoader( Loader ):
	"""Represent the OneDriveDocLoader component.

	Purpose:
		Provides the OneDriveDocLoader object used by Foo workflows. This class keeps its runtime
		state and public interface available for loading, fetching, generation, scraping, or
		supporting operations without altering the executable behavior of the original
		implementation.
	"""
	loader: Optional[ OneDriveLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	client_id: Optional[ str ]
	drive_id: Optional[ str ]
	client_secret: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize instance.

		Purpose:
			Initializes the OneDriveDocLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
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
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
	def file_options( self ) -> List[ str ]:
		"""Return file options.

		Purpose:
			Returns configured option values exposed by this component for selection, validation, or
			display.

		Returns:
			Result produced by the operation.
		"""
		return [ 'pdf', 'doc', 'docx', 'txt' ]
	
	def load( self, id: str ) -> List[ Document ] | None:
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			id (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		try:
			throw_if( 'id', id )
			self.drive_id = id
			self.loader = OneDriveLoader( drive_id=self.drive_id )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def load_folder( self, id: str, path: str ) -> List[ Document ] | None:
		"""Perform the load folder operation.

		Purpose:
			Executes the load folder operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			id (str): Value used by the operation.
			path (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		try:
			throw_if( 'id', id )
			self.drive_id = id
			self.file_path = path
			self.loader = OneDriveLoader( drive_id=self.drive_id, folder_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'WikiLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class GoogleLoader( Loader ):
	"""Represent the GoogleLoader component.

	Purpose:
		Provides the GoogleLoader object used by Foo workflows. This class keeps its runtime state
		and public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
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
			Initializes the GoogleLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
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
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
	def file_options( self ) -> List[ str ]:
		"""Return file options.

		Purpose:
			Returns configured option values exposed by this component for selection, validation, or
			display.

		Returns:
			Result produced by the operation.
		"""
		return [ 'document',
		         'sheet',
		         'pdf' ]
	
	def load_file( self, file_id: str, recursive: bool = False ) -> List[ Document ] | None:
		"""Perform the load file operation.

		Purpose:
			Executes the load file operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			file_id (str): Value used by the operation.
			recursive (bool): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'GoogleDriveLoader'
			exception.method = 'load_File( self, file_id: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def load_folder( self, folder_id: str, recursive: bool = False ) -> List[ Document ] | None:
		"""Perform the load folder operation.

		Purpose:
			Executes the load folder operation using the existing Foo implementation. The method
			preserves original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			folder_id (str): Value used by the operation.
			recursive (bool): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		try:
			throw_if( 'folder_id', folder_id )
			self.folder_id = folder_id
			self.is_recursive = recursive
			self.loader = GoogleDriveLoader( folder_id=self.folder_id, recursive=self.is_recursive )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'GoogleDriveLoader'
			exception.method = 'load_folder( self, path: str ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'GoogleDriveLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class EmailLoader( Loader ):
	"""Represent the EmailLoader component.

	Purpose:
		Provides the EmailLoader object used by Foo workflows. This class keeps its runtime state
		and public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
	loader: Optional[ UnstructuredEmailLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	has_attachments: Optional[ bool ]
	mode: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize instance.

		Purpose:
			Initializes the EmailLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			path (str): Value used by the operation.
			mode (str): Value used by the operation.
			attachments (bool): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			exception.module = 'chonky'
			exception.cause = 'EmailLoader'
			exception.method = ('load( self, path: str, mode: str=elements, '
			                    'include_headers: bool=True ) -> List[ Document ]')
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
		try:
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self.split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'chonky'
			exception.cause = 'EmailLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			Logger( ).write( exception )
			raise exception

class PubMedSearchLoader( Loader ):
	"""Represent the PubMedSearchLoader component.

	Purpose:
		Provides the PubMedSearchLoader object used by Foo workflows. This class keeps its runtime
		state and public interface available for loading, fetching, generation, scraping, or
		supporting operations without altering the executable behavior of the original
		implementation.
	"""
	loader: Optional[ PubMedLoader ]
	documents: Optional[ List[ Document ] ]
	query: Optional[ str ]
	max_docs: Optional[ int ]
	
	def __init__( self ) -> None:
		"""Initialize instance.

		Purpose:
			Initializes the PubMedSearchLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.query = None
		self.max_docs = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			query (str): Value used by the operation.
			max_docs (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
	"""Represent the OpenCityLoader component.

	Purpose:
		Provides the OpenCityLoader object used by Foo workflows. This class keeps its runtime state
		and public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
	loader: Optional[ OpenCityDataLoader ]
	documents: Optional[ List[ Document ] ]
	city_id: Optional[ str ]
	dataset_id: Optional[ str ]
	limit: Optional[ int ]
	
	def __init__( self ) -> None:
		"""Initialize instance.

		Purpose:
			Initializes the OpenCityLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.city_id = None
		self.dataset_id = None
		self.limit = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			city_id (str): Value used by the operation.
			dataset_id (str): Value used by the operation.
			limit (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
	"""Represent the JupyterNotebookLoader component.

	Purpose:
		Provides the JupyterNotebookLoader object used by Foo workflows. This class keeps its
		runtime state and public interface available for loading, fetching, generation, scraping, or
		supporting operations without altering the executable behavior of the original
		implementation.
	"""
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
			Initializes the JupyterNotebookLoader instance with the default runtime state and
			configuration required by later method calls. The constructor preserves the original
			initialization behavior.
		"""
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
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			path (str): Value used by the operation.
			include_outputs (bool): Value used by the operation.
			max_output_length (int): Value used by the operation.
			remove_newline (bool): Value used by the operation.
			traceback (bool): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
	"""Represent the GoogleCloudFileLoader component.

	Purpose:
		Provides the GoogleCloudFileLoader object used by Foo workflows. This class keeps its
		runtime state and public interface available for loading, fetching, generation, scraping, or
		supporting operations without altering the executable behavior of the original
		implementation.
	"""
	loader: Optional[ GCSFileLoader ]
	documents: Optional[ List[ Document ] ]
	project_name: Optional[ str ]
	bucket: Optional[ str ]
	blob: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize instance.

		Purpose:
			Initializes the GoogleCloudFileLoader instance with the default runtime state and
			configuration required by later method calls. The constructor preserves the original
			initialization behavior.
		"""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.project_name = None
		self.bucket = None
		self.blob = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			project_name (str): Value used by the operation.
			bucket (str): Value used by the operation.
			blob (str): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
	"""Represent the AwsFileLoader component.

	Purpose:
		Provides the AwsFileLoader object used by Foo workflows. This class keeps its runtime state
		and public interface available for loading, fetching, generation, scraping, or supporting
		operations without altering the executable behavior of the original implementation.
	"""
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
			Initializes the AwsFileLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
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
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			bucket (str): Value used by the operation.
			key (str): Value used by the operation.
			aws_access_key_id (Optional[str]): Value used by the operation.
			aws_secret_access_key (Optional[str]): Value used by the operation.
			aws_session_token (Optional[str]): Value used by the operation.
			region_name (Optional[str]): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
	"""Represent the GoogleSpeechToTextLoader component.

	Purpose:
		Provides the GoogleSpeechToTextLoader object used by Foo workflows. This class keeps its
		runtime state and public interface available for loading, fetching, generation, scraping, or
		supporting operations without altering the executable behavior of the original
		implementation.
	"""
	loader: Optional[ SpeechToTextLoader ]
	documents: Optional[ List[ Document ] ]
	project_id: Optional[ str ]
	file_path: Optional[ str ]
	config: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		"""Initialize instance.

		Purpose:
			Initializes the GoogleSpeechToTextLoader instance with the default runtime state and
			configuration required by later method calls. The constructor preserves the original
			initialization behavior.
		"""
		super( ).__init__( )
		self.loader = None
		self.documents = None
		self.project_id = None
		self.file_path = None
		self.config = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			project_id (str): Value used by the operation.
			file_path (str): Value used by the operation.
			config (Optional[Dict[str, Any]]): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
	"""Represent the GoogleBucketLoader component.

	Purpose:
		Provides the GoogleBucketLoader object used by Foo workflows. This class keeps its runtime
		state and public interface available for loading, fetching, generation, scraping, or
		supporting operations without altering the executable behavior of the original
		implementation.
	"""
	loader: Optional[ GCSDirectoryLoader ]
	documents: Optional[ List[ Document ] ]
	project_name: Optional[ str ]
	bucket: Optional[ str ]
	prefix: Optional[ str ]
	continue_on_failure: Optional[ bool ]
	
	def __init__( self ) -> None:
		"""Initialize instance.

		Purpose:
			Initializes the GoogleBucketLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
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
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			project_name (str): Value used by the operation.
			bucket (str): Value used by the operation.
			prefix (Optional[str]): Value used by the operation.
			continue_on_failure (bool): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
	"""Represent the AwsBucketLoader component.

	Purpose:
		Provides the AwsBucketLoader object used by Foo workflows. This class keeps its runtime
		state and public interface available for loading, fetching, generation, scraping, or
		supporting operations without altering the executable behavior of the original
		implementation.
	"""
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
			Initializes the AwsBucketLoader instance with the default runtime state and configuration
			required by later method calls. The constructor preserves the original initialization
			behavior.
		"""
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
			Returns the stable list of public members exposed for introspection, documentation, and UI
			display.

		Returns:
			Result produced by the operation.
		"""
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
		"""Perform the load operation.

		Purpose:
			Executes the load operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			bucket (str): Value used by the operation.
			prefix (Optional[str]): Value used by the operation.
			aws_access_key_id (Optional[str]): Value used by the operation.
			aws_secret_access_key (Optional[str]): Value used by the operation.
			aws_session_token (Optional[str]): Value used by the operation.
			region_name (Optional[str]): Value used by the operation.
			endpoint_url (Optional[str]): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			Logger( ).write( exception )
			raise exception
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		"""Perform the split operation.

		Purpose:
			Executes the split operation using the existing Foo implementation. The method preserves
			original runtime behavior while exposing documentation compatible with MkDocs.

		Args:
			chunk (int): Value used by the operation.
			overlap (int): Value used by the operation.

		Returns:
			Result produced by the operation.
		"""
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
			Logger( ).write( exception )
			raise exception
