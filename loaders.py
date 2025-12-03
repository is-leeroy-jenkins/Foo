'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                name.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="guro.py" company="Terry D. Eppler">

	     name.py
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
    name.py
  </summary>
  ******************************************************************************************
'''
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
	from langchain_core.documents import Document
except Exception:
	from langchain.schema import Document

from langchain_community.document_loaders import (
	CSVLoader,
	Docx2txtLoader,
	PyPDFLoader,
	UnstructuredExcelLoader,
	WebBaseLoader,
)
from typing import Optional, List, Dict, Any

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
	loader: Optional[ RecursiveCharacterTextSplitter ]
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
			exception.module = 'Foo'
			exception.cause = 'Loader'
			exception.method = '_ensure_existing_file( self, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def _resolve_paths( self, pattern: str ) -> List[ str ] | None:
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
			exception.module = 'Foo'
			exception.cause = 'Loader'
			exception.method = '_resolve_paths( self, pattern: str ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def _split_documents( self, documents: List[ Document ], chunk: int=1000, overlap: int=200 ) \
			-> \
					List[ Document ] | None:
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
			throw_if( 'docs', documents )
			self.documents = documents
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.loader = RecursiveCharacterTextSplitter( chunk_size=self.chunk_size,
				chunk_overlap=self.overlap_amount )
			return self.loader.split_documents( documents )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Loader'
			exception.method = ('_split_documents( self, docs: List[ Document ], chunk: int=1000, '
			                    'overlap: int=200 ) -> List[ Document ]')
			error = ErrorDialog( exception )
			error.show( )

class CsvLoader( Loader ):
	'''

		Purpose:
		--------
		Provides CSVLoader functionality to parse CSV files into Document objects.


		Attributes:
		----------
		documents: List[ Document ],
		file_path: str,
		pattern: str,
		expanded: List[ str ],
		candidates: List[ str ],
		resolved: List[ str ],
		splitter: RecursiveCharacterTextSplitter,
		chunk_size: int,
		overlap_amount: int,

		Methods:
		-------
		_ensure_existing_file( self, path: str ) -> str,
		_resolve_paths( self, pattern: str ) -> List[ str ],
		_split_documents( self, docs: List[ Document ], chunk: int=1000, overlap: int=200 ) ->
		List[ Document ],
		load( self, path: str, encoding: Optional[ str ]=None,
			csv_args: Optional[ Dict[ str, Any ] ]=None,
			source_column: Optional[ str ]=None ) -> List[ Document ]

	'''
	loader: Optional[ CSVLoader ]
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	pattern: Optional[ List[ str ] ]
	expanded: Optional[ List[ str ] ]
	candidates: Optional[ List[ str ] ]
	resolved: Optional[ List[ str ] ]
	encoding: Optional[ str ]
	csv_args: Optional[ Dict[ str, Any ] ]
	source_column: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.encoding = None
		self.csv_args = None
		self.source_column = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def load( self, path: str, encoding: Optional[ str ], csv_args: Optional[ Dict[ str, Any ] ],
			source_column: Optional[ str ] ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load a CSV file into LangChain Document objects.

			Parameters:
			-----------
			path (str): Path to the CSV file.
			encoding (Optional[str]): File encoding (e.g., 'utf-8') if known.
			csv_args (Optional[Dict[str, Any]]): Additional CSV parsing arguments.
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
			self.loader = CSVLoader( file_path=self.file_path, encoding=self.encoding,
				csv_args=self.csv_args, source_column=self.source_column )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'CSV'
			exception.method = 'loader( )'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, size: int = 1000, amount: int = 200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded CSV documents into smaller text chunks.

			Parameters:
			-----------
			chunk_size (int): Maximum number of characters per chunk.
			chunk_overlap (int): Number of overlapping characters between chunks.

			Returns:
			--------
			List[Document]: List of split Document chunks.

		'''
		try:
			throw_if( 'docs', self.documents )
			self.chunk_size = size
			self.overlap_amount = amount
			_documents = self._split_documents( documents=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return _documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'CSV'
			exception.method = 'split( self, size: int=1000, amount: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class WebLoader( Loader ):
	'''

		Purpose:
		--------
		Provides LangChain's WebBaseLoader functionality to retrieve
		and parse HTML content into Document objects.

	'''
	loader: Optional[ WebBaseLoader ]
	urls: Optional[ List[ str ] ]
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.urls = None
		self.documents = None
		self.file_path = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def load( self, urls: List[ str ] ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load one or more web pages and convert to Document objects.

			Parameters:
			-----------
			urls (str | List[str]): A single URL string or list of URL strings.

			Returns:
			--------
			List[Document]: Parsed Document objects from fetched HTML content.

		'''
		try:
			throw_if( 'urls', urls )
			self.urls = urls
			self.loader = WebBaseLoader( web_path=self.urls )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Web'
			exception.method = 'load( self, urls: List[ str ] ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded web documents into smaller chunks for better LLM processing.

			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlap between chunks in characters.

			Returns:
			--------
			List[Document]: Chunked Document objects.

		'''
		try:
			throw_if( 'docs', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			_documents = self._split_documents( documents=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return _documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Web'
			exception.method = 'split( self, chunk: int=1000 , overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class PdfLoader( Loader ):
	'''

		Purpose:
		--------
		Wrap LangChain's PyPDFLoader to extract and chunk PDF documents.

	'''
	loader: Optional[ PyPDFLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	mode: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_pathpath = None
		self.documents = [ ]
		self.file_path = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
		self.mode = None
	
	def __repr__( self ) -> str:
		return f'PDF(path={self.path!r}, docs={len( self.docs or [ ] )})'
	
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
			exception.module = 'Foo'
			exception.cause = 'PDF'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''


			Purpose:
			--------
			Split loaded PDF documents into smaller chunks for efficient processing.

			Parameters:
			-----------
			chunk_size (int): Max characters allowed per chunk.
			chunk_overlap (int): Characters that overlap between consecutive chunks.

			Returns:
			--------
			List[Document]: Chunked list of Document objects.


		'''
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self._split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.docs
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'PDF'
			exception.method = 'split( self, chunk: int=1000 , overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class ExcelLoader( Loader ):
	'''


		Purpose:
		--------
		Provides LangChain's UnstructuredExcelLoader functionality 
		to parse Excel spreadsheets into documents.


	'''
	loader: Optional[ UnstructuredExcelLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None

	def load( self, path: str, mode: str='elements', headers: bool=True ) -> List[ Document ] | None:
		'''


			Purpose:
			--------
			Load and convert Excel data into LangChain Document objects.

			Parameters:
			-----------
			path (str): File path to the Excel spreadsheet.
			mode (str): Extraction mode, either 'elements' or 'paged'.
			include_headers (bool): Whether to include column headers in parsing.

			Returns:
			--------
			List[Document]: List of parsed Document objects from Excel content.


		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = UnstructuredExcelLoader( file_path=self.file_path, mode=mode,
				include_headers=headers )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Excel'
			exception.method = 'load( self, path: str, mode: str=elements, include_headers: bool=True ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
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
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self._split_documents( self.documents, chunk=self.chunk_size,
				overlap=overlap )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Excel'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class WordLoader( Loader ):
	'''


		Purpose:
		--------
		Provides LangChain's Docx2txtLoader functionality to 
		convert docx files into Document objects.


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
			exception.module = 'Foo'
			exception.cause = 'Word'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
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
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self._split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.docs
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Word'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class MarkLoader( Loader ):
	'''


		Purpose:
		--------
		Wrap LangChain's UnstructuredMarkdownLoader to parse Markdown files into Document objects.


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
	
	def __repr__( self ) -> str:
		return f'Markdown(path={self.path!r}, docs={len( self.docs or [ ] )})'
	
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
			exception.module = 'Foo'
			exception.cause = 'Markdown'
			exception.method = 'load( self, path: str ) -> List[ Document ] '
			error = ErrorDialog( exception )
			error.show( )
	
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
			self.documents = self._split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Markdown'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class HtmlLoader( Loader ):
	'''

		Purpose:
		--------
		Provides the UnstructuredHTMLLoader's functionality
		to parse HTML files into Document objects.

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
	
	def __repr__( self ) -> str:
		return f'HTML(path={self.path!r}, docs={len( self.documents or [ ] )})'
	
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
			exception.module = 'Foo'
			exception.cause = 'HTML'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
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
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self._split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'HTML'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )


