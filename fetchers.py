'''
  ******************************************************************************************
      Assembly:                Foo
      Filename:                fetchers.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file='fetchers.py' company='Terry D. Eppler'>

	     Foo is a python framework for web scraping information into ML pipelines.
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
    fetchers.py
  </summary>
  ******************************************************************************************
  '''
from boogr import Error, ErrorDialog
from __future__ import annotations
import crawl4ai as crl
import config as cfg
from core import Result
from langchain_googledrive.retrievers import GoogleDriveRetriever
from langchain_community.retrievers import ArxivRetriever, WikipediaRetriever
import re
import requests
from requests import Response
from typing import Any, Dict, Optional, Pattern, List

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

class Fetcher:
	'''

		Purpose:
		--------
		Base class for fetchers. Implement `fetch(...)` in concrete subclasses.

	'''
	timeout: Optional[ int ]
	headers: Optional[ Dict[ str, str ] ]
	response: Optional[ Response ]
	url: Optional[ str ]
	result: Optional[ Result ]
	text: Optional[ str ]
	
	def __init__( self ) -> None:
		'''

			Purpose:
			-----------
			Base initializer. Subclasses should set defaults they require.

		'''
		self.timeout = None
		self.headers = None
		self.response = None
		self.url = None
		self.result = None
	
	def __dir__( self ) -> list[ str ]:
		'''

			Purpose:
			-----------
			Control ordering for introspection.

			Parameters:
			-----------
			None

			Returns:
			-----------
			list[str]: Ordered attribute/method names.

		'''
		return [ 'timeout',
		         'headers',
		         'response',
		         'url',
		         'result',
		         'fetch',
		         'html_to_text' ]
	
	def fetch( self, url: str, time: int=10 ) -> Result | None:
		'''

			Purpose:
			--------
			Abstract fetch method to be implemented by subclasses.

			Parameters:
			-----------
			url (str): Resource URL to fetch.
			time (int): Timeout in seconds.
			show_dialog (bool): If True, show an ErrorDialog on exception.

			Returns:
			---------
			Optional[Result]: Should return Result on success or None on failure.

		'''
		raise NotImplementedError( 'Must be implemented by a subclass.' )

class Fetch( ):
	"""

		Purpose:
		---------
		Provides a unified conversational system with explicit methods for
		querying structured data (SQL), unstructured documents, or free-form
		chat with an OpenAI LLM. Each method is deterministic and isolates
		a specific capability.

		Parameters:
		-----------
		db_uri (str):
		URI string for the SQLite database connection.
		doc_paths (List[str]):
		File paths to documents (txt, pdf, csv, html) for ingestion.
		model (str, optional):
		OpenAI model to use (default: 'gpt-4o-mini').
		temperature (float, optional):
		Sampling temperature for the LLM (default: 0.8).

		Attributes:
		----------
		model (str): OpenAI model identifier.
		temperature (float): Temperature setting for sampling.
		llm (ChatOpenAI): Instantiated OpenAI-compatible chat model.
		db_uri (str): SQLite database URI.
		doc_paths (List[str]): Paths to local document sources.
		memory (ConversationBufferMemory): LangChain conversation buffer.
		sql_tool (Optional[Tool]): SQL query tool.
		doc_tool (Optional[Tool]): Vector document retrieval tool.
		api_tools (List[Tool]): List of custom API tools.
		agent (AgentExecutor): LangChain multi-tool agent.
		__tools (List[Tool]): Consolidated tool list used by agent.
		documents (List[str]): Cached document source text or metadata.
		db_toolkit (Optional[object]): SQLDatabaseToolkit instance.
		database (Optional[object]): Underlying SQLAlchemy database.
		loader (Optional[object]): Last-used document loader.
		tool (Optional[object]): Active retrieval tool.
		extension (Optional[str]): File extension for routing.

	"""
	model: str
	temperature: float
	llm: ChatOpenAI
	db_uri: str
	doc_paths: List[ str ]
	memory: ConversationBufferMemory
	sql_tool: Optional[ Tool ]
	doc_tool: Optional[ Tool ]
	api_tools: List[ Tool ]
	agent: AgentExecutor
	__tools: List[ Tool ]
	documents: List[ str ]
	db_toolkit: Optional[ object ]
	database: Optional[ object ]
	loader: Optional[ object ]
	tool: Optional[ object ]
	extension: Optional[ str ]
	answer: Optional[ Dict ]
	sources: Optional[ Dict[ str, str ] ]
	
	def __init__( self, db_uri: str, doc_paths: List[ str ], model: str = 'gpt-4o-mini',
			temperature: float = 0.8 ):
		"""

			Purpose:
			--------
			Initializes the Fetch system and configures tools for SQL,
			document retrieval, and conversational use.

			Parameters:
			-----------
			db_uri (str): Path or URI to SQLite database.
			doc_paths (List[str]): Files to be processed for retrieval.
			model (str): LLM model name (default: gpt-4o-mini).
			temperature (float): Sampling diversity (default: 0.8).

			Returns:
			-----------
			None

		"""
		self.model = model
		self.temperature = temperature
		self.llm = ChatOpenAI( model=self.model, temperature=self.temperature, streaming=True )
		self.db_uri = db_uri
		self.doc_paths = doc_paths
		self.memory = ConversationBufferMemory( memory_key='chat_history', return_messages=True )
		self.sql_tool = self._init_sql_tool( )
		self.doc_tool = self._init_doc_tool( )
		self.api_tools = self._init_api_tools( )
		self.documents = [ ]
		self.db_toolkit = None
		self.database = None
		self.loader = None
		self.tool = None
		self.extension = None
		self.answer = { }
		self.__tools = [ t for t in [ self.sql_tool, self.doc_tool ] + self.api_tools if t is not None ]
		self.agent = initialize_agent( tools=self.__tools, llm=self.llm, memory=self.memory,
			agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True )
	
	def query_sql( self, question: str ) -> str | None:
		"""

			Purpose:
				Answer a question using ONLY the SQL database tool.

			Parameters:
				question (str): Natural language SQL-like question.

			Returns:
				str: Answer from the SQL query tool.

		"""
		try:
			throw_if( 'question', question )
			return self.sql_tool.func( question )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Fetch'
			exception.method = 'query_sql(self, question)'
			error = ErrorDialog( exception )
			error.show( )
	
	def query_docs( self, question: str, with_sources: bool = False ) -> str | None:
		"""

			Purpose:
			-----------
			Answer a question using ONLY the document retrieval tool.

			Parameters:
			-----------
			question (str):
			Natural language question grounded in the loaded documents.
			with_sources (bool):
			If True, returns sources alongside the answer.

			Returns:
			-----------
			str: Response from the document retriever. Includes sources if available.

		"""
		try:
			throw_if( 'question', question )
			if with_sources:
				if self.doc_chain_with_sources is None:
					raise RuntimeError( 'Document chain with sources is not available' )
				
				result = self.doc_chain_with_sources( {
						'question': question } )
				if 'answer' not in result or 'sources' not in result:
					raise RuntimeError( 'Malformed response from doc_chain_with_sources' )
				
				answer = result[ 'answer' ]
				sources = result[ 'sources' ]
				
				if sources:
					return f'{answer}\n\nSOURCES:\n{sources}'
				return answer
			else:
				return self.doc_tool.func( question )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Fetch'
			exception.method = 'query_docs(self, question, with_sources)'
			error = ErrorDialog( exception )
			error.show( )
	
	def query_chat( self, prompt: str ) -> str | None:
		"""

			Purpose:
			-----------
			Send a general-purpose prompt directly to the LLM without using
			tools, but with full memory context.

			Parameters:
			-----------
			prompt (str): User message for free-form reasoning.

			Returns:
			-----------
			str: LLM-generated conversational response.

		"""
		try:
			throw_if( 'prompt', prompt )
			return self.llm.invoke( prompt ).content
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Fetch'
			exception.method = 'query_chat(self, prompt)'
			error = ErrorDialog( exception )
			error.show( )

class WebFetcher( Fetcher ):
	'''
		
		Purpose:
		---------
		Concrete synchronous fetcher using `requests` and minimal HTML→text
		extraction.
		
		Parameters:
		-----------
		headers (Optional[Dict[str, str]]): Optional HTTP headers; User-Agent
		auto-filled if missing.
		
		Returns:
		-----------
		None
		
	'''
	agents: Optional[ str ]
	url: Optional[ str ]
	html: Optional[ str ]
	re_tag: Optional[ Pattern ]
	re_ws: Optional[ Pattern ]
	response: Optional[ Response ]

	def __init__( self ) -> None:
		'''
			Purpose:
			-----------
			Initialize WebFetcher with optional headers and sane defaults.
			
			Parameters:
			-----------
			headers (Optional[Dict[str, str]]): Optional headers for requests.
			
			Returns:
			-----------
			None
		'''
		super( ).__init__( )
		self.timeout = 15
		self.re_tag = re.compile( r'<[^>]+>' )
		self.re_ws = re.compile( r'\s+' )
		self.url = None
		self.html = None
		self.response = None
		self.headers = { }
		self.agents = cfg.AGENTS

		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents

	def __dir__( self ) -> List[ str ]:
		'''
			
			Purpose:
			-----------
			Control visible ordering for WebFetcher.
			
			Parameters:
			-----------
			None
			
			Returns:
			-----------
			list[str]: Ordered attribute/method names.
			
		'''
		return [ 'agents',
		         'url',
		         'html',
		         'timeout',
		         'headers',
		         'fetch',
		         'html_to_text' ]

	def fetch( self, url: str, time: int=10  ) -> Result | None:
		'''
			
			Purpose:
			-------
			Perform an HTTP GET to fetch a page and return canonicalized Result.
				
			Parameters:
			-----------
			url (str): Absolute URL to fetch.
			time (int): Timeout seconds to use for the request.
			show_dialog (bool): If True, show an ErrorDialog on exception.
				
			Returns:
			---------
			Optional[Result]: Result with url, status, text, html, headers on success.
			
		'''
		try:
			throw_if( 'url', url )
			self.url = url
			self.timeout = time
			self.response = requests.get( url=self.url, headers=self.headers, timeout=self.timeout )
			self.response.raise_for_status( )
			self.result = Result( self.response )
			return self.result
		except Exception as exc:  
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebFetcher'
			exception.method = 'fetch( self, url: str, time: int=10  ) -> Result'
			dialog = ErrorDialog( exception )
			dialog.show( )

	def html_to_text( self, html: str ) -> str:
		'''
			
			Purpose:
			--------
			Convert HTML to compact plain text with minimal heuristics (scripts and
			styles removed, tags replaced with whitespace, whitespace normalized).
			
			Parameters:
			---------
			html (str): Raw HTML string.
			show_dialog (bool): If True, show an ErrorDialog on exception.
			
			Returns:
			--------
			str: Plain text extracted from HTML.
			
		'''
		try:
			throw_if( 'html', html )
			html = re.sub( r'<script[\s\S]*?</script>', ' ', html, flags = re.IGNORECASE )
			html = re.sub( r'<style[\s\S]*?</style>', ' ', html, flags = re.IGNORECASE )
			html = re.sub( r'</?(p|div|br|li|h[1-6])[^>]*>', '\n', html, flags = re.IGNORECASE )
			text = re.sub( self.re_tag, ' ', html )
			text = re.sub( self.re_ws, ' ', text ).strip( )
			return text
		except Exception as exc:  
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'scrapers'
			exception.method = 'html2text( )'
			dialog = ErrorDialog( exception )
			dialog.show( )

class WebCrawler( WebFetcher ):
	'''
		
		Purpose:
		-------
		A crawler that attempts `crawl4ai` first (if installed) and falls back to
		Playwright headful rendering only when required. Designed to be used when
		pages require JS to render content.
			
		Parameters:
		----------
		headers (Optional[Dict[str, str]]): Optional headers for requests/playwright.
			
		Returns:
		-------
		None
		
	'''
	use_playwright: Optional[ bool ]
	browser_context: Optional[ Any ]

	def __init__( self, headers: Optional[ Dict[ str, str ] ]=None ) -> None:
		'''
		
			Purpose:
			-------
			Initialize crawler. By default prefer `crawl4ai` when available and
			only enable Playwright when `use_playwright=True`.
				
			Parameters:
			-----------
			headers (Optional[Dict[str, str]]): Optional headers.
			use_playwright (bool): If True, enable Playwright fallback.
				
			Returns:
			--------
			None
			
		'''
		super( ).__init__( )
		self.browser_context = None
		self.raw_url = None
		self.raw_html = None
		self.response = None
		self.headers = headers if headers is not None else {}

	def __dir__( self ) -> list[ str ]:
		'''
		
			Purpose:
			-----------
			Ordering for WebCrawler introspection.
			
			Parameters:
			-----------
			None
			
			Returns:
			-----------
			list[str]: Ordered attribute/method names.
			
		'''
		return [ 'use_playwright',
		         'browser_context',
		         'fetch',
		         'html_to_text',
		         'render_with_playwright' ]

	def fetch( self, url: str, time: int=10 ) -> Result | None:
		'''
			
			Purpose:
			-------
			Try `crawl4ai` (if installed) to fetch JS-rendered content. If not
			available or it returns empty, fall back to the synchronous fetch or
			(optionally) to Playwright rendering.
				
			Parameters:
			-------
			url (str): Absolute URL to fetch.
			time (int): Timeout seconds.
			show_dialog (bool): If True, show an ErrorDialog on exception.
				
			Returns:
			-------
			Optional[Result]: Result with url, status, text, html, headers on success.
				
		'''
		try:
			throw_if( 'url', url )
			configuration = { 'url': url }
			payload = crawl4ai.fetch_and_render( configuration )
			if payload and isinstance( payload, dict ) and 'content' in payload:
				self.raw_html = payload.get( 'content', '' )
				text = self.html_to_text( self.raw_html )
				self.result = Result( url = url, status=200, text=text,
					html=self.raw_html, headers=self.headers )
				return self.result
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebCrawler'
			exception.method = 'fetch( self, url: str, time: int=15 ) -> Result'
			dialog = ErrorDialog( exception )
			dialog.show( )

	def render_with_playwright( self, url: str, timeout: int=15 ) -> str:
		'''
		
			Purpose:
			-----------
			Render the page with Playwright (synchronous API) and return the page HTML.
			This method imports Playwright lazily so the package is optional.
			
			Parameters:
			-----------
			url (str): URL to render.
			timeout (int): Timeout seconds for render.
			
			Returns:
			-----------
			str: Rendered HTML of the page.
			
		'''
		try:
			with sync_playwright( ) as p:
				browser = p.chromium.launch( )
				page = browser.new_page( )
				page.goto( url, timeout = timeout * 1000 )
				page.wait_for_load_state( 'networkidle', timeout = timeout * 1000 )
				html = page.content( )
				browser.close( )
				return html
		except Exception as exc: 
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebCrawler'
			exception.method = 'render_with_playwright'
			dialog = ErrorDialog( exception )
			dialog.show( )

class ArXivFetcher( Fetcher ):
	'''

		Purpose:
		--------
		Provides the Arxiv loading functionality
		to parse video research papers into Document objects.

	'''
	fetcher: Optional[ ArxivRetriever ]
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
	
	def load( self, question: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load an arxiv document and convert its contents into LangChain Document objects.

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
			self.loader = ArxivLoader( query=self.query, max_documents=self.max_documents,
				doc_content_chars_max=self.max_characters )
			self.documents = self.loader.fetch( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'ArxivFetcher'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Split loaded arxiv documents into manageable text chunks.

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
			exception.module = 'Foo'
			exception.cause = 'ArxivFetcher'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class GoogleDriveFetcher( Fetcher ):
	'''

		Purpose:
		--------
		Provides the google drive loading functionality
		to parse items on googke drive into Document objects.

	'''
	fetcher: Optional[ GoogleDriveRetriever ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	num_results: Optional[ int ]
	folder_id: Optional[ str ]
	template: Optional[ str ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.fetcher = None
		self.template = None
		self.folder_id = None
		self.num_results = None
	
	def fetch( self, question: str, folder_id: str= 'root',
			results: int=2, template: str='gdrive-query' ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Load an google drive items and convert its contents into LangChain Document objects.

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
			self.template = template
			self.num_results = results
			self.folder_id = folder_id
			self.fetcher = GoogleDriveRetriever( num_results=self.num_results, template=self.template )
			self.documents = self.fetcher.invoke( input=self.query, folder_id=self.folder_id )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'GoogleDriveFetcher'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
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
			exception.module = 'Foo'
			exception.cause = 'GoogleDriveFetcher'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class WikipediaFetcher( Fetcher ):
	'''

		Purpose:
		--------
		Provides the Arxiv loading functionality
		to parse video research papers into Document objects.

	'''
	fetcher: Optional[ WikipediaRetriever ]
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
			self.loader = WikipediaRetriever( query=self.query, max_documents=self.max_documents,
				doc_content_chars_max=self.max_characters )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'WikiFetcher'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int = 1000, overlap: int = 200 ) -> List[ Document ] | None:
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
			exception.module = 'Foo'
			exception.cause = 'ArxivLoader'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

