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
from __future__ import annotations
from astroquery.simbad import Simbad
import datetime
import matplotlib.pyplot as plt
import base64
from boogr import Error, ErrorDialog
import crawl4ai as crl
import config as cfg
from core import Result, Schema
import datetime as dt
import googlemaps
import http.client
import json
from langchain_googledrive.retrievers import GoogleDriveRetriever
from langchain_community.retrievers import ArxivRetriever, WikipediaRetriever
import re
import requests
from requests import Response
from sscws.sscws import SscWs
from typing import Any, Dict, Optional, Pattern, List
from owslib.wms import WebMapService
import urllib.parse

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

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]


	'''
	timeout: Optional[ int ]
	headers: Optional[ Dict[ str, Any ] ]
	response: Optional[ Response ]
	url: Optional[ str ]
	result: Optional[ Result ]
	query: Optional[ str ]
	
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
		self.query = None
	
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
		         'query',
		         'fetch' ]
	
	def fetch( self, query: str, url:str, time: int=10 ) -> Result | None:
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

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
		

		
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
		self.timeout = 10
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
			exception.module = 'fetchers'
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
			exception.cause = 'WebFetchers'
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

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
		

		
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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
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

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
		


	'''
	fetcher: Optional[ ArxivRetriever ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	max_documents: Optional[ int ]
	full_documents: Optional[ bool ]
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
		self.max_documents = 100
		self.full_documents = True
		self.include_metadata = False
	
	def fetch( self, question: str ) -> List[ Document ] | None:
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
			self.loader = ArxivRetriever( load_max_docs=self.max_documents,
				 get_full_documents=self.full_documents, load_all_available_meta=self.include_metadata )
			self.documents = self.loader.invoke( input=self.query )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'ArxivFetcher'
			exception.method = 'fetch( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> dict:
		"""
		
			Purpose:
			________
			Construct and return a fully dynamic OpenAI Tool API schema definition.
			Supports arbitrary parameters, types, nested objects, and required fields.

			Parameters:
			___________
			function (str):
			The function name exposed to the LLM.

			tool (str):
			The underlying system or service the function wraps
			(e.g., “Google Maps”, “SQLite”, “Weather API”).

			description (str):
			Precise explanation of what the function does.

			parameters (dict):
			A dictionary defining parameter names and JSON schema descriptors.
			Each value must itself be a valid JSON-schema fragment.

				Example:
					{
						"origin": {
							"type": "string",
							"description": "Starting location."
						},
						"destination": {
							"type": "string",
							"description": "Ending location."
						},
						"mode": {
							"type": "string",
							"enum": ["driving", "walking", "bicycling", "transit"],
							"description": "Travel mode."
						}
					}

			required (list[str] | None):
			List of required parameter names.
			If None, required = list(parameters.keys()).

			Returns:
			________
			dict:
			A JSON-compatible dictionary defining the tool schema.
			
		"""
		throw_if( 'function', function )
		throw_if( 'tool', tool )
		throw_if( 'description', description )
		throw_if( 'parameters', parameters )
		if not isinstance( parameters, dict ):
			raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )

		func_name = function.strip( )
		tool_name = tool.strip( )
		desc = description.strip( )
		if required is None:
			required = list( parameters.keys( ) )
		return \
		{
				'name': func_name,
				'description': f'{desc} This function uses the {tool_name} service.',
				'parameters':
				{
					'type': 'object',
					'properties': parameters,
					'required': required
				}
		}


class GoogleDriveFetcher( Fetcher ):
	'''

		Purpose:
		--------
		Provides the google drive loading functionality
		to parse items on googke drive into Document objects.

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
		


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
			self.fetcher = GoogleDriveRetriever( num_results=self.num_results, template=self.template,
				folder_id=self.folder_id )
			self.documents = self.fetcher.invoke( input=self.query  )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'GoogleDriveFetcher'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: List[ str ] ) -> dict:
		"""
		
			Purpose:
			________
			Construct and return a fully dynamic OpenAI Tool API schema definition.
			Supports arbitrary parameters, types, nested objects, and required fields.

			Parameters:
			___________
			function (str):
			The function name exposed to the LLM.

			tool (str):
			The underlying system or service the function wraps
			(e.g., “Google Maps”, “SQLite”, “Weather API”).

			description (str):
			Precise explanation of what the function does.

			parameters (dict):
			A dictionary defining parameter names and JSON schema descriptors.
			Each value must itself be a valid JSON-schema fragment.

				Example:
					{
						"origin": {
							"type": "string",
							"description": "Starting location."
						},
						"destination": {
							"type": "string",
							"description": "Ending location."
						},
						"mode": {
							"type": "string",
							"enum": ["driving", "walking", "bicycling", "transit"],
							"description": "Travel mode."
						}
					}

			required (list[str] | None):
			List of required parameter names.
			If None, required = list(parameters.keys()).

			Returns:
			________
			dict:
			A JSON-compatible dictionary defining the tool schema.
			
		"""
		throw_if( 'function', function )
		throw_if( 'tool', tool )
		throw_if( 'description', description )
		throw_if( 'parameters', parameters )
		if not isinstance( parameters, dict ):
			raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )

		func_name = function.strip( )
		tool_name = tool.strip( )
		desc = description.strip( )
		if required is None:
			required = list( parameters.keys( ) )
		return \
		{
				'name': func_name,
				'description': f'{desc} This function uses the {tool_name} service.',
				'parameters':
				{
					'type': 'object',
					'properties': parameters,
					'required': required
				}
		}
	
class WikipediaFetcher( Fetcher ):
	'''

		Purpose:
		--------
		Provides the wikipedia searchig functionality.

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
		


	'''
	fetcher: Optional[ WikipediaRetriever ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	max_documents: Optional[ int ]
	language: Optional[ str ]
	include_metadata: Optional[ bool ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.query = None
		self.chunk_size = None
		self.overlap_amount = None
		self.fetcher = None
		self.max_documents = 100
		self.language = 'english'
		self.include_metadata = False
	
	def fetch( self, question: str ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Searches wikipedia and provides its contents as LangChain Document objects.

			Parameters:
			-----------
			question (str): query passed to wiki search

			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'question', question )
			self.query = question
			self.fetcher = WikipediaRetriever( lang=self.language, load_all_available_meta=self.include_metadata )
			self.documents = self.fetcher.invoke( input=self.query,  )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'WikipediaFetcher'
			exception.method = 'fetch( self, question: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> dict:
		"""
		
			Purpose:
			________
			Construct and return a fully dynamic OpenAI Tool API schema definition.
			Supports arbitrary parameters, types, nested objects, and required fields.

			Parameters:
			___________
			function (str):
			The function name exposed to the LLM.

			tool (str):
			The underlying system or service the function wraps
			(e.g., “Google Maps”, “SQLite”, “Weather API”).

			description (str):
			Precise explanation of what the function does.

			parameters (dict):
			A dictionary defining parameter names and JSON schema descriptors.
			Each value must itself be a valid JSON-schema fragment.

				Example:
					{
						"origin": {
							"type": "string",
							"description": "Starting location."
						},
						"destination": {
							"type": "string",
							"description": "Ending location."
						},
						"mode": {
							"type": "string",
							"enum": ["driving", "walking", "bicycling", "transit"],
							"description": "Travel mode."
						}
					}

			required (list[str] | None):
			List of required parameter names.
			If None, required = list(parameters.keys()).

			Returns:
			________
			dict:
			A JSON-compatible dictionary defining the tool schema.
			
		"""
		throw_if( 'function', function )
		throw_if( 'tool', tool )
		throw_if( 'description', description )
		throw_if( 'parameters', parameters )
		if not isinstance( parameters, dict ):
			raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )

		func_name = function.strip( )
		tool_name = tool.strip( )
		desc = description.strip( )
		if required is None:
			required = list( parameters.keys( ) )
		return \
		{
				'name': func_name,
				'description': f'{desc} This function uses the {tool_name} service.',
				'parameters':
				{
					'type': 'object',
					'properties': parameters,
					'required': required
				}
		}

class NewsFetcher( Fetcher ):
	'''

		Purpose:
		--------
		Provides the News API functionality.

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
		


	'''
	agents: Optional[ str ]
	url: Optional[ str ]
	html: Optional[ str ]
	re_tag: Optional[ Pattern ]
	re_ws: Optional[ Pattern ]
	response: Optional[ Response ]
	result: Optional[ Result ]
	api_key: Optional[ str ]
	categories: Optional[ str ]
	limit: Optional[ int ]
	params: Optional[ Dict[ str, str ] ]

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
		self.timeout = None
		self.re_tag = re.compile( r'<[^>]+>' )
		self.re_ws = re.compile( r'\s+' )
		self.url = 'api.thenewsapi.com'
		self.html = None
		self.response = None
		self.result = None
		self.headers = { }
		self.params = { }
		self.max_documents = 50
		self.api_key = 'OgP3S1uQpZ73EvUlGlT7NzxS8LqSZijom4LrTKpA'
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

	def fetch( self, query: str,  time: int=10  ) -> Result | None:
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
			throw_if( 'query', query )
			self.query = query
			self.timeout = time
			conn = http.client.HTTPSConnection( self.url )
			params = urllib.parse.urlencode(
			{
					'api_token': self.api_key,
					'search': self.query,
					'language': 'en',
					'limit': self.max_documents,
			} )
			
			conn.request( 'GET', '/v1/news/all?{}'.format( params ) )
			res = conn.getresponse( )
			data = res.read( )
			return data.decode( 'utf-8' )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'NewsFetcher'
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
			exception.cause = 'NewsFetcher'
			exception.method = 'html2text( )'
			dialog = ErrorDialog( exception )
			dialog.show( )
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> dict:
		"""
		
			Purpose:
			________
			Construct and return a fully dynamic OpenAI Tool API schema definition.
			Supports arbitrary parameters, types, nested objects, and required fields.

			Parameters:
			___________
			function (str):
			The function name exposed to the LLM.

			tool (str):
			The underlying system or service the function wraps
			(e.g., “Google Maps”, “SQLite”, “Weather API”).

			description (str):
			Precise explanation of what the function does.

			parameters (dict):
			A dictionary defining parameter names and JSON schema descriptors.
			Each value must itself be a valid JSON-schema fragment.

				Example:
					{
						"origin": {
							"type": "string",
							"description": "Starting location."
						},
						"destination": {
							"type": "string",
							"description": "Ending location."
						},
						"mode": {
							"type": "string",
							"enum": ["driving", "walking", "bicycling", "transit"],
							"description": "Travel mode."
						}
					}

			required (list[str] | None):
			List of required parameter names.
			If None, required = list(parameters.keys()).

			Returns:
			________
			dict:
			A JSON-compatible dictionary defining the tool schema.
			
		"""
		throw_if( 'function', function )
		throw_if( 'tool', tool )
		throw_if( 'description', description )
		throw_if( 'parameters', parameters )
		if not isinstance( parameters, dict ):
			raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )

		func_name = function.strip( )
		tool_name = tool.strip( )
		desc = description.strip( )
		if required is None:
			required = list( parameters.keys( ) )
		return \
		{
				'name': func_name,
				'description': f'{desc} This function uses the {tool_name} service.',
				'parameters':
				{
					'type': 'object',
					'properties': parameters,
					'required': required
				}
		}
			
class GoogleSearch( Fetcher ):
	'''

		Purpose:
		---------
		Class providing the functionality of the google custom search api.

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
		


	'''
	keywords: Optional[ str ]
	url: Optional[ str ]
	re_tag: Optional[ Pattern ]
	re_ws: Optional[ Pattern ]
	response: Optional[ Response ]
	api_key: Optional[ str ]
	cse_id: Optional[ str ]
	params: Optional[ Dict[ str, str ] ]
	
	def __init__( self ) -> None:
		'''
			Purpose:
			-----------
			Initialize GoogleSearch with optional headers and sane defaults.

			Parameters:
			-----------
			headers (Optional[Dict[str, str]]): Optional headers for requests.

			Returns:
			-----------
			None
		'''
		super( ).__init__( )
		self.api_key = cfg.GOOGLE_API_KEY
		self.cse_id = cfg.GOOGLE_CSE_ID
		self.re_tag = re.compile( r'<[^>]+>' )
		self.re_ws = re.compile( r'\s+' )
		self.url = None
		self.headers = { }
		self.timeout = None
		self.keywords = None
		self.params = None
		self.response = None
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
		return [ 'keywords',
		         'url',
		         'timeout',
		         'headers',
		         'fetch',
		         'api_key',
		         'response',
		         'cse_id',
		         'params',
		         'agents,',
		         'fetch' ]
	
	def fetch( self, keywords: str, time: int=10 ) -> Response | None:
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
			throw_if( 'keywords', keywords )
			self.url = r'https://www.googleapis.com/customsearch/v1?'
			self.keywords = keywords
			self.timeout = time
			self.params = \
			{
				'q': self.keywords,
				'key': self.api_key,
				'cx': '376fd5d0d8ae948b2',
				'num': self.timeout
			}
			_response = requests.get( url=self.url, params=self.params )
			return _response
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleSearch'
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
			html = re.sub( r'<script[\s\S]*?</script>', ' ', html, flags=re.IGNORECASE )
			html = re.sub( r'<style[\s\S]*?</style>', ' ', html, flags=re.IGNORECASE )
			html = re.sub( r'</?(p|div|br|li|h[1-6])[^>]*>', '\n', html, flags=re.IGNORECASE )
			text = re.sub( self.re_tag, ' ', html )
			text = re.sub( self.re_ws, ' ', text ).strip( )
			return text
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleSearch'
			exception.method = 'html2text( )'
			dialog = ErrorDialog( exception )
			dialog.show( )
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> dict:
		"""
		
			Purpose:
			________
			Construct and return a fully dynamic OpenAI Tool API schema definition.
			Supports arbitrary parameters, types, nested objects, and required fields.

			Parameters:
			___________
			function (str):
			The function name exposed to the LLM.

			tool (str):
			The underlying system or service the function wraps
			(e.g., “Google Maps”, “SQLite”, “Weather API”).

			description (str):
			Precise explanation of what the function does.

			parameters (dict):
			A dictionary defining parameter names and JSON schema descriptors.
			Each value must itself be a valid JSON-schema fragment.

				Example:
					{
						"origin": {
							"type": "string",
							"description": "Starting location."
						},
						"destination": {
							"type": "string",
							"description": "Ending location."
						},
						"mode": {
							"type": "string",
							"enum": ["driving", "walking", "bicycling", "transit"],
							"description": "Travel mode."
						}
					}

			required (list[str] | None):
			List of required parameter names.
			If None, required = list(parameters.keys()).

			Returns:
			________
			dict:
			A JSON-compatible dictionary defining the tool schema.
			
		"""
		throw_if( 'function', function )
		throw_if( 'tool', tool )
		throw_if( 'description', description )
		throw_if( 'parameters', parameters )
		if not isinstance( parameters, dict ):
			raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )

		func_name = function.strip( )
		tool_name = tool.strip( )
		desc = description.strip( )
		if required is None:
			required = list( parameters.keys( ) )
		return \
		{
				'name': func_name,
				'description': f'{desc} This function uses the {tool_name} service.',
				'parameters':
				{
					'type': 'object',
					'properties': parameters,
					'required': required
				}
		}

class GoogleMaps( Fetcher ):
	'''

		Purpose:
		--------
		Provides the google drive loading functionality
		to parse items on googke drive into Document objects.

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
		


	'''
	file_path: Optional[ str ]
	num_results: Optional[ int ]
	api_key: Optional[ str ]
	mode: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	coordinates: Optional[ Tuple[ float, float ] ]
	address: Optional[ str ]
	directions: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.api_key = cfg.GEOCODING_API_KEY
		self.longitude = None
		self.latitude = None
		self.mode = None
		self.url = None
		self.file_path = None
		self.coordinates = None
		self.fetcher = None
		self.address = None
		self.directions = None
		self.agents = cfg.AGENTS
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def geocode_location( self, address: str ) -> Tuple[ float, float ] | None:
		'''

			Purpose:
			--------
			Uses gmaps to get coordinates from a given address.

			Parameters:
			-----------
			address (str): address

			Returns:
			--------
			Tuple[ float, float ] - a tuple of floats representing latitude and longitude (lat, lng)

		'''
		try:
			throw_if( 'address', address )
			self.address = address
			self.url = r'https://maps.googleapis.com/maps/api/geocode/json?address='
			self.url += f'{self.address}&key={self.api_key}'
			self.response = requests.get( self.url )
			self.response.raise_for_status()
			_response = self.response.json()
			_result = _response[ 'results' ][ 0 ]
			_geo = _result[ 'geometry' ]
			_loc = _geo[ 'location' ]
			_lat = _loc[ 'lat' ]
			_lng = _loc[ 'lng' ]
			return ( _lat, _lng )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'GoogleMaps'
			exception.method = 'fetch_location( self, address: str ) -> Tuple[ float, float ]'
			error = ErrorDialog( exception )
			error.show( )

	def geocode_coordinates( self, lat: float, long: float ) -> str | None:
		'''

			Purpose:
			--------
			Uses the Google Maps API to get address from coordinates.

			Parameters:
			-----------
			address (str): address

			Returns:
			--------
			List[ Document ]: List of Document objects parsed from HTML content.

		'''
		try:
			throw_if( 'latitiude', lat )
			throw_if( 'longitude', long )
			self.latitude = lat
			self.longitude = long
			self.coordinates =  ( lat, long )
			self.url = r'https://maps.googleapis.com/maps/api/geocode/json?latlng='
			self.url += f'{lat},' + f'{long}' + f'&key={self.api_key}'
			_response = requests.get( self.url ).json( )
			_address = _response[ 'results' ][0][ 'formatted_address' ]
			return _address
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'GoogleMaps'
			exception.method = 'fetch_location( self, address: str ) -> Tuple[ float, float ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def validate_address( self, address: List[ str ]  ) -> Dict[ Any, Any ] | None:
		"""
			
			Purpose:
			--------
			Validate an address using Google's Address Validation API.
	
			Parameters:
			-----------
			api_key (str): Your Google Maps API key.
			address_lines (list): List of address lines (e.g. ["1600 Amphitheatre Parkway"]).
			region_code (str): Country code (default "US").
	
			Returns:
			--------
			dict: Parsed JSON response from Google.
			
		"""
		try:
			throw_if( 'address', address )
			url = 'https://addressvalidation.googleapis.com/v1:validateAddress'
			payload = \
			{
				'address':
				{
					'addressLines': address,
				}
			}
			self.params = {'key': self.api_key }
			response = requests.post( url, params=self.params, json=payload )
			if response.status_code != 200:
				msg = f'Request failed: {response.status_code} – {response.text}'
				raise RuntimeError( msg )
			return response.json( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'GoogleMaps'
			exception.method = 'validate_address( self, address: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def request_directions( self, origin: str, destination: str, mode: str='driving' ) -> str | None:
		"""
		
			Purpose:
			----------
			Request route directions from Google Maps Directions API.
		
			Parameters:
			-----------
			api_key     (str): Google Maps Platform API key.
			origin      (str): Starting location (address or lat,lng).
			destination (str): Ending location (address or lat,lng).
			mode        (str): travel mode: 'driving', 'walking', bicycling', or 'transit'.
		
			Returns:
			---------
			dict: Parsed JSON response from Google Directions API.
			
		"""
		try:
			throw_if( 'origin', origin )
			throw_if( 'destination', destination )
			self.mode = mode
			self.url = "https://maps.googleapis.com/maps/api/directions/json"
			self.params = \
			{
				'origin': origin,
				'destination': destination,
				'mode': self.mode,
				'key': self.api_key
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			_route = _results.get( 'routes', [ { } ] )[ 0 ]
			return _route
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'GoogleMaps'
			exception.method = 'request_directions( self, origin: str, destination: str ) -> dict'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> dict:
		"""
		
			Purpose:
			________
			Construct and return a fully dynamic OpenAI Tool API schema definition.
			Supports arbitrary parameters, types, nested objects, and required fields.

			Parameters:
			___________
			function (str):
			The function name exposed to the LLM.

			tool (str):
			The underlying system or service the function wraps
			(e.g., “Google Maps”, “SQLite”, “Weather API”).

			description (str):
			Precise explanation of what the function does.

			parameters (dict):
			A dictionary defining parameter names and JSON schema descriptors.
			Each value must itself be a valid JSON-schema fragment.

				Example:
					{
						"origin": {
							"type": "string",
							"description": "Starting location."
						},
						"destination": {
							"type": "string",
							"description": "Ending location."
						},
						"mode": {
							"type": "string",
							"enum": ["driving", "walking", "bicycling", "transit"],
							"description": "Travel mode."
						}
					}

			required (list[str] | None):
			List of required parameter names.
			If None, required = list(parameters.keys()).

			Returns:
			________
			dict:
			A JSON-compatible dictionary defining the tool schema.
			
		"""
		throw_if( 'function', function )
		throw_if( 'tool', tool )
		throw_if( 'description', description )
		throw_if( 'parameters', parameters )
		if not isinstance( parameters, dict ):
			raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )
		func_name = function.strip( )
		tool_name = tool.strip( )
		desc = description.strip( )
		if required is None:
			required = list( parameters.keys( ) )
		return \
		{
				'name': func_name,
				'description': f'{desc} This function uses the {tool_name} service.',
				'parameters':
				{
					'type': 'object',
					'properties': parameters,
					'required': required
				}
		}

class GoogleWeather( Fetcher ):
	'''

		Purpose:
		--------
		Provides the google drive loading functionality
		to parse items on googke drive into Document objects.

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
		

	
	'''
	gmaps: Optional[ GoogleMaps ]
	file_path: Optional[ str ]
	num_results: Optional[ int ]
	api_key: Optional[ str ]
	mode: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	coordinates: Optional[ Tuple[ float, float ] ]
	address: Optional[ str ]
	directions: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.api_key = cfg.GEOCODING_API_KEY
		self.gmaps = GoogleMaps( )
		self.mode = None
		self.url = None
		self.file_path = None
		self.longitude = None
		self.latitude = None
		self.coordinates = None
		self.fetcher = None
		self.address = None
		self.directions = None
		self.agents = cfg.AGENTS
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def current_observation( self, address: str ) -> Dict[ Any, Any ] | None:
		"""
		
			Purpose:
			--------
			Retrieve current weather conditions for an address using the
			Google Maps Weather API.
			
		"""
		try:
			throw_if( 'address', address )
			self.address = address
			lat, lng = self.gmaps.geocode_location( address )
			self.latitude = lat
			self.longitude = lng
			self.url = "https://maps.googleapis.com/maps/api/weather/v1/lookup"
			self.params = \
			{
				'location': f'{self.latitude},{self.longitude}',
				'key': self.api_key
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'GoogleWeather'
			exception.method = 'validate_address( self, address: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def future_forecast( self, address: str ) -> Dict[ Any, Any ] | None:
		"""
			
			Purpose:
			--------
			Retrieve weather forecast (hourly + daily) for an address using
			the Google Maps Weather API.
			
		"""
		try:
			throw_if( 'address', address )
			self.address = address
			lat, lng = geocode_address( self.api_key, self.address )
			self.latitude = lat
			self.longitude = lng
			self.url = "https://maps.googleapis.com/maps/api/weather/v1/forecast"
			self.params = \
			{
				'location': f'{self.latitude},{self.longitude}',
				'key': self.api_key
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'GoogleWeather'
			exception.method = 'validate_address( self, address: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> dict:
		"""
		
			Purpose:
			________
			Construct and return a fully dynamic OpenAI Tool API schema definition.
			Supports arbitrary parameters, types, nested objects, and required fields.

			Parameters:
			___________
			function (str):
			The function name exposed to the LLM.

			tool (str):
			The underlying system or service the function wraps
			(e.g., “Google Maps”, “SQLite”, “Weather API”).

			description (str):
			Precise explanation of what the function does.

			parameters (dict):
			A dictionary defining parameter names and JSON schema descriptors.
			Each value must itself be a valid JSON-schema fragment.

				Example:
					{
						"origin": {
							"type": "string",
							"description": "Starting location."
						},
						"destination": {
							"type": "string",
							"description": "Ending location."
						},
						"mode": {
							"type": "string",
							"enum": ["driving", "walking", "bicycling", "transit"],
							"description": "Travel mode."
						}
					}

			required (list[str] | None):
			List of required parameter names.
			If None, required = list(parameters.keys()).

			Returns:
			________
			dict:
			A JSON-compatible dictionary defining the tool schema.
			
		"""
		throw_if( 'function', function )
		throw_if( 'tool', tool )
		throw_if( 'description', description )
		throw_if( 'parameters', parameters )
		if not isinstance( parameters, dict ):
			raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )

		func_name = function.strip( )
		tool_name = tool.strip( )
		desc = description.strip( )
		if required is None:
			required = list( parameters.keys( ) )
		return \
		{
				'name': func_name,
				'description': f'{desc} This function uses the {tool_name} service.',
				'parameters':
				{
					'type': 'object',
					'properties': parameters,
					'required': required
				}
		}

class NavalObservatory( Fetcher ):
	'''

		Purpose:
		--------
		Provides access to APIs from the US Naval Observatory's Celestial Navigation Data for
		Assumed Position and Time:  this data service provides all the astronomical information
		necessary to plot navigational lines of position from observations of the altitudes of
		celestial bodies. Simply fill in the form below and click on the "Get Data" button at
		the end of the form.

		The output table gives both almanac data and altitude corrections for each celestial body
		that is above the horizon at the place and time that you specify. Sea-level observations
		are assumed. The almanac data consist of Greenwich hour angle (GHA), declination (Dec),
		computed altitude (Hc), and computed azimuth (Zn). The altitude corrections consist of
		atmospheric refraction (Refr), semidiameter (SD), parallax in altitude (PA), and the sum of
		the altitude corrections (Sum = Refr + SD + PA). The SD and PA values are zero for stars.
		The SD values are non-zero only for the Sun and Moon; for all other objects, it is assumed
		that the center of light is observed.

		The assumed position that you enter below can be your best estimate of your actual
		location (e.g., your DR position); there is no need to round the coordinate values,
		since all data is computed specifically for the exact position you provide without
		any table lookup.

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
		


	'''
	file_path: Optional[ str ]
	api_key: Optional[ str ]
	url: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	coordinates: Optional[ Tuple[ float, float ] ]
	calendar_date: Optional[ dt.datetime ]
	julian_date: Optional[ float ]
	sidereal_time: Optional[ str]
	utc_time: Optonal[ dt.time ]
	local_time: Optonal[ dt.time ]
	params: Optional[ Dict[ str, Any ] ]
	era: Optional[ str ]
	year: Optional[ str]
	month: Optioanl[ str ]
	day: Optional[ str ]
	gmaps: Optional[ GoogleMaps ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.api_key = cfg.GOVINFO_API_KEY
		self.gmaps = GoogleMaps( )
		self.mode = None
		self.url = None
		self.file_path = None
		self.longitude = None
		self.latitude = None
		self.coordinates = None
		self.calendar_date = None
		self.julian_date = None
		self.sidereal_time = None
		self.local_time = None
		self.utc_time = None
		self.agents = cfg.AGENTS
		self.era = None
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def fetch_julian( self, date: dt.date ) -> float | None:
		"""

			Purpose:
			--------
			Converts a given calendar date into a julian date
			
			Parameters:
			----------
			date - dt.date representing a calendar date

		"""
		try:
			throw_if( 'date', date )
			self.calendar_date = date
			self.year = str( date.year )
			self.month = str( date.month )
			self.day = str( date.day )
			self.era = 'AD'
			self.local_time = date.strftime( '%H:%M:%S' )
			self.url = f'https://aa.usno.navy.mil/api/juliandate?'
			self.params = \
			{
				'date': self.calendar_date,
				'time': self.local_time,
				'era': self.era,
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'NavalObservatory'
			exception.method = 'fetch_julian( self, address: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_sidereal( self, coords: Tuple[ float, float ], date: dt.date, time: dt.time ) -> float | None:
		"""

			Purpose:
			--------
			Converts local time into sidereal time given coordinates, date, and time
			
			Specifying the time interval requires the following three components:
				1. reps — the number of iterations
				2. intv_mag — the magnitude of the time interval between iterations
				   (i.e. if an interval of 5 minutes is desired, set as "5")
				3. intv_unit — the units of the time interval between iterations
				   (days, hours, minutes, seconds)
				   
			Parmeters:
			----------
			coords - Tuple[ float, float ] representing geographic coordinates
			date - the datetime
			time - the time (HH:MM:SS)

		"""
		try:
			throw_if( 'coords', coords )
			throw_if( 'date', date )
			throw_if( 'time', time )
			self.coordinates = coords
			self.latitude = coords[ 0 ]
			self.longitude = coords[ 1 ]
			self.calendar_date = f'{date.year}-{date.month}-{date.day}'
			self.year = str( date.year )
			self.month = str( date.month )
			self.day = str( date.day )
			self.local_time = date.strftime( '%H:%M:%S' )
			_coords = f'{sself.declination},{self.longitude}'
			self.url = f'https://aa.usno.navy.mil/api/siderealtime?'
			self.params = \
			{
				'date': self.calendar_date,
				'time': self.local_time,
				'coords': _coords,
				'reps': str( 90 ),
				'intv_mag': str( 5 ),
				'intv_unit': 'minutes',
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'GoogleWeather'
			exception.method = 'fetch_sidereal( self, date: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> dict:
		"""

			Purpose:
			________
			Construct and return a fully dynamic OpenAI Tool API schema definition.
			Supports arbitrary parameters, types, nested objects, and required fields.

			Parameters:
			___________
			function (str):
			The function name exposed to the LLM.

			tool (str):
			The underlying system or service the function wraps
			(e.g., “Google Maps”, “SQLite”, “Weather API”).

			description (str):
			Precise explanation of what the function does.

			parameters (dict):
			A dictionary defining parameter names and JSON schema descriptors.
			Each value must itself be a valid JSON-schema fragment.

				Example:
					{
						"origin": {
							"type": "string",
							"description": "Starting location."
						},
						"destination": {
							"type": "string",
							"description": "Ending location."
						},
						"mode": {
							"type": "string",
							"enum": ["driving", "walking", "bicycling", "transit"],
							"description": "Travel mode."
						}
					}

			required (list[str] | None):
			List of required parameter names.
			If None, required = list(parameters.keys()).

			Returns:
			________
			dict:
			A JSON-compatible dictionary defining the tool schema.

		"""
		throw_if( 'function', function )
		throw_if( 'tool', tool )
		throw_if( 'description', description )
		throw_if( 'parameters', parameters )
		if not isinstance( parameters, dict ):
			raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )
		
		func_name = function.strip( )
		tool_name = tool.strip( )
		desc = description.strip( )
		if required is None:
			required = list( parameters.keys( ) )
		return \
		{
			'name': func_name,
			'description': f'{desc} This function uses the {tool_name} service.',
			'parameters':
			{
				'type': 'object',
				'properties': parameters,
				'required': required
			}
		}
	
class SatelliteCenter( Fetcher ):
	'''

		Purpose:
		--------
		Provides access to APIs from NASA's Satellite Situation Center Web (SSCWeb) service
		that is operated jointly by the NASA/GSFC Space Physics Data Facility (SPDF) and the
		National Space Science Data Center (NSSDC) to support a range of NASA science programs
		and to fulfill key international NASA responsibilities including those of NSSDC and the
		World Data Center-A for Rockets and Satellites.
		
		The software and associated database of SSCWeb together form a system to cast geocentric
		spacecraft location information into a framework of (empirical) geophysical regions and
		mappings of spacecraft locations along lines of the Earth's magnetic field.
		
		This capability is one key to mission science planning (both single missions and
		coordinated observations of multiple spacecraft with ground-based investigations) and to
		subsequent multi-mission data analysis.

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
		


	'''
	ssc: Optional[ SscWs ]
	file_path: Optional[ str ]
	api_key: Optional[ str ]
	url: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	coordinates: Optional[ Tuple[ float, float ] ]
	calendar_date: Optional[ dt.datetime ]
	utc_time: Optonal[ dt.time ]
	local_time: Optonal[ dt.time ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.api_key = cfg.NASA_API_KEY
		self.ssc = None
		self.url = None
		self.file_path = None
		self.longitude = None
		self.latitude = None
		self.coordinates = None
		self.fetcher = None
		self.calendar_date = None
		self.julian_date = None
		self.local_time = None
		self.utc_time = None
		self.agents = cfg.AGENTS
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
	def fetch_observatories( self ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			This service provides descriptions of the observatories that are available from SSC.

		"""
		try:
			self.ssc = SscWs( )
			result = self.ssc.get_observatories( )
			observatories = result[ 'Observatory' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'SatelliteCenter'
			exception.method = 'fetch_observatories( self ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_locations( self ) ->  None:
		"""

			Purpose:
			--------
			This service provides observatory location information.

			Parameters:
			----------
			date - dt.date representing a calendar date

		"""
		try:
			self.ssc = SscWs( )
			result = self.ssc.get_locations( [ observatory ], example_interval )
			data = result[ 'Data' ][ 0 ]
			coords = data[ 'Coordinates' ][ 0 ]
			fig = plt.figure( )
			if version.parse( mpl.__version__ ) < version.parse( '3.4' ):
				ax = fig.gca( projection='3d' )
			else:
				ax = Axes3D( fig, auto_add_to_figure=False )
				fig.add_axes( ax )
			ax.set_xlabel( 'X (km)' )
			ax.set_ylabel( 'Y (km)' )
			ax.set_zlabel( 'Z (km)' )
			title = data[ 'Id' ] + ' Orbit (' + \
			        coords[ 'CoordinateSystem' ].value.upper( ) + ')'
			ax.plot( coords[ 'X' ], coords[ 'Y' ], coords[ 'Z' ], label=title )
			ax.legend( )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'SatelliteCenter'
			exception.method = 'fetch_locations( self )'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_time_interal( self, select_obs, observatories ):
		try:
			self.ssc = SscWs( )
			for obs in observatories:
				if obs[ 'Id' ] == select_obs:
					end = obs[ 'EndTime' ]
					return TimeInterval( end - timedelta( hours=2 ), end )
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'NavalObservatory'
			exception.method = 'fetch_julian( self, address: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_ground_stations( self ):
		try:
			self.ssc = SscWs( )
			result = self.ssc.get_ground_stations( )
			for ground_station in result[ 'GroundStation' ][ :5 ]:
				location = ground_station[ 'Location' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'NavalObservatory'
			exception.method = 'fetch_julian( self, address: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> dict:
		"""

			Purpose:
			________
			Construct and return a fully dynamic OpenAI Tool API schema definition.
			Supports arbitrary parameters, types, nested objects, and required fields.

			Parameters:
			___________
			function (str):
			The function name exposed to the LLM.

			tool (str):
			The underlying system or service the function wraps
			(e.g., “Google Maps”, “SQLite”, “Weather API”).

			description (str):
			Precise explanation of what the function does.

			parameters (dict):
			A dictionary defining parameter names and JSON schema descriptors.
			Each value must itself be a valid JSON-schema fragment.

				Example:
					{
						"origin": {
							"type": "string",
							"description": "Starting location."
						},
						"destination": {
							"type": "string",
							"description": "Ending location."
						},
						"mode": {
							"type": "string",
							"enum": ["driving", "walking", "bicycling", "transit"],
							"description": "Travel mode."
						}
					}

			required (list[str] | None):
			List of required parameter names.
			If None, required = list(parameters.keys()).

			Returns:
			________
			dict:
			A JSON-compatible dictionary defining the tool schema.

		"""
		throw_if( 'function', function )
		throw_if( 'tool', tool )
		throw_if( 'description', description )
		throw_if( 'parameters', parameters )
		if not isinstance( parameters, dict ):
			raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )
		
		func_name = function.strip( )
		tool_name = tool.strip( )
		desc = description.strip( )
		if required is None:
			required = list( parameters.keys( ) )
		return \
			{
					'name': func_name,
					'description': f'{desc} This function uses the {tool_name} service.',
					'parameters':
						{
								'type': 'object',
								'properties': parameters,
								'required': required
						}
			}

class EarthObservatory( Fetcher ):
	'''

		Purpose:
		--------
		Provides access to APIs from NASA's Earth Obseratory Natural Events Tracker (EONET) .
		The Earth Observatory Natural Event Tracker (EONET) is a web service for
		providing a curated source of continuously updated natural event metadata;
		providing a service that links those natural events to thematically-related
		web service-enabled image sources (e.g., via WMS, WMTS, etc.).

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
		


	'''
	file_path: Optional[ str ]
	api_key: Optional[ str ]
	url: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	days: Optional[ int ]
	calendar_date: Optional[ dt.datetime ]
	utc_time: Optonal[ dt.time ]
	local_time: Optonal[ dt.time ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.api_key = cfg.NASA_API_KEY
		self.url = None
		self.days = None
		self.longitude = None
		self.latitude = None
		self.coordinates = None
		self.calendar_date = None
		self.local_time = None
		self.utc_time = None
		self.agents = cfg.AGENTS
		self.era = None
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def fetch_events( self, count: int=30 ) -> Dict[ str, Any ]| None:
		"""

			Purpose:
			--------
			Converts a given calendar date into a julian date

			Parameters:
			----------
			date - dt.date representing a calendar date

		"""
		try:
			throw_if( 'count', count )
			self.days = count
			self.url = f'https://eonet.gsfc.nasa.gov/api/v2.1/events?'
			self.params = \
			{
					'days': f'{ self.days }',
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'EarthObservatory'
			exception.method = 'fetch_julian( self, address: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_categories( self, count: int=30 ) -> Dict[ str, Any ]| None:
		"""

			Purpose:
			--------
			Converts a given calendar date into a julian date

			Parameters:
			----------
			date - dt.date representing a calendar date

		"""
		try:
			throw_if( 'count', count )
			self.days = count
			self.url = f'https://eonet.gsfc.nasa.gov/api/v2.1/events?'
			self.params = \
				{
						'days': f'{self.days}',
				}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'EarthObservatory'
			exception.method = 'fetch_julian( self, address: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
			
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> dict:
		"""

			Purpose:
			________
			Construct and return a fully dynamic OpenAI Tool API schema definition.
			Supports arbitrary parameters, types, nested objects, and required fields.

			Parameters:
			___________
			function (str):
			The function name exposed to the LLM.

			tool (str):
			The underlying system or service the function wraps
			(e.g., “Google Maps”, “SQLite”, “Weather API”).

			description (str):
			Precise explanation of what the function does.

			parameters (dict):
			A dictionary defining parameter names and JSON schema descriptors.
			Each value must itself be a valid JSON-schema fragment.

				Example:
					{
						"origin": {
							"type": "string",
							"description": "Starting location."
						},
						"destination": {
							"type": "string",
							"description": "Ending location."
						},
						"mode": {
							"type": "string",
							"enum": ["driving", "walking", "bicycling", "transit"],
							"description": "Travel mode."
						}
					}

			required (list[str] | None):
			List of required parameter names.
			If None, required = list(parameters.keys()).

			Returns:
			________
			dict:
			A JSON-compatible dictionary defining the tool schema.

		"""
		throw_if( 'function', function )
		throw_if( 'tool', tool )
		throw_if( 'description', description )
		throw_if( 'parameters', parameters )
		if not isinstance( parameters, dict ):
			raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )
		
		func_name = function.strip( )
		tool_name = tool.strip( )
		desc = description.strip( )
		if required is None:
			required = list( parameters.keys( ) )
		return \
		{
				'name': func_name,
				'description': f'{desc} This function uses the {tool_name} service.',
				'parameters':
				{
					'type': 'object',
					'properties': parameters,
					'required': required
				}
		}

class GlobalImagery( Fetcher ):
	'''

		Purpose:
		--------
		Provides access to APIs from NASA's Global Imagery Browse Services (GIBS) that is
		designed to deliver global, full-resolution satellite imagery to users in a highly
		responsive manner, enabling visual discovery of scientific phenomena, supporting timely
		decision-making for natural hazards, educating the next generation of scientists,
		and making imagery of the planet more accessible to the media and public.
		
		GIBS provides quick access to over 1,000 satellite imagery products, covering every part
		of the world. Most imagery is updated daily - available within a few hours after satellite
		observation, and some products span almost 30 years

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
		


	'''
	file_path: Optional[ str ]
	api_key: Optional[ str ]
	url: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	coordinates: Optional[ Tuple[ float, float ] ]
	calendar_date: Optional[ dt.datetime ]
	julian_date: Optional[ float ]
	sidereal_time: Optional[ str ]
	utc_time: Optonal[ dt.time ]
	local_time: Optonal[ dt.time ]
	params: Optional[ Dict[ str, Any ] ]
	era: Optional[ str ]
	year: Optional[ str ]
	month: Optioanl[ str ]
	day: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.api_key = cfg.NASA_API_KEY
		self.mode = None
		self.url = None
		self.file_path = None
		self.longitude = None
		self.latitude = None
		self.coordinates = None
		self.fetcher = None
		self.calendar_date = None
		self.julian_date = None
		self.sidereal_time = None
		self.local_time = None
		self.utc_time = None
		self.agents = cfg.AGENTS
		self.era = None
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def fetch_map_services( self ):
		try:
			# Connect to GIBS WMS Service
			wms = WebMapService( 'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?',
				version='1.1.1' )
			img = wms.getmap( layers=[ 'MODIS_Terra_CorrectedReflectance_TrueColor' ],  # Layers
				srs='epsg:4326',  # Map projection
				bbox=(-180, -90, 180, 90),  # Bounds
				size=( 1200, 600 ),  # Image size
				time='2021-09-21',  # Time of data
				format='image/png',  # Image format
				transparent=True )  # Nodata transparency
			out = open( 'python-examples/MODIS_Terra_CorrectedReflectance_TrueColor.png', 'wb' )
			out.write( img.read( ) )
			out.close( )
			Image( 'python-examples/MODIS_Terra_CorrectedReflectance_TrueColor.png' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'GlobalImageryServices'
			exception.method = 'fetch_sidereal( self, date: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_mercator_map( self ):
		try:
			proj3857 = 'https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi?\
			version=1.3.0&service=WMS&\
			request=GetMap&format=image/png&STYLE=default&bbox=-8000000,-8000000,8000000,8000000&\
			CRS=EPSG:3857&HEIGHT=600&WIDTH=600&TIME=2000-12-01&layers=Landsat_WELD_CorrectedReflectance_Bands157_Global_Annual'
			img = io.imread( proj3857 )
			fig = plt.figure( )
			ax = fig.add_subplot( 1, 1, 1, projection=ccrs.Mercator.GOOGLE )
			extent = (-8000000, 8000000, -8000000, 8000000)
			plt.imshow( img, transform=ccrs.Mercator.GOOGLE, extent=extent, origin='upper' )
			gl = ax.gridlines( ccrs.PlateCarree( ), linewidth=1, color='blue', alpha=0.3, draw_labels=True )
			gl.top_labels = False
			gl.right_labels = False
			gl.xlines = True
			gl.ylines = True
			gl.xlocator = mticker.FixedLocator( [ 0, 30, -30,  0 ] )
			gl.ylocator = mticker.FixedLocator( [ -30, 0, 30 ] )
			gl.xformatter = LONGITUDE_FORMATTER
			gl.yformatter = LATITUDE_FORMATTER
			gl.xlabel_style = { 'color': 'blue' }
			gl.ylabel_style = { 'color': 'blue' }
			plt.title( 'Mercator Projection', fontname='Roboto', fontsize=20, color='green' )
			plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'GoogleWeather'
			exception.method = 'fetch_sidereal( self, date: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> dict:
		"""

			Purpose:
			________
			Construct and return a fully dynamic OpenAI Tool API schema definition.
			Supports arbitrary parameters, types, nested objects, and required fields.

			Parameters:
			___________
			function (str):
			The function name exposed to the LLM.

			tool (str):
			The underlying system or service the function wraps
			(e.g., “Google Maps”, “SQLite”, “Weather API”).

			description (str):
			Precise explanation of what the function does.

			parameters (dict):
			A dictionary defining parameter names and JSON schema descriptors.
			Each value must itself be a valid JSON-schema fragment.

				Example:
					{
						"origin": {
							"type": "string",
							"description": "Starting location."
						},
						"destination": {
							"type": "string",
							"description": "Ending location."
						},
						"mode": {
							"type": "string",
							"enum": ["driving", "walking", "bicycling", "transit"],
							"description": "Travel mode."
						}
					}

			required (list[str] | None):
			List of required parameter names.
			If None, required = list(parameters.keys()).

			Returns:
			________
			dict:
			A JSON-compatible dictionary defining the tool schema.

		"""
		throw_if( 'function', function )
		throw_if( 'tool', tool )
		throw_if( 'description', description )
		throw_if( 'parameters', parameters )
		if not isinstance( parameters, dict ):
			raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )
		
		func_name = function.strip( )
		tool_name = tool.strip( )
		desc = description.strip( )
		if required is None:
			required = list( parameters.keys( ) )
		return \
			{
					'name': func_name,
					'description': f'{desc} This function uses the {tool_name} service.',
					'parameters':
						{
								'type': 'object',
								'properties': parameters,
								'required': required
						}
			}

class NearbyObjects( Fetcher ):
	'''

		Purpose:
		--------
		Provides access to APIs from JPL’s SSD (Solar System Dynamics) and CNEOS
		(Center for Near-Earth Object Studies) API (Application Program Interface) service.
		This service provides an interface to machine-readable data (JSON-format) related to SSD
		and CNEOS.

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]

	'''
	file_path: Optional[ str ]
	api_key: Optional[ str ]
	url: Optional[ str ]
	declination: Optional[ float ]
	right_ascension: Optional[ float ]
	coordinates: Optional[ Tuple[ float, float ] ]
	start_date: Optional[ dt.date ]
	end_date: Optional[ dt.date ]
	distance: Optional[ int ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.api_key = cfg.GOVINFO_API_KEY
		self.url = None
		self.file_path = None
		self.right_ascension = None
		self.declination = None
		self.coordinates = None
		self.start_date = None
		self.end_date = None
		self.distance = None
		self.agents = cfg.AGENTS
		self.era = None
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def fetch_nearby( self, start: dt.date, end: dt.date, min_dist: int=10 ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Converts a given calendar date into a julian date
			*Lunar Distance units are in 'LD', ex. 10LD
			*Date formats are YYYY-MM-DD

			Parameters:
			----------
			start - dt.date representing a calendar date
			end - dt.date representing a calendar date
			min_dist - lunar distance minimum
			
			Returns:
			-------
			Dict[ stc, Any ]

		"""
		try:
			throw_if( 'start', start )
			throw_if( 'end', end )
			self.start_date = start
			self.end_date = end
			self.distance = min_dist
			self.url = f'https://ssd-api.jpl.nasa.gov/cad.api?'
			self.params = \
			{
					'date-min': f'{ self.start_date }',
					'date-max': f'{ self.end_date }',
					'min-dist-min': f'{ self.distance }',
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'NearByObjects'
			exception.method = ('fetch_nearby( self, start: dt.date, end: dt.date, '
			                    'min_dist: int=10 ) -> Dict[ str, Any ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_fireballs( self,  start: dt.date, end: dt.date ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------


			Parmeters:
			----------


		"""
		try:
			throw_if( 'start', start )
			throw_if( 'end', end )
			self.start_date = start
			self.end_date = end
			self.url = f'https://aa.usno.navy.mil/api/siderealtime?'
			self.params = \
			{
					'date-min': f'{ self.start_date }',
					'date-max': f'{ self.end_date }',
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'SolarSystemDynamics'
			exception.method = 'fetch_sidereal( self, date: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_asteroids( self ) -> Dict[ str, Any ] | None:
		try:
			self.url = r'https://ssd-api.jpl.nasa.gov/nhats.api?'
			self.response = requests.get( url=self.url )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'NearByObjects'
			exception.method = 'fetch_fireballs( self,  start: dt.date, end: dt.date ) -> Dict[ str, Any ] '
			error = ErrorDialog( exception )
			error.show( )
		
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> dict:
		"""

			Purpose:
			________
			Construct and return a fully dynamic OpenAI Tool API schema definition.
			Supports arbitrary parameters, types, nested objects, and required fields.

			Parameters:
			___________
			function (str):
			The function name exposed to the LLM.

			tool (str):
			The underlying system or service the function wraps
			(e.g., “Google Maps”, “SQLite”, “Weather API”).

			description (str):
			Precise explanation of what the function does.

			parameters (dict):
			A dictionary defining parameter names and JSON schema descriptors.
			Each value must itself be a valid JSON-schema fragment.

				Example:
					{
						"origin": {
							"type": "string",
							"description": "Starting location."
						},
						"destination": {
							"type": "string",
							"description": "Ending location."
						},
						"mode": {
							"type": "string",
							"enum": ["driving", "walking", "bicycling", "transit"],
							"description": "Travel mode."
						}
					}

			required (list[str] | None):
			List of required parameter names.
			If None, required = list(parameters.keys()).

			Returns:
			________
			dict:
			A JSON-compatible dictionary defining the tool schema.

		"""
		throw_if( 'function', function )
		throw_if( 'tool', tool )
		throw_if( 'description', description )
		throw_if( 'parameters', parameters )
		if not isinstance( parameters, dict ):
			raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )
		
		func_name = function.strip( )
		tool_name = tool.strip( )
		desc = description.strip( )
		if required is None:
			required = list( parameters.keys( ) )
		return \
			{
					'name': func_name,
					'description': f'{desc} This function uses the {tool_name} service.',
					'parameters':
						{
								'type': 'object',
								'properties': parameters,
								'required': required
						}
			}

class OpenScience( Fetcher ):
	'''

		Purpose:
		--------
		Provides access to APIs from NASA's Open Science Data Repostitory (OSDR).
		NASA OSDR provides a RESTful Application Programming Interface (API) to its
		full-text search, data file retrieval, and metadata retrieval capabilities.
		The API provides a choice of standard web output formats,
		either JavaScript Object Notation (JSON) or Hyper Text Markup Language (HTML),
		of query results.
		
		The Data File API returns metadata on data files associated with dataset(s),
		including the location of these files for download via https. The Metadata API returns
		entire sets of metadata for input study dataset accession numbers. The Search API can be
		used to search dataset metadata by keywords and/or metadata. It can also be used to provide
		search of three other omics databases: the National Institutes of Health (NIH) /
		National Center for Biotechnology Information's (NCBI) Gene Expression Omnibus (GEO);
		the European Bioinformatics Institute's (EBI) Proteomics Identification (PRIDE);
		the Argonne National Laboratory's (ANL);
		Metagenomics Rapid Annotations using Subsystems Technology (MG-RAST).

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]

	'''
	file_path: Optional[ str ]
	api_key: Optional[ str ]
	url: Optional[ str ]
	keywords: Optional[ str ]
	size: Optional[ int ]
	datasource: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.api_key = cfg.GOVINFO_API_KEY
		self.url = None
		self.agents = cfg.AGENTS
		self.keywords = None
		self.size = None
		self.datasource = None
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def fetch_dataset( self, keyword: str, results: int=100 ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Requests study data to OSDR given  'keywords' and a limitation ('results')
			
			Parameters:
			----------
			keyword - str, the search criteria
			results - int, the limit
			
			Returns:
			-------
			Dict[ str, Any ]

		"""
		try:
			throw_if( 'keyword', keyword )
			self.keywords = keyword
			self.datasource = 'cgene, nih_geo, ebi_pride, mg_rast'
			self.size = results
			self.url = f'https://osdr.nasa.gov/osdr/data/search?'
			self.params = \
			{
					'term': self.keywords,
					'type': self.datasource,
					'size': f'{ self.size }',
					'api_key': self.api_key
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'OpenScience'
			exception.method = 'fetch_dataset( self, keyword: str, results: int=100 ) -> Dict[ str, Any ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_studies( self, keywords: str ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Requests study data to OSDR given  'keywords'

			Parmeters:
			----------
			keyword - str, the search criteria
			
			Returns:
			-------
			Dict[ str, Any ]

		"""
		try:
			throw_if( 'keywords', keywords )
			self.keywords = keyword
			self.datasource = 'cgene, nih_geo, ebi_pride, mg_rast'
			self.url = f'https://osdr.nasa.gov/bio/repo/search?'
			self.params = \
			{
					'q': self.keywords,
					'type': self.datasource,
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'OpenScience'
			exception.method = 'fetch_studies( self, keywords: str ) -> Dict[ str, Any ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> dict:
		"""

			Purpose:
			________
			Construct and return a fully dynamic OpenAI Tool API schema definition.
			Supports arbitrary parameters, types, nested objects, and required fields.

			Parameters:
			___________
			function (str):
			The function name exposed to the LLM.

			tool (str):
			The underlying system or service the function wraps
			(e.g., “Google Maps”, “SQLite”, “Weather API”).

			description (str):
			Precise explanation of what the function does.

			parameters (dict):
			A dictionary defining parameter names and JSON schema descriptors.
			Each value must itself be a valid JSON-schema fragment.

				Example:
					{
						"origin": {
							"type": "string",
							"description": "Starting location."
						},
						"destination": {
							"type": "string",
							"description": "Ending location."
						},
						"mode": {
							"type": "string",
							"enum": ["driving", "walking", "bicycling", "transit"],
							"description": "Travel mode."
						}
					}

			required (list[str] | None):
			List of required parameter names.
			If None, required = list(parameters.keys()).

			Returns:
			________
			dict:
			A JSON-compatible dictionary defining the tool schema.

		"""
		throw_if( 'function', function )
		throw_if( 'tool', tool )
		throw_if( 'description', description )
		throw_if( 'parameters', parameters )
		if not isinstance( parameters, dict ):
			raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )
		
		func_name = function.strip( )
		tool_name = tool.strip( )
		desc = description.strip( )
		if required is None:
			required = list( parameters.keys( ) )
		return \
			{
					'name': func_name,
					'description': f'{desc} This function uses the {tool_name} service.',
					'parameters':
						{
								'type': 'object',
								'properties': parameters,
								'required': required
						}
			}

class SpaceWeather( Fetcher ):
	'''

		Purpose:
		--------
		Access to The Space Weather Database Of Notifications, Knowledge, Information (DONKI)
		is a comprehensive on-line tool for space weather forecasters, scientists, and the general
		space science community.
		
		DONKI chronicles the daily interpretations of space weather observations, analysis, models,
		forecasts, and notifications provided by the Space Weather Research Center (SWRC),
		comprehensive knowledge-base search functionality to support anomaly resolution and space
		science research, intelligent linkages, relationships, cause-and-effects between  space
		weather activities and comprehensive webservice API access to information stored in DONKI.

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]

	'''
	api_key: Optional[ str ]
	url: Optional[ str ]
	start_date: Optional[ dt.datetime ]
	end_date: Optional[ dt.datetime ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.api_key = cfg.GOVINFO_API_KEY
		self.start_date = None
		self.end_date = None
		self.url = None
		self.agents = cfg.AGENTS
		self.params = None
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def fetch_ejections( self, start: dt.date, end: dt.date ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Retrieves Coronal Mass Ejectionss given a start and end date

			Parameters:
			----------
			start - dt.date representing a calendar date
			end - dt.date
			
			Returns:
			-------
			Dict[ str, Any ]

		"""
		try:
			throw_if( 'start', start )
			throw_if( 'end', end )
			self.start_date = start
			self.end_date = end
			self.url = f'https://api.nasa.gov/DONKI/CME?'
			self.params = \
			{
				'startDate': f'{ self.start_date }',
				'endDate': f'{ self.end_date }',
				'api_key': f'{ self.api_key }',
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'NavalObservatory'
			exception.method = 'fetch_julian( self, address: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_analysis( self, start: dt.date, end: dt.date ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Retrieves CME analysis data given a start and end date

			Parameters:
			----------
			start - dt.date representing a calendar date
			end - dt.date
			
			Returns:
			-------
			Dict[ str, Any ]

		"""
		try:
			throw_if( 'start', start )
			throw_if( 'end', end )
			self.start_date = start
			self.end_date = end
			self.url = f'https://api.nasa.gov/DONKI/CMEAnalysis?'
			self.params = \
			{
					'startDate': f'{ self.start_date }',
					'endDate': f'{ self.end_date }',
					'api_key': f'{ self.api_key }',
					'mostAccurateOnly': 'true',
					'speed': '500',
					'halfAngle': '30',
					'catalog': 'ALL'
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'SpaceWeather'
			exception.method = ('fetch_ejection_analysis( self, start: dt.date, end: dt.date )'
			                    ' -> Dict[ str, Any ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_storms( self, start: dt.date, end: dt.date ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------

			Parmeters:
			----------

		"""
		try:
			throw_if( 'start', start )
			throw_if( 'end', end )
			self.url = f'https://api.nasa.gov/DONKI/GST?'
			self.params = \
			{
				'startDate': f'{ self.start_date }',
				'endDate': f'{ self.end_date }',
				'api_key': f'{ self.api_key }',
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'SpaceWeather'
			exception.method = 'fetch_geomagnetic_storms( self, date: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_solar_flares( self, start: dt.date, end: dt.date ) -> float | None:
		"""

			Purpose:
			--------
		

			Parmeters:
			----------
			

		"""
		try:
			throw_if( 'start', start )
			throw_if( 'end', end )
			self.url = f'https://api.nasa.gov/DONKI/FLR?'
			self.params = \
			{
					'startDate': self.calendar_date,
					'endDate': self.local_time,
					'api_key': self.api_key
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'DONKI'
			exception.method = 'fetch_sidereal( self, date: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> dict:
		"""

			Purpose:
			________
			Construct and return a fully dynamic OpenAI Tool API schema definition.
			Supports arbitrary parameters, types, nested objects, and required fields.

			Parameters:
			___________
			function (str):
			The function name exposed to the LLM.

			tool (str):
			The underlying system or service the function wraps
			(e.g., “Google Maps”, “SQLite”, “Weather API”).

			description (str):
			Precise explanation of what the function does.

			parameters (dict):
			A dictionary defining parameter names and JSON schema descriptors.
			Each value must itself be a valid JSON-schema fragment.

				Example:
					{
						"origin": {
							"type": "string",
							"description": "Starting location."
						},
						"destination": {
							"type": "string",
							"description": "Ending location."
						},
						"mode": {
							"type": "string",
							"enum": ["driving", "walking", "bicycling", "transit"],
							"description": "Travel mode."
						}
					}

			required (list[str] | None):
			List of required parameter names.
			If None, required = list(parameters.keys()).

			Returns:
			________
			dict:
			A JSON-compatible dictionary defining the tool schema.

		"""
		throw_if( 'function', function )
		throw_if( 'tool', tool )
		throw_if( 'description', description )
		throw_if( 'parameters', parameters )
		if not isinstance( parameters, dict ):
			raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )
		
		func_name = function.strip( )
		tool_name = tool.strip( )
		desc = description.strip( )
		if required is None:
			required = list( parameters.keys( ) )
		return \
		{
			'name': func_name,
			'description': f'{desc} This function uses the {tool_name} service.',
			'parameters':
			{
				'type': 'object',
				'properties': parameters,
				'required': required
			}
		}

class AstroCatalog( Fetcher ):
	'''

		Purpose:
		--------
		Access to the Open Astronomy Catalog API (OACAPI) offers a lightweight,
		simple way to access data available via the api.
		
		The pattern for the API is one of the domains listed above
		(e.g. https://api.astrocats.space) followed by
		
			/OBJECT/QUANTITY/ATTRIBUTE?ARGUMENT1=VALUE1&ARGUMENT2=VALUE2&...
		
		where OBJECT is set to a transient's name, QUANTITY is set to a desired
		quantity to retrieve (e.g. redshift), ATTRIBUTE is a property of that quantity,
		and the ARGUMENT variables allow to user to filter data based upon various
		attribute values. The ARGUMENT variables can either be used to guarantee that
		a certain attribute appears in the returned results (e.g. adding &time&e_magnitude to
		the query will guarantee that each returned item has a time and e_magnitude attribute),
		or used to filter via a simple equality such as telescope=HST
		(which would only return QUANTITY objects where the telescope attribute equals "HST"),
		or they can be more powerful for certain filter attributes
		(examples being ra and dec for performing cone searches).

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
		

		
	'''
	api_key: Optional[ str ]
	url: Optional[ str ]
	radius: Optional[ int ]
	format: Optional[ str ]
	name: Optional[ str ]
	declination: Optional[ str ]
	right_ascension: Optional[ str ]
	start_date: Optional[ dt.datetime ]
	end_date: Optional[ dt.datetime ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.api_key = cfg.GOVINFO_API_KEY
		self.radius = None
		self.format = None
		self.name = None
		self.right_ascension = None
		self.declination = None
		self.start_date = None
		self.end_date = None
		self.url = None
		self.agents = cfg.AGENTS
		self.params = None
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def cone_search( self, ra: str, dec: str, radius: int=2 ) -> float | None:
		"""

			Purpose:
			--------
			Submits query that returns all objects within a 2" cone about coordinates (ra, dec)

			Parameters:
			----------
			ra - str, eg. HH:MM:SS.AS
			dec - str, eg. HH:MM:SS.AS
			radius - int

		"""
		try:
			throw_if( 'ra', ra )
			throw_if( 'dec', dec )
			self.declination = dec
			self.right_ascension = ra
			self.radius = radius
			self.url = f'https://api.astrocats.space/catalog?'
			self.params = \
			{
				'ra': self.right_ascension,
				'dec': self.declination,
				'radius': f'{self.radius}',
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'AstroCatalog'
			exception.method = 'cone_search( self, ra: str, dec: str, radius: int=2 ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_supernovae( self ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Return all supernova metadata in CSV format

			Parmeters:
			----------
			Void

		"""
		try:
			self.format = 'CSV'
			self.url = f'https://api.astrocats.space/catalog/sne?'
			self.params = \
			{
				'format': self.format,
			}
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'AstroCatalog'
			exception.method = 'fetch_sidereal( self, date: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_redshift( self, object: str ) -> float | None:
		"""

			Purpose:
			--------
			Sumbits a query that gets the available redshift values for a given object

			Parmeters:
			----------

		"""
		try:
			throw_if( 'object', object )
			self.name = object
			self.url = f'https://api.astrocats.space/{self.name}/redshift'
			self.response = requests.get( url=self.url  )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'AstroCatalog'
			exception.method = 'fetch_sidereal( self, date: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_solar_flares( self, start: dt.date, end: dt.date ) -> float | None:
		"""

			Purpose:
			--------


			Parmeters:
			----------


		"""
		try:
			throw_if( 'start', start )
			throw_if( 'end', end )
			self.url = f'https://api.nasa.gov/DONKI/FLR?'
			self.params = \
				{
						'startDate': self.calendar_date,
						'endDate': self.local_time,
						'api_key': self.api_key
				}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'AstroCatalog'
			exception.method = 'fetch_sidereal( self, date: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> dict:
		"""

			Purpose:
			________
			Construct and return a fully dynamic OpenAI Tool API schema definition.
			Supports arbitrary parameters, types, nested objects, and required fields.

			Parameters:
			___________
			function (str):
			The function name exposed to the LLM.

			tool (str):
			The underlying system or service the function wraps
			(e.g., “Google Maps”, “SQLite”, “Weather API”).

			description (str):
			Precise explanation of what the function does.

			parameters (dict):
			A dictionary defining parameter names and JSON schema descriptors.
			Each value must itself be a valid JSON-schema fragment.

				Example:
					{
						"origin": {
							"type": "string",
							"description": "Starting location."
						},
						"destination": {
							"type": "string",
							"description": "Ending location."
						},
						"mode": {
							"type": "string",
							"enum": ["driving", "walking", "bicycling", "transit"],
							"description": "Travel mode."
						}
					}

			required (list[str] | None):
			List of required parameter names.
			If None, required = list(parameters.keys()).

			Returns:
			________
			dict:
			A JSON-compatible dictionary defining the tool schema.

		"""
		throw_if( 'function', function )
		throw_if( 'tool', tool )
		throw_if( 'description', description )
		throw_if( 'parameters', parameters )
		if not isinstance( parameters, dict ):
			raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )
		
		func_name = function.strip( )
		tool_name = tool.strip( )
		desc = description.strip( )
		if required is None:
			required = list( parameters.keys( ) )
		return \
		{
			'name': func_name,
			'description': f'{desc} This function uses the {tool_name} service.',
			'parameters':
			{
					'type': 'object',
					'properties': parameters,
					'required': required
			}
		}

class AstroQuery( Fetcher ):
	'''

		Purpose:
		--------
		Access to the astropy package that contains key functionality and common tools needed for
		performing astronomy and astrophysics with Python. It is at the core of the Astropy Project,
		which aims to enable the community to develop a robust ecosystem of affiliated packages
		covering a broad range of needs for astronomical research, data processing, and data analysis.

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]

	'''
	api_key: Optional[ str ]
	url: Optional[ str ]
	radius: Optional[ int ]
	format: Optional[ str ]
	name: Optional[ str ]
	catalog: Optional[ str ]
	declination: Optional[ str ]
	right_ascension: Optional[ str ]
	start_date: Optional[ dt.datetime ]
	end_date: Optional[ dt.datetime ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.api_key = cfg.GOVINFO_API_KEY
		self.radius = None
		self.format = None
		self.dso = None
		self.radius = None
		self.right_ascension = None
		self.declination = None
		self.start_date = None
		self.end_date = None
		self.url = None
		self.agents = cfg.AGENTS
		self.params = None
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def dso_search( self, dso: str ) -> Table | None:
		"""

			Purpose:
			--------

			Parameters:
			----------

		"""
		try:
			throw_if( 'dso', dso )
			self.dso = dso
			_results = Simbad.query_object( self.dso )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'AstroQuery'
			exception.method = 'fetch_julian( self, address: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def region_search( self, dso: str, radius: float=0.5 ) -> Table | None:
		"""

			Purpose:
			--------

			Parmeters:
			----------

		"""
		try:
			simbad = Simbad( )
			simbad.ROW_LIMIT = 100
			_result = simbad.query_region( 'm81', radius='0.5d' )
			return _result
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'AstroQuery'
			exception.method = 'fetch_sidereal( self, date: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def catalog_search( self, name: str='ESO' ) -> Table| None:
		"""

			Purpose:
			--------

			Parmeters:
			----------

		"""
		try:
			throw_if( 'name', name )
			self.name = name
			simbad = Simbad( ROW_LIMIT=6 )
			_results = simbad.query_catalog( 'ESO' )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'OpenAstronomyCatalog'
			exception.method = 'catalog_search( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> dict:
		"""

			Purpose:
			________
			Construct and return a fully dynamic OpenAI Tool API schema definition.
			Supports arbitrary parameters, types, nested objects, and required fields.

			Parameters:
			___________
			function (str):
			The function name exposed to the LLM.

			tool (str):
			The underlying system or service the function wraps
			(e.g., “Google Maps”, “SQLite”, “Weather API”).

			description (str):
			Precise explanation of what the function does.

			parameters (dict):
			A dictionary defining parameter names and JSON schema descriptors.
			Each value must itself be a valid JSON-schema fragment.

				Example:
					{
						"origin": {
							"type": "string",
							"description": "Starting location."
						},
						"destination": {
							"type": "string",
							"description": "Ending location."
						},
						"mode": {
							"type": "string",
							"enum": ["driving", "walking", "bicycling", "transit"],
							"description": "Travel mode."
						}
					}

			required (list[str] | None):
			List of required parameter names.
			If None, required = list(parameters.keys()).

			Returns:
			________
			dict:
			A JSON-compatible dictionary defining the tool schema.

		"""
		throw_if( 'function', function )
		throw_if( 'tool', tool )
		throw_if( 'description', description )
		throw_if( 'parameters', parameters )
		if not isinstance( parameters, dict ):
			raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )
		
		func_name = function.strip( )
		tool_name = tool.strip( )
		desc = description.strip( )
		if required is None:
			required = list( parameters.keys( ) )
		return \
		{
			'name': func_name,
			'description': f'{desc} This function uses the {tool_name} service.',
			'parameters':
			{
					'type': 'object',
					'properties': parameters,
					'required': required
			}
		}

class StarMap( Fetcher ):
	'''
		
		Purpose:
		-------
		Class providing access to StarMap.org celestial map generation functionality.

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
	'''
	url: Optional[ str ]
	image_source: Optional[ str ]
	object: Optional[ str ]
	right_ascension: Optional[ float ]
	declination: Optional[ float ]
	data: Optional[ dt.date ]
	time: Optional[ dt.time ]
	box_color: Optional[ str ]
	main_color: Optional[ str ]
	show_box: Optional[ bool ]
	show_grid: Optional[ bool ]
	show_lines: Optional[ bool ]
	show_boundaries: Optional[ bool ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ):
		'''
		
			Purpose:
			-------
			StarMap Class constructor
			
		'''
		super( ).__init__( )
		self.api_key = cfg.GOVINFO_API_KEY
		self.right_ascension = None
		self.declination = None
		self.object = None
		self.date = None
		self.time = None
		self.show_box = True
		self.show_grid = True
		self.show_lines = True
		self.show_boundaries = True
		self.image_source = None
		self.url = None
		self.box_color = '#FFFFFF'
		self.agents = cfg.AGENTS
		self.params = None
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		'''
		
		Returns:
		-------
		A List[ str ] of class members.
		
		'''
		return [ 'url',
		         'object',
		         'right_ascension',
		         'declination',
		         'date',
		         'time',
		         'Show_box',
		         'show_grid',
		         'show_lines',
		         'image_source',
		         'show_boundaries',
		         'params',
		         'box_color' ]

	def fetch_by_coordinates( self, ra: float, dec: float ) -> Dict[ str, Any ] | None:
		'''
			
			Purpose:
			-------
			Generates a skymap from given coordinats
			
			Returns:
			--------
			ra - a float representing right ascension
			dec - a float representing declination
			
		'''
		try:
			throw_if( 'ra', ra )
			throw_if( 'dec', dec )
			self.right_ascension = ra
			self.declination = dec
			self.url = f'http://www.sky-map.org/?'
			self.params = \
			{
				'ra': f'{ self.right_ascension }',
				'dec': f'{ self.declination }',
				'Show_box': f'{ self.show_box }',
				'box_color': f'{ self.box_color }',
				'show_constellation_lines': f'{ self.show_lines }',
				'show_constellation_boundaries': f'{ self.show_boundaries }'
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'StarMap'
			exception.method = 'fetch_by_coordinates( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def ssds_by_coordinates( self, ra: float, dec: float ) -> Dict[ str, Any ] | None:
		'''

			Purpose:
			-------
			Generates a skymap using the Sloan Digital Sky Survey (SDSS) given coordinates

			Returns:
			--------
			ra - a float representing right ascension
			dec - a float representing declination

		'''
		try:
			throw_if( 'ra', ra )
			throw_if( 'dec', dec )
			self.right_ascension = ra
			self.declination = dec
			self.url = f'http://www.sky-map.org/?'
			self.params = \
			{
					'ra': f'{self.right_ascension}',
					'dec': f'{self.declination}',
					'Show_box': f'{self.show_box}',
					'box_color': f'{self.box_color}',
					'show_constellation_lines': f'{self.show_lines}',
					'show_constellation_boundaries': f'{self.show_boundaries}',
					'image_source': 'SDSS'
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'StarMap'
			exception.method = 'sdss_by_coordinates( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
			
	def fetch_by_object( self, name: str ) -> Dict[ str, Any ] | None:
		'''
			
			Purpose:
			-------
			Generates a skymap from given an dso name
			
			Returns:
			--------
			name - str
			

		'''
		try:
			throw_if( 'name', name )
			self.object = name
			self.url = f'http://www.sky-map.org/?'
			self.params = \
			{
					'object': self.object,
					'Show_box': f'{self.show_box}',
					'box_color': f'{self.box_color}',
					'show_constellation_lines': f'{self.show_lines}',
					'show_constellation_boundaries': f'{self.show_boundaries}'
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'StarMap'
			exception.method = 'fetch_by_coordinates( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def ssds_by_object( self, name: str ) -> Dict[ str, Any ] | None:
		'''

			Purpose:
			-------
			Generates a skymap using the Sloan Digital Sky Survey (SDSS) given a dso name

			Returns:
			--------
			name: str

		'''
		try:
			throw_if( 'name', name )
			self.object = name
			self.url = f'http://www.sky-map.org/?'
			self.params = \
			{
					'object': self.object,
					'Show_box': f'{self.show_box}',
					'box_color': f'{self.box_color}',
					'show_constellation_lines': f'{self.show_lines}',
					'show_constellation_boundaries': f'{self.show_boundaries}',
					'image_source': 'SSDS'
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'StarMap'
			exception.method = 'fetch_by_coordinates( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )


class GovData( Fetcher ):
	'''
		
		Purpose:
		--------
		Provides  service that can be used to query the GovInfo search engine and return results
		that are the equivalent to what is returned by the main user interface.
		
		You can use field operators, such as congress, publishdate, branch, and others to construct
		complex queries that will return only matching documents.
		
		bill_type - [ hr, s, hjres, sjres, hconres, sconres, hres, sres ]

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
	
	'''
	api_key: Optional[ str ]
	congress_number: Optional[ int ]
	bill_number: Optional[ int ]
	bill_type: Optional[ str ]
	law_type: Optional[ str ]
	law_number: Optional[ str ]
	part_number: Optional[ int ]
	title_number: Optional[ int ]
	date: Optional[ dt.date ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	query: Optional[ str ]
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = cfg.GOVINFO_API_KEY
		self.date = None
		self.congress_number = None
		self.bill_number = None
		self.bill_type = None
		self.law_type = None
		self.law_number = None
		self.part_number = None
		self.url = None
		self.query = None
		self.params = None
		
	def search_criteria( self, criteria: str ) -> Dict[ str, Any ] | None:
		'''
		
			Returns:
			-------
			Starmap
			
		'''
		try:
			throw_if( 'criteria', criteria )
			self.query = criteria
			self.url = f'https://api.govinfo.gov/search?'
			self.params = \
			{
				'api_key': self.api_key,
				'q': self.query,
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'GovInfo'
			exception.method = 'fetch_by_location( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_regulations( self, title: int, part: int ) -> Dict[ str, Any ] | None:
		'''

			Returns:
			-------
			Starmap

		'''
		try:
			throw_if( 'title', title )
			throw_if( 'part', part )
			self.title_number = title
			self.part_number = part
			self.url = f'https://www.govinfo.gov/link/cfr/?'
			self.params = \
			{
					'api_key': self.api_key,
					'titlenum': self.title_number,
					'partnum': self.part_number
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'GovInfo'
			exception.method = 'fetch_by_location( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )

			
	def fetch_bills( self, congress: int, billtype: str, billnum: int ) -> Dict[ str, Any ] | None:
		'''

			Returns:
			-------
			Starmap

		'''
		try:
			throw_if( 'congress', congress )
			throw_if( 'type', type )
			throw_if( 'part', part )
			self.congress_number = congress
			self.bill_type = billtype
			self.bill_number = billnum
			self.url = f'https://www.govinfo.gov/link/bills/?'
			self.params = \
			{
					'api_key': self.api_key,
					'congress': self.congress_number,
					'billtype': self.bill_type,
					'billnum': self.bill_number
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'GovInfo'
			exception.method = 'fetch_by_location( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
			
	def fetch_statutes( self, congress: int, lawtype: str, lawnum: int ) -> Dict[ str, Any ] | None:
		'''

			Returns:
			-------
			Starmap

		'''
		try:
			throw_if( 'congress', congress )
			throw_if( 'lawtype', lawtype )
			throw_if( 'lawnum', lawnum )
			self.congress_number = congress
			self.law_type = lawtype
			self.law_number = lawnum
			self.url = f'https://www.govinfo.gov/link/statute/?'
			self.params = \
			{
					'congress': self.congress_number,
					'lawtype': self.law_type,
					'lawnum': self.law_number
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'GovInfo'
			exception.method = 'fetch_statutes( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
			
	def fetch_records( self, congress: str, billtype: str,  billnum: int ) -> Dict[ str, Any ] | None:
		'''

			Returns:
			-------
			Starmap

		'''
		try:
			throw_if( 'congress', congress )
			self.congress_number = congress
			self.bill_type = lawtype
			self.bill_number = lawnum
			self.url = f'https://www.govinfo.gov/link/crec/cas/?'
			self.params = \
			{
					'congress': self.congress_number,
					'billtype': self.bill_type,
					'billnum': self.bill_number
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'GovInfo'
			exception.method = 'fetch_public_laws( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
			
	def fetch_laws( self, congress: str, lawtype: str,  lawnum: int ) -> Dict[ str, Any ] | None:
		'''

			Returns:
			-------
			Starmap

		'''
		try:
			throw_if( 'congress', congress )
			self.congress_number = congress
			self.law_type = lawtype
			self.law_number = lawnum
			self.url = f'https://www.govinfo.gov/link/plaw/?'
			self.params = \
			{
					'congress': self.congress_number,
					'lawtype': self.law_type,
					'lawnum': self.law_number
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'GovInfo'
			exception.method = 'fetch_public_laws( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
			
class StarChart( Fetcher ):
	'''

		Purpose:
		--------
		Provides functionality via the Astronomy API for creating a Star Chart
		given a location and date
		
		style options [ default, inverted, navy, red ]

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]

	'''
	app_id: Optional[ str ]
	app_token: Optional[ str ]
	userpass: Optional[ str ]
	authString: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	date: Optional[ dt.date ]
	url: Optional[ str ]
	style: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ):
		super( ).__init__( )
		self.app_id = r'565d535c-be49-42ab-b960-73b8aaafb0e5'
		self.app_token = cfg.SKYMAP_TOKEN
		self.userpass = f'{self.app_id}:{self.app_token}'
		self.authString = None
		self.latitude = None
		self.longitude = None
		self.date = None
		self.url = None
		self.params = None
		self.headers = { }
	
	def fetch_by_location( self, lat: float, lng: float, date: dt.date, style: str='red' ) -> str | None:
		'''

			Returns:
			-------
			Starmap

		'''
		try:
			throw_if( 'lat', lat )
			throw_if( 'lng', lng )
			throw_if( 'date', date )
			self.latitude = lat
			self.longitude = lng
			self.date = date
			self.style = style
			self.authString = base64.b64encode( self.userpass.encode( ) ).decode( )
			self.url = f'https://api.astronomyapi.com/api/v2/studio/star-chart?'
			self.headers[ 'Authorizeion' ] = 'Basic ' + authString
			self.params = \
			{
                'style': self.style,
				'observer':
				{
						'latitude': self.latitude,
						'longitude': self.longitude,
						'date': f'{ self.date }'
				},
				'view':
				{
					'type': 'area',
					'parameters':
					{
						'position':
						{
							'equatorial':
							{
								'rightAscension':  14.83,
								'declination': 33.3
							}
						}
					}
				}
			}
			
			self.response = requests.post( url=self.url, params=self.params, headers=self.headers )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'StarChart'
			exception.method = 'fetch_by_location( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )

class Congress( Fetcher ):
	'''

		Purpose:
		--------
		Provides  service that can be used to query the GovInfo search engine and return results
		that are the equivalent to what is returned by the main user interface.

		You can use field operators, such as congress, publishdate, branch, and others to construct
		complex queries that will return only matching documents.

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
	'''
	api_key: Optional[ str ]
	congress_number: Optional[ int ]
	bill_number: Optional[ int ]
	bill_type: Optional[ str ]
	law_type: Optional[ str ]
	law_number: Optional[ str ]
	part_number: Optional[ int ]
	title_number: Optional[ int ]
	date: Optional[ dt.date ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	query: Optional[ str ]
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = cfg.CONGRESS_API_KEY
		self.date = None
		self.congress_number = None
		self.bill_number = None
		self.bill_type = None
		self.law_type = None
		self.law_number = None
		self.part_number = None
		self.url = None
		self.query = None
		self.params = None
	
	def fetch_bills( self, congress: num ) -> Dict[ str, Any ] | None:
		'''

			Returns:
			-------
			All congressional bills given a Congress (eg, 117)

		'''
		try:
			throw_if( 'congress', congress)
			self.congress_number = congress
			self.url = f'https://api.congress.gov/v3/bill/?'
			self.params = \
			{
				'api_key': self.api_key,
				'congress': self.congress_number,
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Congress'
			exception.method = 'fetch_by_location( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_laws( self, congress: num ) -> Dict[ str, Any ] | None:
		'''

			Returns:
			-------
			All laws passed given a Congress (eg, 117)

		'''
		try:
			throw_if( 'congress', congress )
			self.congress_number = congress
			self.url = f'https://api.congress.gov/v3/law/?'
			self.params = \
			{
				'api_key': self.api_key,
				'congress': self.congress_number,
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Congress'
			exception.method = 'fetch_by_location( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_reports( self, congress: int ) -> Dict[ str, Any ] | None:
		'''

			Returns:
			-------
			All congressional reports given a Congress (eg, 117)

		'''
		try:
			throw_if( 'congress', congress )
			self.congress_number = congress
			self.url = f'https://api.congress.gov/v3/law/?'
			self.params = \
			{
				'api_key': self.api_key,
				'congress': self.congress_number,
			}
			
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Congress'
			exception.method = 'fetch_by_location( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )

class InternetArchive( Fetcher ):
	'''

		Purpose:
		---------
		Class providing the functionality of the Internet Archive Search api.

		Attribues:
		-----------
		timeout - int
		headers - Dict[ str, Any ]
		response - requests.Response
		url - str
		result - core.Result
		query - string

		Methods:
		-----------
		fetch( ) -> Dict[ str, Any ]
	'''
	keywords: Optional[ str ]
	url: Optional[ str ]
	re_tag: Optional[ Pattern ]
	re_ws: Optional[ Pattern ]
	response: Optional[ Response ]
	api_key: Optional[ str ]
	fields: Optional[ List[ str ] ]
	count: Optional[ int ]
	params: Optional[ Dict[ str, str ] ]
	
	def __init__( self ) -> None:
		'''
			Purpose:
			-----------
			Initialize InternetArchive with optional headers and sane defaults.

			Parameters:
			-----------
			headers (Optional[Dict[str, str]]): Optional headers for requests.

			Returns:
			-----------
			None
		'''
		super( ).__init__( )
		self.re_tag = re.compile( r'<[^>]+>' )
		self.re_ws = re.compile( r'\s+' )
		self.url = None
		self.headers = { }
		self.timeout = None
		self.keywords = None
		self.params = None
		self.fields = ['identifier', 'name', 'subject', 'title', 'source', 'type', 'publicdate']
		self.response = None
		self.agents = cfg.AGENTS
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-----------
			Internet Archive list of members.

			Parameters:
			-----------
			None

			Returns:
			-----------
			list[str]: Ordered attribute/method names.

		'''
		return [ 'keywords',
		         'url',
		         'timeout',
		         'headers',
		         'fetch',
		         'api_key',
		         'response',
		         'cse_id',
		         'params',
		         'agents,',
		         'fetch' ]
	
	def fetch( self, keywords: str, time: int = 10 ) -> Response | None:
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
			throw_if( 'keywords', keywords )
			self.url = r'https://archive.org/advancedsearch.php?'
			self.keywords = keywords
			self.timeout = time
			self.params = \
			{
					'q': self.keywords,
					'fields': self.fields,
					'num': self.timeout
			}
			_response = requests.get( url=self.url, params=self.params )
			return _response
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'InternetArchive'
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
			html = re.sub( r'<script[\s\S]*?</script>', ' ', html, flags=re.IGNORECASE )
			html = re.sub( r'<style[\s\S]*?</style>', ' ', html, flags=re.IGNORECASE )
			html = re.sub( r'</?(p|div|br|li|h[1-6])[^>]*>', '\n', html, flags=re.IGNORECASE )
			text = re.sub( self.re_tag, ' ', html )
			text = re.sub( self.re_ws, ' ', text ).strip( )
			return text
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleSearch'
			exception.method = 'html2text( )'
			dialog = ErrorDialog( exception )
			dialog.show( )
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> dict:
		"""

			Purpose:
			________
			Construct and return a fully dynamic OpenAI Tool API schema definition.
			Supports arbitrary parameters, types, nested objects, and required fields.

			Parameters:
			___________
			function (str):
			The function name exposed to the LLM.

			tool (str):
			The underlying system or service the function wraps
			(e.g., “Google Maps”, “SQLite”, “Weather API”).

			description (str):
			Precise explanation of what the function does.

			parameters (dict):
			A dictionary defining parameter names and JSON schema descriptors.
			Each value must itself be a valid JSON-schema fragment.

				Example:
					{
						"origin": {
							"type": "string",
							"description": "Starting location."
						},
						"destination": {
							"type": "string",
							"description": "Ending location."
						},
						"mode": {
							"type": "string",
							"enum": ["driving", "walking", "bicycling", "transit"],
							"description": "Travel mode."
						}
					}

			required (list[str] | None):
			List of required parameter names.
			If None, required = list(parameters.keys()).

			Returns:
			________
			dict:
			A JSON-compatible dictionary defining the tool schema.

		"""
		throw_if( 'function', function )
		throw_if( 'tool', tool )
		throw_if( 'description', description )
		throw_if( 'parameters', parameters )
		if not isinstance( parameters, dict ):
			raise ValueError( 'parameters must be a dict of param_name → schema definitions.' )
		
		func_name = function.strip( )
		tool_name = tool.strip( )
		desc = description.strip( )
		if required is None:
			required = list( parameters.keys( ) )
		return \
			{
					'name': func_name,
					'description': f'{desc} This function uses the {tool_name} service.',
					'parameters':
						{
								'type': 'object',
								'properties': parameters,
								'required': required
						}
			}