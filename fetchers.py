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

import base64
import datetime as dt
import http.client
import re
import urllib.parse
from typing import Any, Dict, Optional, Pattern, List

import matplotlib.pyplot as plt
import openmeteo_requests
import requests
import requests_cache
from anthropic import Anthropic
from astroquery.simbad import Simbad
from google.genai.types import HttpOptions
from grokipedia_api import GrokipediaClient
from groq import Groq
from langchain_core.documents import Document
from langchain_community.retrievers import ArxivRetriever, WikipediaRetriever
from langchain_googledrive.retrievers import GoogleDriveRetriever
from openai import OpenAI
from owslib.wms import WebMapService
from requests import Response
from retry_requests import retry
from sscws.sscws import SscWs

import config as cfg
from boogr import Error, ErrorDialog
from core import Result

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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
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
	soup: Optional[ BeautifulSoup ]
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
		         'html_to_text',
		         'scrape_images',
		         'scrape_hyperlinks',
		         'scrape_images',
		         'scrape_hyperlinks',
		         'scrape_blockquotes',
		         'scrape_sections',
		         'scrape_divisions',
		         'sracpe_headings',
		         'scrape_tables',
		         'scrape_lists',
		         'scrape_paragraphse', ]
	
	def fetch( self, url: str, time: int=10 ) -> Result | None:
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
			self.response = requests.get( url=self.url, headers=self.headers,
				timeout=self.timeout )
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
			html = re.sub( r'<script[\s\S]*?</script>', ' ', html, flags=re.IGNORECASE )
			html = re.sub( r'<style[\s\S]*?</style>', ' ', html, flags=re.IGNORECASE )
			html = re.sub( r'</?(p|div|br|li|h[1-6])[^>]*>', '\n', html, flags=re.IGNORECASE )
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
	
	def scrape_paragraphs( self, uri: str ) -> List[ str ] | None:
		"""
		
	
			Purpose:
			--------
			Extract readable text from all <p> elements on a page.
	
			Parameters:
			-----------
			uri (str):
			Fully-qualified URI of the target HTML document.
	
			Returns:
			--------
			List[str]:
			Cleaned paragraph text entries.
			
		"""
		try:
			throw_if( 'uri', uri )
			self.response = requests.get( uri, timeout=10 )
			self.response.raise_for_status( )
			self.soup = BeautifulSoup( self.response.text, 'html.parser' )
			blocks = [ p.get_text( ' ', strip=True ) for p in self.soup.find_all( 'p' ) ]
			return [ b for b in blocks if b ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetchers'
			exception.method = 'scrape_paragraphs( self, uri: str ) -> List[ str ]'
			dialog = ErrorDialog( exception )
			dialog.show( )

	def scrape_lists( self, uri: str ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Extract text from <li> elements found in ordered and unordered lists.
	
			Parameters:
			-----------
			uri (str):
			Fully-qualified URI of the HTML page.
	
			Returns:
			--------
			List[str]:
			Clean list item text segments.
			
		"""
		try:
			throw_if( 'uri', uri )
			self.response = requests.get( uri, timeout=10 )
			self.response.raise_for_status( )
			self.soup = BeautifulSoup( self.response.text, 'html.parser' )
			items = [ li.get_text( ' ', strip=True ) for li in self.soup.find_all( 'li' ) ]
			return [ i for i in items if i ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetchers'
			exception.method = 'scrape_lists( self, uri: str ) -> List[ str ]'
			dialog = ErrorDialog( exception )
			dialog.show( )
	
	def scrape_tables( self, uri: str ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Extract flattened table cell contents from all <table> structures on the
			page.
		
			Parameters:
			-----------
			uri (str):
			URI of the HTML document.
		
			Returns:
			--------
			List[str]:
			Table cell values (one entry per <td> or <th>).
			
		"""
		try:
			throw_if( 'uri', uri )
			self.response = requests.get( uri, timeout=10 )
			self.response.raise_for_status( )
			self.soup = BeautifulSoup( self.response.text, 'html.parser' )
			_results: List[ str ] = [ ]
			for table in self.soup.find_all( 'table' ):
				for row in table.find_all( 'tr' ):
					for cell in row.find_all( [ 'td',  'th' ] ):
						text = cell.get_text( ' ', strip=True )
						if text:
							_results.append( text )
			
			return _results
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetchers'
			exception.method = 'scrape_tables( self, uri: str ) -> List[ str ]'
			dialog = ErrorDialog( exception )
			dialog.show( )
	
	def scrape_articles( self, uri: str ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Extract consolidated text from <article> elements. Each article is
			returned as a single cleaned string.
		
			Parameters:
			-----------
			uri (str):
			URI of the HTML page.
		
			Returns:
			--------
			List[str]:
			Article-level text blocks.
			
		"""
		try:
			throw_if( 'uri', uri )
			self.response = requests.get( uri, timeout=10 )
			self.response.raise_for_status( )
			self.soup = BeautifulSoup( self.response.text, 'html.parser' )
			blocks = [ art.get_text( " ", strip=True ) for art in soup.find_all( 'article' ) ]
			return [ b for b in blocks if b ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetchers'
			exception.method = 'scrape_articles( self, uri: str ) -> List[ str ]'
			dialog = ErrorDialog( exception )
			dialog.show( )
	
	def scrape_headings( self, uri: str ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Extract text from heading tags (h1–h6).
		
			Parameters:
			-----------
			uri (str):
			Fully-qualified document URI.
		
			Returns:
			--------
			List[str]:
			Clean heading strings.
			
		"""
		try:
			throw_if( "uri", uri )
			self.response = requests.get( uri, timeout=10 )
			self.response.raise_for_status( )
			self.soup = BeautifulSoup( self.response.text, "html.parser" )
			heading_tags = [ 'h1',
			                 'h2',
			                 'h3',
			                 'h4',
			                 'h5',
			                 'h6' ]
			blocks = [ h.get_text( ' ', strip=True ) for h in self.soup.find_all( heading_tags ) ]
			return [ b for b in blocks if b ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetchers'
			exception.method = 'scrape_headings( self, uri: str ) -> List[ str ]'
			dialog = ErrorDialog( exception )
			dialog.show( )
	
	def scrape_divisions( self, uri: str ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Extract cleaned text from <div> elements on the page.
		
			Parameters:
			-----------
			uri (str):
			URI of the HTML document.
		
			Returns:
			--------
			List[str]:
			Clean division text blocks.
			
		"""
		try:
			throw_if( 'uri', uri )
			self.response = requests.get( uri, timeout=10 )
			self.response.raise_for_status( )
			self.soup = BeautifulSoup( self.response.text, 'html.parser' )
			blocks = [ div.get_text( " ", strip=True ) for div in self.soup.find_all( 'div' ) ]
			return [ b for b in blocks if b ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetchers'
			exception.method = 'scrape_divisions( self, uri: str ) -> List[ str ]'
			dialog = ErrorDialog( exception )
			dialog.show( )
	
	def scrape_sections( self, uri: str ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Extract readable text from <section> elements.
		
			Parameters:
			-----------
			uri (str):
			Fully-qualified document URI.
		
			Returns:
			--------
			List[str]:
			Clean section text blocks.
			
		"""
		try:
			throw_if( 'uri', uri )
			self.response = requests.get( uri, timeout=10 )
			rself.esponse.raise_for_status( )
			self.soup = BeautifulSoup( self.response.text, 'html.parser' )
			blocks = [ sec.get_text( " ", strip=True ) for sec in self.soup.find_all( 'section' ) ]
			return [ b for b in blocks if b ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetchers'
			exception.method = 'scrape_sections( self, uri: str ) -> List[ str ]'
			dialog = ErrorDialog( exception )
			dialog.show( )
	
	def scrape_blockquotes( self, uri: str ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Extract text from <blockquote> elements.
	
			Parameters:
			-----------
			uri (str):
			Document URI.
	
			Returns:
			--------
			List[str]:
			Cleaned blockquote text entries.
			
			
		"""
		try:
			throw_if( 'uri', uri )
			self.response = requests.get( uri, timeout=10 )
			self.response.raise_for_status( )
			self.soup = BeautifulSoup( self.response.text, 'html.parser' )
			blocks = [ bq.get_text( ' ', strip=True ) for bq in self.soup.find_all( 'blockquote' ) ]
			return [ b for b in blocks if b ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetchers'
			exception.method = 'scrape_blockquotes( self, uri: str ) -> List[ str ]'
			dialog = ErrorDialog( exception )
			dialog.show( )
	
	def scrape_hyperlinks( self, uri: str ) -> List[ str ] | None:
		"""
		
			Purpose:
			--------
			Extract hyperlink values (href attributes) from <a> tags.
		
			Parameters:
			-----------
			uri (str):
			URI of the web page.
		
			Returns:
			--------
			List[str]:
			List of hyperlink paths.
		
		"""
		try:
			throw_if( 'uri', uri )
			self.response = requests.get( uri, timeout=10 )
			self.response.raise_for_status( )
			self.soup = BeautifulSoup( self.response.text, 'html.parser' )
			links = [ a.get( 'href' ) for a in self.soup.find_all( 'a' ) if a.get( 'href' ) ]
			return links
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetchers'
			exception.method = 'scrape_hyperlinks( self, uri: str ) -> List[ str ]'
			dialog = ErrorDialog( exception )
			dialog.show( )
	
	def scrape_images( self, uri: str ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Extract image references (<img src="...">) from the target document.
		
			Parameters:
			-----------
			uri (str):
			Fully-qualified HTML page URI.
		
			Returns:
			--------
			List[str]:
			Image source values extracted from <img> elements.
		
		"""
		try:
			throw_if( 'uri', uri )
			self.response = requests.get( uri, timeout=10 )
			self.response.raise_for_status( )
			self.soup = BeautifulSoup( self.response.text, 'html.parser' )
			images = [ img.get( 'src' ) for img in self.soup.find_all( 'img' ) if img.get( 'src' ) ]
			return images
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'WebFetchers'
			exception.method = 'scrape_images( self, uri: str ) -> List[ str ] '
			dialog = ErrorDialog( exception )
			dialog.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = ''
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )
			

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

class ArXiv( Fetcher ):
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
			question: query

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
			exception.module = 'fetchers'
			exception.cause = 'ArXiv'
			exception.method = 'fetch( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'ArXiv'
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )


class GoogleDrive( Fetcher ):
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
	
	def fetch( self, question: str, folder_id: str='root',
			results: int=10, template: str='gdrive-query' ) -> List[ Document ] | None:
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
			st.error( str( e ) )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GoogleDrive'
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )
	
class Wikipedia( Fetcher ):
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
		except Exception as exc:
			st.error( str( exc ) )
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Wikipedia'
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

class TheNews( Fetcher ):
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
			exception.cause = 'TheNews'
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
			exception.cause = 'TheNews'
			exception.method = 'html2text( )'
			dialog = ErrorDialog( exception )
			dialog.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'TheNews'
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )
			
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
	params: Optional[ Dict[ str, Any ] ]
	
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
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = ''
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
			exception.cause = 'GoogleMaps'
			exception.method = 'request_directions( self, origin: str, destination: str ) -> dict'
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GoogleMaps'
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
			exception.cause = 'GoogleWeather'
			exception.method = 'validate_address( self, address: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GoogleWeather'
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
			exception.cause = 'GoogleWeather'
			exception.method = 'fetch_sidereal( self, date: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NavalObservatory'
			exception.method = ('create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]')
			error = ErrorDialog( exception )
			error.show( )
	
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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
			exception.cause = 'SatelliteCenter'
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
			exception.module = 'fetchers'
			exception.cause = 'SatelliteCenter'
			exception.method = 'fetch_julian( self, address: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'SatelliteCenter'
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
			exception.cause = 'EarthObservatory'
			exception.method = 'fetch_julian( self, address: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = ''
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

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
			exception.module = 'fetchers'
			exception.cause = 'GlobalImagery'
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
			exception.module = 'fetchers'
			exception.cause = 'GlobalImagery'
			exception.method = 'fetch_sidereal( self, date: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = ''
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
			exception.cause = 'NearByObjects'
			exception.method = 'fetch_fireballs( self,  start: dt.date, end: dt.date ) -> Dict[ str, Any ] '
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = ''
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
			exception.cause = 'OpenScience'
			exception.method = 'fetch_studies( self, keywords: str ) -> Dict[ str, Any ]'
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = ''
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
			exception.cause = 'DONKI'
			exception.method = 'fetch_sidereal( self, date: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = ''
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
			exception.cause = 'AstroCatalog'
			exception.method = 'fetch_sidereal( self, date: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = ''
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
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
			exception.module = 'fetchers'
			exception.cause = 'OpenAstronomyCatalog'
			exception.method = 'catalog_search( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = ''
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

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
			self.url = f'http://www.sky-map.org/'
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
			exception.module = 'fetchers'
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
			self.url = f'http://www.sky-map.org/'
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
			exception.module = 'fetchers'
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
			self.url = f'http://www.sky-map.org/'
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
			exception.module = 'fetchers'
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
			self.url = f'http://www.sky-map.org/'
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
			exception.module = 'fetchers'
			exception.cause = 'StarMap'
			exception.method = 'fetch_by_coordinates( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = ''
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
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
			self.url = f'https://api.govinfo.gov/search'
			self.params = \
			{
				'query': self.query +'&api_key=' + self.api_key,
			}
			
			self.response = requests.post( url=self.url, data=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
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
			self.url = r'https://www.govinfo.gov/link/cfr/'
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
			exception.module = 'fetchers'
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
			self.url = f'https://www.govinfo.gov/link/bills/'
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
			exception.module = 'fetchers'
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
			self.url = f'https://www.govinfo.gov/link/statute/'
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
			exception.module = 'fetchers'
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
			self.url = f'https://www.govinfo.gov/link/crec/cas/'
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
			exception.module = 'fetchers'
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
			self.url = f'https://www.govinfo.gov/link/plaw/'
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
			exception.module = 'fetchers'
			exception.cause = 'GovInfo'
			exception.method = 'fetch_public_laws( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = ''
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
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
			exception.module = 'fetchers'
			exception.cause = 'StarChart'
			exception.method = 'fetch_by_location( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = ''
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
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
			self.url = f'https://api.congress.gov/v3/bill/'
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
			exception.module = 'fetchers'
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
			self.url = f'https://api.congress.gov/v3/law/'
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
			exception.module = 'fetchers'
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
			self.url = f'https://api.congress.gov/v3/law/'
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
			exception.module = 'fetchers'
			exception.cause = 'Congress'
			exception.method = 'fetch_by_location( self, name: str ) -> float'
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = ''
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
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
		self.fields = [ 'identifier', 'name', 'subject', 'title', 'source', 'type', 'publicdate' ]
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
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = ''
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

class OpenWeather( Fetcher ):
	'''

		Purpose:
		--------
		Provides access to weather data via the OpenMeteo API.

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
	client: Optional[ openmeteo_requests.Client ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	timezone: Optional[ str ]
	daily_metrics: Optional[ List[ str ] ]
	hourly_metrics: Optional[ List[ str ] ]
	current_metrics: Optional[ List[ str ] ]
	windspeed_unit: Optional[ str ]
	temperature_unit: Optional[ str ]
	precipitation_unit: Optional[ str ]
	current_forecast: Optional[ DataFrame ]
	hourly_forecast: Optional[ DataFrame ]
	daily_forecast: Optional[ DataFrame ]
	cache_session: Optional[ requests_cache.CachedSession ]
	retry_session: Optional[ retry ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.client = None
		self.url = None
		self.longitude = None
		self.latitude = None
		self.timezone = None
		self.windspeed_unit = 'kn'
		self.temperature_unit = 'fahrenheit'
		self.precipitation_unit = 'inches'
		self.cache_session = None
		self.retry_session = None
		self.agents = cfg.AGENTS
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		self.current_metrics = [ 'temperature_2m',
		                         'relative_humidity_2m',
		                         'apparent_temperature',
		                         'is_day',
		                         'snowfall',
		                         'showers',
		                         'rain',
		                         'precipitation',
		                         'weather_code',
		                         'cloud_cover',
		                         'pressure_msl',
		                         'surface_pressure',
		                         'wind_gusts_10m',
		                         'wind_direction_10m',
		                         'wind_speed_10m' ]
		self.hourly_metrics = [ 'temperature_2m',
		                        'uv_index',
		                        'uv_index_clear_sky',
		                        'is_day',
		                        'sunshine_duration',
		                        'relative_humidity_2m',
		                        'dew_point_2m',
		                        'apparent_temperature',
		                        'precipitation_probability',
		                        'precipitation',
		                        'rain',
		                        'showers',
		                        'snowfall',
		                        'snow_depth',
		                        'cloud_cover_high',
		                        'visibility',
		                        'cloud_cover_mid',
		                        'cloud_cover_low',
		                        'cloud_cover',
		                        'surface_pressure',
		                        'pressure_msl',
		                        'weather_code',
		                        'evapotranspiration',
		                        'et0_fao_evapotranspiration',
		                        'vapour_pressure_deficit',
		                        'wind_speed_10m',
		                        'wind_direction_10m',
		                        'wind_gusts_10m' ]
		self.daily_metrics = [ 'weather_code',
		                       'temperature_2m_max',
		                       'temperature_2m_min',
		                       'apparent_temperature_max',
		                       'apparent_temperature_min',
		                       'uv_index_clear_sky_max',
		                       'uv_index_max',
		                       'sunshine_duration',
		                       'daylight_duration',
		                       'sunset',
		                       'sunrise',
		                       'rain_sum',
		                       'showers_sum',
		                       'snowfall_sum',
		                       'precipitation_sum',
		                       'precipitation_hours',
		                       'precipitation_probability_max',
		                       'et0_fao_evapotranspiration',
		                       'shortwave_radiation_sum',
		                       'wind_direction_10m_dominant',
		                       'wind_gusts_10m_max',
		                       'wind_speed_10m_max',
		                       'temperature_2m_mean',
		                       'apparent_temperature_mean',
		                       'dew_point_2m_mean',
		                       'precipitation_probability_mean',
		                       'relative_humidity_2m_mean',
		                       'pressure_msl_mean',
		                       'visibility_mean',
		                       'surface_pressure_mean',
		                       'wind_gusts_10m_mean',
		                       'wind_speed_10m_mean' ]
	
	def fetch_current( self, lat: float, long: float, zone: str ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Retrieves current weather data given a location in coordinates and a timezone.

			Parameters:
			----------
			lat - float representing a location's latitude
			long - float representing a location's longitude
			zone - str representing a location's timezone

		"""
		try:
			throw_if( 'lat', lat )
			throw_if( 'long', long )
			throw_if( 'zone', zone )
			self.latitude = lat
			self.longitude = long
			self.timezone = zone
			self.cache_session = requests_cache.CachedSession( '.cache', expire_after=-1 )
			self.retry_session = retry( self.cache_session, retries=5, backoff_factor=0.2 )
			self.client = openmeteo_requests.Client( session=self.retry_session )
			self.url = r'https://api.open-meteo.com/v1/forecast'
			self.params = \
			{
				'longitude': self.longitude,
				'latitude': self.latitude,
				'daily': self.daily_metrics,
				'hourly': self.hourly_metrics,
				'current': self.current_metrics,
				'timezone': self.timezone,
				'windspeed_unit': self.windspeed_unit,
				'temperature_unit': self.temperature_unit,
				'precipitation_unit': self.precipitation_unit,
			}
			
			self.response = self.client.weather_api( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenWeather'
			exception.method = 'fetch_current( self, lat: float, long: float, zone: str ) -> Dict[ str, Any ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_hourly( self, lat: float, long: float, zone: str ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Retrieves hourly forecast data given a location in coordinates and a timezone.

			Parameters:
			----------
			lat - float representing a location's latitude
			long - float representing a location's longitude
			zone - str representing a location's timezone

		"""
		try:
			throw_if( 'lat', lat )
			throw_if( 'long', long )
			throw_if( 'zone', zone )
			self.latitude = lat
			self.longitude = long
			self.timezone = zone
			self.cache_session = requests_cache.CachedSession( '.cache', expire_after=-1 )
			self.retry_session = retry( self.cache_session, retries=5, backoff_factor=0.2 )
			self.client = openmeteo_requests.Client( session=self.retry_session )
			self.url = r'https://api.open-meteo.com/v1/forecast'
			self.params = \
			{
				'longitude': self.longitude,
				'latitude': self.latitude,
				'daily': self.daily_metrics,
				'hourly': self.hourly_metrics,
				'current': self.current_metrics,
				'timezone': self.timezone,
				'windspeed_unit': self.windspeed_unit,
				'temperature_unit': self.temperature_unit,
				'precipitation_unit': self.precipitation_unit,
			}
			
			self.response = self.client.weather_api( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenWeather'
			exception.method = 'fetch_hourly( self, lat: float, long: float, zone: str ) -> Dict[ str, Any ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_daily( self, lat: float, long: float, zone: str ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Retrieves daily forecast data given a location in coordinates and a timezone.

			Parameters:
			----------
			lat - float representing a location's latitude
			long - float representing a location's longitude
			zone - str representing a location's timezone

		"""
		try:
			throw_if( 'lat', lat )
			throw_if( 'long', long )
			throw_if( 'zone', zone )
			self.latitude = lat
			self.longitude = long
			self.timezone = zone
			self.cache_session = requests_cache.CachedSession( '.cache', expire_after=-1 )
			self.retry_session = retry( self.cache_session, retries=5, backoff_factor=0.2 )
			self.client = openmeteo_requests.Client( session=self.retry_session )
			self.url = r'https://api.open-meteo.com/v1/forecast'
			self.params = \
			{
				'longitude': self.longitude,
				'latitude': self.latitude,
				'daily': self.daily_metrics,
				'hourly': self.hourly_metrics,
				'current': self.current_metrics,
				'timezone': self.timezone,
				'windspeed_unit': self.windspeed_unit,
				'temperature_unit': self.temperature_unit,
				'precipitation_unit': self.precipitation_unit,
			}
			
			self.response = self.client.weather_api( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenWeather'
			exception.method = 'fetch_daily( self, lat: float, long: float, zone: str ) -> Dict[ str, Any ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_historical( self, lat: float, long: float, zone: str, start: dt.date, end: dt.date ) -> \
	Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Retrieves historical weather data given a location in coordinates, a timezone, a start
			and end date

			Parameters:
			----------
			lat - float representing a location's latitude
			long - float representing a location's longitude
			zone - str representing a location's timezone
			start - datetime representing the beginning of a historical timeframe
			end - datetime representing the ending of a historical timeframe

		"""
		try:
			throw_if( 'lat', lat )
			throw_if( 'long', long )
			throw_if( 'zone', zone )
			self.latitude = lat
			self.longitude = long
			self.timezone = zone
			self.cache_session = requests_cache.CachedSession( '.cache', expire_after=-1 )
			self.retry_session = retry( self.cache_session, retries=5, backoff_factor=0.2 )
			self.client = openmeteo_requests.Client( session=self.retry_session )
			self.url = 'https://archive-api.open-meteo.com/v1/archive'
			self.params = \
			{
				'longitude': self.longitude,
				'latitude': self.latitude,
				'daily': self.daily_metrics,
				'hourly': self.hourly_metrics,
				'current': self.current_metrics,
				'timezone': self.timezone,
				'windspeed_unit': self.windspeed_unit,
				'temperature_unit': self.temperature_unit,
				'precipitation_unit': self.precipitation_unit,
			}
			
			self.response = self.client.weather_api( url=self.url, params=self.params )
			self.response.raise_for_status( )
			_results = self.response.json( )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenWeather'
			exception.method = 'fetch_daily( self, lat: float, long: float, zone: str ) -> Dict[ str, Any ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenWeather'
			exception.method = ('create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]')
			error = ErrorDialog( exception )
			error.show( )

class Groq( Fetcher ):
	'''

		Purpose:
		---------
		Class providing to the Groq API

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
	client: Optional[ Groq ]
	model: Optional[ str ]
	keywords: Optional[ str ]
	url: Optional[ str ]
	file_path: Optional[ str ]
	response: Optional[ Response ]
	api_key: Optional[ str ]
	query: Optional[  str  ]
	params: Optional[ Dict[ str, str ] ]
	temperature: Optional[ float ]
	max_tokens: Optional[ int ]
	top_p: Optioanl[ float ]
	reasonging_effort: Optional[ float ]
	stream: Optional[ bool ]
	store: Optional[ bool ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	
	def __init__( self ) -> None:
		'''
		
			Purpose:
			-----------
			Initialize Groq API.

			Parameters:
			-----------
			headers (Optional[Dict[str, str]]): Optional headers for requests.

			Returns:
			-----------
			None
			
		'''
		super( ).__init__( )
		self.api_key = cfg.GROQ_API_KE
		self.model = 'openai/gpt-oss-120b'
		self.url = r'https://api.groq.com/openai/v1?'
		self.messages = None
		self.temperature = 0.8
		self.top_p =  0.9
		self.max_tokens = 8192
		self.reasonging_effort = 'medium'
		self.headers = { }
		self.timeout = None
		self.file_path = None
		self.content = None
		self.params = None
		self.response = None
		self.agents = cfg.AGENTS
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-----------
			Groq list of members.

			Parameters:
			-----------
			None

			Returns:
			-----------
			list[str]: Ordered attribute/method names.

		'''
		return [ 'content',
		         'url',
		         'client',
		         'timeout',
		         'headers',
		         'fetch',
		         'file_path',
		         'messages',
		         'content',
		         'temperature',
		         'top_p',
		         'reasoning_effort',
		         'max_tokens',
		         'api_key',
		         'response',
		         'params',
		         'agents' ]
	
	def fetch( self, query: str, time: int=10 ) -> str | None:
		'''

			Purpose:
			-------
			Sends an API request to Groq given a query as input

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
			self.client = Groq( )
			self.messages = [
			{
				'role': 'user',
				'content': self.query
			} ]
			completion = self.client.chat.completions.create( model=self.model,
				messages=self.messages,
				temperature=1,
				max_completion_tokens=8192,
				top_p=1,
				reasoning_effort='medium',
				stream=True,
				stop=None
			)
			_results = completion.choices[0].message
			return _results
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Groq'
			exception.method = 'fetch( self, query: str, time: int=10 ) -> str'
			dialog = ErrorDialog( exception )
			dialog.show( )
	
	def analyze_image( self, path: str, prompt: str, is_url=False ):
		'''
		
			Purpose:
			--------
			Uses the Groq API to analyze an image given a prompt and path
			
		'''
		throw_if( 'prompt', prompt )
		throw_if( 'path', path )
		self.query = prompt
	
		self.client = Groq( api_key=self.api_key )
		if is_url:
			image_content = \
			{
					'type': 'image_url',
					'image_url':
					{
						'url': path
					}
			}
		else:
			base64_image = encode_image( path )
			image_content = {
					"type": "image_url",
					"image_url": {
							"url": f"data:image/jpeg;base64,{base64_image}" } }
		
		try:
			chat_completion = client.chat.completions.create(
				messages=[
						{
								"role": "user",
								"content": [
										{
												"type": "text",
												"text": prompt },
										image_content,
								],
						}
				],
				model="llava-v1.5-7b-4096-preview",
			)
			return chat_completion.choices[ 0 ].message.content
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Groq'
			exception.method = 'fetch( self, query: str, time: int=10 ) -> str'
			dialog = ErrorDialog( exception )
			dialog.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
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
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Groq'
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

class Gemini( Fetcher ):
	'''

		Purpose:
		---------
		Class providing the Google's Gemini API

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
	client: Optional[ genai.Client ]
	prompt: Optional[ str ]
	file_path: Optional[ str ]
	response: Optional[ Response ]
	mime_type: Optional[ str ]
	api_key: Optional[ str ]
	id: Optional[ str ]
	location: Optional[ str ]
	use_vertex: Optional[ bool ]
	contents: Optional[  str  ]
	model: Optional[ str ]
	params: Optional[ Dict[ str, str ] ]
	temperature: Optional[ float ]
	max_tokens: Optional[ int ]
	top_p: Optioanl[ float ]
	reasonging_effort: Optional[ float ]
	http_options: Optional[ str ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	
	def __init__( self ) -> None:
		'''
		
			Purpose:
			-----------
			Initialize the Gmemini Class
			
		'''
		super( ).__init__( )
		self.api_key = cfg.GEMINI_API_KEY
		self.id = cfg.GOOGLE_PROJECT_ID
		self.location = cfg.GOOGLE_CLOUD_LOCATION
		self.use_vertex = cfg.GOOGLE_GENAI_USE_VERTEXAI
		self.model = 'gemini-2.5-flash'
		self.headers = { }
		self.timeout = None
		self.contents = None
		self.params = None
		self.response = None
		self.agents = cfg.AGENTS
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-----------
			Groq list of members.

			Parameters:
			-----------
			None

			Returns:
			-----------
			list[str]: Ordered attribute/method names.

		'''
		return [ 'query',
		         'url',
		         'client',
		         'id',
		         'location',
		         'use_vertex',
		         'headers',
		         'fetch',
		         'api_key',
		         'response',
		         'cse_id',
		         'params',
		         'agents,',
		         'fetch' ]
	
	def fetch( self, query: str ) -> str | None:
		'''

			Purpose:
			-------
			Sends query/content to the Gemini API

			Parameters:
			-----------
			query (str): Absolute URL to fetch.
			

			Returns:
			---------
			str

		'''
		try:
			throw_if( 'query', query )
			self.contents = query
			self.client = genai.Client( http_options=HttpOptions( api_version='v1' ) )
			_response = self.client.models.generate_content( model=self.model, contents=self.contents, )
			return _response.text
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Gemini'
			exception.method = 'fetch( self, query: str ) -> str '
			dialog = ErrorDialog( exception )
			dialog.show( )
	
	def analyze( self, query: str, path: str ) -> str | None:
		'''

			Purpose:
			-------
			Sends image request to the Gemini API given a query and path

			Parameters:
			-----------
			query (str): content/query passed to the Gemini API
			path (str): path to image file

			Returns:
			---------
			str.

		'''
		try:
			throw_if( 'query', query )
			throw_if( 'path', path )
			self.contents = query
			self.file_path = path
			_client = genai.Client( http_options=HttpOptions( api_version='v1' ) )
			_response = client.models.generate_content( model=self.model, contents=self.contents, )
			return _response.text
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Gemini'
			exception.method = 'analyze( self, query: str, path: str ) -> str'
			dialog = ErrorDialog( exception )
			dialog.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Gemini'
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

class Claude( Fetcher ):
	'''

		Purpose:
		---------
		Class providing to the Groq API

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
	client: Optional[ Anthropic ]
	model: Optional[ str ]
	keywords: Optional[ str ]
	response: Optional[ Response ]
	api_key: Optional[ str ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	params: Optional[ Dict[ str, str ] ]
	temperature: Optional[ float ]
	max_tokens: Optional[ int ]
	top_p: Optioanl[ float ]
	reasonging_effort: Optional[ float ]
	
	def __init__( self ) -> None:
		'''
		
			Purpose:
			-----------
			Initialize Groq API.

			Parameters:
			-----------
			headers (Optional[Dict[str, str]]): Optional headers for requests.

			Returns:
			-----------
			None
			
		'''
		super( ).__init__( )
		self.api_key = cfg.CLAUDE_API_KEY
		self.url = r'https://api.anthropic.com'
		self.client = None
		self.messages = None
		self.model = 'claude-sonnet-4-5'
		self.max_tokens = 1000
		self.headers = { }
		self.timeout = None
		self.content = None
		self.params = None
		self.response = None
		self.agents = cfg.AGENTS
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-----------
			Groq list of members.

			Parameters:
			-----------
			None

			Returns:
			-----------
			list[str]: Ordered attribute/method names.

		'''
		return [ 'content',
		         'url',
		         'client',
		         'timeout',
		         'headers',
		         'fetch',
		         'api_key',
		         'response',
		         'cse_id',
		         'params',
		         'agents,',
		         'fetch' ]
	
	def fetch( self, query: str ) -> str | None:
		'''

			Purpose:
			-------
			Sends an API request to Groq given a query as input

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
			self.keywords = query
			self.client = Anthropic( )
			message = self.client.messages.create( model=self.model, max_tokens=self.max_tokens,
		    messages=[
	        {
	            'role': 'user',
	            'content': self.query
	        }])
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Claude'
			exception.method = 'fetch( self, query: str, time: int=10 ) -> str'
			dialog = ErrorDialog( exception )
			dialog.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Claude'
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

class Mistral( Fetcher ):
	'''

		Purpose:
		---------
		Class providing to the Mistral API

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
	client: Optional[ Mistral ]
	model: Optional[ str ]
	response: Optional[ Response ]
	api_key: Optional[ str ]
	query: Optional[  str  ]
	params: Optional[ Dict[ str, str ] ]
	temperature: Optional[ float ]
	max_tokens: Optional[ int ]
	top_p: Optioanl[ float ]
	reasonging_effort: Optional[ float ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	
	def __init__( self ) -> None:
		'''
		
			Purpose:
			-----------
			Initialize Groq API.

			Parameters:
			-----------
			headers (Optional[Dict[str, str]]): Optional headers for requests.

			Returns:
			-----------
			None
			
		'''
		super( ).__init__( )
		self.api_key = cfg.MISTRAL_API_KEY
		self.model = 'mistral-medium-lates'
		self.headers = { }
		self.timeout = None
		self.content = None
		self.params = None
		self.response = None
		self.query = None
		self.agents = cfg.AGENTS
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-----------
			Groq list of members.

			Parameters:
			-----------
			None

			Returns:
			-----------
			list[str]: Ordered attribute/method names.

		'''
		return [ 'content',
		         'url',
		         'client',
		         'timeout',
		         'headers',
		         'fetch',
		         'api_key',
		         'response',
		         'cse_id',
		         'params',
		         'agents,',
		         'fetch' ]
	
	def fetch( self, query: str ) -> List[ str ] | None:
		'''

			Purpose:
			-------
			Sends an API request to Groq given a query as input

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
			self.client = Mistral( api_key=self.api_key )
			self.messages = [
	        {
	            'role': 'user',
	            'content': self.query,
	        } ]
			
			_response = self.client.chat.complete(  model=self.model, messages=self.messages)
			return _response
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Mistral'
			exception.method = 'fetch( self, query: str ) -> List[ str ]'
			dialog = ErrorDialog( exception )
			dialog.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Mistral'
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

class Chat( Fetcher ):
	"""
	
	    Purpose
	    ___________
	    Class used for interacting with a Data Science & Programming assistant
	
	
	    Parameters
	    ------------
	    num: int=1
	    temp: float=0.8
	    top: float=0.9
	    freq: float=0.0
	    pres: float=0.0
	    iters: int=10000
	    store: bool=True
	    stream: bool=True
	
	
	    Methods
	    ------------
	    get_model_options( self ) -> str
	    generate_text( self, prompt: str ) -> str:
	    analyze_image( self, prompt: str, url: str ) -> str:
	    summarize_document( self, prompt: str, path: str ) -> str
	    search_web( self, prompt: str ) -> str
	    search_files( self, prompt: str ) -> str
	    dump( self ) -> str
	    get_data( self ) -> { }



    """
	
	def __init__( self, num: int=1, temp: float=0.8, top: float=0.9,
			freq: float=0.0, pres: float=0.0, iters: int=10000, store: bool=True, stream: bool=True, ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.system_instructions = None
		self.client = OpenAI( api_key=self.api_key )
		self.client.api_key = cfg.OPENAI_API_KEY
		self.model = ''
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.max_completion_tokens = iters
		self.store = store
		self.stream = stream
		self.modalities = [ 'text', 'audio' ]
		self.stops = [ '#', ';' ]
		self.response_format = 'auto'
		self.reasoning_effort = None
		self.input_text = None
		self.name = 'Bro'
		self.description = 'A Computer Programming and Data Science Assistant'
		self.id = 'asst_2Yu2yfINGD5en4e0aUXAKxyu'
		self.vector_store_ids = [ 'vs_67e83bdf8abc81918bda0d6b39a19372', ]
		self.metadata = { }
		self.tools = [ ]
		self.vector_stores = { 'Code': 'vs_67e83bdf8abc81918bda0d6b39a19372', }
	
	def generate_text( self, prompt: str ) -> str:
		"""
	
	        Purpose
	        _______
	        Generates a chat completion given a prompt
	
	
	        Parameters
	        ----------
	        prompt: str
	
	
	        Returns
	        -------
	        str

        """
		try:
			throw_if( 'prompt', prompt )
			self.input_text = prompt
			self.response = self.client.responses.create( model=self.model, input=self.input_text )
			generated_text = self.response.output_text
			return generated_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	def generate_image( self, prompt: str ) -> str:
		"""
	
	        Purpose
	        _______
	        Generates a chat completion given a prompt
	
	
	        Parameters
	        ----------
	        prompt: str
	
	
	        Returns
	        -------
	        str

        """
		try:
			throw_if( 'prompt', prompt )
			self.input_text = prompt
			self.response = self.client.images.generate( model='dall-e-3', prompt=self.input_text,
				size='1024x1024', quality='standard', n=1, )
			generated_image = self.response.data[ 0 ].url
			return generated_image
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = 'generate_image( self, prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze_image( self, prompt: str, url: str ) -> str:
		"""

	        Purpose
	        _______
	        Method that analyzeses an image given a prompt,
	
	        Parameters
	        ----------
	        prompt (str) - user input text
	        url: str - file path to image
	
	        Returns
	        -------
	        str

        """
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'url', url )
			self.input_text = prompt
			self.image_url = url
			self.input = [
			{
				'role': 'user',
				'content': [
				{
					'type': 'input_text',
					'text': self.input_text
				},
				{
					'type': 'input_image',
					'image_url': self.image_url
				},],
			} ]
			self.response = self.client.responses.create( model=self.model, input=self.input )
			image_analysis = self.response.output_text
			return image_analysis
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = 'analyze_image( self, prompt: str, url: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	def summarize_document( self, prompt: str, path: str ) -> str:
		"""

	        Purpose
	        _______
	        Method that summarizes a document given a
	        path prompt, and a path
	
	        Parameters
	        ----------
	        prompt: str
	        path: str
	
	        Returns
	        -------
	        str

        """
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'path', path )
			self.input_text = prompt
			self.file_path = path
			self.file = self.client.files.create( file=open( self.file_path, 'rb' ), purpose='user_data' )
			self.messages = [
			{
				'role': 'user',
				'content': [
				{
					'type': 'file',
					'file':
					{
						'file_id': self.file.id,
					},
				},
				{
					'type': 'text',
					'text': self.input_text,
				}, ],
			}, ]
			self.response = self.client.responses.create( model=self.model, inputs=self.messages )
			document_summary = self.reponse.output_text
			return document_summary
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = 'summarize_document( self, prompt: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def search_web( self, prompt: str ) -> str:
		"""

                Purpose
                _______
                Use web_search_options to retrieve and synthesize
                recent web results for `prompt`.


                Parameters
                ----------
                prompt: str
                url: str

                Returns
                -------
                str

        """
		try:
			throw_if( 'prompt', prompt )
			self.web_options = { 'search_recency_days': 30, 'max_search_results': 8 }
			self.messages = [ {'role': 'user', 'content': prompt,} ]
			self.response = self.client.responses.create( model=self.model,
				web_search_options=self.web_options, input=self.messages )
			web_results = self.response.output_text
			return web_results
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = 'search_web( self, prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def search_files( self, prompt: str ) -> str:
		"""

            Purpose
	        -------
	        Run a file-search tool call against configured vector stores using
	        the Responses API, and return the textual result.


            Parameters
            ----------
            prompt: str

            Returns
            -------
            str

        """
		try:
			throw_if( 'prompt', prompt )
			self.query = prompt
			self.tools = [
			{
				'type': 'file_search',
				'vector_store_ids': self.vector_store_ids,
				'max_num_results': 20,
			} ]
			self.response = self.client.responses.create( model=self.model, tools=self.tools,
				input=prompt )
			file_search = self.response.output_text
			return file_search
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = 'search_files( self, prompt: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def translate( self, text: str ) -> str:
		pass
	
	def transcribe( self, text: str ) -> str:
		pass
	
	def get_format_options( self ) -> List[ str ]:
		'''
	
	        Purpose:
	        ---------
	        Method that returns a list of formatting options

        '''
		return [ 'auto', 'text', 'json' ]
	
	def get_model_options( self ) -> List[ str ]:
		'''
	
	        Purpose:
	        ---------
	        Method that returns a list of available models

        '''
		return [ 'gpt-4-0613',
		         'gpt-4-0314',
				 'gpt-4-turbo-2024-04-09',
				 'gpt-4o-2024-08-06',
				 'gpt-4o-2024-11-20',
				 'gpt-4o-2024-05-13',
				 'gpt-4o-mini-2024-07-18',
				 'o1-2024-12-17',
				 'o1-mini-2024-09-12',
				 'o3-mini-2025-01-31',
				 'ft:gpt-4.1-2025-04-14:leeroy-jenkins:bro-gpt-4-1-df-analysis-2025-21-05:BZetxEQa',
				 'ft:gpt-4o-2024-08-06:leeroy-jenkins:bro-fine-tuned-05052025:BTryvkMx',
				 'ft:gpt-4o-2024-08-06:leeroy-jenkins:bro-analytics:BTX4TYqY',
				 'ft:gpt-4o-2024-08-06:leeroy-jenkins:bro-fine-tuned-05052025:BTryvkMx', ]
	
	def get_effort_options( self ) -> List[ str ]:
		'''
	
	        Purpose:
	        ---------
	        Method that returns a list of available models

        '''
		return [ 'auto',
		         'low',
		         'high' ]
	
	def get_data( self ) -> Dict[ str, Any ]:
		'''
	
	        Purpose:
	        ---------
	        Returns: dict[ str ] of members

        '''
		return \
		{
			'num': self.number,
			'temperature': self.temperature,
			'top_percent': self.top_percent,
			'frequency_penalty': self.frequency_penalty,
			'presence_penalty': self.presence_penalty,
			'store': self.store,
			'stream': self.stream,
			'size': self.size,
		}
	
	def dump( self ) -> str:
		'''

	        Purpose:
	        ---------
	        Returns: dict of members

        '''
		new = '\r\n'
		return ( 'num' + f' = {self.number}' + new
				+ 'temperature' + f' = {self.temperature}' + new
				+ 'top_percent' + f' = {self.top_percent}' + new
				+ 'frequency_penalty' + f' = {self.frequency_penalty}' + new
				+ 'presence_penalty' + f' = {self.presence_penalty}' + new
				+ 'max_completion_tokens' + f' = {self.max_completion_tokens}' + new
				+ 'store' + f' = {self.store}' + new
				+ 'stream' + f' = {self.stream}' + new )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema  = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = ( 'create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]' )
			error = ErrorDialog( exception )
			error.show( )

class Grokipedia( Fetcher ):
	'''
		
			Purpose:
			-------
			Class providing access to the Grokipedia API
			
			Attributes:
			----------
			client - GrokipediaClient
			query - str
			response - Response
			page - str
			api_key - str
			params - Dict[ str, Any ]
			
			
			Methods:
			--------
			fetch( )
			fetch_page( )
			fetch_pages( )
			create_schema( )
		
	
	'''
	api_key: Optional[ str ]
	client: Optional[ GrokipediaClient ]
	query: Optional[ str ]
	page: Optional[ str ]
	limit: Optional[ int ]
	include_content: Optional[ bool ]
	response: Optional[ Response ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ):
		'''
		
		Purpose:
		-------
		Class constructor: initializes fields
		
		'''
		super( ).__init__( )
		self.api_key = cfg.GROQ_API_KEY
		self.url = None
		self.client = None
		self.query = None
		self.page = None
		self.response = None
		self.params = None
		self.limit = 12
		self.include_content = True
		
	def fetch( self, query: str ) -> Dict[ str, Any ] | None:
		'''
			
			Purpose:
			-------
			Method for submitting request to the Grokipedia API
			
			Paramters:
			---------
			query - str
			
			Returns:
			-------
			Dict[ str, Any ]
			
			
		'''
		try:
			throw_if( 'query', query )
			self.query = query
			self.client = GrokipediaClient( )
			_results = self.client.search( query=self.query )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Grokipedia'
			exception.method = 'fetch( self, query: str ) -> Dict[ str, Any ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_page( self, query: str ) -> Dict[ str, Any ] | None:
		'''
			
			Purpose:
			-------
			Method for submitting request to the Grokipedia API for a specific page
			
			Paramters:
			---------
			query - str
			
			Returns:
			-------
			Dict[ str, Any ]
			
			
		'''
		try:
			throw_if( 'query', query )
			self.query = query
			self.client = GrokipediaClient( )
			_results = self.client.get_page( query=self.query, include_content=True )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Grokipedia'
			exception.method = 'fetch_page( self, query: str ) -> Dict[ str, Any ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_pages( self, query: str ) -> Dict[ str, Any ] | None:
		'''
			
			Purpose:
			-------
			Method for submitting request to the Grokipedia API for a specific page
			
			Paramters:
			---------
			query - str
			
			Returns:
			-------
			Dict[ str, Any ]
			
			
		'''
		try:
			throw_if( 'query', query )
			self.query = query
			self.client = GrokipediaClient( )
			_results = self.client.get_page( query=self.query, include_content=True )
			return _results
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Grokipedia'
			exception.method = 'fetch_pages( self, query: str ) -> Dict[ str, Any ]'
			error = ErrorDialog( exception )
			error.show( )
		
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[
				                                                               str, str ] | None:
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
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			if not isinstance( parameters, dict ):
				msg = 'parameters must be a dict of param_name → schema definitions.'
				raise ValueError( msg )
			func_name = function.strip( )
			tool_name = tool.strip( )
			desc = description.strip( )
			if required is None:
				required = list( parameters.keys( ) )
			_schema = \
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
			return _schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Grokipedia'
			exception.method = (
				'create_schema( self, function: str, tool: str, description: str, '
				'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]')
			error = ErrorDialog( exception )
			error.show( )