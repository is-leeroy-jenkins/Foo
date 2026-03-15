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
import io
import re
import urllib.parse
from typing import Any, Dict, Optional, Pattern, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from PIL.Image import Image
from astropy.table import Table
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_core.tools import Tool
from pandas import DataFrame
import requests
import requests_cache
from anthropic import Anthropic
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from google import genai
from grokipedia_api import GrokipediaClient
from groq import Groq as GroqClient
import googlemaps as gmaps
from langchain_core.documents import Document
from langchain_classic.agents import AgentExecutor
from langchain_community.retrievers import ArxivRetriever, WikipediaRetriever
from langchain_googledrive.retrievers import GoogleDriveRetriever
import os
from openai import OpenAI
import openmeteo_requests
from pathlib import Path
from owslib.wms import WebMapService
from requests import Response
from retry_requests import retry
from sscws.sscws import SscWs
import config as cfg
from boogr import Error
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

def encode_image( path: str ) -> str:
	"""
	
		Purpose:
		_________
		
		Parametes:
		----------
		
		
		Returns:
		--------
		
		
	"""
	data = Path( path ).read_bytes( )
	return base64.b64encode( data ).decode( "utf-8" )

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
	
	def __init__( self, db_uri: str, doc_paths: List[ str ], model: str='gpt-5-mini',
			temperature: float=0.8 ):
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
			raise exception
	
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
			raise exception
	
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
			raise exception
			
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
			raise exception
			
	
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
			raise exception
			
	
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
			raise exception
			

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
			raise exception
			
	
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
			raise exception
			
	
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
			raise exception
			
	
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
			raise exception
			
	
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
			raise exception
			
	
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
			raise exception
			
	
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
			raise exception
			
	
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
			raise exception
			
	
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
			raise exception
			
		
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
			raise exception
			
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
			raise exception
			

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
			raise exception

class ArXiv( Fetcher ):
	'''

		Purpose:
		--------
		Provides ArXiv retrieval functionality and converts returned entries into
		LangChain Document objects.

	'''
	fetcher: Optional[ ArxivRetriever ]
	documents: Optional[ List[ Document ] ]
	max_documents: Optional[ int ]
	full_documents: Optional[ bool ]
	include_metadata: Optional[ bool ]
	query: Optional[ str ]
	
	def __init__( self, max_documents: int = 5, full_documents: bool = False,
			include_metadata: bool = False ) -> None:
		super( ).__init__( )
		self.fetcher = None
		self.documents = None
		self.query = None
		self.max_documents = max( 1, min( int( max_documents ), 300 ) )
		self.full_documents = bool( full_documents )
		self.include_metadata = bool( include_metadata )
	
	def fetch( self, question: str, max_documents: int | None = None,
			full_documents: bool | None = None,
			include_metadata: bool | None = None ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Query ArXiv through LangChain's ArxivRetriever and return LangChain
			Document objects.

			Parameters:
			-----------
			question:
				Free-text search query or arXiv identifier.
			max_documents:
				Optional override for maximum number of returned documents.
			full_documents:
				Optional override indicating whether full document text should be
				fetched instead of summary-oriented retrieval.
			include_metadata:
				Optional override indicating whether all available metadata should
				be included.

			Returns:
			--------
			List[Document] | None

		'''
		try:
			throw_if( 'question', question )
			self.query = question.strip( )
			
			max_docs = self.max_documents if max_documents is None else \
				max( 1, min( int( max_documents ), 300 ) )
			
			get_full = self.full_documents if full_documents is None else \
				bool( full_documents )
			
			load_meta = self.include_metadata if include_metadata is None else \
				bool( include_metadata )
			
			self.fetcher = ArxivRetriever(
				load_max_docs=max_docs,
				get_full_documents=get_full,
				load_all_available_meta=load_meta )
			
			self.documents = self.fetcher.invoke( self.query )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'ArXiv'
			exception.method = (
					'fetch( self, question: str, max_documents: int | None=None, '
					'full_documents: bool | None=None, include_metadata: bool | None=None ) '
					'-> List[ Document ]'
			)
			raise exception

class GoogleDrive( Fetcher ):
	'''
	
		Purpose:
		--------
		Provides Google Drive retrieval functionality and converts returned
		items into LangChain Document objects.
	
	'''
	fetcher: Optional[ GoogleDriveRetriever ]
	documents: Optional[ List[ Document ] ]
	num_results: Optional[ int ]
	folder_id: Optional[ str ]
	template: Optional[ str ]
	query: Optional[ str ]
	mime_type: Optional[ str ]
	mode: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.fetcher = None
		self.documents = None
		self.query = None
		self.template = 'gdrive-query'
		self.folder_id = 'root'
		self.num_results = 10
		self.mime_type = None
		self.mode = 'documents'
	
	@property
	def mime_options( self ) -> List[ str ]:
		'''
		
			Purpose:
			--------
			Return supported MIME types aligned to the Google Drive retriever docs.
		
			Returns:
			--------
			List[str]
		
		'''
		return [
				'',
				'text/text',
				'text/plain',
				'text/html',
				'text/csv',
				'text/markdown',
				'image/png',
				'image/jpeg',
				'application/epub+zip',
				'application/pdf',
				'application/rtf',
				'application/vnd.google-apps.document',
				'application/vnd.google-apps.presentation',
				'application/vnd.google-apps.spreadsheet',
				'application/vnd.google.colaboratory',
				'application/vnd.openxmlformats-officedocument.presentationml.presentation',
				'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
		]
	
	@property
	def template_options( self ) -> List[ str ]:
		'''
		
			Purpose:
			--------
			Return predefined template options supported by GoogleDriveRetriever.
		
			Returns:
			--------
			List[str]
		
		'''
		return [
				'gdrive-all-in-folder',
				'gdrive-query',
				'gdrive-by-name',
				'gdrive-query-in-folder',
				'gdrive-mime-type',
				'gdrive-mime-type-in-folder',
				'gdrive-query-with-mime-type',
				'gdrive-query-with-mime-type-and-folder',
		]
	
	@property
	def mode_options( self ) -> List[ str ]:
		'''
		
			Purpose:
			--------
			Return supported retrieval display modes.
		
			Returns:
			--------
			List[str]
		
		'''
		return [ 'documents', 'snippets' ]
	
	def fetch( self, question: str, folder_id: str = 'root', results: int = 10,
			template: str = 'gdrive-query', mime_type: str | None = None,
			mode: str = 'documents' ) -> List[ Document ] | None:
		'''
		
			Purpose:
			--------
			Query Google Drive through LangChain's GoogleDriveRetriever and
			return LangChain Document objects.
		
			Parameters:
			-----------
			question:
				Free-text query used by the retriever. For templates that do not
				require a query, a placeholder value may still be passed.
			folder_id:
				Google Drive folder id. Use 'root' for the user's root Drive.
			results:
				Maximum number of returned documents.
			template:
				Predefined GoogleDriveRetriever selection template.
			mime_type:
				Optional MIME type filter.
			mode:
				Retrieval mode, typically 'documents' or 'snippets'.
		
			Returns:
			--------
			List[Document] | None
		
		'''
		try:
			throw_if( 'template', template )
			throw_if( 'folder_id', folder_id )
			
			self.query = (question or '').strip( )
			self.folder_id = folder_id.strip( ) if folder_id else 'root'
			self.num_results = max( 1, min( int( results ), 100 ) )
			self.template = template.strip( )
			self.mime_type = mime_type.strip( ) if isinstance( mime_type, str ) and mime_type.strip( ) else None
			self.mode = mode.strip( ) if mode else 'documents'
			
			retriever_kwargs: Dict[ str, Any ] = {
					'folder_id': self.folder_id,
					'template': self.template,
					'num_results': self.num_results,
					'mode': self.mode,
			}
			
			if self.mime_type:
				retriever_kwargs[ 'mime_type' ] = self.mime_type
			
			if cfg.GOOGLE_ACCOUNT_FILE:
				retriever_kwargs[ 'credentials_path' ] = cfg.GOOGLE_ACCOUNT_FILE
			
			if cfg.GOOGLE_DRIVE_TOKEN_PATH:
				retriever_kwargs[ 'token_path' ] = cfg.GOOGLE_DRIVE_TOKEN_PATH
			
			self.fetcher = GoogleDriveRetriever( **retriever_kwargs )
			
			invoke_query = self.query
			if not invoke_query:
				if self.template in ('gdrive-all-in-folder', 'gdrive-mime-type',
				                     'gdrive-mime-type-in-folder'):
					invoke_query = '*'
				else:
					raise ValueError(
						'A query is required for the selected Google Drive template.'
					)
			
			self.documents = self.fetcher.invoke( invoke_query )
			return self.documents
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'GoogleDrive'
			exception.method = (
					'fetch( self, question: str, folder_id: str=root, results: int=10, '
					'template: str=gdrive-query, mime_type: str | None=None, '
					'mode: str=documents ) -> List[ Document ]'
			)
			raise exception

class Wikipedia( Fetcher ):
	'''

		Purpose:
		--------
		Provides Wikipedia retrieval functionality and converts returned entries
		into LangChain Document objects.

	'''
	fetcher: Optional[ WikipediaRetriever ]
	documents: Optional[ List[ Document ] ]
	max_documents: Optional[ int ]
	include_metadata: Optional[ bool ]
	language: Optional[ str ]
	query: Optional[ str ]
	
	def __init__( self, language: str = 'en', max_documents: int = 5,
			include_metadata: bool = False ) -> None:
		super( ).__init__( )
		self.fetcher = None
		self.documents = None
		self.query = None
		self.language = (language or 'en').strip( ) or 'en'
		self.max_documents = max( 1, min( int( max_documents ), 300 ) )
		self.include_metadata = bool( include_metadata )
	
	def fetch( self, question: str, language: str | None = None,
			max_documents: int | None = None,
			include_metadata: bool | None = None ) -> List[ Document ] | None:
		'''

			Purpose:
			--------
			Query Wikipedia through LangChain's WikipediaRetriever and return
			LangChain Document objects.

			Parameters:
			-----------
			question:
				Free-text Wikipedia query.
			language:
				Optional language code override, e.g. "en", "fr", "de", "ja".
			max_documents:
				Optional override for maximum number of returned documents.
				Hard-capped at 300.
			include_metadata:
				Optional override indicating whether all available metadata
				should be included.

			Returns:
			--------
			List[Document] | None

		'''
		try:
			throw_if( 'question', question )
			self.query = question.strip( )
			
			lang = self.language if language is None else \
				(language.strip( ) if language else 'en')
			
			max_docs = self.max_documents if max_documents is None else \
				max( 1, min( int( max_documents ), 300 ) )
			
			load_meta = self.include_metadata if include_metadata is None else \
				bool( include_metadata )
			
			self.fetcher = WikipediaRetriever(
				lang=lang,
				load_max_docs=max_docs,
				load_all_available_meta=load_meta )
			
			self.documents = self.fetcher.invoke( input=self.query )
			return self.documents
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Wikipedia'
			exception.method = (
					'fetch( self, question: str, language: str | None=None, '
					'max_documents: int | None=None, include_metadata: bool | None=None ) '
					'-> List[ Document ]'
			)
			raise exception

class TheNews( Fetcher ):
	'''

		Purpose:
		--------
		Provides a structured wrapper around The News API endpoints.

	'''
	agents: Optional[ str ]
	url: Optional[ str ]
	response: Optional[ Response ]
	api_key: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	endpoint: Optional[ str ]
	limit: Optional[ int ]
	page: Optional[ int ]
	
	def __init__( self ) -> None:
		'''
			Purpose:
			-----------
			Initialize The News API wrapper with sane defaults and environment-
			based authentication.
		'''
		super( ).__init__( )
		self.timeout = 10
		self.url = 'https://api.thenewsapi.com/v1/news'
		self.response = None
		self.result = None
		self.headers = { }
		self.params = { }
		self.endpoint = 'all'
		self.limit = 10
		self.page = 1
		self.api_key = cfg.THENEWS_API_KEY
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		if 'Accept' not in self.headers:
			self.headers[ 'Accept' ] = 'application/json'
	
	def __dir__( self ) -> List[ str ]:
		'''
			Purpose:
			-----------
			Return visible member ordering.
		'''
		return [
				'api_key',
				'url',
				'timeout',
				'headers',
				'endpoint',
				'limit',
				'page',
				'params',
				'fetch',
		]
	
	def fetch( self, endpoint: str = 'all', query: str = '', language: str = 'en',
			categories: str = '', exclude_categories: str = '', locale: str = '',
			domains: str = '', exclude_domains: str = '', source_ids: str = '',
			exclude_source_ids: str = '', published_after: str = '', published_before: str = '',
			published_on: str = '', sort: str = 'published_at',
			limit: int = 10, page: int = 1, include_similar: bool = True,
			headlines_per_category: int = 6, time: int = 10,
			api_key: str | None = None ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Send a request to The News API using one of the documented endpoints
			and return the parsed JSON response.

			Parameters:
			-----------
			endpoint:
				One of: all, top, headlines, sources
			query:
				Search query for endpoints that support search.
			language:
				Comma-separated language codes.
			categories:
				Comma-separated category filter.
			exclude_categories:
				Comma-separated excluded categories.
			locale:
				Comma-separated country codes where supported.
			domains:
				Comma-separated included domains.
			exclude_domains:
				Comma-separated excluded domains.
			source_ids:
				Comma-separated included source ids.
			exclude_source_ids:
				Comma-separated excluded source ids.
			published_after:
				Date/datetime filter when supported.
			published_before:
				Date/datetime filter when supported.
			published_on:
				Single publication date filter when supported.
			sort:
				Sort order for applicable endpoints.
			limit:
				Result size.
			page:
				Page number.
			include_similar:
				Headlines-only switch.
			headlines_per_category:
				Headlines-only per-category limit.
			time:
				Request timeout in seconds.
			api_key:
				Optional runtime override. Falls back to cfg.THENEWS_API_KEY.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			self.endpoint = (endpoint or 'all').strip( ).lower( )
			self.timeout = int( time )
			self.limit = max( 1, min( int( limit ), 50 ) )
			self.page = max( 1, int( page ) )
			
			active_key = (api_key or self.api_key or '').strip( )
			if not active_key:
				raise ValueError(
					'The News API key is required. Set THENEWSAPI_API_KEY or enter one in the UI.'
				)
			
			valid_endpoints = { 'all', 'top', 'headlines', 'sources' }
			if self.endpoint not in valid_endpoints:
				raise ValueError(
					f"Unsupported endpoint '{self.endpoint}'. "
					f"Supported endpoints: {', '.join( sorted( valid_endpoints ) )}."
				)
			
			self.params = { 'api_token': active_key }
			
			if self.endpoint in ('all', 'top'):
				if query and query.strip( ):
					self.params[ 'search' ] = query.strip( )
				
				if language and language.strip( ):
					self.params[ 'language' ] = language.strip( )
				
				if categories and categories.strip( ):
					self.params[ 'categories' ] = categories.strip( )
				
				if exclude_categories and exclude_categories.strip( ):
					self.params[ 'exclude_categories' ] = exclude_categories.strip( )
				
				if domains and domains.strip( ):
					self.params[ 'domains' ] = domains.strip( )
				
				if exclude_domains and exclude_domains.strip( ):
					self.params[ 'exclude_domains' ] = exclude_domains.strip( )
				
				if source_ids and source_ids.strip( ):
					self.params[ 'source_ids' ] = source_ids.strip( )
				
				if exclude_source_ids and exclude_source_ids.strip( ):
					self.params[ 'exclude_source_ids' ] = exclude_source_ids.strip( )
				
				if published_after and published_after.strip( ):
					self.params[ 'published_after' ] = published_after.strip( )
				
				if published_before and published_before.strip( ):
					self.params[ 'published_before' ] = published_before.strip( )
				
				if published_on and published_on.strip( ):
					self.params[ 'published_on' ] = published_on.strip( )
				
				if sort and sort.strip( ):
					self.params[ 'sort' ] = sort.strip( )
				
				self.params[ 'limit' ] = self.limit
				self.params[ 'page' ] = self.page
				
				if self.endpoint == 'top' and locale and locale.strip( ):
					self.params[ 'locale' ] = locale.strip( )
			
			elif self.endpoint == 'headlines':
				if locale and locale.strip( ):
					self.params[ 'locale' ] = locale.strip( )
				
				if domains and domains.strip( ):
					self.params[ 'domains' ] = domains.strip( )
				
				if exclude_domains and exclude_domains.strip( ):
					self.params[ 'exclude_domains' ] = exclude_domains.strip( )
				
				if source_ids and source_ids.strip( ):
					self.params[ 'source_ids' ] = source_ids.strip( )
				
				if exclude_source_ids and exclude_source_ids.strip( ):
					self.params[ 'exclude_source_ids' ] = exclude_source_ids.strip( )
				
				if language and language.strip( ):
					self.params[ 'language' ] = language.strip( )
				
				if published_on and published_on.strip( ):
					self.params[ 'published_on' ] = published_on.strip( )
				
				self.params[ 'headlines_per_category' ] = max(
					1,
					min( int( headlines_per_category ), 10 ) )
				
				self.params[ 'include_similar' ] = \
					'true' if bool( include_similar ) else 'false'
			
			elif self.endpoint == 'sources':
				if categories and categories.strip( ):
					self.params[ 'categories' ] = categories.strip( )
				
				if exclude_categories and exclude_categories.strip( ):
					self.params[ 'exclude_categories' ] = exclude_categories.strip( )
				
				if language and language.strip( ):
					self.params[ 'language' ] = language.strip( )
				
				self.params[ 'page' ] = self.page
			
			request_url = f'{self.url}/{self.endpoint}'
			self.response = requests.get(
				url=request_url,
				params=self.params,
				headers=self.headers,
				timeout=self.timeout )
			
			self.response.raise_for_status( )
			return self.response.json( )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'TheNews'
			exception.method = (
					'fetch( self, endpoint: str=all, query: str=, language: str=en, '
					'categories: str=, exclude_categories: str=, locale: str=, '
					'domains: str=, exclude_domains: str=, source_ids: str=, '
					'exclude_source_ids: str=, published_after: str=, '
					'published_before: str=, published_on: str=, '
					'sort: str=published_at, limit: int=10, page: int=1, '
					'include_similar: bool=True, headlines_per_category: int=6, '
					'time: int=10, api_key: str | None=None ) -> Dict[ str, Any ]'
			)
			raise exception

class GoogleSearch( Fetcher ):
	'''

		Purpose:
		---------
		Class providing structured access to the Google Custom Search JSON API.

	'''
	keywords: Optional[ str ]
	url: Optional[ str ]
	re_tag: Optional[ Pattern ]
	re_ws: Optional[ Pattern ]
	response: Optional[ Response ]
	api_key: Optional[ str ]
	cse_id: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	results: Optional[ int ]
	start: Optional[ int ]
	
	def __init__( self ) -> None:
		'''
			Purpose:
			-----------
			Initialize GoogleSearch with environment-based credentials and sane defaults.

			Returns:
			-----------
			None
		'''
		super( ).__init__( )
		self.api_key = cfg.GOOGLE_API_KEY
		self.cse_id = cfg.GOOGLE_CSE_ID
		self.re_tag = re.compile( r'<[^>]+>' )
		self.re_ws = re.compile( r'\s+' )
		self.url = 'https://customsearch.googleapis.com/customsearch/v1'
		self.headers = { }
		self.timeout = 10
		self.keywords = None
		self.params = { }
		self.response = None
		self.results = 10
		self.start = 1
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		if 'Accept' not in self.headers:
			self.headers[ 'Accept' ] = 'application/json'
	
	def __dir__( self ) -> List[ str ]:
		'''
			Purpose:
			-----------
			Control visible ordering for GoogleSearch.
		'''
		return [
				'keywords',
				'url',
				'timeout',
				'headers',
				'fetch',
				'api_key',
				'response',
				'cse_id',
				'params',
				'agents',
				'results',
				'start',
		]
	
	def fetch( self, keywords: str, results: int = 10,
			start: int = 1, exact_terms: str = '', exclude_terms: str = '',
			file_type: str = '', date_restrict: str = '', gl: str = '', lr: str = '',
			safe: str = 'off', search_type: str = '', site_search: str = '', site_search_filter: str = '',
			sort: str = '', img_size: str = '', img_type: str = '', img_color_type: str = '',
			img_dominant_color: str = '', time: int = 10, api_key: str | None = None,
			cse_id: str | None = None ) -> Dict[ str, Any ] | None:
		'''

			Purpose:
			--------
			Send a request to the Google Custom Search JSON API and return the
			parsed JSON response.

			Parameters:
			-----------
			keywords:
				Search query string.
			results:
				Number of results per request. Google supports up to 10 per request.
			start:
				Index of the first result to return. Combined paging is limited to
				the first 100 results.
			exact_terms:
				Phrase that all documents must contain.
			exclude_terms:
				Words or phrases that must not appear.
			file_type:
				File extension filter, e.g. pdf, docx.
			date_restrict:
				Date restriction such as d7, w2, m1, y1.
			gl:
				Country boost code, e.g. us.
			lr:
				Language restrict code, e.g. lang_en.
			safe:
				Safe search value, typically active or off.
			search_type:
				Set to image for image search.
			site_search:
				Restrict to a site or domain.
			site_search_filter:
				i to include, e to exclude the specified site.
			sort:
				Sort expression when supported by the engine.
			img_size:
				Image size filter when search_type=image.
			img_type:
				Image type filter when search_type=image.
			img_color_type:
				Image color type filter when search_type=image.
			img_dominant_color:
				Image dominant color filter when search_type=image.
			time:
				Request timeout in seconds.
			api_key:
				Optional runtime override. Falls back to cfg.GOOGLE_API_KEY.
			cse_id:
				Optional runtime override. Falls back to cfg.GOOGLE_CSE_ID.

			Returns:
			--------
			Dict[str, Any] | None

		'''
		try:
			throw_if( 'keywords', keywords )
			
			active_key = (api_key or self.api_key or '').strip( )
			active_cse = (cse_id or self.cse_id or '').strip( )
			
			if not active_key:
				raise ValueError(
					'Google API key is required. Set GOOGLE_API_KEY or enter one in the UI.'
				)
			
			if not active_cse:
				raise ValueError(
					'Google CSE ID is required. Set GOOGLE_CSE_ID or enter one in the UI.'
				)
			
			self.timeout = int( time )
			self.keywords = keywords.strip( )
			self.results = max( 1, min( int( results ), 10 ) )
			self.start = max( 1, min( int( start ), 91 ) )
			
			self.params = {
					'q': self.keywords,
					'key': active_key,
					'cx': active_cse,
					'num': self.results,
					'start': self.start,
					'safe': (safe or 'off').strip( ),
			}
			
			if exact_terms and exact_terms.strip( ):
				self.params[ 'exactTerms' ] = exact_terms.strip( )
			
			if exclude_terms and exclude_terms.strip( ):
				self.params[ 'excludeTerms' ] = exclude_terms.strip( )
			
			if file_type and file_type.strip( ):
				self.params[ 'fileType' ] = file_type.strip( )
			
			if date_restrict and date_restrict.strip( ):
				self.params[ 'dateRestrict' ] = date_restrict.strip( )
			
			if gl and gl.strip( ):
				self.params[ 'gl' ] = gl.strip( )
			
			if lr and lr.strip( ):
				self.params[ 'lr' ] = lr.strip( )
			
			if search_type and search_type.strip( ):
				self.params[ 'searchType' ] = search_type.strip( )
			
			if site_search and site_search.strip( ):
				self.params[ 'siteSearch' ] = site_search.strip( )
			
			if site_search_filter and site_search_filter.strip( ):
				self.params[ 'siteSearchFilter' ] = site_search_filter.strip( )
			
			if sort and sort.strip( ):
				self.params[ 'sort' ] = sort.strip( )
			
			if search_type.strip( ).lower( ) == 'image':
				if img_size and img_size.strip( ):
					self.params[ 'imgSize' ] = img_size.strip( )
				
				if img_type and img_type.strip( ):
					self.params[ 'imgType' ] = img_type.strip( )
				
				if img_color_type and img_color_type.strip( ):
					self.params[ 'imgColorType' ] = img_color_type.strip( )
				
				if img_dominant_color and img_dominant_color.strip( ):
					self.params[ 'imgDominantColor' ] = img_dominant_color.strip( )
			
			self.response = requests.get(
				url=self.url,
				params=self.params,
				headers=self.headers,
				timeout=self.timeout )
			
			self.response.raise_for_status( )
			return self.response.json( )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleSearch'
			exception.method = (
					'fetch( self, keywords: str, results: int=10, start: int=1, '
					'exact_terms: str=, exclude_terms: str=, file_type: str=, '
					'date_restrict: str=, gl: str=, lr: str=, safe: str=off, '
					'search_type: str=, site_search: str=, site_search_filter: str=, '
					'sort: str=, img_size: str=, img_type: str=, '
					'img_color_type: str=, img_dominant_color: str=, time: int=10, '
					'api_key: str | None=None, cse_id: str | None=None ) -> Dict[ str, Any ]'
			)
			raise exception
			
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
	headers: Optional[ Dict[ str, Any ] ]
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
		self.api_key = cfg.GOOGLE_API_KEY
		self.headers = { }
		self.params = { }
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
	
	def geocode_location( self, address: str ) -> Tuple[ float, float ]:
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
			self.url = "https://maps.googleapis.com/maps/api/geocode/json"
			self.params = {
				"address": self.address,
				"key": self.api_key,
				"headers": self.headers
			}
	
			self.response = requests.get( url=self.url, params=self.params )
			self.response.raise_for_status( )
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
			raise exception
			

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
			raise exception
			
	
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
			
			self.params = \
			{
				'key': self.api_key
			}
			
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
			raise exception
			
	
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
			raise exception
			
		
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
			raise exception

class GoogleWeather( Fetcher ):
	'''
	
		Purpose:
		--------
		Provides structured access to the Google Maps Platform Weather API.
	
	'''
	gmaps: Optional[ GoogleMaps ]
	headers: Optional[ Dict[ str, Any ] ]
	api_key: Optional[ str ]
	mode: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	coordinates: Optional[ Tuple[ float, float ] ]
	address: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	response: Optional[ Response ]
	result: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.api_key = cfg.GOOGLE_WEATHER_API_KEY
		self.headers = { }
		self.gmaps = GoogleMaps( )
		self.mode = None
		self.url = 'https://weather.googleapis.com/v1'
		self.longitude = None
		self.latitude = None
		self.coordinates = None
		self.fetcher = None
		self.address = None
		self.params = { }
		self.response = None
		self.result = None
		self.timeout = 10
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		if 'Accept' not in self.headers:
			self.headers[ 'Accept' ] = 'application/json'
	
	def __dir__( self ) -> List[ str ]:
		return [
				'api_key',
				'url',
				'timeout',
				'headers',
				'fetch_current',
				'fetch_hourly_forecast',
				'fetch_daily_forecast',
				'fetch_alerts',
		]
	
	def _resolve_coordinates( self, address: str ) -> Tuple[ float, float ]:
		'''
		
			Purpose:
			--------
			Resolve a user-supplied address into latitude and longitude by using
			the existing GoogleMaps helper.
		
			Returns:
			--------
			Tuple[float, float]
		
		'''
		try:
			throw_if( 'address', address )
			self.address = address.strip( )
			lat, lng = self.gmaps.geocode_location( address=self.address )
			self.latitude = lat
			self.longitude = lng
			self.coordinates = (lat, lng)
			return self.coordinates
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleWeather'
			exception.method = '_resolve_coordinates( self, address: str ) -> Tuple[ float, float ]'
			raise exception
	
	def _request( self, path: str, params: Dict[ str, Any ], time: int = 10 ) -> Dict[ str, Any ] | None:
		'''
		
			Purpose:
			--------
			Send a GET request to a Google Weather API endpoint and return JSON.
		
			Returns:
			--------
			Dict[str, Any] | None
		
		'''
		try:
			active_key = (self.api_key or '').strip( )
			if not active_key:
				raise ValueError( 'Google Weather API key is required. Set GOOGLE_WEATHER_API_KEY.' )
			
			self.timeout = int( time )
			request_params = dict( params or { } )
			request_params[ 'key' ] = active_key
			
			request_url = f'{self.url}/{path}'
			self.response = requests.get(
				url=request_url,
				params=request_params,
				headers=self.headers,
				timeout=self.timeout )
			
			self.response.raise_for_status( )
			self.result = self.response.json( )
			return self.result
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleWeather'
			exception.method = '_request( self, path: str, params: Dict[ str, Any ], time: int=10 ) -> Dict[ str, Any ]'
			raise exception
	
	def fetch_current( self, address: str, units_system: str = 'METRIC',
			language_code: str = 'en', time: int = 10 ) -> Dict[ str, Any ] | None:
		'''
		
			Purpose:
			--------
			Retrieve current weather conditions for an address.
		
			Returns:
			--------
			Dict[str, Any] | None
		
		'''
		try:
			lat, lng = self._resolve_coordinates( address )
			params = {
					'location.latitude': lat,
					'location.longitude': lng,
					'unitsSystem': units_system,
					'languageCode': language_code,
			}
			return self._request(
				path='currentConditions:lookup',
				params=params,
				time=time )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleWeather'
			exception.method = (
					'fetch_current( self, address: str, units_system: str=METRIC, '
					'language_code: str=en, time: int=10 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch_hourly_forecast( self, address: str, hours: int = 24, units_system: str = 'METRIC',
			language_code: str = 'en', time: int = 10 ) -> Dict[ str, Any ] | None:
		'''
		
			Purpose:
			--------
			Retrieve hourly weather forecast for an address.
		
			Returns:
			--------
			Dict[str, Any] | None
		
		'''
		try:
			lat, lng = self._resolve_coordinates( address )
			params = {
					'location.latitude': lat,
					'location.longitude': lng,
					'hours': max( 1, min( int( hours ), 240 ) ),
					'unitsSystem': units_system,
					'languageCode': language_code,
			}
			return self._request(
				path='forecast/hours:lookup',
				params=params,
				time=time )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleWeather'
			exception.method = (
					'fetch_hourly_forecast( self, address: str, hours: int=24, '
					'units_system: str=METRIC, language_code: str=en, time: int=10 ) '
					'-> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch_daily_forecast( self, address: str, days: int = 5, units_system: str = 'METRIC',
			language_code: str = 'en', time: int = 10 ) -> Dict[ str, Any ] | None:
		'''
		
			Purpose:
			--------
			Retrieve daily weather forecast for an address.
		
			Returns:
			--------
			Dict[str, Any] | None
		
		'''
		try:
			lat, lng = self._resolve_coordinates( address )
			params = {
					'location.latitude': lat,
					'location.longitude': lng,
					'days': max( 1, min( int( days ), 10 ) ),
					'unitsSystem': units_system,
					'languageCode': language_code,
			}
			return self._request(
				path='forecast/days:lookup',
				params=params,
				time=time )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleWeather'
			exception.method = (
					'fetch_daily_forecast( self, address: str, days: int=5, '
					'units_system: str=METRIC, language_code: str=en, time: int=10 ) '
					'-> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch_alerts( self, address: str, language_code: str = 'en',
			time: int = 10 ) -> Dict[ str, Any ] | None:
		'''
		
			Purpose:
			--------
			Retrieve public weather alerts for an address.
		
			Returns:
			--------
			Dict[str, Any] | None
		
		'''
		try:
			lat, lng = self._resolve_coordinates( address )
			params = {
					'location.latitude': lat,
					'location.longitude': lng,
					'languageCode': language_code,
			}
			return self._request(
				path='publicAlerts:lookup',
				params=params,
				time=time )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'GoogleWeather'
			exception.method = (
					'fetch_alerts( self, address: str, language_code: str=en, '
					'time: int=10 ) -> Dict[ str, Any ]'
			)
			raise exception

class NavalObservatory( Fetcher ):
	'''

		Purpose:
		--------
		Provides access to the U.S. Naval Observatory Astronomical Applications
		API for Celestial Navigation Data for Assumed Position and Time.

		This class is aligned to the current documented API endpoint:

			https://aa.usno.navy.mil/api/celnav

		with required parameters:

			- date
			- time
			- coords

		Referenced API Requirements:
		----------------------------
		Celestial Navigation Data for Assumed Position and Time:
			- Endpoint:
				https://aa.usno.navy.mil/api/celnav
			- Required parameters:
				- date (YYYY-MM-DD)
				- time (HH:MM or HH:MM:SS.S)
				- coords (latitude,longitude)

		Attributes:
		-----------
		base_url: Optional[str]
			Base USNO API URL.

		url: Optional[str]
			Resolved request URL.

		params: Optional[Dict[str, Any]]
			Request parameters.

		date_value: Optional[str]
			Request date in YYYY-MM-DD format.

		time_value: Optional[str]
			Request UT1/UTC time in HH:MM or HH:MM:SS.S format.

		latitude: Optional[float]
			Latitude in decimal degrees, north positive.

		longitude: Optional[float]
			Longitude in decimal degrees, east positive.

		location_label: Optional[str]
			Optional display label retained client-side.

		agents: Optional[str]
			User-Agent string.

		Methods:
		--------
		__init__() -> None
			Initialize fetcher defaults.

		__dir__() -> List[str]
			Provide ordered member visibility.

		fetch_celnav(...) -> Dict[str, Any] | None
			Fetch celestial navigation data from the USNO API.

		fetch(...) -> Dict[str, Any] | None
			Unified dispatcher for the celnav service.

		create_schema(...) -> Dict[str, str] | None
			Construct a dynamic tool schema.

	'''
	base_url: Optional[ str ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	date_value: Optional[ str ]
	time_value: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	location_label: Optional[ str ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		'''
			Purpose:
			--------
			Initialize the Naval Observatory fetcher with current API defaults.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		'''
		super( ).__init__( )
		self.headers = { }
		self.base_url = 'https://aa.usno.navy.mil/api'
		self.url = None
		self.params = { }
		self.date_value = ''
		self.time_value = ''
		self.latitude = 38.9072
		self.longitude = -77.0369
		self.location_label = ''
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		'''
			Purpose:
			--------
			Provide ordered member visibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		'''
		return [
				'base_url',
				'url',
				'params',
				'date_value',
				'time_value',
				'latitude',
				'longitude',
				'location_label',
				'fetch_celnav',
				'fetch',
				'create_schema'
		]
	
	def fetch_celnav( self, date_value: str, time_value: str, latitude: float,
			longitude: float, location_label: str = '',
			time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Fetch celestial navigation data for an assumed position and time.

			Parameters:
			-----------
			date_value (str):
				Date in YYYY-MM-DD format. Example: 2026-03-15

			time_value (str):
				UT1/UTC time in HH:MM or HH:MM:SS.S format.
				Examples:
				- 16:11
				- 16:11:00
				- 16:11:00.0

			latitude (float):
				Latitude in decimal degrees. North positive.

			longitude (float):
				Longitude in decimal degrees. East positive.

			location_label (str):
				Optional label preserved in the returned result for UI display.

			time (int):
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			throw_if( 'date_value', date_value )
			throw_if( 'time_value', time_value )
			
			self.date_value = str( date_value ).strip( )
			self.time_value = str( time_value ).strip( )
			self.latitude = float( latitude )
			self.longitude = float( longitude )
			self.location_label = str( location_label or '' ).strip( )
			
			self.url = f'{self.base_url}/celnav'
			self.params = {
					'date': self.date_value,
					'time': self.time_value,
					'coords': f'{self.latitude},{self.longitude}'
			}
			
			self.response = requests.get(
				url=self.url,
				params=self.params,
				headers=self.headers,
				timeout=int( time )
			)
			self.response.raise_for_status( )
			
			payload = self.response.json( ) or { }
			
			return {
					'mode': 'celnav',
					'url': self.url,
					'params': self.params,
					'location_label': self.location_label,
					'data': payload
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NavalObservatory'
			exception.method = (
					'fetch_celnav( self, date_value: str, time_value: str, '
					'latitude: float, longitude: float, location_label: str=, '
					'time: int=20 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch( self, mode: str = 'celnav', date_value: str = '',
			time_value: str = '', latitude: float = 0.0, longitude: float = 0.0,
			location_label: str = '', time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Unified dispatcher for Naval Observatory requests.

			Parameters:
			-----------
			mode (str):
				Currently supported:
				- celnav

			date_value (str):
				Date in YYYY-MM-DD format.

			time_value (str):
				Time in HH:MM or HH:MM:SS.S format.

			latitude (float):
				Latitude in decimal degrees.

			longitude (float):
				Longitude in decimal degrees.

			location_label (str):
				Optional display label.

			time (int):
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			active_mode = str( mode or 'celnav' ).strip( ).lower( )
			
			if active_mode == 'celnav':
				return self.fetch_celnav(
					date_value=date_value,
					time_value=time_value,
					latitude=latitude,
					longitude=longitude,
					location_label=location_label,
					time=time
				)
			
			raise ValueError( "Unsupported mode. Use 'celnav'." )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'NavalObservatory'
			exception.method = (
					'fetch( self, mode: str=celnav, date_value: str=, time_value: str=, '
					'latitude: float=0.0, longitude: float=0.0, location_label: str=, '
					'time: int=20 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict,
			required: list[ str ] ) -> Dict[ str, str ] | None:
		'''
			Purpose:
			--------
			Construct and return a fully dynamic OpenAI Tool API schema definition.

			Parameters:
			-----------
			function (str):
				The function name exposed to the LLM.

			tool (str):
				The underlying system or service the function wraps.

			description (str):
				Precise explanation of what the function does.

			parameters (dict):
				A dictionary defining parameter names and JSON schema descriptors.

			required (list[str]):
				List of required parameter names.

			Returns:
			--------
			Dict[str, str] | None
		'''
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return {
					'name': function.strip( ),
					'description': f'{description.strip( )} This function uses the {tool.strip( )} service.',
					'parameters': {
							'type': 'object',
							'properties': parameters,
							'required': required
					}
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NavalObservatory'
			exception.method = (
					'create_schema( self, function: str, tool: str, description: str, '
					'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]'
			)
			raise exception

class SatelliteCenter( Fetcher ):
	'''

		Purpose:
		--------
		Provides access to NASA SSCWeb / Satellite Situation Center data for
		observatories, ground stations, and location trajectories.

	'''
	ssc: Optional[ SscWs ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	observatories: Optional[ List[ Dict[ str, Any ] ] ]
	ground_stations: Optional[ List[ Dict[ str, Any ] ] ]
	timeout: Optional[ int ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.ssc = None
		self.url = 'https://sscweb.gsfc.nasa.gov/WS/sscr/2'
		self.params = { }
		self.observatories = [ ]
		self.ground_stations = [ ]
		self.timeout = 20
		self.headers = { }
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		if 'Accept' not in self.headers:
			self.headers[ 'Accept' ] = 'application/json'
	
	def __dir__( self ) -> List[ str ]:
		return [
				'url',
				'timeout',
				'headers',
				'fetch_observatories',
				'fetch_ground_stations',
				'fetch_locations',
				'fetch',
		]
	
	def fetch_observatories( self ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Get descriptions of the observatories available from SSC.

			Returns:
			--------
			Dict[str, Any] | None

		"""
		try:
			self.ssc = SscWs( user_agent=self.agents, timeout=self.timeout )
			result = self.ssc.get_observatories( )
			return result
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'SatelliteCenter'
			exception.method = 'fetch_observatories( self ) -> Dict[ str, Any ]'
			raise exception
	
	def fetch_ground_stations( self ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Get descriptions of the ground stations available from SSC.

			Returns:
			--------
			Dict[str, Any] | None

		"""
		try:
			self.ssc = SscWs( user_agent=self.agents, timeout=self.timeout )
			result = self.ssc.get_ground_stations( )
			return result
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'SatelliteCenter'
			exception.method = 'fetch_ground_stations( self ) -> Dict[ str, Any ]'
			raise exception
	
	def fetch_locations( self, observatories: str, start_time: str, end_time: str,
			coordinate_systems: str = 'gse', resolution_factor: int = 1,
			time: int = 20 ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Get location data for one or more observatories over a time range using
			the documented SSC REST GET endpoint.

			Parameters:
			-----------
			observatories:
				Comma-separated observatory identifiers such as "iss" or "mms1,mms2".
			start_time:
				ISO 8601 UTC start like "2026-03-15T00:00:00Z".
			end_time:
				ISO 8601 UTC end like "2026-03-15T02:00:00Z".
			coordinate_systems:
				Comma-separated coordinate systems such as "gse", "geo", "gsm".
			resolution_factor:
				Return one out of every N values.
			time:
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None

		"""
		try:
			throw_if( 'observatories', observatories )
			throw_if( 'start_time', start_time )
			throw_if( 'end_time', end_time )
			
			self.timeout = int( time )
			
			obs = observatories.strip( )
			time_range = f'{start_time.strip( )},{end_time.strip( )}'
			coords = (coordinate_systems or 'gse').strip( )
			
			request_url = (
					f'{self.url}/locations/'
					f'{obs}/'
					f'{time_range}/'
					f'{coords}/'
			)
			
			self.params = {
					'resolutionFactor': max( 1, int( resolution_factor ) )
			}
			
			self.response = requests.get(
				url=request_url,
				params=self.params,
				headers=self.headers,
				timeout=self.timeout )
			
			self.response.raise_for_status( )
			return self.response.json( )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'SatelliteCenter'
			exception.method = (
					'fetch_locations( self, observatories: str, start_time: str, '
					'end_time: str, coordinate_systems: str=gse, '
					'resolution_factor: int=1, time: int=20 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch( self, mode: str = 'observatories', query: str = '', start_time: str = '',
			end_time: str = '', coordinate_systems: str = 'gse', resolution_factor: int = 1,
			time: int = 20 ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Unified dispatch method for Satellite Center requests.

			Returns:
			--------
			Dict[str, Any] | None

		"""
		try:
			active_mode = (mode or 'observatories').strip( ).lower( )
			
			if active_mode == 'observatories':
				return self.fetch_observatories( )
			
			if active_mode == 'ground_stations':
				return self.fetch_ground_stations( )
			
			if active_mode == 'locations':
				return self.fetch_locations( observatories=query, start_time=start_time,
					end_time=end_time, coordinate_systems=coordinate_systems,
					resolution_factor=resolution_factor, time=time )
			
			raise ValueError(
				"Unsupported mode. Use 'observatories', 'ground_stations', or 'locations'."
			)
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'SatelliteCenter'
			exception.method = (
					'fetch( self, mode: str=observatories, query: str=, '
					'start_time: str=, end_time: str=, coordinate_systems: str=gse, '
					'resolution_factor: int=1, time: int=20 ) -> Dict[ str, Any ]'
			)
			raise exception

class EarthObservatory( Fetcher ):
	'''

		Purpose:
		--------
		NASA Earth Observatory's Natural Event Tracker (EONET) allows users to access imagery,
		often in near real-time (NRT), of natural events such as dust storms, forest fires, and
		tropical cyclones—empowering people all across the planet to locate, track, and potentially
		prepare for and manage events that affect communities in their paths.
		Version 3 API for events, categories, sources, and layers.

		This class is aligned to the current documented EONET v3 API and supports:
		- events
		- categories
		- sources
		- layers

		Referenced API Requirements:
		----------------------------
		Base:
			https://eonet.gsfc.nasa.gov/api/v3

		Events:
			https://eonet.gsfc.nasa.gov/api/v3/events
			Optional parameters:
				- source
				- category
				- status
				- limit
				- days
				- start
				- end

		Categories:
			https://eonet.gsfc.nasa.gov/api/v3/categories

		Sources:
			https://eonet.gsfc.nasa.gov/api/v3/sources

		Layers:
			https://eonet.gsfc.nasa.gov/api/v3/layers
			Optional category-specific path:
				https://eonet.gsfc.nasa.gov/api/v3/layers/{category}

		Attributes:
		-----------
		base_url: Optional[str]
			Base EONET API URL.

		url: Optional[str]
			Resolved request URL.

		params: Optional[Dict[str, Any]]
			Request parameters.

		mode: Optional[str]
			Selected API mode.

		query: Optional[str]
			Reserved generic query field.

		status: Optional[str]
			Event status filter.

		category: Optional[str]
			Event or layer category filter.

		source: Optional[str]
			Event source filter.

		days: Optional[int]
			Prior-day filter.

		limit: Optional[int]
			Returned record limit.

		start_date: Optional[str]
			Event-range start date in YYYY-MM-DD format.

		end_date: Optional[str]
			Event-range end date in YYYY-MM-DD format.

		agents: Optional[str]
			User-Agent string.

		Methods:
		--------
		__init__() -> None
			Initialize fetcher defaults.

		__dir__() -> List[str]
			Provide ordered member visibility.

		fetch_events(...) -> Dict[str, Any] | None
			Fetch event records.

		fetch_categories() -> Dict[str, Any] | None
			Fetch category metadata.

		fetch_sources() -> Dict[str, Any] | None
			Fetch source metadata.

		fetch_layers(...) -> Dict[str, Any] | None
			Fetch layer metadata, optionally filtered by category.

		fetch(...) -> Dict[str, Any] | None
			Unified dispatcher.

		create_schema(...) -> Dict[str, str] | None
			Construct a dynamic tool schema.

	'''
	base_url: Optional[ str ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	mode: Optional[ str ]
	status: Optional[ str ]
	category: Optional[ str ]
	source: Optional[ str ]
	days: Optional[ int ]
	limit: Optional[ int ]
	start_date: Optional[ str ]
	end_date: Optional[ str ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		'''
			Purpose:
			--------
			Initialize the EONET fetcher with current API defaults.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		'''
		super( ).__init__( )
		self.headers = { }
		self.base_url = 'https://eonet.gsfc.nasa.gov/api/v3'
		self.url = None
		self.params = { }
		self.mode = 'events'
		self.status = 'open'
		self.category = ''
		self.source = ''
		self.days = 30
		self.limit = 20
		self.start_date = ''
		self.end_date = ''
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		'''
			Purpose:
			--------
			Provide ordered member visibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		'''
		return [
				'base_url',
				'url',
				'params',
				'mode',
				'status',
				'category',
				'source',
				'days',
				'limit',
				'start_date',
				'end_date',
				'fetch_events',
				'fetch_categories',
				'fetch_sources',
				'fetch_layers',
				'fetch',
				'create_schema'
		]
	
	def fetch_events( self, status: str = 'open', category: str = '',
			source: str = '', limit: int = 20, days: int = 30,
			start_date: str = '', end_date: str = '', time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Fetch EONET events using documented v3 filters.

			Parameters:
			-----------
			status (str):
				Event status filter. Typical values: open, closed, all.

			category (str):
				Optional category slug or comma-separated category list.

			source (str):
				Optional source id or comma-separated source ids.

			limit (int):
				Maximum number of events to return.

			days (int):
				Number of prior days, including today, from which to return events.

			start_date (str):
				Optional inclusive start date in YYYY-MM-DD format.

			end_date (str):
				Optional inclusive end date in YYYY-MM-DD format.

			time (int):
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			self.mode = 'events'
			self.status = str( status or 'open' ).strip( ).lower( )
			self.category = str( category or '' ).strip( )
			self.source = str( source or '' ).strip( )
			self.limit = int( limit )
			self.days = int( days )
			self.start_date = str( start_date or '' ).strip( )
			self.end_date = str( end_date or '' ).strip( )
			self.url = f'{self.base_url}/events'
			self.params = { }
			
			if self.status:
				self.params[ 'status' ] = self.status
			
			if self.category:
				self.params[ 'category' ] = self.category
			
			if self.source:
				self.params[ 'source' ] = self.source
			
			if self.limit > 0:
				self.params[ 'limit' ] = self.limit
			
			if self.start_date and self.end_date:
				self.params[ 'start' ] = self.start_date
				self.params[ 'end' ] = self.end_date
			elif self.days > 0:
				self.params[ 'days' ] = self.days
			
			self.response = requests.get(
				url=self.url,
				params=self.params,
				headers=self.headers,
				timeout=int( time )
			)
			self.response.raise_for_status( )
			
			payload = self.response.json( ) or { }
			
			return {
					'mode': self.mode,
					'url': self.url,
					'params': self.params,
					'events': payload.get( 'events', [ ] ),
					'title': payload.get( 'title', '' ),
					'description': payload.get( 'description', '' )
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EarthObservatory'
			exception.method = (
					'fetch_events( self, status: str=open, category: str=, source: str=, '
					'limit: int=20, days: int=30, start_date: str=, end_date: str=, '
					'time: int=20 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch_categories( self, time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Fetch EONET category metadata.

			Parameters:
			-----------
			time (int):
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			self.mode = 'categories'
			self.url = f'{self.base_url}/categories'
			self.params = { }
			
			self.response = requests.get(
				url=self.url,
				params=self.params,
				headers=self.headers,
				timeout=int( time )
			)
			self.response.raise_for_status( )
			
			payload = self.response.json( ) or { }
			
			return {
					'mode': self.mode,
					'url': self.url,
					'params': self.params,
					'categories': payload.get( 'categories', [ ] ),
					'title': payload.get( 'title', '' ),
					'description': payload.get( 'description', '' )
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EarthObservatory'
			exception.method = 'fetch_categories( self, time: int=20 ) -> Dict[ str, Any ]'
			raise exception
	
	def fetch_sources( self, time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Fetch EONET source metadata.

			Parameters:
			-----------
			time (int):
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			self.mode = 'sources'
			self.url = f'{self.base_url}/sources'
			self.params = { }
			
			self.response = requests.get(
				url=self.url,
				params=self.params,
				headers=self.headers,
				timeout=int( time )
			)
			self.response.raise_for_status( )
			
			payload = self.response.json( ) or { }
			
			return {
					'mode': self.mode,
					'url': self.url,
					'params': self.params,
					'sources': payload.get( 'sources', [ ] ),
					'title': payload.get( 'title', '' ),
					'description': payload.get( 'description', '' )
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EarthObservatory'
			exception.method = 'fetch_sources( self, time: int=20 ) -> Dict[ str, Any ]'
			raise exception
	
	def fetch_layers( self, category: str = '', time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Fetch EONET layer metadata, optionally scoped to a category.

			Parameters:
			-----------
			category (str):
				Optional category slug.

			time (int):
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			self.mode = 'layers'
			self.category = str( category or '' ).strip( )
			
			if self.category:
				self.url = f'{self.base_url}/layers/{self.category}'
			else:
				self.url = f'{self.base_url}/layers'
			
			self.params = { }
			
			self.response = requests.get(
				url=self.url,
				params=self.params,
				headers=self.headers,
				timeout=int( time )
			)
			self.response.raise_for_status( )
			
			payload = self.response.json( ) or { }
			
			return {
					'mode': self.mode,
					'url': self.url,
					'params': self.params,
					'category': self.category,
					'layers': payload.get( 'layers', [ ] ),
					'title': payload.get( 'title', '' ),
					'description': payload.get( 'description', '' )
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EarthObservatory'
			exception.method = (
					'fetch_layers( self, category: str=, time: int=20 ) '
					'-> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch( self, mode: str = 'events', status: str = 'open',
			category: str = '', source: str = '', limit: int = 20,
			days: int = 30, start_date: str = '', end_date: str = '',
			time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Unified dispatcher for EONET v3 operations.

			Parameters:
			-----------
			mode (str):
				One of: events, categories, sources, layers

			status (str):
				Event status filter for events mode.

			category (str):
				Category filter for events mode or category path for layers mode.

			source (str):
				Source filter for events mode.

			limit (int):
				Event record limit for events mode.

			days (int):
				Prior-day window for events mode.

			start_date (str):
				Optional start date in YYYY-MM-DD format.

			end_date (str):
				Optional end date in YYYY-MM-DD format.

			time (int):
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			active_mode = (mode or 'events').strip( ).lower( )
			
			if active_mode == 'events':
				return self.fetch_events(
					status=status,
					category=category,
					source=source,
					limit=limit,
					days=days,
					start_date=start_date,
					end_date=end_date,
					time=time
				)
			
			if active_mode == 'categories':
				return self.fetch_categories( time=time )
			
			if active_mode == 'sources':
				return self.fetch_sources( time=time )
			
			if active_mode == 'layers':
				return self.fetch_layers( category=category, time=time )
			
			raise ValueError(
				"Unsupported mode. Use 'events', 'categories', 'sources', or 'layers'."
			)
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'EarthObservatory'
			exception.method = (
					'fetch( self, mode: str=events, status: str=open, category: str=, '
					'source: str=, limit: int=20, days: int=30, start_date: str=, '
					'end_date: str=, time: int=20 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict,
			required: list[ str ] ) -> Dict[ str, str ] | None:
		'''
			Purpose:
			--------
			Construct and return a fully dynamic OpenAI Tool API schema definition.

			Parameters:
			-----------
			function (str):
				The function name exposed to the LLM.

			tool (str):
				The underlying system or service the function wraps.

			description (str):
				Precise explanation of what the function does.

			parameters (dict):
				A dictionary defining parameter names and JSON schema descriptors.

			required (list[str]):
				List of required parameter names.

			Returns:
			--------
			Dict[str, str] | None
		'''
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return {
					'name': function.strip( ),
					'description': f'{description.strip( )} This function uses the {tool.strip( )} service.',
					'parameters': {
							'type': 'object',
							'properties': parameters,
							'required': required
					}
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'EarthObservatory'
			exception.method = (
					'create_schema( self, function: str, tool: str, description: str, '
					'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]'
			)
			raise exception
			
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
	utc_time: Optional[ dt.time ]
	local_time: Optional[ dt.time ]
	params: Optional[ Dict[ str, Any ] ]
	era: Optional[ str ]
	year: Optional[ str ]
	month: Optional[ str ]
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
		self.headers[ 'User-Agent' ] = self.agents
		self.era = None
	
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
			raise exception
			
	
	def fetch_mercator_map( self , ccrs=None ):
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
			raise exception
			
		
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
			raise exception

class NearbyObjects( Fetcher ):
	'''

		Purpose:
		--------
		Provides access to current JPL SSD / CNEOS APIs relevant to near-Earth
		objects, close approaches, and human-accessible target screening.

		This class is aligned to the current documented APIs and supports the
		following modes:

		- close_approaches
		- object_lookup
		- nhats_summary
		- nhats_object
		- fireballs

		Referenced API Requirements:
		----------------------------
		CAD API:
			GET https://ssd-api.jpl.nasa.gov/cad.api
			Common parameters used here:
				- date-min
				- date-max
				- dist-max
				- body
				- sort
				- limit

		SBDB API:
			GET https://ssd-api.jpl.nasa.gov/sbdb.api
			One and only one of:
				- sstr
				- spk
				- des
			Optional parameters used here:
				- phys-par
				- ca-data
				- ca-body
				- discovery

		NHATS API:
			GET https://ssd-api.jpl.nasa.gov/nhats.api
			Summary filters optionally include:
				- dv
				- dur
				- stay
				- launch
				- h
				- occ
			Object-specific details use:
				- des

		Fireball API:
			GET https://ssd-api.jpl.nasa.gov/fireball.api
			Common parameters used here:
				- date-min
				- limit

		Attributes:
		-----------
		base_url: Optional[str]
			Base SSD API URL.

		url: Optional[str]
			Resolved request URL.

		params: Optional[Dict[str, Any]]
			Request parameters for the active call.

		mode: Optional[str]
			Selected operating mode.

		start_date: Optional[str]
			Inclusive start date in YYYY-MM-DD format.

		end_date: Optional[str]
			Inclusive end date in YYYY-MM-DD format.

		query: Optional[str]
			Generic object lookup string or designation.

		dist_max: Optional[str]
			Close-approach maximum distance filter.

		body: Optional[str]
			Close-approach body selector, typically Earth.

		sort: Optional[str]
			CAD sorting key.

		limit: Optional[int]
			Result limit.

		agents: Optional[str]
			User-Agent string.

		Methods:
		--------
		__init__() -> None
			Initialize fetcher defaults.

		__dir__() -> List[str]
			Provide ordered member visibility.

		fetch_close_approaches(...) -> Dict[str, Any] | None
			Fetch close-approach records from the CAD API.

		fetch_object_lookup(...) -> Dict[str, Any] | None
			Fetch detailed object information from the SBDB API.

		fetch_nhats_summary(...) -> Dict[str, Any] | None
			Fetch NHATS summary rows using screening constraints.

		fetch_nhats_object(...) -> Dict[str, Any] | None
			Fetch NHATS details for a single designation.

		fetch_fireballs(...) -> Dict[str, Any] | None
			Fetch fireball atmospheric impact records.

		fetch(...) -> Dict[str, Any] | None
			Unified dispatcher for NEO-related operations.

		create_schema(...) -> Dict[str, str] | None
			Construct a dynamic tool schema.

	'''
	base_url: Optional[ str ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	mode: Optional[ str ]
	start_date: Optional[ str ]
	end_date: Optional[ str ]
	query: Optional[ str ]
	dist_max: Optional[ str ]
	body: Optional[ str ]
	sort: Optional[ str ]
	limit: Optional[ int ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		'''
			Purpose:
			--------
			Initialize the NearbyObjects fetcher with current JPL SSD defaults.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		'''
		super( ).__init__( )
		self.headers = { }
		self.base_url = 'https://ssd-api.jpl.nasa.gov'
		self.url = None
		self.params = { }
		self.mode = 'close_approaches'
		self.start_date = ''
		self.end_date = ''
		self.query = ''
		self.dist_max = '10LD'
		self.body = 'Earth'
		self.sort = 'date'
		self.limit = 20
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		'''
			Purpose:
			--------
			Provide ordered member visibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		'''
		return [
				'base_url',
				'url',
				'params',
				'mode',
				'start_date',
				'end_date',
				'query',
				'dist_max',
				'body',
				'sort',
				'limit',
				'fetch_close_approaches',
				'fetch_object_lookup',
				'fetch_nhats_summary',
				'fetch_nhats_object',
				'fetch_fireballs',
				'fetch',
				'create_schema'
		]
	
	def fetch_close_approaches( self, start_date: str, end_date: str,
			dist_max: str = '10LD', body: str = 'Earth',
			sort: str = 'date', limit: int = 20,
			time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Fetch close-approach data from the JPL SB Close Approach Data API.

			Parameters:
			-----------
			start_date (str):
				Inclusive lower date bound in YYYY-MM-DD format.

			end_date (str):
				Inclusive upper date bound in YYYY-MM-DD format.

			dist_max (str):
				Maximum close-approach distance. Examples:
				- 10LD
				- 0.05AU

			body (str):
				Close-approach body selector. Example values include Earth, Moon,
				Mars, Juptr.

			sort (str):
				Sort key for the returned records. Example values include date and dist.

			limit (int):
				Maximum number of rows to return.

			time (int):
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			throw_if( 'start_date', start_date )
			throw_if( 'end_date', end_date )
			
			self.mode = 'close_approaches'
			self.start_date = str( start_date ).strip( )
			self.end_date = str( end_date ).strip( )
			self.dist_max = str( dist_max or '10LD' ).strip( )
			self.body = str( body or 'Earth' ).strip( )
			self.sort = str( sort or 'date' ).strip( )
			self.limit = int( limit )
			
			self.url = f'{self.base_url}/cad.api'
			self.params = {
					'date-min': self.start_date,
					'date-max': self.end_date,
					'dist-max': self.dist_max,
					'body': self.body,
					'sort': self.sort,
					'limit': self.limit
			}
			
			self.response = requests.get(
				url=self.url,
				params=self.params,
				headers=self.headers,
				timeout=int( time )
			)
			self.response.raise_for_status( )
			payload = self.response.json( ) or { }
			
			return {
					'mode': self.mode,
					'url': self.url,
					'params': self.params,
					'count': payload.get( 'count', 0 ),
					'fields': payload.get( 'fields', [ ] ),
					'data': payload.get( 'data', [ ] ),
					'signature': payload.get( 'signature', { } )
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NearbyObjects'
			exception.method = (
					'fetch_close_approaches( self, start_date: str, end_date: str, '
					'dist_max: str=10LD, body: str=Earth, sort: str=date, '
					'limit: int=20, time: int=20 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch_object_lookup( self, query: str, query_type: str = 'sstr',
			include_physical: bool = True, include_close_approaches: bool = True,
			ca_body: str = 'Earth', include_discovery: bool = True,
			time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Fetch a single-object record from the JPL SBDB API.

			Parameters:
			-----------
			query (str):
				Object identifier or name. Examples:
				- Apophis
				- Eros
				- 2000 SG344
				- 99942

			query_type (str):
				Exactly one of:
				- sstr
				- spk
				- des

			include_physical (bool):
				If True, request physical parameters.

			include_close_approaches (bool):
				If True, request close-approach data.

			ca_body (str):
				Body filter for close-approach data, typically Earth.

			include_discovery (bool):
				If True, request discovery circumstances when available.

			time (int):
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			throw_if( 'query', query )
			throw_if( 'query_type', query_type )
			
			self.mode = 'object_lookup'
			self.query = str( query ).strip( )
			active_type = str( query_type ).strip( ).lower( )
			
			if active_type not in [ 'sstr', 'spk', 'des' ]:
				raise ValueError( "query_type must be 'sstr', 'spk', or 'des'." )
			
			self.url = f'{self.base_url}/sbdb.api'
			self.params = {
					active_type: self.query,
					'phys-par': '1' if bool( include_physical ) else '0',
					'ca-data': '1' if bool( include_close_approaches ) else '0',
					'discovery': '1' if bool( include_discovery ) else '0'
			}
			
			if include_close_approaches and str( ca_body or '' ).strip( ):
				self.params[ 'ca-body' ] = str( ca_body ).strip( )
			
			self.response = requests.get(
				url=self.url,
				params=self.params,
				headers=self.headers,
				timeout=int( time )
			)
			self.response.raise_for_status( )
			payload = self.response.json( ) or { }
			
			return {
					'mode': self.mode,
					'url': self.url,
					'params': self.params,
					'data': payload
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NearbyObjects'
			exception.method = (
					'fetch_object_lookup( self, query: str, query_type: str=sstr, '
					'include_physical: bool=True, include_close_approaches: bool=True, '
					'ca_body: str=Earth, include_discovery: bool=True, time: int=20 ) '
					'-> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch_nhats_summary( self, dv: float = 6.0, dur: int = 360,
			stay: int = 8, launch: str = '2020-2045',
			h: float = 26.0, occ: int = 7,
			time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Fetch NHATS summary data using standard screening constraints.

			Parameters:
			-----------
			dv (float):
				Maximum total delta-V in km/s.

			dur (int):
				Maximum mission duration in days.

			stay (int):
				Minimum stay duration in days.

			launch (str):
				Launch window year range. Example: 2020-2045.

			h (float):
				Maximum H magnitude.

			occ (int):
				Maximum Orbit Condition Code.

			time (int):
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			self.mode = 'nhats_summary'
			self.url = f'{self.base_url}/nhats.api'
			self.params = {
					'dv': float( dv ),
					'dur': int( dur ),
					'stay': int( stay ),
					'launch': str( launch ).strip( ),
					'h': float( h ),
					'occ': int( occ )
			}
			
			self.response = requests.get(
				url=self.url,
				params=self.params,
				headers=self.headers,
				timeout=int( time )
			)
			self.response.raise_for_status( )
			payload = self.response.json( ) or { }
			
			return {
					'mode': self.mode,
					'url': self.url,
					'params': self.params,
					'data': payload
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NearbyObjects'
			exception.method = (
					'fetch_nhats_summary( self, dv: float=6.0, dur: int=360, stay: int=8, '
					'launch: str=2020-2045, h: float=26.0, occ: int=7, time: int=20 ) '
					'-> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch_nhats_object( self, designation: str, dv: float = 6.0,
			dur: int = 360, stay: int = 8, launch: str = '2020-2045',
			time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Fetch NHATS details for a single object designation.

			Parameters:
			-----------
			designation (str):
				Designation of the NEO. Examples:
				- 99942
				- 2000 SG344

			dv (float):
				Maximum total delta-V in km/s.

			dur (int):
				Maximum mission duration in days.

			stay (int):
				Minimum stay duration in days.

			launch (str):
				Launch window year range.

			time (int):
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			throw_if( 'designation', designation )
			
			self.mode = 'nhats_object'
			self.query = str( designation ).strip( )
			self.url = f'{self.base_url}/nhats.api'
			self.params = {
					'des': self.query,
					'dv': float( dv ),
					'dur': int( dur ),
					'stay': int( stay ),
					'launch': str( launch ).strip( )
			}
			
			self.response = requests.get(
				url=self.url,
				params=self.params,
				headers=self.headers,
				timeout=int( time )
			)
			self.response.raise_for_status( )
			payload = self.response.json( ) or { }
			
			return {
					'mode': self.mode,
					'url': self.url,
					'params': self.params,
					'data': payload
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NearbyObjects'
			exception.method = (
					'fetch_nhats_object( self, designation: str, dv: float=6.0, '
					'dur: int=360, stay: int=8, launch: str=2020-2045, time: int=20 ) '
					'-> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch_fireballs( self, date_min: str = '', limit: int = 20,
			time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Fetch atmospheric fireball records from the JPL Fireball API.

			Parameters:
			-----------
			date_min (str):
				Optional lower date bound in YYYY-MM-DD or
				YYYY-MM-DDThh:mm:ss format.

			limit (int):
				Maximum number of rows to return.

			time (int):
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			self.mode = 'fireballs'
			self.url = f'{self.base_url}/fireball.api'
			self.params = { 'limit': int( limit ) }
			
			if str( date_min or '' ).strip( ):
				self.params[ 'date-min' ] = str( date_min ).strip( )
			
			self.response = requests.get(
				url=self.url,
				params=self.params,
				headers=self.headers,
				timeout=int( time )
			)
			self.response.raise_for_status( )
			payload = self.response.json( ) or { }
			
			return {
					'mode': self.mode,
					'url': self.url,
					'params': self.params,
					'count': payload.get( 'count', 0 ),
					'fields': payload.get( 'fields', [ ] ),
					'data': payload.get( 'data', [ ] ),
					'signature': payload.get( 'signature', { } )
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NearbyObjects'
			exception.method = (
					'fetch_fireballs( self, date_min: str=, limit: int=20, '
					'time: int=20 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch( self, mode: str = 'close_approaches', start_date: str = '',
			end_date: str = '', query: str = '', query_type: str = 'sstr',
			dist_max: str = '10LD', body: str = 'Earth', sort: str = 'date',
			limit: int = 20, dv: float = 6.0, dur: int = 360,
			stay: int = 8, launch: str = '2020-2045', h: float = 26.0,
			occ: int = 7, include_physical: bool = True,
			include_close_approaches: bool = True, ca_body: str = 'Earth',
			include_discovery: bool = True, time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Unified dispatcher for JPL SSD / CNEOS NEO-related endpoints.

			Parameters:
			-----------
			mode (str):
				One of:
				- close_approaches
				- object_lookup
				- nhats_summary
				- nhats_object
				- fireballs

			start_date (str):
				Date lower bound for close_approaches.

			end_date (str):
				Date upper bound for close_approaches.

			query (str):
				Object query or designation for object_lookup or nhats_object.

			query_type (str):
				Object lookup selector: sstr, spk, or des.

			dist_max (str):
				Close-approach distance ceiling.

			body (str):
				Close-approach body selector.

			sort (str):
				Close-approach sort key.

			limit (int):
				Result limit.

			dv (float):
				NHATS delta-V filter.

			dur (int):
				NHATS duration filter.

			stay (int):
				NHATS stay filter.

			launch (str):
				NHATS launch window.

			h (float):
				NHATS H-magnitude filter.

			occ (int):
				NHATS OCC filter.

			include_physical (bool):
				SBDB physical-parameter switch.

			include_close_approaches (bool):
				SBDB close-approach section switch.

			ca_body (str):
				SBDB close-approach body filter.

			include_discovery (bool):
				SBDB discovery-data switch.

			time (int):
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			active_mode = str( mode or 'close_approaches' ).strip( ).lower( )
			
			if active_mode == 'close_approaches':
				return self.fetch_close_approaches(
					start_date=start_date,
					end_date=end_date,
					dist_max=dist_max,
					body=body,
					sort=sort,
					limit=limit,
					time=time
				)
			
			if active_mode == 'object_lookup':
				return self.fetch_object_lookup(
					query=query,
					query_type=query_type,
					include_physical=include_physical,
					include_close_approaches=include_close_approaches,
					ca_body=ca_body,
					include_discovery=include_discovery,
					time=time
				)
			
			if active_mode == 'nhats_summary':
				return self.fetch_nhats_summary(
					dv=dv,
					dur=dur,
					stay=stay,
					launch=launch,
					h=h,
					occ=occ,
					time=time
				)
			
			if active_mode == 'nhats_object':
				return self.fetch_nhats_object(
					designation=query,
					dv=dv,
					dur=dur,
					stay=stay,
					launch=launch,
					time=time
				)
			
			if active_mode == 'fireballs':
				return self.fetch_fireballs(
					date_min=start_date,
					limit=limit,
					time=time
				)
			
			raise ValueError(
				"Unsupported mode. Use 'close_approaches', 'object_lookup', "
				"'nhats_summary', 'nhats_object', or 'fireballs'."
			)
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'NearbyObjects'
			exception.method = (
					'fetch( self, mode: str=close_approaches, start_date: str=, '
					'end_date: str=, query: str=, query_type: str=sstr, '
					'dist_max: str=10LD, body: str=Earth, sort: str=date, '
					'limit: int=20, dv: float=6.0, dur: int=360, stay: int=8, '
					'launch: str=2020-2045, h: float=26.0, occ: int=7, '
					'include_physical: bool=True, include_close_approaches: bool=True, '
					'ca_body: str=Earth, include_discovery: bool=True, '
					'time: int=20 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict,
			required: list[ str ] ) -> Dict[ str, str ] | None:
		'''
			Purpose:
			--------
			Construct and return a fully dynamic OpenAI Tool API schema definition.

			Parameters:
			-----------
			function (str):
				The function name exposed to the LLM.

			tool (str):
				The underlying system or service the function wraps.

			description (str):
				Precise explanation of what the function does.

			parameters (dict):
				A dictionary defining parameter names and JSON schema descriptors.

			required (list[str]):
				List of required parameter names.

			Returns:
			--------
			Dict[str, str] | None
		'''
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return {
					'name': function.strip( ),
					'description': f'{description.strip( )} This function uses the {tool.strip( )} service.',
					'parameters': {
							'type': 'object',
							'properties': parameters,
							'required': required
					}
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'NearbyObjects'
			exception.method = (
					'create_schema( self, function: str, tool: str, description: str, '
					'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]'
			)
			raise exception
			
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
			raise exception
			
	
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
			self.keywords = keywords
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
			raise exception
			
		
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
			raise exception

class SpaceWeather( Fetcher ):
	'''

		Purpose:
		--------
		Provides access to NASA DONKI space weather endpoints through the
		NASA Open APIs gateway.

		This class is aligned to the currently documented DONKI endpoints and
		supports the following modes:

		- cme
		- cme_analysis
		- gst
		- ips
		- flr
		- sep
		- mpc
		- rbe
		- hss
		- wsa_enlil
		- notifications

		Referenced API Requirements:
		----------------------------
		Base:
			https://api.nasa.gov/DONKI

		Common parameters:
			- startDate
			- endDate
			- api_key

		Endpoint-specific parameters supported here:
			- location
			- catalog
			- type
			- mostAccurateOnly
			- completeEntryOnly
			- speed
			- halfAngle
			- keyword

		Attributes:
		-----------
		base_url: Optional[str]
			Base DONKI API URL.

		api_key: Optional[str]
			NASA API key from configuration.

		url: Optional[str]
			Resolved endpoint URL.

		params: Optional[Dict[str, Any]]
			Request parameters sent to DONKI.

		mode: Optional[str]
			Selected DONKI endpoint mode.

		start_date: Optional[str]
			Inclusive start date in YYYY-MM-DD format.

		end_date: Optional[str]
			Inclusive end date in YYYY-MM-DD format.

		location: Optional[str]
			IPS endpoint location filter.

		catalog: Optional[str]
			CMEAnalysis or IPS catalog filter.

		notification_type: Optional[str]
			Notifications endpoint event type filter.

		limit_note: Optional[str]
			Reserved descriptive note.

		agents: Optional[str]
			User-Agent string.

		Methods:
		--------
		__init__() -> None
			Initialize fetcher defaults.

		__dir__() -> List[str]
			Provide ordered member visibility.

		fetch_endpoint(...) -> Dict[str, Any] | None
			Fetch a single DONKI endpoint with normalized parameters.

		fetch(...) -> Dict[str, Any] | None
			Unified dispatcher for DONKI modes.

		create_schema(...) -> Dict[str, str] | None
			Construct a dynamic tool schema.

	'''
	base_url: Optional[ str ]
	api_key: Optional[ str ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	mode: Optional[ str ]
	start_date: Optional[ str ]
	end_date: Optional[ str ]
	location: Optional[ str ]
	catalog: Optional[ str ]
	notification_type: Optional[ str ]
	limit_note: Optional[ str ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		'''
			Purpose:
			--------
			Initialize the DONKI fetcher with current endpoint defaults.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		'''
		super( ).__init__( )
		self.headers = { }
		self.base_url = 'https://api.nasa.gov/DONKI'
		self.api_key = cfg.NASA_API_KEY
		self.url = None
		self.params = { }
		self.mode = 'cme'
		self.start_date = ''
		self.end_date = ''
		self.location = 'ALL'
		self.catalog = 'ALL'
		self.notification_type = 'all'
		self.limit_note = None
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		'''
			Purpose:
			--------
			Provide ordered member visibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		'''
		return [
				'base_url',
				'api_key',
				'url',
				'params',
				'mode',
				'start_date',
				'end_date',
				'location',
				'catalog',
				'notification_type',
				'fetch_endpoint',
				'fetch',
				'create_schema'
		]
	
	def fetch_endpoint( self, endpoint: str, start_date: str, end_date: str,
			time: int = 20, location: str = '', catalog: str = '',
			notification_type: str = '', most_accurate_only: bool = True,
			complete_entry_only: bool = True, speed: int = 0,
			half_angle: int = 0, keyword: str = '',
			api_key: str | None = None ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Send a request to a specific DONKI endpoint and return normalized JSON.

			Parameters:
			-----------
			endpoint (str):
				DONKI endpoint path fragment.

			start_date (str):
				Inclusive start date in YYYY-MM-DD format.

			end_date (str):
				Inclusive end date in YYYY-MM-DD format.

			time (int):
				Request timeout in seconds.

			location (str):
				IPS location filter.

			catalog (str):
				CMEAnalysis or IPS catalog filter.

			notification_type (str):
				Notifications type filter.

			most_accurate_only (bool):
				CMEAnalysis filter.

			complete_entry_only (bool):
				CMEAnalysis filter.

			speed (int):
				CMEAnalysis lower-bound speed filter.

			half_angle (int):
				CMEAnalysis lower-bound half-angle filter.

			keyword (str):
				CMEAnalysis keyword filter.

			api_key (str | None):
				Optional runtime override for NASA API key.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			throw_if( 'endpoint', endpoint )
			throw_if( 'start_date', start_date )
			throw_if( 'end_date', end_date )
			
			active_key = str( api_key or self.api_key or '' ).strip( )
			if not active_key:
				raise ValueError( 'NASA API key is required for DONKI requests.' )
			
			self.url = f'{self.base_url}/{endpoint}'
			self.params = {
					'startDate': str( start_date ).strip( ),
					'endDate': str( end_date ).strip( ),
					'api_key': active_key
			}
			
			if endpoint == 'IPS' and location.strip( ):
				self.params[ 'location' ] = location.strip( )
			
			if endpoint == 'IPS' and catalog.strip( ):
				self.params[ 'catalog' ] = catalog.strip( )
			
			if endpoint == 'CMEAnalysis':
				self.params[ 'mostAccurateOnly' ] = str( bool( most_accurate_only ) ).lower( )
				self.params[ 'completeEntryOnly' ] = str( bool( complete_entry_only ) ).lower( )
				self.params[ 'speed' ] = int( speed )
				self.params[ 'halfAngle' ] = int( half_angle )
				
				if catalog.strip( ):
					self.params[ 'catalog' ] = catalog.strip( )
				
				if keyword.strip( ):
					self.params[ 'keyword' ] = keyword.strip( )
			
			if endpoint == 'notifications' and notification_type.strip( ):
				self.params[ 'type' ] = notification_type.strip( )
			
			self.response = requests.get(
				url=self.url,
				params=self.params,
				headers=self.headers,
				timeout=int( time )
			)
			self.response.raise_for_status( )
			payload = self.response.json( )
			
			return {
					'mode': self.mode,
					'endpoint': endpoint,
					'url': self.url,
					'params': self.params,
					'data': payload
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'SpaceWeather'
			exception.method = (
					'fetch_endpoint( self, endpoint: str, start_date: str, end_date: str, '
					'time: int=20, location: str=, catalog: str=, notification_type: str=, '
					'most_accurate_only: bool=True, complete_entry_only: bool=True, '
					'speed: int=0, half_angle: int=0, keyword: str=, '
					'api_key: str|None=None ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch( self, mode: str = 'cme', start_date: str = '', end_date: str = '',
			time: int = 20, location: str = 'ALL', catalog: str = 'ALL',
			notification_type: str = 'all', most_accurate_only: bool = True,
			complete_entry_only: bool = True, speed: int = 0,
			half_angle: int = 0, keyword: str = '',
			api_key: str | None = None ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Unified dispatcher for NASA DONKI endpoints.

			Parameters:
			-----------
			mode (str):
				One of:
				- cme
				- cme_analysis
				- gst
				- ips
				- flr
				- sep
				- mpc
				- rbe
				- hss
				- wsa_enlil
				- notifications

			start_date (str):
				Inclusive start date in YYYY-MM-DD format.

			end_date (str):
				Inclusive end date in YYYY-MM-DD format.

			time (int):
				Request timeout in seconds.

			location (str):
				IPS location filter.

			catalog (str):
				CMEAnalysis or IPS catalog filter.

			notification_type (str):
				Notifications type filter.

			most_accurate_only (bool):
				CMEAnalysis filter.

			complete_entry_only (bool):
				CMEAnalysis filter.

			speed (int):
				CMEAnalysis speed filter.

			half_angle (int):
				CMEAnalysis half-angle filter.

			keyword (str):
				CMEAnalysis keyword filter.

			api_key (str | None):
				Optional runtime override for NASA API key.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			active_mode = str( mode or 'cme' ).strip( ).lower( )
			self.mode = active_mode
			
			endpoint_map = {
					'cme': 'CME',
					'cme_analysis': 'CMEAnalysis',
					'gst': 'GST',
					'ips': 'IPS',
					'flr': 'FLR',
					'sep': 'SEP',
					'mpc': 'MPC',
					'rbe': 'RBE',
					'hss': 'HSS',
					'wsa_enlil': 'WSAEnlilSimulations',
					'notifications': 'notifications'
			}
			
			if active_mode not in endpoint_map:
				raise ValueError(
					"Unsupported mode. Use 'cme', 'cme_analysis', 'gst', 'ips', "
					"'flr', 'sep', 'mpc', 'rbe', 'hss', 'wsa_enlil', or 'notifications'."
				)
			
			return self.fetch_endpoint(
				endpoint=endpoint_map[ active_mode ],
				start_date=str( start_date ).strip( ),
				end_date=str( end_date ).strip( ),
				time=int( time ),
				location=str( location or 'ALL' ).strip( ),
				catalog=str( catalog or 'ALL' ).strip( ),
				notification_type=str( notification_type or 'all' ).strip( ),
				most_accurate_only=bool( most_accurate_only ),
				complete_entry_only=bool( complete_entry_only ),
				speed=int( speed ),
				half_angle=int( half_angle ),
				keyword=str( keyword or '' ).strip( ),
				api_key=api_key
			)
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'SpaceWeather'
			exception.method = (
					'fetch( self, mode: str=cme, start_date: str=, end_date: str=, '
					'time: int=20, location: str=ALL, catalog: str=ALL, '
					'notification_type: str=all, most_accurate_only: bool=True, '
					'complete_entry_only: bool=True, speed: int=0, half_angle: int=0, '
					'keyword: str=, api_key: str|None=None ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict,
			required: list[ str ] ) -> Dict[ str, str ] | None:
		'''
			Purpose:
			--------
			Construct and return a fully dynamic OpenAI Tool API schema definition.

			Parameters:
			-----------
			function (str):
				The function name exposed to the LLM.

			tool (str):
				The underlying system or service the function wraps.

			description (str):
				Precise explanation of what the function does.

			parameters (dict):
				A dictionary defining parameter names and JSON schema descriptors.

			required (list[str]):
				List of required parameter names.

			Returns:
			--------
			Dict[str, str] | None
		'''
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return {
					'name': function.strip( ),
					'description': f'{description.strip( )} This function uses the {tool.strip( )} service.',
					'parameters': {
							'type': 'object',
							'properties': parameters,
							'required': required
					}
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'SpaceWeather'
			exception.method = (
					'create_schema( self, function: str, tool: str, description: str, '
					'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]'
			)
			raise exception

class AstroCatalog( Fetcher ):
	'''

		Purpose:
		--------
		Provides structured access to the Open Astronomy Catalog API (OACAPI).

	'''
	base_url: Optional[ str ]
	format: Optional[ str ]
	name: Optional[ str ]
	declination: Optional[ str ]
	right_ascension: Optional[ str ]
	radius: Optional[ int ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ):
		super( ).__init__( )
		self.base_url = 'https://api.astrocats.space'
		self.format = 'json'
		self.name = None
		self.right_ascension = None
		self.declination = None
		self.radius = None
		self.params = { }
		self.headers = { }
		self.timeout = 20
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		if 'Accept' not in self.headers:
			self.headers[ 'Accept' ] = 'application/json'
	
	def __dir__( self ) -> List[ str ]:
		return [
				'base_url',
				'timeout',
				'headers',
				'fetch_object',
				'cone_search',
				'fetch',
		]
	
	def _normalize_attribute_path( self, quantity: str = '', attributes: str = '' ) -> str:
		"""
			Purpose:
			--------
			Build the OAC route path segment from quantity and attribute inputs.

			Returns:
			--------
			str
		"""
		parts: list[ str ] = [ ]
		if quantity and quantity.strip( ):
			parts.append( quantity.strip( ) )
		
		if attributes and attributes.strip( ):
			attr_parts = [ a.strip( ) for a in attributes.split( ',' ) if a.strip( ) ]
			parts.extend( attr_parts )
		
		return '/'.join( parts )
	
	def _parse_argument_string( self, argument_string: str ) -> Dict[ str, Any ]:
		"""
			Purpose:
			--------
			Parse a comma-separated or newline-separated list of OAC query
			arguments into a dictionary.

			Examples:
			---------
			band=R,time,e_magnitude,complete
		"""
		params: Dict[ str, Any ] = { }
		
		if not argument_string or not argument_string.strip( ):
			return params
		
		raw_items = re.split( r'[\n,]+', argument_string )
		items = [ item.strip( ) for item in raw_items if item and item.strip( ) ]
		
		for item in items:
			if '=' in item:
				k, v = item.split( '=', 1 )
				params[ k.strip( ) ] = v.strip( )
			else:
				params[ item ] = ''
		
		return params
	
	def _request( self, route: str, params: Dict[ str, Any ] | None = None,
			time: int = 20 ) -> Any:
		"""
			Purpose:
			--------
			Send an HTTP request to the OAC API and return parsed JSON when possible.

			Returns:
			--------
			Any
		"""
		try:
			self.timeout = int( time )
			self.url = f'{self.base_url}/{route.lstrip( "/" )}'
			self.params = params or { }
			
			self.response = requests.get(
				url=self.url,
				params=self.params,
				headers=self.headers,
				timeout=self.timeout )
			
			self.response.raise_for_status( )
			
			content_type = (self.response.headers.get( 'Content-Type', '' ) or '').lower( )
			if 'json' in content_type:
				return self.response.json( )
			
			return self.response.text
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroCatalog'
			exception.method = '_request( self, route: str, params: Dict[ str, Any ] | None=None, time: int=20 ) -> Any'
			raise exception
	
	def fetch_object( self, name: str, quantity: str = '', attributes: str = '',
			arguments: str = '', data_format: str = 'json', time: int = 20 ) -> Any:
		"""
			Purpose:
			--------
			Query OAC by object/event name using the documented route pattern.

			Returns:
			--------
			Any
		"""
		try:
			throw_if( 'name', name )
			self.name = name.strip( )
			self.format = (data_format or 'json').strip( ).lower( )
			
			route_parts = [ urllib.parse.quote( self.name ) ]
			attr_path = self._normalize_attribute_path( quantity, attributes )
			if attr_path:
				route_parts.append( attr_path )
			
			route = '/'.join( route_parts )
			params = self._parse_argument_string( arguments )
			
			if self.format:
				params[ 'format' ] = self.format
			
			return self._request( route=route, params=params, time=time )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroCatalog'
			exception.method = (
					'fetch_object( self, name: str, quantity: str=, attributes: str=, '
					'arguments: str=, data_format: str=json, time: int=20 ) -> Any'
			)
			raise exception
	
	def cone_search( self, ra: str, dec: str, radius: int = 2, quantity: str = '',
			attributes: str = '', arguments: str = '', data_format: str = 'json', time: int = 20 ) -> Any:
		"""
			Purpose:
			--------
			Query OAC using a coordinate cone search via special arguments.

			Returns:
			--------
			Any
		"""
		try:
			throw_if( 'ra', ra )
			throw_if( 'dec', dec )
			
			self.right_ascension = ra.strip( )
			self.declination = dec.strip( )
			self.radius = max( 1, int( radius ) )
			self.format = (data_format or 'json').strip( ).lower( )
			
			route = 'catalog'
			attr_path = self._normalize_attribute_path( quantity, attributes )
			if attr_path:
				route = f'{route}/{attr_path}'
			
			params = self._parse_argument_string( arguments )
			params[ 'ra' ] = self.right_ascension
			params[ 'dec' ] = self.declination
			params[ 'radius' ] = str( self.radius )
			
			if self.format:
				params[ 'format' ] = self.format
			
			return self._request( route=route, params=params, time=time )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroCatalog'
			exception.method = (
					'cone_search( self, ra: str, dec: str, radius: int=2, '
					'quantity: str=, attributes: str=, arguments: str=, '
					'data_format: str=json, time: int=20 ) -> Any'
			)
			raise exception
	
	def fetch( self, mode: str = 'object_query', query: str = '', quantity: str = '',
			attributes: str = '', arguments: str = '', ra: str = '', dec: str = '',
			radius: int = 2, data_format: str = 'json', time: int = 20 ) -> Any:
		"""
			Purpose:
			--------
			Unified dispatch for Astronomy Catalog operations.

			Returns:
			--------
			Any
		"""
		try:
			active_mode = (mode or 'object_query').strip( ).lower( )
			
			if active_mode == 'object_query':
				return self.fetch_object(
					name=query,
					quantity=quantity,
					attributes=attributes,
					arguments=arguments,
					data_format=data_format,
					time=time )
			
			if active_mode == 'cone_search':
				return self.cone_search(
					ra=ra,
					dec=dec,
					radius=radius,
					quantity=quantity,
					attributes=attributes,
					arguments=arguments,
					data_format=data_format,
					time=time )
			
			raise ValueError( "Unsupported mode. Use 'object_query' or 'cone_search'." )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroCatalog'
			exception.method = (
					'fetch( self, mode: str=object_query, query: str=, quantity: str=, '
					'attributes: str=, arguments: str=, ra: str=, dec: str=, '
					'radius: int=2, data_format: str=json, time: int=20 ) -> Any'
			)
			raise exception

class AstroQuery( Fetcher ):
	'''

		Purpose:
		--------
		Provides a focused astroquery wrapper around the SIMBAD service for
		named-object lookups, identifier retrieval, and cone/region searches.

		Notes:
		------
		This wrapper does not require an API key for standard SIMBAD queries.

	'''
	url: Optional[ str ]
	radius: Optional[ float ]
	name: Optional[ str ]
	declination: Optional[ str ]
	right_ascension: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	row_limit: Optional[ int ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.url = None
		self.radius = None
		self.name = None
		self.right_ascension = None
		self.declination = None
		self.params = { }
		self.row_limit = 100
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		return [
				'headers',
				'row_limit',
				'object_search',
				'object_ids',
				'region_search',
				'fetch',
		]
	
	def _table_to_records( self, table: Table | None ) -> List[ Dict[ str, Any ] ]:
		"""

			Purpose:
			--------
			Convert an Astropy Table into a list of row dictionaries that can be
			rendered easily in Streamlit.

			Parameters:
			-----------
			table:
				An astropy.table.Table returned by astroquery.

			Returns:
			--------
			List[Dict[str, Any]]

		"""
		try:
			if table is None:
				return [ ]
			
			records: List[ Dict[ str, Any ] ] = [ ]
			for row in table:
				record: Dict[ str, Any ] = { }
				for col in table.colnames:
					try:
						value = row[ col ]
						if hasattr( value, 'item' ):
							try:
								value = value.item( )
							except Exception:
								pass
						record[ str( col ) ] = str( value )
					except Exception:
						record[ str( col ) ] = ''
				records.append( record )
			
			return records
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroQuery'
			exception.method = '_table_to_records( self, table: Table | None ) -> List[ Dict[ str, Any ] ]'
			raise exception
	
	def object_search( self, name: str, row_limit: int = 100 ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Query SIMBAD for a named astronomical object.

			Parameters:
			-----------
			name:
				Object identifier or common name such as "M81", "Sirius", or
				"NGC 1300".
			row_limit:
				Maximum number of rows to return.

			Returns:
			--------
			Dict[str, Any] | None

		"""
		try:
			throw_if( 'name', name )
			self.name = name.strip( )
			self.row_limit = max( 1, int( row_limit ) )
			
			simbad = Simbad( )
			simbad.ROW_LIMIT = self.row_limit
			result_table = simbad.query_object( self.name )
			
			return {
					'mode': 'object_search',
					'query': self.name,
					'row_limit': self.row_limit,
					'columns': list( result_table.colnames ) if result_table is not None else [ ],
					'rows': self._table_to_records( result_table ),
			}
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroQuery'
			exception.method = 'object_search( self, name: str, row_limit: int=100 ) -> Dict[ str, Any ]'
			raise exception
	
	def object_ids( self, name: str, row_limit: int = 100 ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Query SIMBAD for alternate identifiers of a named astronomical object.

			Parameters:
			-----------
			name:
				Object identifier or common name such as "M81" or "Sirius".
			row_limit:
				Maximum number of rows to return.

			Returns:
			--------
			Dict[str, Any] | None

		"""
		try:
			throw_if( 'name', name )
			self.name = name.strip( )
			self.row_limit = max( 1, int( row_limit ) )
			
			simbad = Simbad( )
			simbad.ROW_LIMIT = self.row_limit
			result_table = simbad.query_objectids( self.name )
			
			return {
					'mode': 'object_ids',
					'query': self.name,
					'row_limit': self.row_limit,
					'columns': list( result_table.colnames ) if result_table is not None else [ ],
					'rows': self._table_to_records( result_table ),
			}
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroQuery'
			exception.method = 'object_ids( self, name: str, row_limit: int=100 ) -> Dict[ str, Any ]'
			raise exception
	
	def region_search( self, ra: str, dec: str, radius: float = 0.5,
			radius_unit: str = 'deg', row_limit: int = 100 ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Query SIMBAD in a cone around a sky position.

			Parameters:
			-----------
			ra:
				Right Ascension of the search center. This is the east-west sky
				coordinate. Example values:
				- "13:09:48.09"
				- "197.45037"

			dec:
				Declination of the search center. This is the north-south sky
				coordinate. Example values:
				- "-23:22:53.3"
				- "-23.38148"

			radius:
				Angular search radius around the sky position.

			radius_unit:
				Unit for the radius. Supported values here are "deg", "arcmin",
				and "arcsec".

			row_limit:
				Maximum number of rows to return.

			Returns:
			--------
			Dict[str, Any] | None

		"""
		try:
			throw_if( 'ra', ra )
			throw_if( 'dec', dec )
			
			self.right_ascension = ra.strip( )
			self.declination = dec.strip( )
			self.radius = float( radius )
			self.row_limit = max( 1, int( row_limit ) )
			
			unit_map = {
					'deg': u.deg,
					'arcmin': u.arcmin,
					'arcsec': u.arcsec,
			}
			
			active_unit = unit_map.get( (radius_unit or 'deg').strip( ).lower( ), u.deg )
			
			coord = SkyCoord(
				ra=self.right_ascension,
				dec=self.declination,
				unit=(u.hourangle, u.deg) )
			
			simbad = Simbad( )
			simbad.ROW_LIMIT = self.row_limit
			result_table = simbad.query_region(
				coordinates=coord,
				radius=self.radius * active_unit )
			
			return {
					'mode': 'region_search',
					'ra': self.right_ascension,
					'dec': self.declination,
					'radius': self.radius,
					'radius_unit': (radius_unit or 'deg').strip( ).lower( ),
					'row_limit': self.row_limit,
					'columns': list( result_table.colnames ) if result_table is not None else [ ],
					'rows': self._table_to_records( result_table ),
			}
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroQuery'
			exception.method = (
					'region_search( self, ra: str, dec: str, radius: float=0.5, '
					'radius_unit: str=deg, row_limit: int=100 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch( self, mode: str = 'object_search', query: str = '',
			ra: str = '', dec: str = '', radius: float = 0.5,
			radius_unit: str = 'deg', row_limit: int = 100 ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Unified dispatch for AstroQuery / SIMBAD operations.

			Parameters:
			-----------
			mode:
				One of:
				- "object_search"
				- "object_ids"
				- "region_search"

			query:
				Named object for object-based modes.

			ra:
				Right Ascension used for region_search.

			dec:
				Declination used for region_search.

			radius:
				Angular radius used for region_search.

			radius_unit:
				Unit for radius used for region_search.

			row_limit:
				Maximum number of rows to return.

			Returns:
			--------
			Dict[str, Any] | None

		"""
		try:
			active_mode = (mode or 'object_search').strip( ).lower( )
			
			if active_mode == 'object_search':
				return self.object_search(
					name=query,
					row_limit=row_limit )
			
			if active_mode == 'object_ids':
				return self.object_ids(
					name=query,
					row_limit=row_limit )
			
			if active_mode == 'region_search':
				return self.region_search(
					ra=ra,
					dec=dec,
					radius=radius,
					radius_unit=radius_unit,
					row_limit=row_limit )
			
			raise ValueError(
				"Unsupported mode. Use 'object_search', 'object_ids', or 'region_search'."
			)
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'AstroQuery'
			exception.method = (
					'fetch( self, mode: str=object_search, query: str=, ra: str=, '
					'dec: str=, radius: float=0.5, radius_unit: str=deg, '
					'row_limit: int=100 ) -> Dict[ str, Any ]'
			)
			raise exception

class StarMap( Fetcher ):
	'''

		Purpose:
		--------
		Provides structured access to Sky-Map.org interactive links and
		static snapshot generation.

		Notes:
		------
		No API key is required for the public Site Link / Image Generator
		behavior used here.

	'''
	base_url: Optional[ str ]
	snapshot_url: Optional[ str ]
	image_source: Optional[ str ]
	object: Optional[ str ]
	right_ascension: Optional[ float ]
	declination: Optional[ float ]
	box_color: Optional[ str ]
	show_box: Optional[ bool ]
	show_grid: Optional[ bool ]
	show_lines: Optional[ bool ]
	show_boundaries: Optional[ bool ]
	show_const_names: Optional[ bool ]
	zoom: Optional[ int ]
	params: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		'''
		
			Purpose:
			--------
			Initialize the StarMap

			Returns:
			--------
			None

		'''
		super( ).__init__( )
		self.base_url = 'https://www.sky-map.org/'
		self.snapshot_url = 'https://www.sky-map.org/snapshot'
		self.image_source = 'DSS2'
		self.object = None
		self.right_ascension = None
		self.declination = None
		self.box_color = 'yellow'
		self.show_box = True
		self.show_grid = True
		self.show_lines = True
		self.show_boundaries = True
		self.show_const_names = False
		self.zoom = 5
		self.params = { }
		self.timeout = 20
		self.headers = { }
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		if 'Accept' not in self.headers:
			self.headers[
				'Accept' ] = 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
	
	def __dir__( self ) -> List[ str ]:
		return [
				'base_url',
				'snapshot_url',
				'object',
				'right_ascension',
				'declination',
				'image_source',
				'zoom',
				'show_box',
				'show_grid',
				'show_lines',
				'show_boundaries',
				'show_const_names',
				'box_color',
				'params',
				'fetch_object_link',
				'fetch_coordinate_link',
				'fetch_snapshot',
				'fetch',
		]
	
	def _normalize_bool( self, value: bool ) -> str:
		'''
		
			Purpose:
			--------
			Convert a Python bool into the integer-style string form frequently
			used by Sky-Map query parameters.

			Parameters:
			-----------
			value:
				Boolean value to convert.

			Returns:
			--------
			str

		'''
		return '1' if bool( value ) else '0'
	
	def _extract_snapshot_links( self, html: str, base_url: str ) -> Dict[ str, str ]:
		'''
		
			Purpose:
			--------
			Parse the snapshot HTML page and extract save-as image links for
			formats like jpeg, png, gif, bmp, and tiff.

			Parameters:
			-----------
			html:
				Raw HTML returned by the snapshot endpoint.
			base_url:
				Base URL used to resolve relative hyperlinks.

			Returns:
			--------
			Dict[str, str]

		'''
		try:
			links: Dict[ str, str ] = { }
			if not html or not isinstance( html, str ):
				return links
			
			pattern = re.compile(
				r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>\s*(jpeg|png|gif|bmp|tiff)\s*</a>',
				flags=re.IGNORECASE )
			
			for match in pattern.finditer( html ):
				href = match.group( 1 )
				label = match.group( 2 ).lower( )
				links[ label ] = urllib.parse.urljoin( base_url, href )
			
			return links
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'StarMap'
			exception.method = '_extract_snapshot_links( self, html: str, base_url: str ) -> Dict[ str, str ]'
			raise exception
	
	def fetch_object_link(
			self,
			name: str,
			zoom: int = 5,
			box_color: str = 'yellow',
			show_box: bool = True,
			time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
		
			Purpose:
			--------
			Construct an interactive Sky-Map link centered on a named object.

			Parameters:
			-----------
			name:
				Object name or identifier such as "Polaris", "M31", or
				"NGC 1300".
			zoom:
				Map zoom level. Smaller values show a wider field; larger values
				zoom further in.
			box_color:
				Color of the selection/highlight box.
			show_box:
				Whether to show the highlight box around the object.
			time:
				Request timeout in seconds used for validation.

			Returns:
			--------
			Dict[str, Any] | None

		'''
		try:
			throw_if( 'name', name )
			
			self.object = name.strip( )
			self.zoom = max( 1, min( int( zoom ), 18 ) )
			self.box_color = box_color or 'yellow'
			self.show_box = bool( show_box )
			self.timeout = int( time )
			
			self.params = {
					'object': self.object,
					'show_box': self._normalize_bool( self.show_box ),
					'zoom': str( self.zoom ),
					'box_color': self.box_color,
					'box_width': '50',
					'box_height': '50',
			}
			
			interactive_url = f'{self.base_url}?{urllib.parse.urlencode( self.params )}'
			
			self.response = requests.get(
				url=self.base_url,
				params=self.params,
				headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			return {
					'mode': 'object_link',
					'object': self.object,
					'zoom': self.zoom,
					'params': self.params,
					'interactive_url': interactive_url,
					'status_code': self.response.status_code,
					'html_preview': self.response.text[ : 2000 ],
			}
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'StarMap'
			exception.method = (
					'fetch_object_link( self, name: str, zoom: int=5, box_color: str=yellow, '
					'show_box: bool=True, time: int=20 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch_coordinate_link(
			self,
			ra: float,
			dec: float,
			zoom: int = 5,
			box_color: str = 'yellow',
			show_box: bool = True,
			show_grid: bool = True,
			show_lines: bool = True,
			show_boundaries: bool = True,
			time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
		
			Purpose:
			--------
			Construct an interactive Sky-Map link centered on sky coordinates.

			Parameters:
			-----------
			ra:
				Right Ascension of the map center in hours.
				Example values:
				- 15.2976
				- 5.9195

			dec:
				Declination of the map center in degrees.
				Example values:
				- -17.5892
				- 41.2692

			zoom:
				Map zoom level.
			box_color:
				Color of the selection/highlight box.
			show_box:
				Whether to show the highlight box.
			show_grid:
				Whether to display coordinate grid lines.
			show_lines:
				Whether to display constellation lines.
			show_boundaries:
				Whether to display constellation boundaries.
			time:
				Request timeout in seconds used for validation.

			Returns:
			--------
			Dict[str, Any] | None

		'''
		try:
			throw_if( 'ra', ra )
			throw_if( 'dec', dec )
			
			self.right_ascension = float( ra )
			self.declination = float( dec )
			self.zoom = max( 1, min( int( zoom ), 18 ) )
			self.box_color = box_color or 'yellow'
			self.show_box = bool( show_box )
			self.show_grid = bool( show_grid )
			self.show_lines = bool( show_lines )
			self.show_boundaries = bool( show_boundaries )
			self.timeout = int( time )
			
			self.params = {
					'ra': f'{self.right_ascension}',
					'de': f'{self.declination}',
					'show_box': self._normalize_bool( self.show_box ),
					'zoom': str( self.zoom ),
					'box_color': self.box_color,
					'show_grid': self._normalize_bool( self.show_grid ),
					'show_constellation_lines': self._normalize_bool( self.show_lines ),
					'show_constellation_boundaries': self._normalize_bool( self.show_boundaries ),
			}
			
			interactive_url = f'{self.base_url}?{urllib.parse.urlencode( self.params )}'
			
			self.response = requests.get(
				url=self.base_url,
				params=self.params,
				headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			return {
					'mode': 'coordinate_link',
					'ra': self.right_ascension,
					'dec': self.declination,
					'zoom': self.zoom,
					'params': self.params,
					'interactive_url': interactive_url,
					'status_code': self.response.status_code,
					'html_preview': self.response.text[ : 2000 ],
			}
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'StarMap'
			exception.method = (
					'fetch_coordinate_link( self, ra: float, dec: float, zoom: int=5, '
					'box_color: str=yellow, show_box: bool=True, show_grid: bool=True, '
					'show_lines: bool=True, show_boundaries: bool=True, time: int=20 ) '
					'-> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch_snapshot(
			self,
			ra: float,
			dec: float,
			zoom: int = 10,
			image_source: str = 'DSS2',
			show_grid: bool = True,
			show_lines: bool = True,
			show_boundaries: bool = True,
			show_const_names: bool = False,
			time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
		
			Purpose:
			--------
			Request the Sky-Map snapshot generator page and extract the available
			static image links.

			Parameters:
			-----------
			ra:
				Right Ascension of the image center in hours.
				Example values:
				- 15.2976
				- 5.9195

			dec:
				Declination of the image center in degrees.
				Example values:
				- -17.5892
				- 41.2692

			zoom:
				Snapshot zoom level / field scale.
			image_source:
				Survey source such as DSS2, SDSS, GALEX, IRAS, or RASS.
			show_grid:
				Whether to display the coordinate grid.
			show_lines:
				Whether to display constellation lines.
			show_boundaries:
				Whether to display constellation boundaries.
			show_const_names:
				Whether to display constellation names.
			time:
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None

		'''
		try:
			throw_if( 'ra', ra )
			throw_if( 'dec', dec )
			
			self.right_ascension = float( ra )
			self.declination = float( dec )
			self.zoom = max( 1, min( int( zoom ), 18 ) )
			self.image_source = (image_source or 'DSS2').strip( )
			self.show_grid = bool( show_grid )
			self.show_lines = bool( show_lines )
			self.show_boundaries = bool( show_boundaries )
			self.show_const_names = bool( show_const_names )
			self.timeout = int( time )
			
			self.params = {
					'ra': f'{self.right_ascension}',
					'de': f'{self.declination}',
					'zoom': str( self.zoom ),
					'img_source': self.image_source,
					'show_grid': self._normalize_bool( self.show_grid ),
					'show_constellation_lines': self._normalize_bool( self.show_lines ),
					'show_constellation_boundaries': self._normalize_bool( self.show_boundaries ),
					'show_const_names': self._normalize_bool( self.show_const_names ),
			}
			
			page_url = f'{self.snapshot_url}?{urllib.parse.urlencode( self.params )}'
			
			self.response = requests.get(
				url=self.snapshot_url,
				params=self.params,
				headers=self.headers,
				timeout=self.timeout )
			self.response.raise_for_status( )
			
			image_links = self._extract_snapshot_links(
				html=self.response.text,
				base_url=self.snapshot_url )
			
			return {
					'mode': 'snapshot',
					'ra': self.right_ascension,
					'dec': self.declination,
					'zoom': self.zoom,
					'image_source': self.image_source,
					'params': self.params,
					'snapshot_page_url': page_url,
					'image_links': image_links,
					'preferred_image_url': image_links.get( 'png' ) or image_links.get( 'jpeg', '' ),
					'status_code': self.response.status_code,
					'html_preview': self.response.text[ : 2000 ],
			}
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'StarMap'
			exception.method = (
					'fetch_snapshot( self, ra: float, dec: float, zoom: int=10, '
					'image_source: str=DSS2, show_grid: bool=True, show_lines: bool=True, '
					'show_boundaries: bool=True, show_const_names: bool=False, '
					'time: int=20 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch(
			self,
			mode: str = 'object_link',
			query: str = '',
			ra: float = 0.0,
			dec: float = 0.0,
			zoom: int = 5,
			image_source: str = 'DSS2',
			box_color: str = 'yellow',
			show_box: bool = True,
			show_grid: bool = True,
			show_lines: bool = True,
			show_boundaries: bool = True,
			show_const_names: bool = False,
			time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
		
			Purpose:
			--------
			Unified dispatch for StarMap object links, coordinate links, and
			static snapshot generation.

			Parameters:
			-----------
			mode:
				One of:
				- "object_link"
				- "coordinate_link"
				- "snapshot"

			query:
				Object name used for object_link.

			ra:
				Right Ascension used for coordinate_link and snapshot.

			dec:
				Declination used for coordinate_link and snapshot.

			zoom:
				Zoom level.

			image_source:
				Sky survey source used for snapshot.

			box_color:
				Highlight box color for interactive modes.

			show_box:
				Show highlight box for interactive modes.

			show_grid:
				Show grid for coordinate/snapshot modes.

			show_lines:
				Show constellation lines.

			show_boundaries:
				Show constellation boundaries.

			show_const_names:
				Show constellation names for snapshot mode.

			time:
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None

		'''
		try:
			active_mode = (mode or 'object_link').strip( ).lower( )
			
			if active_mode == 'object_link':
				return self.fetch_object_link(
					name=query,
					zoom=zoom,
					box_color=box_color,
					show_box=show_box,
					time=time )
			
			if active_mode == 'coordinate_link':
				return self.fetch_coordinate_link(
					ra=ra,
					dec=dec,
					zoom=zoom,
					box_color=box_color,
					show_box=show_box,
					show_grid=show_grid,
					show_lines=show_lines,
					show_boundaries=show_boundaries,
					time=time )
			
			if active_mode == 'snapshot':
				return self.fetch_snapshot(
					ra=ra,
					dec=dec,
					zoom=zoom,
					image_source=image_source,
					show_grid=show_grid,
					show_lines=show_lines,
					show_boundaries=show_boundaries,
					show_const_names=show_const_names,
					time=time )
			
			raise ValueError(
				"Unsupported mode. Use 'object_link', 'coordinate_link', or 'snapshot'."
			)
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'StarMap'
			exception.method = (
					'fetch( self, mode: str=object_link, query: str=, ra: float=0.0, '
					'dec: float=0.0, zoom: int=5, image_source: str=DSS2, '
					'box_color: str=yellow, show_box: bool=True, show_grid: bool=True, '
					'show_lines: bool=True, show_boundaries: bool=True, '
					'show_const_names: bool=False, time: int=20 ) -> Dict[ str, Any ]'
			)
			raise exception
			
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
		
	def fetch( self, criteria: str ) -> Dict[ str, Any ] | None:
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
			raise exception
			
	
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
			raise exception
			

	def fetch_bills( self, congress: int, billtype: str, billnum: int ) -> Dict[ str, Any ] | None:
		'''

			Returns:
			-------
			Starmap

		'''
		try:
			throw_if( 'congress', congress )
			throw_if( 'billtype', billtype )
			throw_if( 'billnum', billnum )
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
			raise exception
			
			
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
			raise exception
			
			
	def fetch_records( self, congress: str, billtype: str,  billnum: int ) -> Dict[ str, Any ] | None:
		'''

			Returns:
			-------
			Starmap

		'''
		try:
			throw_if( 'congress', congress )
			throw_if( 'billtype', billtype )
			throw_if( 'billnum', billnum )
			self.congress_number = congress
			self.bill_type = billtype
			self.bill_number = billnum
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
			raise exception
			
			
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
			raise exception
			
		
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
			raise exception

class StarChart( Fetcher ):
	'''

		Purpose:
		--------
		Provides static and link-based star chart generation using the SKY-MAP.ORG
		XML API, Site Linker, and Image Generator interfaces.

		This class is intentionally chart-focused and kept separate from StarMap.

		Referenced API Requirements:
		----------------------------
		XML Search:
			- Endpoint: https://server1.sky-map.org/search
			- Required parameter:
				- star

		Site Linker:
			- Endpoint: https://www.sky-map.org/
			- Supported parameters used here:
				- object
				- ra
				- de
				- zoom
				- show_box
				- box_color
				- show_grid
				- show_constellation_lines
				- show_constellation_boundaries
				- img_source

		Image Generator:
			- Endpoint: https://server2.sky-map.org/map
			- Supported parameters used here:
				- ra
				- de
				- zoom
				- show_grid
				- show_constellation_lines
				- show_constellation_boundaries
				- show_const_names
				- img_source
				- w
				- h
				- mag

		Attributes:
		-----------
		search_url: Optional[str]
			SKY-MAP XML search endpoint.

		link_url: Optional[str]
			SKY-MAP site-link endpoint.

		image_url: Optional[str]
			SKY-MAP image-generator endpoint.

		url: Optional[str]
			Resolved request URL.

		params: Optional[Dict[str, Any]]
			Request parameters.

		mode: Optional[str]
			Selected chart mode.

		query: Optional[str]
			Object query string.

		ra: Optional[float]
			Right Ascension in decimal hours.

		dec: Optional[float]
			Declination in decimal degrees.

		zoom: Optional[int]
			Chart zoom level.

		image_source: Optional[str]
			Chart image source.

		box_color: Optional[str]
			Pointer box color.

		show_box: Optional[bool]
			Show highlight box.

		show_grid: Optional[bool]
			Show coordinate grid.

		show_lines: Optional[bool]
			Show constellation lines.

		show_boundaries: Optional[bool]
			Show constellation boundaries.

		show_const_names: Optional[bool]
			Show constellation names.

		width: Optional[int]
			Generated image width.

		height: Optional[int]
			Generated image height.

		magnitude: Optional[float]
			Image generator limiting magnitude.

		agents: Optional[str]
			User-Agent string.

		Methods:
		--------
		__init__() -> None
			Initialize chart defaults.

		__dir__() -> List[str]
			Provide ordered member visibility.

		search_object(...) -> Dict[str, Any] | None
			Resolve an object name through the SKY-MAP XML API.

		fetch_object_chart(...) -> Dict[str, Any] | None
			Build an object-based chart link.

		fetch_coordinate_chart(...) -> Dict[str, Any] | None
			Build a coordinate-based chart link.

		fetch_static_chart(...) -> Dict[str, Any] | None
			Build a static chart image URL.

		fetch(...) -> Dict[str, Any] | None
			Unified dispatcher.

		create_schema(...) -> Dict[str, str] | None
			Construct a dynamic tool schema.

	'''
	search_url: Optional[ str ]
	link_url: Optional[ str ]
	image_url: Optional[ str ]
	url: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	mode: Optional[ str ]
	query: Optional[ str ]
	ra: Optional[ float ]
	dec: Optional[ float ]
	zoom: Optional[ int ]
	image_source: Optional[ str ]
	box_color: Optional[ str ]
	show_box: Optional[ bool ]
	show_grid: Optional[ bool ]
	show_lines: Optional[ bool ]
	show_boundaries: Optional[ bool ]
	show_const_names: Optional[ bool ]
	width: Optional[ int ]
	height: Optional[ int ]
	magnitude: Optional[ float ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		'''
			Purpose:
			--------
			Initialize the StarChart fetcher with current SKY-MAP defaults.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		'''
		super( ).__init__( )
		self.headers = { }
		self.search_url = 'https://server1.sky-map.org/search'
		self.link_url = 'https://www.sky-map.org/'
		self.image_url = 'https://server2.sky-map.org/map'
		self.url = None
		self.params = { }
		self.mode = 'object_chart'
		self.query = ''
		self.ra = 0.0
		self.dec = 0.0
		self.zoom = 5
		self.image_source = 'DSS2'
		self.box_color = 'yellow'
		self.show_box = True
		self.show_grid = True
		self.show_lines = True
		self.show_boundaries = True
		self.show_const_names = False
		self.width = 900
		self.height = 450
		self.magnitude = 7.5
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		'''
			Purpose:
			--------
			Provide ordered member visibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
			
		'''
		return [
				'search_url',
				'link_url',
				'image_url',
				'url',
				'params',
				'mode',
				'query',
				'ra',
				'dec',
				'zoom',
				'image_source',
				'box_color',
				'show_box',
				'show_grid',
				'show_lines',
				'show_boundaries',
				'show_const_names',
				'width',
				'height',
				'magnitude',
				'search_object',
				'fetch_object_chart',
				'fetch_coordinate_chart',
				'fetch_static_chart',
				'fetch',
				'create_schema'
		]
	
	def _flag( self, value: bool, invert: bool = False ) -> int:
		'''
			Purpose:
			--------
			Convert boolean UI flags into SKY-MAP numeric flags.

			Parameters:
			-----------
			value (bool):
				Input boolean value.

			invert (bool):
				If True, invert the SKY-MAP convention.

			Returns:
			--------
			int
		'''
		if invert:
			return 0 if bool( value ) else 1
		
		return 1 if bool( value ) else 0
	
	def search_object( self, name: str, time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Resolve an object name into SKY-MAP coordinates using the XML API.

			Parameters:
			-----------
			name (str):
				Object name or catalog id. Examples:
				- Polaris
				- M31
				- NGC 1300

			time (int):
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			throw_if( 'name', name )
			self.query = str( name ).strip( )
			self.url = self.search_url
			self.params = { 'star': self.query }
			
			self.response = requests.get(
				url=self.url,
				params=self.params,
				headers=self.headers,
				timeout=int( time )
			)
			self.response.raise_for_status( )
			
			root = ET.fromstring( self.response.text )
			status = root.findtext( 'status', default='' )
			verbiage = root.findtext( 'verbiage', default='' )
			star = root.find( 'star' )
			
			if star is None:
				return {
						'mode': 'object_search',
						'url': self.url,
						'params': self.params,
						'status': status,
						'verbiage': verbiage,
						'data': { }
				}
			
			result = {
					'id': star.attrib.get( 'id', '' ),
					'catalog_id': star.findtext( 'catId', default='' ),
					'constellation': star.findtext( 'constellation', default='' ),
					'ra': float( star.findtext( 'ra', default='0' ) ),
					'dec': float( star.findtext( 'de', default='0' ) ),
					'magnitude': star.findtext( 'mag', default='' )
			}
			
			self.ra = result[ 'ra' ]
			self.dec = result[ 'dec' ]
			
			return {
					'mode': 'object_search',
					'url': self.url,
					'params': self.params,
					'status': status,
					'verbiage': verbiage,
					'data': result
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'StarChart'
			exception.method = (
					'search_object( self, name: str, time: int=20 ) '
					'-> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch_object_chart( self, name: str, zoom: int = 5,
			box_color: str = 'yellow', show_box: bool = True,
			image_source: str = '', time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Build an object-based SKY-MAP chart link.

			Parameters:
			-----------
			name (str):
				Object name or catalog id.

			zoom (int):
				Chart zoom level.

			box_color (str):
				Pointer box color.

			show_box (bool):
				Show pointer box.

			image_source (str):
				Optional image source such as SDSS.

			time (int):
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			throw_if( 'name', name )
			self.mode = 'object_chart'
			self.query = str( name ).strip( )
			self.zoom = int( zoom )
			self.box_color = str( box_color or 'yellow' ).strip( )
			self.show_box = bool( show_box )
			self.image_source = str( image_source or '' ).strip( )
			
			self.url = self.link_url
			self.params = {
					'object': self.query,
					'zoom': self.zoom,
					'show_box': self._flag( self.show_box ),
					'box_color': self.box_color
			}
			
			if self.image_source:
				self.params[ 'img_source' ] = self.image_source
			
			link = requests.Request( 'GET', self.url, params=self.params ).prepare( ).url
			
			search = self.search_object( name=self.query, time=time ) or { }
			
			return {
					'mode': self.mode,
					'url': self.url,
					'params': self.params,
					'chart_url': link,
					'search': search
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'StarChart'
			exception.method = (
					'fetch_object_chart( self, name: str, zoom: int=5, '
					'box_color: str=yellow, show_box: bool=True, image_source: str=, '
					'time: int=20 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch_coordinate_chart( self, ra: float, dec: float, zoom: int = 5,
			box_color: str = 'yellow', show_box: bool = True,
			show_grid: bool = True, show_lines: bool = True,
			show_boundaries: bool = True, image_source: str = '' ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Build a coordinate-based SKY-MAP chart link.

			Parameters:
			-----------
			ra (float):
				Right Ascension in decimal hours.

			dec (float):
				Declination in decimal degrees.

			zoom (int):
				Chart zoom level.

			box_color (str):
				Pointer box color.

			show_box (bool):
				Show pointer box.

			show_grid (bool):
				Show coordinate grid.

			show_lines (bool):
				Show constellation lines.

			show_boundaries (bool):
				Show constellation boundaries.

			image_source (str):
				Optional image source such as SDSS.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			self.mode = 'coordinate_chart'
			self.ra = float( ra )
			self.dec = float( dec )
			self.zoom = int( zoom )
			self.box_color = str( box_color or 'yellow' ).strip( )
			self.show_box = bool( show_box )
			self.show_grid = bool( show_grid )
			self.show_lines = bool( show_lines )
			self.show_boundaries = bool( show_boundaries )
			self.image_source = str( image_source or '' ).strip( )
			
			self.url = self.link_url
			self.params = {
					'ra': self.ra,
					'de': self.dec,
					'zoom': self.zoom,
					'show_box': self._flag( self.show_box ),
					'box_color': self.box_color,
					'show_grid': self._flag( self.show_grid, invert=True ),
					'show_constellation_lines': self._flag( self.show_lines, invert=True ),
					'show_constellation_boundaries': self._flag( self.show_boundaries, invert=True )
			}
			
			if self.image_source:
				self.params[ 'img_source' ] = self.image_source
			
			link = requests.Request( 'GET', self.url, params=self.params ).prepare( ).url
			
			return {
					'mode': self.mode,
					'url': self.url,
					'params': self.params,
					'chart_url': link
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'StarChart'
			exception.method = (
					'fetch_coordinate_chart( self, ra: float, dec: float, zoom: int=5, '
					'box_color: str=yellow, show_box: bool=True, show_grid: bool=True, '
					'show_lines: bool=True, show_boundaries: bool=True, image_source: str= ) '
					'-> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch_static_chart( self, ra: float, dec: float, zoom: int = 5,
			image_source: str = 'DSS2', show_grid: bool = True,
			show_lines: bool = True, show_boundaries: bool = True,
			show_const_names: bool = False, width: int = 900,
			height: int = 450, magnitude: float = 7.5 ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Build a static SKY-MAP chart image URL.

			Parameters:
			-----------
			ra (float):
				Right Ascension in decimal hours.

			dec (float):
				Declination in decimal degrees.

			zoom (int):
				Chart zoom level.

			image_source (str):
				Image survey source.

			show_grid (bool):
				Show grid.

			show_lines (bool):
				Show constellation lines.

			show_boundaries (bool):
				Show constellation boundaries.

			show_const_names (bool):
				Show constellation names.

			width (int):
				Image width in pixels.

			height (int):
				Image height in pixels.

			magnitude (float):
				Limiting magnitude.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			self.mode = 'static_chart'
			self.ra = float( ra )
			self.dec = float( dec )
			self.zoom = int( zoom )
			self.image_source = str( image_source or 'DSS2' ).strip( )
			self.show_grid = bool( show_grid )
			self.show_lines = bool( show_lines )
			self.show_boundaries = bool( show_boundaries )
			self.show_const_names = bool( show_const_names )
			self.width = int( width )
			self.height = int( height )
			self.magnitude = float( magnitude )
			
			self.url = self.image_url
			self.params = {
					'type': 'FULL',
					'w': self.width,
					'h': self.height,
					'ra': self.ra,
					'de': self.dec,
					'zoom': self.zoom,
					'mag': self.magnitude,
					'show_grid': self._flag( self.show_grid ),
					'grid_color': '404040',
					'grid_color_zero': '808080',
					'show_constellation_lines': self._flag( self.show_lines ),
					'show_constellation_boundaries': self._flag( self.show_boundaries ),
					'show_const_names': self._flag( self.show_const_names ),
					'img_source': self.image_source
			}
			
			image_link = requests.Request( 'GET', self.url, params=self.params ).prepare( ).url
			
			return {
					'mode': self.mode,
					'url': self.url,
					'params': self.params,
					'image_url': image_link
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'StarChart'
			exception.method = (
					'fetch_static_chart( self, ra: float, dec: float, zoom: int=5, '
					'image_source: str=DSS2, show_grid: bool=True, show_lines: bool=True, '
					'show_boundaries: bool=True, show_const_names: bool=False, '
					'width: int=900, height: int=450, magnitude: float=7.5 ) '
					'-> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch( self, mode: str = 'object_chart', query: str = '',
			ra: float = 0.0, dec: float = 0.0, zoom: int = 5,
			image_source: str = 'DSS2', box_color: str = 'yellow',
			show_box: bool = True, show_grid: bool = True,
			show_lines: bool = True, show_boundaries: bool = True,
			show_const_names: bool = False, width: int = 900,
			height: int = 450, magnitude: float = 7.5,
			time: int = 20 ) -> Dict[ str, Any ] | None:
		'''
			Purpose:
			--------
			Unified dispatcher for SKY-MAP chart generation.

			Parameters:
			-----------
			mode (str):
				One of:
				- object_search
				- object_chart
				- coordinate_chart
				- static_chart

			query (str):
				Object query for object_search and object_chart.

			ra (float):
				Right Ascension for coordinate_chart and static_chart.

			dec (float):
				Declination for coordinate_chart and static_chart.

			zoom (int):
				Chart zoom level.

			image_source (str):
				Image source.

			box_color (str):
				Pointer box color.

			show_box (bool):
				Show pointer box.

			show_grid (bool):
				Show coordinate grid.

			show_lines (bool):
				Show constellation lines.

			show_boundaries (bool):
				Show constellation boundaries.

			show_const_names (bool):
				Show constellation names.

			width (int):
				Static image width.

			height (int):
				Static image height.

			magnitude (float):
				Static image limiting magnitude.

			time (int):
				Request timeout in seconds.

			Returns:
			--------
			Dict[str, Any] | None
		'''
		try:
			active_mode = (mode or 'object_chart').strip( ).lower( )
			
			if active_mode == 'object_search':
				return self.search_object(
					name=query,
					time=time
				)
			
			if active_mode == 'object_chart':
				return self.fetch_object_chart(
					name=query,
					zoom=zoom,
					box_color=box_color,
					show_box=show_box,
					image_source=image_source if image_source != 'DSS2' else '',
					time=time
				)
			
			if active_mode == 'coordinate_chart':
				return self.fetch_coordinate_chart(
					ra=ra,
					dec=dec,
					zoom=zoom,
					box_color=box_color,
					show_box=show_box,
					show_grid=show_grid,
					show_lines=show_lines,
					show_boundaries=show_boundaries,
					image_source=image_source if image_source != 'DSS2' else ''
				)
			
			if active_mode == 'static_chart':
				return self.fetch_static_chart(
					ra=ra,
					dec=dec,
					zoom=zoom,
					image_source=image_source,
					show_grid=show_grid,
					show_lines=show_lines,
					show_boundaries=show_boundaries,
					show_const_names=show_const_names,
					width=width,
					height=height,
					magnitude=magnitude
				)
			
			raise ValueError(
				"Unsupported mode. Use 'object_search', 'object_chart', "
				"'coordinate_chart', or 'static_chart'."
			)
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'StarChart'
			exception.method = (
					'fetch( self, mode: str=object_chart, query: str=, ra: float=0.0, '
					'dec: float=0.0, zoom: int=5, image_source: str=DSS2, '
					'box_color: str=yellow, show_box: bool=True, show_grid: bool=True, '
					'show_lines: bool=True, show_boundaries: bool=True, '
					'show_const_names: bool=False, width: int=900, height: int=450, '
					'magnitude: float=7.5, time: int=20 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict,
			required: list[ str ] ) -> Dict[ str, str ] | None:
		'''
			Purpose:
			--------
			Construct and return a fully dynamic OpenAI Tool API schema definition.

			Parameters:
			-----------
			function (str):
				The function name exposed to the LLM.

			tool (str):
				The underlying system or service the function wraps.

			description (str):
				Precise explanation of what the function does.

			parameters (dict):
				A dictionary defining parameter names and JSON schema descriptors.

			required (list[str]):
				List of required parameter names.

			Returns:
			--------
			Dict[str, str] | None
		'''
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return {
					'name': function.strip( ),
					'description': f'{description.strip( )} This function uses the {tool.strip( )} service.',
					'parameters': {
							'type': 'object',
							'properties': parameters,
							'required': required
					}
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'StarChart'
			exception.method = (
					'create_schema( self, function: str, tool: str, description: str, '
					'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]'
			)
			raise exception
			
class Congress( Fetcher ):
	'''

		Purpose:
		--------
		Provides  service that can be used to query the GovInfo search engine and return results
		that are the equivalent to what is returned by the main user interface.

		You can use field operators, such as congress, published date, branch, and others to construct
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
	
	def fetch_bills( self, congress: int ) -> Dict[ str, Any ] | None:
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
			raise exception
			
	
	def fetch_laws( self, congress: int ) -> Dict[ str, Any ] | None:
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
			raise exception
			
	
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
			raise exception
			
		
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
			raise exception
			
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
			raise exception
			
	
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
			raise exception
			
		
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
			raise exception

class OpenWeather( Fetcher ):
	'''

		Purpose:
		--------
		Provides forecast weather retrieval by location name using the Open-Meteo
		Geocoding API and Open-Meteo Forecast API.

		This class is forecast-only by design and intentionally excludes archive /
		historical date-based retrieval so it does not overlap with the separate
		HistoricalWeather class.

		Referenced API Requirements:
		----------------------------
		Geocoding API:
			- Endpoint: https://geocoding-api.open-meteo.com/v1/search
			- Required parameter: name
			- Optional parameter: count

		Forecast API:
			- Endpoint: https://api.open-meteo.com/v1/forecast
			- Required parameters: latitude, longitude
			- Optional parameters used here:
				- current
				- hourly
				- daily
				- timezone
				- forecast_days
				- past_days
				- temperature_unit
				- wind_speed_unit
				- precipitation_unit

		Attributes:
		-----------
		geocode_url: Optional[str]
			The Open-Meteo geocoding endpoint.

		forecast_url: Optional[str]
			The Open-Meteo forecast endpoint.

		location: Optional[str]
			User-supplied location query.

		latitude: Optional[float]
			Resolved latitude from geocoding.

		longitude: Optional[float]
			Resolved longitude from geocoding.

		timezone: Optional[str]
			Resolved or user-requested timezone.

		mode: Optional[str]
			Forecast mode: current, hourly, or daily.

		current_metrics: Optional[List[str]]
			Current weather metrics requested from the API.

		hourly_metrics: Optional[List[str]]
			Hourly metrics requested from the API.

		daily_metrics: Optional[List[str]]
			Daily metrics requested from the API.

		windspeed_unit: Optional[str]
			Wind speed unit passed to the API.

		temperature_unit: Optional[str]
			Temperature unit passed to the API.

		precipitation_unit: Optional[str]
			Precipitation unit passed to the API.

		params: Optional[Dict[str, Any]]
			Request parameters for the forecast call.

		geocode_params: Optional[Dict[str, Any]]
			Request parameters for the geocoding call.

		result_limit: Optional[int]
			Maximum number of geocoding candidates to request.

		Methods:
		--------
		__init__() -> None
			Initialize the fetcher and default metric sets.

		__dir__() -> List[str]
			Provide ordered member visibility.

		geocode_location(...) -> Dict[str, Any] | None
			Resolve a place name into a selected geocoding record.

		fetch(...) -> Dict[str, Any] | None
			Resolve a location string and retrieve forecast weather.

		fetch_current(...) -> Dict[str, Any] | None
			Retrieve current forecast conditions only.

		fetch_hourly(...) -> Dict[str, Any] | None
			Retrieve hourly forecast data.

		fetch_daily(...) -> Dict[str, Any] | None
			Retrieve daily forecast data.

		create_schema(...) -> Dict[str, str] | None
			Generate a dynamic tool schema definition.

	'''
	geocode_url: Optional[ str ]
	forecast_url: Optional[ str ]
	location: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	timezone: Optional[ str ]
	mode: Optional[ str ]
	current_metrics: Optional[ List[ str ] ]
	hourly_metrics: Optional[ List[ str ] ]
	daily_metrics: Optional[ List[ str ] ]
	windspeed_unit: Optional[ str ]
	temperature_unit: Optional[ str ]
	precipitation_unit: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	geocode_params: Optional[ Dict[ str, Any ] ]
	result_limit: Optional[ int ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		'''

			Purpose:
			--------
			Initialize the OpenWeather forecast fetcher with forecast-only defaults,
			endpoints, headers, unit selections, and metric collections.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		'''
		super( ).__init__( )
		self.headers = { }
		self.agents = cfg.AGENTS
		self.location = None
		self.latitude = None
		self.longitude = None
		self.timezone = 'auto'
		self.mode = 'current'
		self.result_limit = 10
		self.geocode_url = 'https://geocoding-api.open-meteo.com/v1/search'
		self.forecast_url = 'https://api.open-meteo.com/v1/forecast'
		self.temperature_unit = 'fahrenheit'
		self.windspeed_unit = 'kn'
		self.precipitation_unit = 'inch'
		self.geocode_params = None
		self.params = None
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		self.current_metrics = [
				'temperature_2m',
				'relative_humidity_2m',
				'apparent_temperature',
				'is_day',
				'precipitation',
				'rain',
				'showers',
				'snowfall',
				'weather_code',
				'cloud_cover',
				'pressure_msl',
				'surface_pressure',
				'wind_speed_10m',
				'wind_direction_10m',
				'wind_gusts_10m'
		]
		
		self.hourly_metrics = [
				'temperature_2m',
				'relative_humidity_2m',
				'apparent_temperature',
				'precipitation_probability',
				'precipitation',
				'rain',
				'showers',
				'snowfall',
				'weather_code',
				'cloud_cover',
				'pressure_msl',
				'surface_pressure',
				'visibility',
				'wind_speed_10m',
				'wind_direction_10m',
				'wind_gusts_10m'
		]
		
		self.daily_metrics = [
				'weather_code',
				'temperature_2m_max',
				'temperature_2m_min',
				'apparent_temperature_max',
				'apparent_temperature_min',
				'sunrise',
				'sunset',
				'daylight_duration',
				'sunshine_duration',
				'precipitation_sum',
				'rain_sum',
				'showers_sum',
				'snowfall_sum',
				'precipitation_hours',
				'precipitation_probability_max',
				'wind_speed_10m_max',
				'wind_gusts_10m_max',
				'wind_direction_10m_dominant'
		]
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			--------
			Provide ordered member visibility for introspection and editor discovery.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Ordered attribute and method names.

		'''
		return [
				'geocode_url',
				'forecast_url',
				'location',
				'latitude',
				'longitude',
				'timezone',
				'mode',
				'current_metrics',
				'hourly_metrics',
				'daily_metrics',
				'windspeed_unit',
				'temperature_unit',
				'precipitation_unit',
				'params',
				'geocode_params',
				'result_limit',
				'geocode_location',
				'fetch',
				'fetch_current',
				'fetch_hourly',
				'fetch_daily',
				'create_schema'
		]
	
	def geocode_location( self, location: str, count: int = 10 ) -> Dict[ str, Any ] | None:
		'''

			Purpose:
			--------
			Resolve a user-supplied location string into a geocoding result from
			the Open-Meteo Geocoding API.

			Parameters:
			-----------
			location (str):
				The place name or postal code to search for.

			count (int):
				The maximum number of geocoding matches to request.

			Returns:
			--------
			Dict[str, Any] | None:
				The selected geocoding record, typically the first result.

		'''
		try:
			throw_if( 'location', location )
			self.location = location.strip( )
			self.geocode_params = {
					'name': self.location,
					'count': int( count )
			}
			
			self.url = self.geocode_url
			self.response = requests.get(
				self.url,
				params=self.geocode_params,
				headers=self.headers,
				timeout=20
			)
			self.response.raise_for_status( )
			
			payload = self.response.json( ) or { }
			results = payload.get( 'results', [ ] ) or [ ]
			
			if not results:
				return None
			
			selected = results[ 0 ]
			self.latitude = selected.get( 'latitude', None )
			self.longitude = selected.get( 'longitude', None )
			
			resolved_timezone = selected.get( 'timezone', None )
			if resolved_timezone:
				self.timezone = resolved_timezone
			
			return selected
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenWeather'
			exception.method = (
					'geocode_location( self, location: str, count: int=10 ) '
					'-> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch_current( self, lat: float, long: float, zone: str = 'auto',
			past_days: int = 0 ) -> Dict[ str, Any ] | None:
		'''

			Purpose:
			--------
			Retrieve current forecast conditions for a coordinate pair.

			Parameters:
			-----------
			lat (float):
				The latitude of the resolved location.

			long (float):
				The longitude of the resolved location.

			zone (str):
				The timezone for the response. Supports 'auto'.

			past_days (int):
				Optional number of previous days to include when supported by
				the forecast API.

			Returns:
			--------
			Dict[str, Any] | None:
				Normalized forecast response containing current conditions.

		'''
		try:
			throw_if( 'lat', lat )
			throw_if( 'long', long )
			throw_if( 'zone', zone )
			
			self.mode = 'current'
			self.latitude = float( lat )
			self.longitude = float( long )
			self.timezone = str( zone or 'auto' ).strip( )
			
			self.params = {
					'latitude': self.latitude,
					'longitude': self.longitude,
					'current': self.current_metrics,
					'timezone': self.timezone,
					'temperature_unit': self.temperature_unit,
					'wind_speed_unit': self.windspeed_unit,
					'precipitation_unit': self.precipitation_unit,
					'past_days': int( past_days )
			}
			
			self.url = self.forecast_url
			self.response = requests.get(
				self.url,
				params=self.params,
				headers=self.headers,
				timeout=30
			)
			self.response.raise_for_status( )
			
			payload = self.response.json( ) or { }
			
			return {
					'mode': self.mode,
					'location': self.location,
					'latitude': payload.get( 'latitude', self.latitude ),
					'longitude': payload.get( 'longitude', self.longitude ),
					'timezone': payload.get( 'timezone', self.timezone ),
					'url': self.url,
					'params': self.params,
					'data': payload
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenWeather'
			exception.method = (
					'fetch_current( self, lat: float, long: float, zone: str=auto, '
					'past_days: int=0 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch_hourly( self, lat: float, long: float, zone: str = 'auto',
			forecast_days: int = 7, past_days: int = 0 ) -> Dict[ str, Any ] | None:
		'''

			Purpose:
			--------
			Retrieve hourly forecast data for a coordinate pair.

			Parameters:
			-----------
			lat (float):
				The latitude of the resolved location.

			long (float):
				The longitude of the resolved location.

			zone (str):
				The timezone for the response. Supports 'auto'.

			forecast_days (int):
				Number of forecast days to request.

			past_days (int):
				Optional number of previous days to include.

			Returns:
			--------
			Dict[str, Any] | None:
				Normalized forecast response containing hourly data.

		'''
		try:
			throw_if( 'lat', lat )
			throw_if( 'long', long )
			throw_if( 'zone', zone )
			
			self.mode = 'hourly'
			self.latitude = float( lat )
			self.longitude = float( long )
			self.timezone = str( zone or 'auto' ).strip( )
			
			self.params = {
					'latitude': self.latitude,
					'longitude': self.longitude,
					'hourly': self.hourly_metrics,
					'timezone': self.timezone,
					'forecast_days': int( forecast_days ),
					'past_days': int( past_days ),
					'temperature_unit': self.temperature_unit,
					'wind_speed_unit': self.windspeed_unit,
					'precipitation_unit': self.precipitation_unit
			}
			
			self.url = self.forecast_url
			self.response = requests.get(
				self.url,
				params=self.params,
				headers=self.headers,
				timeout=30
			)
			self.response.raise_for_status( )
			
			payload = self.response.json( ) or { }
			
			return {
					'mode': self.mode,
					'location': self.location,
					'latitude': payload.get( 'latitude', self.latitude ),
					'longitude': payload.get( 'longitude', self.longitude ),
					'timezone': payload.get( 'timezone', self.timezone ),
					'url': self.url,
					'params': self.params,
					'data': payload
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenWeather'
			exception.method = (
					'fetch_hourly( self, lat: float, long: float, zone: str=auto, '
					'forecast_days: int=7, past_days: int=0 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch_daily( self, lat: float, long: float, zone: str = 'auto',
			forecast_days: int = 7, past_days: int = 0 ) -> Dict[ str, Any ] | None:
		'''

			Purpose:
			--------
			Retrieve daily forecast data for a coordinate pair.

			Parameters:
			-----------
			lat (float):
				The latitude of the resolved location.

			long (float):
				The longitude of the resolved location.

			zone (str):
				The timezone for the response. Supports 'auto'.

			forecast_days (int):
				Number of forecast days to request.

			past_days (int):
				Optional number of previous days to include.

			Returns:
			--------
			Dict[str, Any] | None:
				Normalized forecast response containing daily data.

		'''
		try:
			throw_if( 'lat', lat )
			throw_if( 'long', long )
			throw_if( 'zone', zone )
			
			self.mode = 'daily'
			self.latitude = float( lat )
			self.longitude = float( long )
			self.timezone = str( zone or 'auto' ).strip( )
			
			self.params = {
					'latitude': self.latitude,
					'longitude': self.longitude,
					'daily': self.daily_metrics,
					'timezone': self.timezone,
					'forecast_days': int( forecast_days ),
					'past_days': int( past_days ),
					'temperature_unit': self.temperature_unit,
					'wind_speed_unit': self.windspeed_unit,
					'precipitation_unit': self.precipitation_unit
			}
			
			self.url = self.forecast_url
			self.response = requests.get(
				self.url,
				params=self.params,
				headers=self.headers,
				timeout=30
			)
			self.response.raise_for_status( )
			
			payload = self.response.json( ) or { }
			
			return {
					'mode': self.mode,
					'location': self.location,
					'latitude': payload.get( 'latitude', self.latitude ),
					'longitude': payload.get( 'longitude', self.longitude ),
					'timezone': payload.get( 'timezone', self.timezone ),
					'url': self.url,
					'params': self.params,
					'data': payload
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenWeather'
			exception.method = (
					'fetch_daily( self, lat: float, long: float, zone: str=auto, '
					'forecast_days: int=7, past_days: int=0 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch( self, location: str, mode: str = 'current', zone: str = 'auto',
			forecast_days: int = 7, past_days: int = 0,
			count: int = 10 ) -> Dict[ str, Any ] | None:
		'''

			Purpose:
			--------
			Resolve a location string to coordinates, then retrieve forecast weather
			using the selected non-historical mode.

			Parameters:
			-----------
			location (str):
				The place name entered by the user.

			mode (str):
				Forecast mode. Supported values are current, hourly, and daily.

			zone (str):
				The timezone requested for the forecast response. If 'auto', the API
				will resolve the local timezone for the coordinates.

			forecast_days (int):
				Number of forecast days to request for hourly and daily modes.

			past_days (int):
				Optional number of previous days to include.

			count (int):
				Maximum number of geocoding matches to request before selecting the
				first result.

			Returns:
			--------
			Dict[str, Any] | None:
				Combined geocoding and forecast result.

		'''
		try:
			throw_if( 'location', location )
			throw_if( 'mode', mode )
			throw_if( 'zone', zone )
			
			selected = self.geocode_location( location=location, count=count )
			
			if not selected:
				return {
						'location': location,
						'mode': mode,
						'message': 'No geocoding results found.',
						'data': { }
				}
			
			lat = selected.get( 'latitude', None )
			long = selected.get( 'longitude', None )
			
			if lat is None or long is None:
				return {
						'location': location,
						'mode': mode,
						'message': 'Geocoding returned no usable coordinates.',
						'geocoding': selected,
						'data': { }
				}
			
			active_mode = str( mode ).strip( ).lower( )
			if active_mode == 'current':
				result = self.fetch_current(
					lat=float( lat ),
					long=float( long ),
					zone=str( zone or 'auto' ).strip( ),
					past_days=int( past_days )
				) or { }
			
			elif active_mode == 'hourly':
				result = self.fetch_hourly(
					lat=float( lat ),
					long=float( long ),
					zone=str( zone or 'auto' ).strip( ),
					forecast_days=int( forecast_days ),
					past_days=int( past_days )
				) or { }
			
			elif active_mode == 'daily':
				result = self.fetch_daily(
					lat=float( lat ),
					long=float( long ),
					zone=str( zone or 'auto' ).strip( ),
					forecast_days=int( forecast_days ),
					past_days=int( past_days )
				) or { }
			
			else:
				raise ValueError(
					"Unsupported mode. Use 'current', 'hourly', or 'daily'."
				)
			
			result[ 'geocoding' ] = selected
			return result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenWeather'
			exception.method = (
					'fetch( self, location: str, mode: str=current, zone: str=auto, '
					'forecast_days: int=7, past_days: int=0, count: int=10 ) '
					'-> Dict[ str, Any ]'
			)
			raise exception
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict,
			required: list[ str ] ) -> Dict[ str, str ] | None:
		'''

			Purpose:
			--------
			Construct and return a fully dynamic OpenAI Tool API schema definition.

			Parameters:
			-----------
			function (str):
				The function name exposed to the LLM.

			tool (str):
				The underlying system or service the function wraps.

			description (str):
				Precise explanation of what the function does.

			parameters (dict):
				A dictionary defining parameter names and JSON schema descriptors.

			required (list[str]):
				List of required parameter names.

			Returns:
			--------
			Dict[str, str] | None:
				A JSON-compatible dictionary defining the tool schema.

		'''
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return {
					'name': function.strip( ),
					'description': f"{description.strip( )} This function uses the {tool.strip( )} service.",
					'parameters': {
							'type': 'object',
							'properties': parameters,
							'required': required
					}
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'OpenWeather'
			exception.method = (
					'create_schema( self, function: str, tool: str, description: str, '
					'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]'
			)
			raise exception

class HistoricalWeather( Fetcher ):
	'''

		Purpose:
		--------
		Provides historical weather retrieval by location name and date using the
		Open-Meteo Geocoding API and Open-Meteo Historical Weather API.

		This class is intentionally designed around the actual user-facing need in
		the Foo fetcher expander: enter a location and a date, resolve that location
		to coordinates, then retrieve historical weather for that date.

		Referenced API Requirements:
		----------------------------
		Geocoding API:
			- Endpoint: https://geocoding-api.open-meteo.com/v1/search
			- Required parameter: name
			- Optional parameter: count

		Historical Weather API:
			- Endpoint: https://archive-api.open-meteo.com/v1/archive
			- Required parameters: latitude, longitude, start_date, end_date
			- Optional parameters used here:
				- timezone
				- daily
				- hourly
				- temperature_unit
				- wind_speed_unit
				- precipitation_unit

		Attributes:
		-----------
		geocode_url: Optional[str]
			The Open-Meteo geocoding endpoint.

		archive_url: Optional[str]
			The Open-Meteo historical weather endpoint.

		location: Optional[str]
			User-supplied location query.

		latitude: Optional[float]
			Resolved latitude from geocoding.

		longitude: Optional[float]
			Resolved longitude from geocoding.

		timezone: Optional[str]
			Resolved or user-requested timezone. Defaults to 'auto'.

		target_date: Optional[dt.date]
			The requested historical date.

		daily_metrics: Optional[List[str]]
			Daily historical metrics requested from the archive API.

		hourly_metrics: Optional[List[str]]
			Hourly historical metrics requested from the archive API.

		windspeed_unit: Optional[str]
			Wind speed unit passed to the API.

		temperature_unit: Optional[str]
			Temperature unit passed to the API.

		precipitation_unit: Optional[str]
			Precipitation unit passed to the API.

		params: Optional[Dict[str, Any]]
			Request parameters for the historical weather call.

		geocode_params: Optional[Dict[str, Any]]
			Request parameters for the geocoding call.

		result_limit: Optional[int]
			Maximum number of geocoding candidates to request.

		Methods:
		--------
		__init__() -> None
			Initialize the fetcher and default metrics.

		__dir__() -> List[str]
			Provide ordered introspection members.

		fetch(...) -> Dict[str, Any] | None
			Resolve a location string and retrieve historical weather for one date.

		geocode_location(...) -> Dict[str, Any] | None
			Resolve a place name into a selected geocoding record.

		fetch_historical(...) -> Dict[str, Any] | None
			Retrieve historical weather for resolved coordinates and a date.

		create_schema(...) -> Dict[str, str] | None
			Generate a dynamic tool schema definition.

	'''
	geocode_url: Optional[ str ]
	archive_url: Optional[ str ]
	location: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	timezone: Optional[ str ]
	target_date: Optional[ dt.date ]
	daily_metrics: Optional[ List[ str ] ]
	hourly_metrics: Optional[ List[ str ] ]
	windspeed_unit: Optional[ str ]
	temperature_unit: Optional[ str ]
	precipitation_unit: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	geocode_params: Optional[ Dict[ str, Any ] ]
	result_limit: Optional[ int ]
	agents: Optional[ str ]
	
	def __init__( self ) -> None:
		'''

			Purpose:
			--------
			Initialize the HistoricalWeather fetcher with default endpoints,
			request headers, metric collections, and unit selections.

			Parameters:
			-----------
			None

			Returns:
			--------
			None

		'''
		super( ).__init__( )
		self.headers = { }
		self.agents = cfg.AGENTS
		self.location = None
		self.latitude = None
		self.longitude = None
		self.timezone = 'auto'
		self.target_date = None
		self.result_limit = 10
		self.geocode_url = 'https://geocoding-api.open-meteo.com/v1/search'
		self.archive_url = 'https://archive-api.open-meteo.com/v1/archive'
		self.temperature_unit = 'fahrenheit'
		self.windspeed_unit = 'kn'
		self.precipitation_unit = 'inch'
		self.geocode_params = None
		self.params = None
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
		
		self.daily_metrics = [
				'weather_code',
				'temperature_2m_max',
				'temperature_2m_min',
				'apparent_temperature_max',
				'apparent_temperature_min',
				'sunrise',
				'sunset',
				'daylight_duration',
				'sunshine_duration',
				'precipitation_sum',
				'rain_sum',
				'showers_sum',
				'snowfall_sum',
				'precipitation_hours',
				'wind_speed_10m_max',
				'wind_gusts_10m_max',
				'wind_direction_10m_dominant'
		]
		
		self.hourly_metrics = [
				'temperature_2m',
				'relative_humidity_2m',
				'apparent_temperature',
				'precipitation',
				'rain',
				'showers',
				'snowfall',
				'weather_code',
				'cloud_cover',
				'pressure_msl',
				'surface_pressure',
				'wind_speed_10m',
				'wind_direction_10m',
				'wind_gusts_10m'
		]
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			--------
			Provide ordered member visibility for introspection and editor discovery.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Ordered attribute and method names.

		'''
		return [
				'geocode_url',
				'archive_url',
				'location',
				'latitude',
				'longitude',
				'timezone',
				'target_date',
				'daily_metrics',
				'hourly_metrics',
				'windspeed_unit',
				'temperature_unit',
				'precipitation_unit',
				'params',
				'geocode_params',
				'result_limit',
				'fetch',
				'geocode_location',
				'fetch_historical',
				'create_schema'
		]
	
	def geocode_location( self, location: str, count: int = 10 ) -> Dict[ str, Any ] | None:
		'''

			Purpose:
			--------
			Resolve a user-supplied location string into a geocoding result from
			the Open-Meteo Geocoding API.

			Parameters:
			-----------
			location (str):
				The place name or postal code to search for.

			count (int):
				The maximum number of geocoding matches to request.

			Returns:
			--------
			Dict[str, Any] | None:
				The selected geocoding record, typically the first result.

		'''
		try:
			throw_if( 'location', location )
			self.location = location.strip( )
			self.geocode_params = {
					'name': self.location,
					'count': int( count )
			}
			
			self.url = self.geocode_url
			self.response = requests.get(
				self.url,
				params=self.geocode_params,
				headers=self.headers,
				timeout=20
			)
			self.response.raise_for_status( )
			
			payload = self.response.json( ) or { }
			results = payload.get( 'results', [ ] ) or [ ]
			
			if not results:
				return None
			
			selected = results[ 0 ]
			self.latitude = selected.get( 'latitude', None )
			self.longitude = selected.get( 'longitude', None )
			
			resolved_timezone = selected.get( 'timezone', None )
			if resolved_timezone:
				self.timezone = resolved_timezone
			
			return selected
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'HistoricalWeather'
			exception.method = (
					'geocode_location( self, location: str, count: int=10 ) '
					'-> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch_historical( self, lat: float, long: float, date: dt.date,
			zone: str = 'auto' ) -> Dict[ str, Any ] | None:
		'''

			Purpose:
			--------
			Retrieve historical weather for a single date using the Open-Meteo
			Historical Weather API.

			Parameters:
			-----------
			lat (float):
				The latitude of the resolved location.

			long (float):
				The longitude of the resolved location.

			date (dt.date):
				The requested historical date. This is used for both start_date
				and end_date so the response is limited to that day.

			zone (str):
				The timezone to use in the response. Supports 'auto'.

			Returns:
			--------
			Dict[str, Any] | None:
				Normalized response payload including request metadata, selected
				location metadata, and archive data.

		'''
		try:
			throw_if( 'lat', lat )
			throw_if( 'long', long )
			throw_if( 'date', date )
			throw_if( 'zone', zone )
			
			self.latitude = float( lat )
			self.longitude = float( long )
			self.target_date = date
			self.timezone = zone.strip( ) if zone else 'auto'
			
			self.params = {
					'latitude': self.latitude,
					'longitude': self.longitude,
					'start_date': self.target_date.isoformat( ),
					'end_date': self.target_date.isoformat( ),
					'timezone': self.timezone,
					'daily': self.daily_metrics,
					'hourly': self.hourly_metrics,
					'temperature_unit': self.temperature_unit,
					'wind_speed_unit': self.windspeed_unit,
					'precipitation_unit': self.precipitation_unit
			}
			
			self.url = self.archive_url
			self.response = requests.get(
				self.url,
				params=self.params,
				headers=self.headers,
				timeout=30
			)
			self.response.raise_for_status( )
			
			payload = self.response.json( ) or { }
			
			return {
					'location': self.location,
					'latitude': self.latitude,
					'longitude': self.longitude,
					'timezone': payload.get( 'timezone', self.timezone ),
					'date': self.target_date.isoformat( ),
					'url': self.url,
					'params': self.params,
					'data': payload
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'HistoricalWeather'
			exception.method = (
					'fetch_historical( self, lat: float, long: float, date: dt.date, '
					'zone: str=auto ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def fetch( self, location: str, date: dt.date,
			zone: str = 'auto', count: int = 10 ) -> Dict[ str, Any ] | None:
		'''

			Purpose:
			--------
			Resolve a location string to coordinates, then retrieve historical
			weather for the requested date.

			Parameters:
			-----------
			location (str):
				The place name entered by the user.

			date (dt.date):
				The requested historical date.

			zone (str):
				The timezone requested for the archive response. If 'auto', the API
				will resolve the local timezone for the coordinates.

			count (int):
				Maximum number of geocoding matches to request before selecting the
				first result.

			Returns:
			--------
			Dict[str, Any] | None:
				Combined geocoding and historical weather result.

		'''
		try:
			throw_if( 'location', location )
			throw_if( 'date', date )
			throw_if( 'zone', zone )
			
			selected = self.geocode_location( location=location, count=count )
			
			if not selected:
				return {
						'location': location,
						'date': date.isoformat( ),
						'message': 'No geocoding results found.',
						'data': { }
				}
			
			lat = selected.get( 'latitude', None )
			long = selected.get( 'longitude', None )
			
			if lat is None or long is None:
				return {
						'location': location,
						'date': date.isoformat( ),
						'message': 'Geocoding returned no usable coordinates.',
						'geocoding': selected,
						'data': { }
				}
			
			result = self.fetch_historical(
				lat=float( lat ),
				long=float( long ),
				date=date,
				zone=str( zone or 'auto' ).strip( )
			) or { }
			
			result[ 'geocoding' ] = selected
			return result
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'HistoricalWeather'
			exception.method = (
					'fetch( self, location: str, date: dt.date, zone: str=auto, '
					'count: int=10 ) -> Dict[ str, Any ]'
			)
			raise exception
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict,
			required: list[ str ] ) -> Dict[ str, str ] | None:
		'''

			Purpose:
			--------
			Construct and return a fully dynamic OpenAI Tool API schema definition.

			Parameters:
			-----------
			function (str):
				The function name exposed to the LLM.

			tool (str):
				The underlying system or service the function wraps.

			description (str):
				Precise explanation of what the function does.

			parameters (dict):
				A dictionary defining parameter names and JSON schema descriptors.

			required (list[str]):
				List of required parameter names.

			Returns:
			--------
			Dict[str, str] | None:
				A JSON-compatible dictionary defining the tool schema.

		'''
		try:
			throw_if( 'function', function )
			throw_if( 'tool', tool )
			throw_if( 'description', description )
			throw_if( 'parameters', parameters )
			
			if required is None:
				required = list( parameters.keys( ) )
			
			return {
					'name': function.strip( ),
					'description': f"{description.strip( )} This function uses the {tool.strip( )} service.",
					'parameters': {
							'type': 'object',
							'properties': parameters,
							'required': required
					}
			}
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'HistoricalWeather'
			exception.method = (
					'create_schema( self, function: str, tool: str, description: str, '
					'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]'
			)
			raise exception
			
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
	client: Optional[ GroqClient ]
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
	top_p: Optional[ float ]
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
		self.api_key = cfg.GROQ_API_KEY
		self.model = 'openai/gpt-oss-120b'
		self.url = r'https://api.groq.com/openai/v1?'
		self.client = None
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
			self.client = GroqClient( api_key=self.api_key )
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
			raise exception
			
	
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
			chat_completion = self.client.chat.completions.create(
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
			raise exception
			
		
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
			raise exception
			
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
	top_p: Optional[ float ]
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
		self.client = None
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
			self.client = genai.Client( api_key=self.api_key )
			_response = self.client.models.generate_content( model=self.model, contents=self.contents, )
			return _response.text
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Gemini'
			exception.method = 'fetch( self, query: str ) -> str '
			raise exception
			
	
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
			self.client = genai.Client( api_key=self.api_key )
			_response = self.client.models.generate_content( model=self.model, contents=self.contents, )
			return _response.text
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Gemini'
			exception.method = 'analyze( self, query: str, path: str ) -> str'
			raise exception
			
		
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
			raise exception
			
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
	top_p: Optional[ float ]
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
			raise exception
			
		
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
			raise exception
			
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
	top_p: Optional[ float ]
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
		self.client = None
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
			raise exception
			
		
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
			raise exception
			
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
			raise exception
			
	
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
			raise exception
			
	
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
			raise exception
			
	
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
			raise exception
			
	
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
			raise exception
			
	
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
			raise exception
			
	
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
			raise exception
			
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
			raise exception
			
	
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
			raise exception
			
	
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
			raise exception
			
		
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
			raise exception
			