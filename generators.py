'''
  ******************************************************************************************
      Assembly:                Foo
      Filename:                generators.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="generators.py" company="Terry D. Eppler">

	     generators.py
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
    generators.py
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

from anthropic import Anthropic as Claude
import base64
from boogr import Error
from core import Result
import config as cfg
from google import genai
from openai import OpenAI
from pathlib import Path
from typing import Any, Dict, Optional, Pattern, List, Tuple
from requests import Response
from xai_sdk import Client as Xai
from mistralai import Mistral as MistralAI
import re
import urllib

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

class Generator:
	'''

		Purpose:
		--------
		Base class for Generator subclasses

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
	
	def fetch( self, query: str, url: str, time: int = 10 ) -> Result | None:
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

class Grok( Generator ):
	'''
	
		Purpose:
		---------
		Class providing xAI Grok text-generation and search functionality through
		the OpenAI-compatible xAI Responses API.
	
		Attribues:
		-----------
		client - Grok
		model - str
		response - Any
		api_key - str
		query - str
		params - Dict[ str, Any ]
		temperature - float
		max_tokens - int
		top_p - float
		reasoning_effort - str | None
		stream - bool
		store - bool
		messages - List[ Dict[ str, Any ] ]
	
		Methods:
		-----------
		fetch( ) -> str
		generate_text( ) -> str
		search_web( ) -> str
	
	'''
	client: Optional[ Xai ]
	model: Optional[ str ]
	response: Optional[ Response ]
	api_key: Optional[ str ]
	query: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	temperature: Optional[ float ]
	max_tokens: Optional[ int ]
	top_p: Optional[ float ]
	reasoning_effort: Optional[ str ]
	stream: Optional[ bool ]
	store: Optional[ bool ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	system_instructions: Optional[ str ]
	web_search: Optional[ bool ]
	search_domains: Optional[ List[ str ] ]
	parallel_tool_calls: Optional[ bool ]
	tool_choice: Optional[ str ]
	tools: Optional[ List[ Dict[ str, Any ] ] ]
	
	def __init__( self ) -> None:
		'''
		
			Purpose:
			-----------
			Initialize xAI Grok via the OpenAI-compatible Responses API.
			
			Parameters:
			-----------
			None
			
			Returns:
			-----------
			None
			
		'''
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.model = 'grok-4-fast-reasoning'
		self.client = Xai( api_key=self.api_key, base_url='https://api.x.ai/v1' )
		self.messages = None
		self.temperature = 0.7
		self.top_p = 1.0
		self.max_tokens = 2048
		self.reasoning_effort = None
		self.headers = { }
		self.timeout = None
		self.file_path = None
		self.content = None
		self.params = None
		self.response = None
		self.query = None
		self.system_instructions = None
		self.web_search = False
		self.search_domains = [ ]
		self.parallel_tool_calls = True
		self.tool_choice = 'auto'
		self.tools = [ ]
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		'''
		
			Purpose:
			-----------
			Grok list of members.
			
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
		         'agents',
		         'system_instructions',
		         'web_search',
		         'search_domains',
		         'parallel_tool_calls',
		         'tool_choice',
		         'tools' ]
	
	def _normalize_domains( self, domains: Any ) -> List[ str ]:
		'''
		
			Purpose:
			-----------
			Normalize domain input into a canonical, de-duplicated list.
			
			Parameters:
			-----------
			domains (Any): String, list, tuple, set, or None.
			
			Returns:
			-----------
			List[str]
			
		'''
		try:
			if domains is None:
				return [ ]
			
			if isinstance( domains, str ):
				_parts = re.split( r'[\n,;]+', domains )
			elif isinstance( domains, (list, tuple, set) ):
				_parts = [ str( x ) for x in domains if x is not None ]
			else:
				_parts = [ str( domains ) ]
			
			_values = [ ]
			for _entry in _parts:
				_value = str( _entry ).strip( ).lower( )
				if not _value:
					continue
				
				if not _value.startswith( 'http://' ) and not _value.startswith( 'https://' ):
					_value = f'https://{_value}'
				
				_parsed = urllib.parse.urlparse( _value )
				_domain = (_parsed.netloc or _parsed.path or '').strip( ).lower( )
				_domain = re.sub( r':\d+$', '', _domain )
				_domain = _domain.lstrip( '.' )
				
				if _domain.startswith( 'www.' ):
					_domain = _domain[ 4: ]
				
				if _domain and _domain not in _values:
					_values.append( _domain )
			
			return _values
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Grok'
			exception.method = '_normalize_domains( self, domains: Any ) -> List[ str ]'
			raise exception
	
	def _supports_reasoning_effort( self, model: str ) -> bool:
		'''
		
			Purpose:
			-----------
			Determine whether the selected xAI model accepts the reasoning_effort
			parameter.
			
			Parameters:
			-----------
			model (str): Model name.
			
			Returns:
			-----------
			bool
			
		'''
		try:
			throw_if( 'model', model )
			_name = str( model ).strip( ).lower( )
			return 'grok-3-mini' in _name
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Grok'
			exception.method = '_supports_reasoning_effort( self, model: str ) -> bool'
			raise exception
	
	def _is_reasoning_model( self, model: str ) -> bool:
		'''
		
			Purpose:
			-----------
			Identify models where reasoning is native or otherwise model-defined.
			
			Parameters:
			-----------
			model (str): Model name.
			
			Returns:
			-----------
			bool
			
		'''
		try:
			throw_if( 'model', model )
			_name = str( model ).strip( ).lower( )
			return 'grok-4' in _name or 'reasoning' in _name
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Grok'
			exception.method = '_is_reasoning_model( self, model: str ) -> bool'
			raise exception
	
	def _build_instructions( self, system: str | None = None, response_format: str | None = None ) -> str | None:
		'''
		
			Purpose:
			-----------
			Build the final instruction block sent to xAI.
			
			Parameters:
			-----------
			system (str | None): Optional system instructions.
			response_format (str | None): Optional output mode.
			
			Returns:
			-----------
			str | None
			
		'''
		try:
			_parts = [ ]
			if system and str( system ).strip( ):
				_parts.append( str( system ).strip( ) )
			if response_format and str( response_format ).strip( ).lower( ) == 'json':
				_parts.append( 'Return valid JSON only. Do not include markdown fences or commentary.' )
			
			if _parts:
				return '\n\n'.join( _parts )
			
			return None
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Grok'
			exception.method = ('_build_instructions( self, system: str | None=None, '
			                    'response_format: str | None=None ) -> str | None')
			raise exception
	
	def fetch( self, query: str, model: str = 'grok-4-fast-reasoning', temperature: float = 0.7,
			max_tokens: int = 2048, top_p: float = 1.0, seed: int | None = None,
			system: str | None = None, response_format: str | None = None,
			reasoning_effort: str | None = None, web_search: bool = False,
			search_domains: Any = None, stop: List[ str ] | None = None, stream: bool = False,
			store: bool = True, parallel_tool_calls: bool = True,
			tool_choice: str = 'auto' ) -> str | None:
		'''
		
			Purpose:
			-------
			Send an xAI Responses API request for Grok text generation, optional
			reasoning effort, and optional server-side web search with domain filtering.
			
			Parameters:
			-----------
			query (str): User prompt.
			model (str): Grok model name.
			temperature (float): Sampling temperature.
			max_tokens (int): Max output tokens.
			top_p (float): Top-p nucleus sampling parameter.
			seed (int | None): Optional deterministic seed.
			system (str | None): Optional system prompt.
			response_format (str | None): Optional output mode.
			reasoning_effort (str | None): Optional xAI reasoning effort.
			web_search (bool): Enable xAI web search.
			search_domains (Any): Optional domain filter input.
			stop (List[str] | None): Optional stop sequences.
			stream (bool): Stream response.
			store (bool): Store response server-side.
			parallel_tool_calls (bool): Allow server-side parallel tool calls.
			tool_choice (str): Tool choice behavior.
			
			Returns:
			---------
			str | None
			
		'''
		try:
			throw_if( 'query', query )
			throw_if( 'model', model )
			self.query = query
			self.model = str( model ).strip( )
			self.temperature = float( temperature )
			self.max_tokens = int( max_tokens )
			self.top_p = float( top_p )
			self.reasoning_effort = reasoning_effort if reasoning_effort else None
			self.stream = bool( stream )
			self.store = bool( store )
			self.web_search = bool( web_search )
			self.search_domains = self._normalize_domains( search_domains )
			self.parallel_tool_calls = bool( parallel_tool_calls )
			self.tool_choice = tool_choice or 'auto'
			self.system_instructions = self._build_instructions( system=system,
				response_format=response_format, )
			
			self.tools = [ ]
			if self.web_search:
				_web_tool = { 'type': 'web_search' }
				if self.search_domains:
					_web_tool[ 'allowed_domains' ] = self.search_domains
				self.tools.append( _web_tool )
			
			self.request = \
			{
				'model': self.model,
				'input': self.query,
				'max_output_tokens': self.max_tokens,
				'temperature': self.temperature,
				'top_p': self.top_p,
				'stream': self.stream,
				'store': self.store,
				'parallel_tool_calls': self.parallel_tool_calls,
			}
			
			if seed is not None:
				self.request[ 'seed' ] = int( seed )
			
			if self.system_instructions:
				self.request[ 'instructions' ] = self.system_instructions
			
			if self.tools:
				self.request[ 'tools' ] = self.tools
				self.request[ 'tool_choice' ] = self.tool_choice
			
			if self._supports_reasoning_effort( self.model ) and self.reasoning_effort:
				self.request[ 'reasoning_effort' ] = self.reasoning_effort
			
			if stop and not self._is_reasoning_model( self.model ):
				self.request[ 'stop' ] = stop
			
			self.response = self.client.responses.create( **self.request )
			return self.response.output_text
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Grok'
			exception.method = (
					'fetch( self, query: str, model: str="grok-4-fast-reasoning", '
					'temperature: float=0.7, max_tokens: int=2048, top_p: float=1.0, '
					'seed: int | None=None, system: str | None=None, '
					'response_format: str | None=None, reasoning_effort: str | None=None, '
					'web_search: bool=False, search_domains: Any=None, '
					'stop: List[ str ] | None=None, stream: bool=False, store: bool=True, '
					'parallel_tool_calls: bool=True, tool_choice: str="auto" ) -> str | None'
			)
			raise exception
	
	def generate_text( self, query: str, model: str = 'grok-4-fast-reasoning', temperature: float = 0.7,
			max_tokens: int = 2048, top_p: float = 1.0, seed: int | None = None,
			system: str | None = None, response_format: str | None = None,
			reasoning_effort: str | None = None, web_search: bool = False,
			search_domains: Any = None, stop: List[ str ] | None = None, stream: bool = False,
			store: bool = True, parallel_tool_calls: bool = True,
			tool_choice: str = 'auto' ) -> str | None:
		'''
		
			Purpose:
			-----------
			Convenience wrapper around fetch for text generation.
			
			Parameters:
			-----------
			query (str): User prompt.
			
			Returns:
			-----------
			str | None
			
		'''
		try:
			return self.fetch(
				query=query,
				model=model,
				temperature=temperature,
				max_tokens=max_tokens,
				top_p=top_p,
				seed=seed,
				system=system,
				response_format=response_format,
				reasoning_effort=reasoning_effort,
				web_search=web_search,
				search_domains=search_domains,
				stop=stop,
				stream=stream,
				store=store,
				parallel_tool_calls=parallel_tool_calls,
				tool_choice=tool_choice,
			)
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Grok'
			exception.method = 'generate_text( self, query: str, ... ) -> str | None'
			raise exception
	
	def search_web( self, query: str, model: str = 'grok-4-fast-reasoning', temperature: float = 0.7,
			max_tokens: int = 2048, top_p: float = 1.0, seed: int | None = None,
			system: str | None = None, response_format: str | None = None,
			reasoning_effort: str | None = None, search_domains: Any = None,
			stream: bool = False, store: bool = True, parallel_tool_calls: bool = True,
			tool_choice: str = 'auto' ) -> str | None:
		'''
		
			Purpose:
			-----------
			Convenience wrapper around fetch with web search enabled.
			
			Parameters:
			-----------
			query (str): User prompt.
			
			Returns:
			-----------
			str | None
			
		'''
		try:
			return self.fetch(
				query=query,
				model=model,
				temperature=temperature,
				max_tokens=max_tokens,
				top_p=top_p,
				seed=seed,
				system=system,
				response_format=response_format,
				reasoning_effort=reasoning_effort,
				web_search=True,
				search_domains=search_domains,
				stop=None,
				stream=stream,
				store=store,
				parallel_tool_calls=parallel_tool_calls,
				tool_choice=tool_choice,
			)
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Grok'
			exception.method = 'search_web( self, query: str, ... ) -> str | None'
			raise exception

class Gemini( Generator ):
	'''

		Purpose:
		---------
		Class providing Google's Gemini API text generation, optional grounding
		with Google Search, optional JSON output, and optional reasoning through
		thinking configuration.

		Attribues:
		-----------
		client - genai.Client
		prompt - str
		response - Any
		api_key - str
		id - str
		location - str
		use_vertex - bool
		model - str
		params - Dict[ str, Any ]
		temperature - float
		max_tokens - int
		top_p - float
		top_k - int | None
		candidate_count - int
		thinking_level - str | None
		include_thoughts - bool
		system_instructions - str | None
		grounding - bool
		search_domains - List[ str ]

		Methods:
		-----------
		fetch( ) -> str | None
		generate_text( ) -> str | None
		search_web( ) -> str | None

	'''
	client: Optional[ genai.Client ]
	prompt: Optional[ str ]
	response: Optional[ Response ]
	api_key: Optional[ str ]
	id: Optional[ str ]
	location: Optional[ str ]
	use_vertex: Optional[ bool ]
	model: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	temperature: Optional[ float ]
	max_tokens: Optional[ int ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	candidate_count: Optional[ int ]
	thinking_level: Optional[ str ]
	include_thoughts: Optional[ bool ]
	system_instructions: Optional[ str ]
	grounding: Optional[ bool ]
	search_domains: Optional[ List[ str ] ]
	http_options: Optional[ Dict[ str, Any ] ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	
	def __init__( self ) -> None:
		'''
		
			Purpose:
			-----------
			Initialize the Gemini class.
			
		'''
		super( ).__init__( )
		self.api_key = cfg.GEMINI_API_KEY
		self.id = cfg.GOOGLE_PROJECT_ID
		self.location = cfg.GOOGLE_CLOUD_LOCATION
		self.use_vertex = self._to_bool( cfg.GOOGLE_GENAI_USE_VERTEXAI )
		self.model = 'gemini-2.5-flash'
		self.headers = { }
		self.client = self._create_client( )
		self.timeout = None
		self.params = None
		self.response = None
		self.agents = cfg.AGENTS
		self.temperature = 0.7
		self.max_tokens = 2048
		self.top_p = 1.0
		self.top_k = None
		self.candidate_count = 1
		self.thinking_level = None
		self.include_thoughts = False
		self.system_instructions = None
		self.grounding = False
		self.search_domains = [ ]
		self.http_options = None
		self.messages = None
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-----------
			Gemini list of members.

			Parameters:
			-----------
			None

			Returns:
			-----------
			list[str]: Ordered attribute/method names.

		'''
		return [ 'client',
		         'prompt',
		         'response',
		         'api_key',
		         'id',
		         'location',
		         'use_vertex',
		         'model',
		         'params',
		         'temperature',
		         'max_tokens',
		         'top_p',
		         'top_k',
		         'candidate_count',
		         'thinking_level',
		         'include_thoughts',
		         'system_instructions',
		         'grounding',
		         'search_domains',
		         'fetch',
		         'generate_text',
		         'search_web' ]
	
	def _to_bool( self, value: Any ) -> bool:
		'''
		
			Purpose:
			-----------
			Convert configuration values into a stable boolean.
			
			Parameters:
			-----------
			value (Any): Source value.
			
			Returns:
			-----------
			bool
			
		'''
		try:
			if isinstance( value, bool ):
				return value
			
			if value is None:
				return False
			
			return str( value ).strip( ).lower( ) in [ '1', 'true', 'yes', 'y', 'on' ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Gemini'
			exception.method = '_to_bool( self, value: Any ) -> bool'
			raise exception
	
	def _create_client( self ) -> genai.Client:
		'''
		
			Purpose:
			-----------
			Create a Gemini client against either the Gemini Developer API or
			Vertex AI, depending on configuration.
			
			Parameters:
			-----------
			None
			
			Returns:
			-----------
			genai.Client
			
		'''
		try:
			if self.use_vertex:
				return genai.Client(
					vertexai=True,
					project=self.id,
					location=self.location,
				)
			
			return genai.Client( api_key=self.api_key )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Gemini'
			exception.method = '_create_client( self ) -> genai.Client'
			raise exception
	
	def _normalize_domains( self, domains: Any ) -> List[ str ]:
		'''
		
			Purpose:
			-----------
			Normalize domain input into a canonical, de-duplicated list.
			
			Parameters:
			-----------
			domains (Any): String, list, tuple, set, or None.
			
			Returns:
			-----------
			List[str]
			
		'''
		try:
			if domains is None:
				return [ ]
			
			if isinstance( domains, str ):
				_parts = re.split( r'[\n,;]+', domains )
			elif isinstance( domains, (list, tuple, set) ):
				_parts = [ str( x ) for x in domains if x is not None ]
			else:
				_parts = [ str( domains ) ]
			
			_values = [ ]
			for _entry in _parts:
				_value = str( _entry ).strip( ).lower( )
				if not _value:
					continue
				
				if not _value.startswith( 'http://' ) and not _value.startswith( 'https://' ):
					_value = f'https://{_value}'
				
				_parsed = urllib.parse.urlparse( _value )
				_domain = (_parsed.netloc or _parsed.path or '').strip( ).lower( )
				_domain = re.sub( r':\d+$', '', _domain )
				_domain = _domain.lstrip( '.' )
				
				if _domain.startswith( 'www.' ):
					_domain = _domain[ 4: ]
				
				if _domain and _domain not in _values:
					_values.append( _domain )
			
			return _values
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Gemini'
			exception.method = '_normalize_domains( self, domains: Any ) -> List[ str ]'
			raise exception
	
	def _supports_thinking( self, model: str ) -> bool:
		'''
		
			Purpose:
			-----------
			Determine whether the selected Gemini model should receive a thinking
			configuration.
			
			Parameters:
			-----------
			model (str): Model name.
			
			Returns:
			-----------
			bool
			
		'''
		try:
			throw_if( 'model', model )
			_name = str( model ).strip( ).lower( )
			return _name.startswith( 'gemini-3' )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Gemini'
			exception.method = '_supports_thinking( self, model: str ) -> bool'
			raise exception
	
	def _build_system_instruction( self, system: str | None = None, response_format: str | None = None,
			search_domains: Any = None ) -> str | None:
		'''
		
			Purpose:
			-----------
			Build the final Gemini system instruction string.
			
			Parameters:
			-----------
			system (str | None): Optional system prompt.
			response_format (str | None): Optional output mode.
			search_domains (Any): Optional domain preference input.
			
			Returns:
			-----------
			str | None
			
		'''
		try:
			_parts = [ ]
			
			if system and str( system ).strip( ):
				_parts.append( str( system ).strip( ) )
			
			if response_format and str( response_format ).strip( ).lower( ) == 'json':
				_parts.append( 'Return valid JSON only. Do not include markdown fences or commentary.' )
			
			_domains = self._normalize_domains( search_domains )
			if _domains:
				_parts.append(
					'When grounding or searching the web, prefer sources from these domains '
					f'when relevant and available: {", ".join( _domains )}.'
				)
			
			if _parts:
				return '\n\n'.join( _parts )
			
			return None
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Gemini'
			exception.method = ('_build_system_instruction( self, system: str | None=None, '
			                    'response_format: str | None=None, search_domains: Any=None ) -> str | None')
			raise exception
	
	def _extract_text( self, response: Any ) -> str:
		'''
		
			Purpose:
			-----------
			Extract plain text from a Gemini response.
			
			Parameters:
			-----------
			response (Any): Gemini response object.
			
			Returns:
			-----------
			str
			
		'''
		try:
			if response is None:
				return ''
			
			if hasattr( response, 'text' ) and response.text:
				return str( response.text )
			
			if hasattr( response, 'candidates' ) and response.candidates:
				_parts = [ ]
				for _candidate in response.candidates:
					_content = getattr( _candidate, 'content', None )
					if _content is None:
						continue
					
					for _part in getattr( _content, 'parts', [ ] ):
						_text = getattr( _part, 'text', None )
						if _text:
							_parts.append( str( _text ) )
				
				if _parts:
					return '\n'.join( _parts ).strip( )
			
			return str( response )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Gemini'
			exception.method = '_extract_text( self, response: Any ) -> str'
			raise exception
	
	def fetch( self, prompt: str, model: str = 'gemini-2.5-flash', temperature: float = 0.7,
			max_tokens: int = 2048, top_p: float = 1.0, top_k: int | None = None,
			candidate_count: int = 1, seed: int | None = None, system: str | None = None,
			response_format: str | None = None, stop_sequences: List[ str ] | None = None,
			grounding: bool = False, search_domains: Any = None, reasoning: bool = False,
			thinking_level: str | None = None, include_thoughts: bool = False ) -> str | None:
		'''
		
			Purpose:
			-------
			Send a Gemini GenerateContent request for text generation with optional
			grounding, JSON output, and thinking configuration.
			
			Parameters:
			-----------
			prompt (str): User prompt.
			model (str): Gemini model name.
			temperature (float): Sampling temperature.
			max_tokens (int): Max output tokens.
			top_p (float): Top-p nucleus sampling parameter.
			top_k (int | None): Top-k sampling parameter.
			candidate_count (int): Number of candidates.
			seed (int | None): Optional deterministic seed.
			system (str | None): Optional system prompt.
			response_format (str | None): Optional output mode.
			stop_sequences (List[str] | None): Optional stop sequences.
			grounding (bool): Enable grounding with Google Search.
			search_domains (Any): Optional preferred domains.
			reasoning (bool): Enable thinking configuration where supported.
			thinking_level (str | None): Optional Gemini thinking level.
			include_thoughts (bool): Include thoughts in response where available.
			
			Returns:
			---------
			str | None
			
		'''
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.prompt = prompt
			self.model = str( model ).strip( )
			self.temperature = float( temperature )
			self.max_tokens = int( max_tokens )
			self.top_p = float( top_p )
			self.top_k = int( top_k ) if top_k is not None else None
			self.candidate_count = int( candidate_count )
			self.thinking_level = thinking_level if thinking_level else None
			self.include_thoughts = bool( include_thoughts )
			self.grounding = bool( grounding )
			self.search_domains = self._normalize_domains( search_domains )
			self.system_instructions = self._build_system_instruction(
				system=system,
				response_format=response_format,
				search_domains=self.search_domains,
			)
			
			self.tools = [ ]
			if self.grounding:
				self.tools.append(
					genai.types.Tool(
						google_search=genai.types.GoogleSearch( )
					)
				)
			
			self.config = genai.types.GenerateContentConfig(
				max_output_tokens=self.max_tokens,
				temperature=self.temperature,
				top_p=self.top_p,
				candidate_count=self.candidate_count,
			)
			
			if self.top_k is not None and self.top_k > 0:
				self.config.top_k = self.top_k
			
			if seed is not None:
				self.config.seed = int( seed )
			
			if stop_sequences:
				self.config.stop_sequences = stop_sequences
			
			if response_format and str( response_format ).strip( ).lower( ) == 'json':
				self.config.response_mime_type = 'application/json'
			
			if self.system_instructions:
				self.config.system_instruction = self.system_instructions
			
			if self.tools:
				self.config.tools = self.tools
			
			if reasoning and self._supports_thinking( self.model ):
				self.config.thinking_config = genai.types.ThinkingConfig(
					thinking_level=(self.thinking_level or 'low'),
					include_thoughts=self.include_thoughts,
				)
			
			self.response = self.client.models.generate_content(
				model=self.model,
				contents=self.prompt,
				config=self.config,
			)
			
			return self._extract_text( self.response )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Gemini'
			exception.method = (
					'fetch( self, prompt: str, model: str="gemini-2.5-flash", '
					'temperature: float=0.7, max_tokens: int=2048, top_p: float=1.0, '
					'top_k: int | None=None, candidate_count: int=1, seed: int | None=None, '
					'system: str | None=None, response_format: str | None=None, '
					'stop_sequences: List[ str ] | None=None, grounding: bool=False, '
					'search_domains: Any=None, reasoning: bool=False, '
					'thinking_level: str | None=None, include_thoughts: bool=False ) -> str | None'
			)
			raise exception
	
	def generate_text( self, prompt: str, model: str = 'gemini-2.5-flash', temperature: float = 0.7,
			max_tokens: int = 2048, top_p: float = 1.0, top_k: int | None = None,
			candidate_count: int = 1, seed: int | None = None, system: str | None = None,
			response_format: str | None = None, stop_sequences: List[ str ] | None = None,
			grounding: bool = False, search_domains: Any = None, reasoning: bool = False,
			thinking_level: str | None = None, include_thoughts: bool = False ) -> str | None:
		'''
		
			Purpose:
			-----------
			Convenience wrapper around fetch for text generation.
			
			Parameters:
			-----------
			prompt (str): User prompt.
			
			Returns:
			-----------
			str | None
			
		'''
		try:
			return self.fetch(
				prompt=prompt,
				model=model,
				temperature=temperature,
				max_tokens=max_tokens,
				top_p=top_p,
				top_k=top_k,
				candidate_count=candidate_count,
				seed=seed,
				system=system,
				response_format=response_format,
				stop_sequences=stop_sequences,
				grounding=grounding,
				search_domains=search_domains,
				reasoning=reasoning,
				thinking_level=thinking_level,
				include_thoughts=include_thoughts,
			)
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Gemini'
			exception.method = 'generate_text( self, prompt: str, ... ) -> str | None'
			raise exception
	
	def search_web( self, prompt: str, model: str = 'gemini-2.5-flash', temperature: float = 0.7,
			max_tokens: int = 2048, top_p: float = 1.0, top_k: int | None = None,
			candidate_count: int = 1, seed: int | None = None, system: str | None = None,
			response_format: str | None = None, stop_sequences: List[ str ] | None = None,
			search_domains: Any = None, reasoning: bool = False, thinking_level: str | None = None,
			include_thoughts: bool = False ) -> str | None:
		'''
		
			Purpose:
			-----------
			Convenience wrapper around fetch with grounding enabled.
			
			Parameters:
			-----------
			prompt (str): User prompt.
			
			Returns:
			-----------
			str | None
			
		'''
		try:
			return self.fetch(
				prompt=prompt,
				model=model,
				temperature=temperature,
				max_tokens=max_tokens,
				top_p=top_p,
				top_k=top_k,
				candidate_count=candidate_count,
				seed=seed,
				system=system,
				response_format=response_format,
				stop_sequences=stop_sequences,
				grounding=True,
				search_domains=search_domains,
				reasoning=reasoning,
				thinking_level=thinking_level,
				include_thoughts=include_thoughts,
			)
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Gemini'
			exception.method = 'search_web( self, prompt: str, ... ) -> str | None'
			raise exception

class Claude( Generator ):
	'''
	
		Purpose:
		---------
		Class providing Anthropic Claude text-generation, extended thinking,
		and optional web search support through the Messages API.
	
		Attribues:
		-----------
		client - Anthropic
		model - str
		response - Any
		api_key - str
		messages - List[ Dict[ str, Any ] ]
		params - Dict[ str, Any ]
		temperature - float
		max_tokens - int
		top_p - float
		top_k - int | None
		thinking_budget - int | None
		system_instructions - str | None
		web_search - bool
		search_domains - List[ str ]
		blocked_domains - List[ str ]
	
		Methods:
		-----------
		fetch( ) -> str | None
		generate_text( ) -> str | None
		search_web( ) -> str | None
		
	'''
	client: Optional[ Claude ]
	model: Optional[ str ]
	response: Optional[ Any ]
	api_key: Optional[ str ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	params: Optional[ Dict[ str, Any ] ]
	temperature: Optional[ float ]
	max_tokens: Optional[ int ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	thinking_budget: Optional[ int ]
	system_instructions: Optional[ str ]
	web_search: Optional[ bool ]
	search_domains: Optional[ List[ str ] ]
	blocked_domains: Optional[ List[ str ] ]
	
	def __init__( self ) -> None:
		'''
		
			Purpose:
			-----------
			Initialize Anthropic Claude API client.
			
			Parameters:
			-----------
			None
			
			Returns:
			-----------
			None
			
		'''
		super( ).__init__( )
		self.api_key = cfg.CLAUDE_API_KEY
		self.url = r'https://api.anthropic.com'
		self.client = None
		self.messages = None
		self.model = 'claude-sonnet-4-6'
		self.max_tokens = 2048
		self.temperature = 0.7
		self.top_p = 1.0
		self.top_k = None
		self.thinking_budget = None
		self.headers = { }
		self.timeout = None
		self.content = None
		self.params = None
		self.response = None
		self.system_instructions = None
		self.web_search = False
		self.search_domains = [ ]
		self.blocked_domains = [ ]
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		'''
		
			Purpose:
			-----------
			Claude list of members.
			
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
		         'params',
		         'agents',
		         'messages',
		         'temperature',
		         'top_p',
		         'top_k',
		         'thinking_budget',
		         'system_instructions',
		         'web_search',
		         'search_domains',
		         'blocked_domains' ]
	
	def _normalize_domains( self, domains: Any ) -> List[ str ]:
		'''
		
			Purpose:
			-----------
			Normalize domain input into a canonical, de-duplicated list.
			
			Parameters:
			-----------
			domains (Any): String, list, tuple, set, or None.
			
			Returns:
			-----------
			List[str]
			
		'''
		try:
			if domains is None:
				return [ ]
			
			if isinstance( domains, str ):
				_parts = re.split( r'[\n,;]+', domains )
			elif isinstance( domains, (list, tuple, set) ):
				_parts = [ str( x ) for x in domains if x is not None ]
			else:
				_parts = [ str( domains ) ]
			
			_values = [ ]
			for _entry in _parts:
				_value = str( _entry ).strip( ).lower( )
				if not _value:
					continue
				
				if not _value.startswith( 'http://' ) and not _value.startswith( 'https://' ):
					_value = f'https://{_value}'
				
				_parsed = urllib.parse.urlparse( _value )
				_domain = (_parsed.netloc or _parsed.path or '').strip( ).lower( )
				_domain = re.sub( r':\d+$', '', _domain )
				_domain = _domain.lstrip( '.' )
				
				if _domain.startswith( 'www.' ):
					_domain = _domain[ 4: ]
				
				if _domain and _domain not in _values:
					_values.append( _domain )
			
			return _values
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Claude'
			exception.method = '_normalize_domains( self, domains: Any ) -> List[ str ]'
			raise exception
	
	def _supports_thinking( self, model: str ) -> bool:
		'''
		
			Purpose:
			-----------
			Determine whether the selected Claude model supports extended thinking.
			
			Parameters:
			-----------
			model (str): Model name.
			
			Returns:
			-----------
			bool
			
		'''
		try:
			throw_if( 'model', model )
			_name = str( model ).strip( ).lower( )
			return _name.startswith( 'claude-' )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Claude'
			exception.method = '_supports_thinking( self, model: str ) -> bool'
			raise exception
	
	def _extract_text( self, response: Any ) -> str:
		'''
		
			Purpose:
			-----------
			Extract plain text from an Anthropic Messages API response.
			
			Parameters:
			-----------
			response (Any): Anthropic response object.
			
			Returns:
			-----------
			str
			
		'''
		try:
			if response is None:
				return ''
			
			if hasattr( response, 'content' ) and response.content:
				_parts = [ ]
				for _block in response.content:
					_type = getattr( _block, 'type', None )
					if _type == 'text':
						_text = getattr( _block, 'text', '' )
						if _text:
							_parts.append( _text )
				if _parts:
					return '\n'.join( _parts ).strip( )
			
			return str( response )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Claude'
			exception.method = '_extract_text( self, response: Any ) -> str'
			raise exception
	
	def fetch( self, query: str, model: str = 'claude-sonnet-4-6', temperature: float = 0.7,
			max_tokens: int = 2048, top_p: float = 1.0, top_k: int | None = None,
			system: str | None = None, stop_sequences: List[ str ] | None = None,
			thinking: bool = False, thinking_budget: int | None = None, web_search: bool = False,
			search_domains: Any = None, blocked_domains: Any = None ) -> str | None:
		'''
		
			Purpose:
			-------
			Send an Anthropic Messages API request for Claude text generation,
			optional extended thinking, and optional server-side web search.
			
			Parameters:
			-----------
			query (str): User prompt.
			model (str): Claude model name.
			temperature (float): Sampling temperature.
			max_tokens (int): Max output tokens.
			top_p (float): Top-p nucleus sampling parameter.
			top_k (int | None): Top-k sampling parameter.
			system (str | None): Optional system prompt.
			stop_sequences (List[str] | None): Optional stop sequences.
			thinking (bool): Enable extended thinking when supported.
			thinking_budget (int | None): Claude thinking budget in tokens.
			web_search (bool): Enable Claude web search tool.
			search_domains (Any): Optional allowlist domain input.
			blocked_domains (Any): Optional blocklist domain input.
			
			Returns:
			---------
			str | None
			
		'''
		try:
			throw_if( 'query', query )
			throw_if( 'model', model )
			
			self.query = query
			self.model = str( model ).strip( )
			self.temperature = float( temperature )
			self.max_tokens = int( max_tokens )
			self.top_p = float( top_p )
			self.top_k = int( top_k ) if top_k is not None else None
			self.system_instructions = system if system and str( system ).strip( ) else None
			self.web_search = bool( web_search )
			self.client = Claude( api_key=self.api_key )
			self.search_domains = self._normalize_domains( search_domains )
			self.blocked_domains = self._normalize_domains( blocked_domains )
			self.thinking_budget = int( thinking_budget ) if thinking_budget is not None else None
			self.messages = [ { 'role': 'user', 'content': self.query } ]
			self.params = \
			{
				'model': self.model,
				'max_tokens': self.max_tokens,
				'messages': self.messages,
			}
			
			if self.system_instructions:
				self.params[ 'system' ] = self.system_instructions
			
			if stop_sequences:
				self.params[ 'stop_sequences' ] = stop_sequences
			
			if thinking and self._supports_thinking( self.model ):
				_budget = self.thinking_budget if self.thinking_budget is not None else 1024
				if _budget < 1024:
					_budget = 1024
				
				self.params[ 'thinking' ] = \
				{
					'type': 'enabled',
					'budget_tokens': _budget,
				}
			
				if self.top_p is not None:
					self.params[ 'top_p' ] = min( 1.0, max( 0.95, self.top_p ) )
			else:
				self.params[ 'temperature' ] = self.temperature
				self.params[ 'top_p' ] = self.top_p
				
				if self.top_k is not None and self.top_k > 0:
					self.params[ 'top_k' ] = self.top_k
			
			if self.web_search:
				self.tools = [ ]
				self.web_tool = \
				{
					'type': 'web_search_20250305',
					'name': 'web_search',
				}
				
				if self.search_domains:
					self.web_tool[ 'allowed_domains' ] = self.search_domains
				
				if self.blocked_domains:
					self.web_tool[ 'blocked_domains' ] = self.blocked_domains
				
				self.tools.append( self.web_tool )
				self.params[ 'tools' ] = self.tools
			
			self.response = self.client.messages.create( **self.params )
			return self._extract_text( self.response )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Claude'
			exception.method = (
					'fetch( self, query: str, model: str="claude-sonnet-4-6", '
					'temperature: float=0.7, max_tokens: int=2048, top_p: float=1.0, '
					'top_k: int | None=None, system: str | None=None, '
					'stop_sequences: List[ str ] | None=None, thinking: bool=False, '
					'thinking_budget: int | None=None, web_search: bool=False, '
					'search_domains: Any=None, blocked_domains: Any=None ) -> str | None'
			)
			raise exception
	
	def generate_text( self, query: str, model: str = 'claude-sonnet-4-6', temperature: float = 0.7,
			max_tokens: int = 2048, top_p: float = 1.0, top_k: int | None = None,
			system: str | None = None, stop_sequences: List[ str ] | None = None,
			thinking: bool = False, thinking_budget: int | None = None, web_search: bool = False,
			search_domains: Any = None, blocked_domains: Any = None ) -> str | None:
		'''
		
			Purpose:
			-----------
			Convenience wrapper around fetch for text generation.
			
			Parameters:
			-----------
			query (str): User prompt.
			
			Returns:
			-----------
			str | None
			
		'''
		try:
			return self.fetch(
				query=query,
				model=model,
				temperature=temperature,
				max_tokens=max_tokens,
				top_p=top_p,
				top_k=top_k,
				system=system,
				stop_sequences=stop_sequences,
				thinking=thinking,
				thinking_budget=thinking_budget,
				web_search=web_search,
				search_domains=search_domains,
				blocked_domains=blocked_domains,
			)
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Claude'
			exception.method = 'generate_text( self, query: str, ... ) -> str | None'
			raise exception
	
	def search_web( self, query: str, model: str = 'claude-sonnet-4-6', temperature: float = 0.7,
			max_tokens: int = 2048, top_p: float = 1.0, top_k: int | None = None,
			system: str | None = None, stop_sequences: List[ str ] | None = None,
			thinking: bool = False, thinking_budget: int | None = None,
			search_domains: Any = None, blocked_domains: Any = None ) -> str | None:
		'''
		
			Purpose:
			-----------
			Convenience wrapper around fetch with web search enabled.
			
			Parameters:
			-----------
			query (str): User prompt.
			
			Returns:
			-----------
			str | None
			
		'''
		try:
			return self.fetch(
				query=query,
				model=model,
				temperature=temperature,
				max_tokens=max_tokens,
				top_p=top_p,
				top_k=top_k,
				system=system,
				stop_sequences=stop_sequences,
				thinking=thinking,
				thinking_budget=thinking_budget,
				web_search=True,
				search_domains=search_domains,
				blocked_domains=blocked_domains,
			)
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Claude'
			exception.method = 'search_web( self, query: str, ... ) -> str | None'
			raise exception

class Mistral( Generator ):
	'''

		Purpose:
		---------
		Class providing access to the Mistral chat-completions API.

		Attribues:
		-----------
		client - Optional[ MistralAI ]
		model - Optional[ str ]
		response - Optional[ Any ]
		api_key - Optional[ str ]
		query - Optional[ str ]
		params - Optional[ Dict[ str, Any ] ]
		temperature - Optional[ float ]
		max_tokens - Optional[ int ]
		top_p - Optional[ float ]
		messages - Optional[ List[ Dict[ str, Any ] ] ]
		system_instructions - Optional[ str ]
		seed - Optional[ int ]
		safe_prompt - Optional[ bool ]

		Methods:
		-----------
		_extract_text( self, response: Any ) -> str
		fetch( self, query: str, model: str='mistral-large-latest',
		       temperature: float=0.7, max_tokens: int=1024, top_p: float=1.0,
		       seed: int | None=None, safe_mode: bool=False,
		       system: str | None=None ) -> str | None
		create_schema( ... ) -> Dict[ str, str ] | None

	'''
	client: Optional[ MistralAI ]
	model: Optional[ str ]
	response: Optional[ Any ]
	api_key: Optional[ str ]
	query: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	temperature: Optional[ float ]
	max_tokens: Optional[ int ]
	top_p: Optional[ float ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	system_instructions: Optional[ str ]
	seed: Optional[ int ]
	safe_prompt: Optional[ bool ]
	
	def __init__( self ) -> None:
		'''

			Purpose:
			-----------
			Initialize the Mistral API wrapper.

			Parameters:
			-----------
			None

			Returns:
			-----------
			None

		'''
		super( ).__init__( )
		self.api_key = cfg.MISTRAL_API_KEY
		self.model = 'mistral-large-latest'
		self.headers = { }
		self.client = None
		self.timeout = None
		self.content = None
		self.params = None
		self.response = None
		self.query = None
		self.temperature = None
		self.max_tokens = None
		self.top_p = None
		self.messages = [ ]
		self.system_instructions = None
		self.seed = None
		self.safe_prompt = False
		self.agents = cfg.AGENTS
		
		if 'User-Agent' not in self.headers:
			self.headers[ 'User-Agent' ] = self.agents
	
	def __dir__( self ) -> List[ str ]:
		'''

			Purpose:
			-----------
			Return the ordered list of members exposed by this wrapper.

			Parameters:
			-----------
			None

			Returns:
			-----------
			list[str]

		'''
		return [
				'content',
				'client',
				'timeout',
				'headers',
				'fetch',
				'api_key',
				'response',
				'params',
				'agents',
				'model',
				'temperature',
				'max_tokens',
				'top_p',
				'messages',
				'system_instructions',
				'seed',
				'safe_prompt',
				'_extract_text',
				'create_schema',
		]
	
	def _extract_text( self, response: Any ) -> str:
		'''

			Purpose:
			-----------
			Extract plain text from a Mistral chat completion response.

			Parameters:
			-----------
			response (Any): Raw SDK response object.

			Returns:
			-----------
			str

		'''
		try:
			if response is None:
				return ''
			
			if hasattr( response, 'choices' ) and response.choices:
				choice = response.choices[ 0 ]
				
				if hasattr( choice, 'message' ) and choice.message is not None:
					message = choice.message
					
					if hasattr( message, 'content' ):
						content = message.content
						
						if isinstance( content, str ):
							return content.strip( )
						
						if isinstance( content, list ):
							parts: List[ str ] = [ ]
							for item in content:
								if isinstance( item, str ) and item.strip( ):
									parts.append( item.strip( ) )
								elif hasattr( item, 'text' ):
									text_value = getattr( item, 'text', '' )
									if text_value:
										parts.append( str( text_value ).strip( ) )
							
							if parts:
								return '\n'.join( parts ).strip( )
						
						return str( content ).strip( )
			
			return str( response )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Mistral'
			exception.method = '_extract_text( self, response: Any ) -> str'
			raise exception
	
	def fetch( self, query: str, model: str = 'mistral-large-latest', temperature: float = 0.7,
			max_tokens: int = 1024, top_p: float = 1.0, seed: int | None = None,
			safe_mode: bool = False, system: str | None = None ) -> str | None:
		'''

			Purpose:
			-------
			Send a Mistral chat-completions request for text generation.

			Parameters:
			-----------
			query (str): User prompt.
			model (str): Mistral model name.
			temperature (float): Sampling temperature.
			max_tokens (int): Maximum output tokens.
			top_p (float): Top-p nucleus sampling parameter.
			seed (int | None): Deterministic random seed when provided.
			safe_mode (bool): If True, enable Mistral safe prompt injection.
			system (str | None): Optional system instructions.

			Returns:
			---------
			str | None

		'''
		try:
			throw_if( 'query', query )
			throw_if( 'model', model )
			
			self.query = str( query ).strip( )
			self.model = str( model ).strip( )
			self.temperature = float( temperature )
			self.max_tokens = int( max_tokens )
			self.top_p = float( top_p )
			self.seed = int( seed ) if seed is not None else None
			self.safe_prompt = bool( safe_mode )
			self.system_instructions = system if system and str( system ).strip( ) else None
			self.client = MistralAI( api_key=self.api_key )
			self.messages = [ ]
			if self.system_instructions:
				self.messages.append(
					{
							'role': 'system',
							'content': self.system_instructions,
					}
				)
			
			self.messages.append(
				{
						'role': 'user',
						'content': self.query,
				}
			)
			
			self.params = {
					'model': self.model,
					'messages': self.messages,
					'temperature': self.temperature,
					'max_tokens': self.max_tokens,
					'top_p': self.top_p,
					'stream': False,
					'response_format': { 'type': 'text' },
					'safe_prompt': self.safe_prompt,
			}
			
			if self.seed is not None and self.seed > 0:
				self.params[ 'random_seed' ] = self.seed
			
			self.response = self.client.chat.complete( **self.params )
			return self._extract_text( self.response )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Mistral'
			exception.method = (
					'fetch( self, query: str, model: str="mistral-large-latest", '
					'temperature: float=0.7, max_tokens: int=1024, top_p: float=1.0, '
					'seed: int | None=None, safe_mode: bool=False, '
					'system: str | None=None ) -> str | None'
			)
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
			exception.cause = 'Mistral'
			exception.method = ('create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]')
			raise exception

class Chat( Generator ):
	"""
	
	    Purpose
	    ___________
	    Class used for interacting with OpenAI text-generation and reasoning models
	    through the Responses API, including optional built-in web search support.
	
	
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
	    fetch( self, prompt: str, ... ) -> str
	    get_model_options( self ) -> List[ str ]
	    get_effort_options( self ) -> List[ str ]
	    generate_text( self, prompt: str, ... ) -> str
	    analyze_image( self, prompt: str, url: str ) -> str
	    summarize_document( self, prompt: str, path: str ) -> str
	    search_web( self, prompt: str, ... ) -> str
	    search_files( self, prompt: str ) -> str
	    dump( self ) -> str
	    get_data( self ) -> Dict[ str, Any ]

    """
	
	def __init__( self, num: int = 1, temp: float = 0.8, top: float = 0.9,
			freq: float = 0.0, pres: float = 0.0, iters: int = 10000, store: bool = True,
			stream: bool = True ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = OpenAI( api_key=self.api_key )
		self.client.api_key = cfg.OPENAI_API_KEY
		self.system_instructions = None
		self.model = 'gpt-5-mini'
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
		self.id = 'asst_2Yu2yfINGD5en4e0aUXAKxyu'
		self.vector_store_ids = [ 'vs_67e83bdf8abc81918bda0d6b39a19372', ]
		self.metadata = { }
		self.tools = [ ]
		self.vector_stores = { 'Code': 'vs_67e83bdf8abc81918bda0d6b39a19372', }
		self.web_search = False
		self.search_domains = [ ]
		self.parallel_tool_calls = True
		self.tool_choice = 'auto'
	
	def _supports_reasoning( self, model: str ) -> bool:
		"""
	
	        Purpose
	        _______
	        Indicates whether the selected model family should receive
	        reasoning options.
	
	
	        Parameters
	        ----------
	        model: str
	
	
	        Returns
	        -------
	        bool

        """
		try:
			throw_if( 'model', model )
			_name = model.strip( ).lower( )
			return _name.startswith( 'gpt-5' ) or _name.startswith( 'o' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = '_supports_reasoning( self, model: str ) -> bool'
			raise exception
	
	def _normalize_domains( self, domains: Any ) -> List[ str ]:
		"""
	
	        Purpose
	        _______
	        Normalize a string or list of domain entries into a canonical,
	        de-duplicated list suitable for provider routing.
	
	
	        Parameters
	        ----------
	        domains: Any
	
	
	        Returns
	        -------
	        List[ str ]

        """
		try:
			if domains is None:
				return [ ]
			
			if isinstance( domains, str ):
				_parts = re.split( r'[\n,;]+', domains )
			elif isinstance( domains, list ) or isinstance( domains, tuple ) or isinstance( domains, set ):
				_parts = [ str( x ) for x in domains if x is not None ]
			else:
				_parts = [ str( domains ) ]
			
			_values = [ ]
			for _entry in _parts:
				_value = str( _entry ).strip( ).lower( )
				if not _value:
					continue
				
				if not _value.startswith( 'http://' ) and not _value.startswith( 'https://' ):
					_value = f'https://{_value}'
				
				_parsed = urllib.parse.urlparse( _value )
				_domain = (_parsed.netloc or _parsed.path or '').strip( ).lower( )
				_domain = re.sub( r':\d+$', '', _domain )
				_domain = _domain.lstrip( '.' )
				
				if _domain.startswith( 'www.' ):
					_domain = _domain[ 4: ]
				
				if _domain and _domain not in _values:
					_values.append( _domain )
			
			return _values
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = '_normalize_domains( self, domains: Any ) -> List[ str ]'
			raise exception
	
	def _build_instructions( self, system: str | None = None, response_format: str | None = None,
			web_search: bool = False, search_domains: Any = None ) -> str | None:
		"""
	
	        Purpose
	        _______
	        Build the final instruction block sent to the OpenAI Responses API.
	
	
	        Parameters
	        ----------
	        system: str | None=None
	        response_format: str | None=None
	        web_search: bool=False
	        search_domains: Any=None
	
	
	        Returns
	        -------
	        str | None

        """
		try:
			_parts = [ ]
			
			if system and str( system ).strip( ):
				_parts.append( str( system ).strip( ) )
			
			if response_format and str( response_format ).strip( ).lower( ) == 'json':
				_parts.append( 'Return valid JSON only. Do not include markdown fences or commentary.' )
			
			_domains = self._normalize_domains( search_domains )
			if web_search and _domains:
				_parts.append(
					'When using web search, strongly prefer sources from the following domains '
					f'when they are relevant and available: {", ".join( _domains )}.'
				)
			
			if _parts:
				return '\n\n'.join( _parts )
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = ('_build_instructions( self, system: str | None=None, '
			                    'response_format: str | None=None, web_search: bool=False, '
			                    'search_domains: Any=None ) -> str | None')
			raise exception
	
	def fetch( self, prompt: str, model: str = 'gpt-5-mini', temperature: float = 0.7,
			max_tokens: int = 1024, top_p: float = 1.0, seed: int | None = None,
			system: str | None = None, response_format: str | None = None,
			reasoning_effort: str | None = None, web_search: bool = False,
			search_domains: Any = None, store: bool = True, stream: bool = False,
			parallel_tool_calls: bool = True, tool_choice: str = 'auto' ) -> str:
		"""
	
	        Purpose
	        _______
	        Primary provider entry point used by the UI. Executes text generation
	        or reasoning through OpenAI's Responses API with optional built-in
	        web search.
	
	
	        Parameters
	        ----------
	        prompt: str
	        model: str='gpt-5-mini'
	        temperature: float=0.7
	        max_tokens: int=1024
	        top_p: float=1.0
	        seed: int | None=None
	        system: str | None=None
	        response_format: str | None=None
	        reasoning_effort: str | None=None
	        web_search: bool=False
	        search_domains: Any=None
	        store: bool=True
	        stream: bool=False
	        parallel_tool_calls: bool=True
	        tool_choice: str='auto'
	
	
	        Returns
	        -------
	        str

        """
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			
			self.model = str( model ).strip( )
			self.input_text = str( prompt )
			self.temperature = float( temperature )
			self.top_percent = float( top_p )
			self.max_completion_tokens = int( max_tokens )
			self.response_format = response_format or 'auto'
			self.reasoning_effort = reasoning_effort if reasoning_effort else None
			self.store = bool( store )
			self.stream = bool( stream )
			self.web_search = bool( web_search )
			self.search_domains = self._normalize_domains( search_domains )
			self.parallel_tool_calls = bool( parallel_tool_calls )
			self.tool_choice = tool_choice or 'auto'
			self.system_instructions = self._build_instructions(
				system=system,
				response_format=self.response_format,
				web_search=self.web_search,
				search_domains=self.search_domains,
			)
			
			self.tools = [ ]
			if self.web_search:
				self.tools.append( { 'type': 'web_search_preview' } )
			
			self.request = \
				{
						'model': self.model,
						'input': self.input_text,
						'max_output_tokens': self.max_completion_tokens,
						'store': self.store,
						'stream': self.stream,
						'parallel_tool_calls': self.parallel_tool_calls,
				}
			
			if self.system_instructions:
				self.request[ 'instructions' ] = self.system_instructions
			
			if self.tools:
				self.request[ 'tools' ] = self.tools
				self.request[ 'tool_choice' ] = self.tool_choice
			
			if self.temperature is not None:
				self.request[ 'temperature' ] = self.temperature
			
			if self.top_percent is not None:
				self.request[ 'top_p' ] = self.top_percent
			
			if seed is not None:
				self.request[ 'seed' ] = int( seed )
			
			if self._supports_reasoning( self.model ) and self.reasoning_effort:
				self.request[ 'reasoning' ] = { 'effort': self.reasoning_effort }
			
			self.response = self.client.responses.create( **self.request )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = (
					'fetch( self, prompt: str, model: str="gpt-5-mini", temperature: float=0.7, '
					'max_tokens: int=1024, top_p: float=1.0, seed: int | None=None, '
					'system: str | None=None, response_format: str | None=None, '
					'reasoning_effort: str | None=None, web_search: bool=False, '
					'search_domains: Any=None, store: bool=True, stream: bool=False, '
					'parallel_tool_calls: bool=True, tool_choice: str="auto" ) -> str'
			)
			raise exception
	
	def generate_text( self, prompt: str, model: str = 'gpt-5-mini', temperature: float = 0.7,
			max_tokens: int = 1024, top_p: float = 1.0, seed: int | None = None,
			system: str | None = None, response_format: str | None = None,
			reasoning_effort: str | None = None, web_search: bool = False,
			search_domains: Any = None, store: bool = True, stream: bool = False,
			parallel_tool_calls: bool = True, tool_choice: str = 'auto' ) -> str:
		"""
	
	        Purpose
	        _______
	        Convenience wrapper around fetch for text generation.
	
	
	        Parameters
	        ----------
	        prompt: str
	
	
	        Returns
	        -------
	        str

        """
		try:
			return self.fetch(
				prompt=prompt,
				model=model,
				temperature=temperature,
				max_tokens=max_tokens,
				top_p=top_p,
				seed=seed,
				system=system,
				response_format=response_format,
				reasoning_effort=reasoning_effort,
				web_search=web_search,
				search_domains=search_domains,
				store=store,
				stream=stream,
				parallel_tool_calls=parallel_tool_calls,
				tool_choice=tool_choice,
			)
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str, ... ) -> str'
			raise exception
	
	def generate_image( self, prompt: str ) -> str:
		"""
	
	        Purpose
	        _______
	        Generate an image using the OpenAI image API.
	
	
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
			self.response = self.client.images.generate(
				model='gpt-image-1',
				prompt=self.input_text,
				size='1024x1024',
			)
			if hasattr( self.response, 'data' ) and self.response.data:
				_image = self.response.data[ 0 ]
				if hasattr( _image, 'url' ) and _image.url:
					return _image.url
			return str( self.response )
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
	        Analyze an image using a multimodal OpenAI model.
	
	
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
			throw_if( 'url', url )
			self.input_text = prompt
			self.image_url = url
			self.input = [
					{
							'role': 'user',
							'content': [
									{
											'type': 'input_text',
											'text': self.input_text,
									},
									{
											'type': 'input_image',
											'image_url': self.image_url,
									}, ],
					} ]
			self.response = self.client.responses.create(
				model=self.model,
				input=self.input,
			)
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = 'analyze_image( self, prompt: str, url: str ) -> str'
			raise exception
	
	def summarize_document( self, prompt: str, path: str ) -> str:
		"""
	
	        Purpose
	        _______
	        Summarize a document provided by file path.
	
	
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
			
			with open( self.file_path, 'rb' ) as _handle:
				self.file = self.client.files.create( file=_handle, purpose='user_data' )
			
			self.messages = [
					{
							'role': 'user',
							'content': [
									{
											'type': 'input_file',
											'file_id': self.file.id,
									},
									{
											'type': 'input_text',
											'text': self.input_text,
									}, ],
					} ]
			
			self.response = self.client.responses.create(
				model=self.model,
				input=self.messages,
			)
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = 'summarize_document( self, prompt: str, path: str ) -> str'
			raise exception
	
	def search_web( self, prompt: str, model: str = 'gpt-5-mini', temperature: float = 0.7,
			max_tokens: int = 1024, top_p: float = 1.0, seed: int | None = None,
			system: str | None = None, response_format: str | None = None,
			reasoning_effort: str | None = None, search_domains: Any = None,
			store: bool = True, stream: bool = False, parallel_tool_calls: bool = True,
			tool_choice: str = 'auto' ) -> str:
		"""
	
	        Purpose
	        _______
	        Execute a Responses API request with the built-in web search tool enabled.
	
	
	        Parameters
	        ----------
	        prompt: str
	
	
	        Returns
	        -------
	        str

        """
		try:
			return self.fetch(
				prompt=prompt,
				model=model,
				temperature=temperature,
				max_tokens=max_tokens,
				top_p=top_p,
				seed=seed,
				system=system,
				response_format=response_format,
				reasoning_effort=reasoning_effort,
				web_search=True,
				search_domains=search_domains,
				store=store,
				stream=stream,
				parallel_tool_calls=parallel_tool_calls,
				tool_choice=tool_choice,
			)
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = 'search_web( self, prompt: str, ... ) -> str'
			raise exception
	
	def search_files( self, prompt: str ) -> str:
		"""
	
	        Purpose
	        _______
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
			self.response = self.client.responses.create(
				model=self.model,
				tools=self.tools,
				input=prompt,
			)
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = 'search_files( self, prompt: str ) -> str'
			raise exception
	
	def translate( self, text: str ) -> str:
		"""
	
	        Purpose
	        _______
	        Translate text using the currently selected text model.
	
	
	        Parameters
	        ----------
	        text: str
	
	
	        Returns
	        -------
	        str

        """
		try:
			throw_if( 'text', text )
			return self.fetch(
				prompt=f'Translate the following text faithfully and preserve meaning:\n\n{text}',
				model=self.model,
				temperature=0.2,
				max_tokens=self.max_completion_tokens,
				top_p=self.top_percent,
				system=self.system_instructions,
			)
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = 'translate( self, text: str ) -> str'
			raise exception
	
	def transcribe( self, text: str ) -> str:
		"""
	
	        Purpose
	        _______
	        Placeholder passthrough for compatibility until audio transcription
	        is split into its own provider path.
	
	
	        Parameters
	        ----------
	        text: str
	
	
	        Returns
	        -------
	        str

        """
		try:
			throw_if( 'text', text )
			return text
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = 'transcribe( self, text: str ) -> str'
			raise exception
	
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
		if hasattr( cfg, 'GPT_MODELS' ) and cfg.GPT_MODELS:
			return list( cfg.GPT_MODELS )
		
		return [ 'gpt-5.4',
		         'gpt-5',
		         'gpt-5-mini',
		         'gpt-5-nano',
		         'gpt-5.1',
		         'gpt-5.2',
		         'gpt-4.1', ]
	
	def get_effort_options( self ) -> List[ str ]:
		'''
	
	        Purpose:
	        ---------
	        Method that returns a list of available reasoning effort levels

        '''
		return [ 'minimal', 'low', 'medium', 'high' ]
	
	def get_data( self ) -> Dict[ str, Any ]:
		'''
	
	        Purpose:
	        ---------
	        Returns: dict[ str, Any ] of members

        '''
		return \
			{
					'num': self.number,
					'model': self.model,
					'temperature': self.temperature,
					'top_percent': self.top_percent,
					'frequency_penalty': self.frequency_penalty,
					'presence_penalty': self.presence_penalty,
					'max_completion_tokens': self.max_completion_tokens,
					'store': self.store,
					'stream': self.stream,
					'response_format': self.response_format,
					'reasoning_effort': self.reasoning_effort,
					'web_search': self.web_search,
					'search_domains': self.search_domains,
					'parallel_tool_calls': self.parallel_tool_calls,
					'tool_choice': self.tool_choice,
					'vector_store_ids': self.vector_store_ids,
			}
	
	def dump( self ) -> str:
		'''
	
	        Purpose:
	        ---------
	        Returns: JSON-like string representation of members

        '''
		try:
			return str( self.get_data( ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'fetchers'
			exception.cause = 'Chat'
			exception.method = 'dump( self ) -> str'
			raise exception
