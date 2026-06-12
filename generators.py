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
    generators.py

    Purpose:
        Provides Foo's generative AI provider wrappers and shared generation utilities.
        The module centralizes request validation, provider-specific payload construction,
        model option helpers, response-text extraction, web-search configuration, and
        structured exception wrapping for OpenAI, xAI Grok, Google Gemini, Anthropic
        Claude, and Mistral workflows.
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

from anthropic import Anthropic as Claude
import base64
from boogr import Error, Logger
from core import Result
import config as cfg
from google import genai
from google import genai
from google.genai import types
from openai import OpenAI
from pathlib import Path
from typing import Any, Dict, Optional, Pattern, List, Tuple
from requests import Response
from xai_sdk import Client as Xai
from mistralai import Mistral as MistralAI
import re
import urllib

def throw_if( name: str, value: object ) -> None:
	"""Throw if.

	Purpose:
			Validates that a required argument contains a usable value before the
			surrounding workflow continues. This guard centralizes early validation so
			provider wrappers fail with consistent, readable error messages.

	Args:
			name (str): Name value used by the operation.
			value (object): Value value used by the operation.

	Raises:
			ValueError: Raised when a required value is missing or empty.
		"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be None.' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty.' )

def encode_image( path: str ) -> str:
	"""Encode image.

	Purpose:
			Reads an image file from disk and returns a Base64-encoded string suitable for
			provider APIs that accept inline image data.

	Args:
			path (str): Path value used by the operation.

	Returns:
			Value produced by the operation.
		"""
	data = Path( path ).read_bytes( )
	return base64.b64encode( data ).decode( "utf-8" )

class Generator:
	"""Generator component.

	Purpose:
			Defines the common base contract for Foo generation providers. The class stores
			shared request, response, header, URL, timeout, result, and query state used by
			concrete provider wrappers.

	Attributes:
			timeout (Optional[int]): Runtime state or configuration value maintained by the
			component.
			headers (Optional[Dict[str, Any]]): Runtime state or configuration value
			maintained by the component.
			response (Optional[Response]): Runtime state or configuration value maintained
			by the component.
			url (Optional[str]): Runtime state or configuration value maintained by the
			component.
			result (Optional[Result]): Runtime state or configuration value maintained by
			the component.
			query (Optional[str]): Runtime state or configuration value maintained by the
			component.
		"""
	timeout: Optional[ int ]
	headers: Optional[ Dict[ str, Any ] ]
	response: Optional[ Response ]
	url: Optional[ str ]
	result: Optional[ Result ]
	query: Optional[ str ]
	
	def __init__( self ) -> None:
		"""Initialize instance.

		Purpose:
				Initializes the object with default configuration, runtime state, and
				compatibility fields required by later method calls. The constructor performs
				local setup and does not execute a provider request.
			"""
		self.timeout = None
		self.headers = None
		self.response = None
		self.url = None
		self.result = None
		self.query = None
	
	def __dir__( self ) -> list[ str ]:
		"""Return member names.

		Purpose:
				Returns a stable ordering of public attributes and methods for interactive
				inspection, documentation surfaces, and UI member displays.

		Returns:
				Ordered public member names exposed by the object.
			"""
		return [ 'timeout',
		         'headers',
		         'response',
		         'url',
		         'result',
		         'query',
		         'fetch' ]
	
	def fetch( self, query: str, url: str, time: int = 10 ) -> Result | None:
		"""Fetch.

		Purpose:
				Executes the provider-specific generation or retrieval workflow after validating
				input values and constructing the request payload required by the underlying
				provider API.

		Args:
				query (str): Input text submitted to the provider workflow.
				url (str): Url value used by the operation.
				time (int): Time value used by the operation.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				NotImplementedError: Raised when a subclass has not implemented the required
				operation.
			"""
		raise NotImplementedError( 'Must be implemented by a subclass.' )

class Grok( Generator ):
	"""Grok component.

	Purpose:
			Provides xAI Grok text generation and web-search support through provider-
			specific request construction, response execution, domain normalization, output
			extraction, and compatibility helper methods.

	Attributes:
			client (Optional[Xai]): Runtime state or configuration value maintained by the
			component.
			model (Optional[str]): Runtime state or configuration value maintained by the
			component.
			response (Optional[Any]): Runtime state or configuration value maintained by the
			component.
			api_key (Optional[str]): Runtime state or configuration value maintained by the
			component.
			query (Optional[str]): Runtime state or configuration value maintained by the
			component.
			params (Optional[Dict[str, Any]]): Runtime state or configuration value
			maintained by the component.
			temperature (Optional[float]): Runtime state or configuration value maintained
			by the component.
			max_tokens (Optional[int]): Runtime state or configuration value maintained by
			the component.
			top_p (Optional[float]): Runtime state or configuration value maintained by the
			component.
			reasoning_effort (Optional[str]): Runtime state or configuration value
			maintained by the component.
			stream (Optional[bool]): Runtime state or configuration value maintained by the
			component.
			store (Optional[bool]): Runtime state or configuration value maintained by the
			component.
			messages (Optional[List[Dict[str, Any]]]): Runtime state or configuration value
			maintained by the component.
			system_instructions (Optional[str]): Runtime state or configuration value
			maintained by the component.
			web_search (Optional[bool]): Runtime state or configuration value maintained by
			the component.
			search_domains (Optional[List[str]]): Runtime state or configuration value
			maintained by the component.
			parallel_tool_calls (Optional[bool]): Runtime state or configuration value
			maintained by the component.
			tool_choice (Optional[str]): Runtime state or configuration value maintained by
			the component.
			tools (Optional[List[Dict[str, Any]]]): Runtime state or configuration value
			maintained by the component.
		"""
	
	client: Optional[ Xai ]
	model: Optional[ str ]
	response: Optional[ Any ]
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
		"""Initialize instance.

		Purpose:
				Initializes the object with default configuration, runtime state, and
				compatibility fields required by later method calls. The constructor performs
				local setup and does not execute a provider request.
			"""
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
		self.query = None
		self.params = None
		self.response = None
		self.system_instructions = None
		self.web_search = False
		self.search_domains = [ ]
		self.parallel_tool_calls = True
		self.tool_choice = 'auto'
		self.tools = [ ]
		self.store = True
		self.stream = False
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.

		Purpose:
				Returns a stable ordering of public attributes and methods for interactive
				inspection, documentation surfaces, and UI member displays.

		Returns:
				Ordered public member names exposed by the object.
			"""
		return [
				'client',
				'model',
				'response',
				'api_key',
				'query',
				'params',
				'temperature',
				'max_tokens',
				'top_p',
				'reasoning_effort',
				'stream',
				'store',
				'messages',
				'system_instructions',
				'web_search',
				'search_domains',
				'parallel_tool_calls',
				'tool_choice',
				'tools',
				'normalize_domains',
				'supports_reasoning_effort',
				'is_reasoning_model',
				'build_instructions',
				'build_tools',
				'build_response_format',
				'extract_output_text',
				'fetch',
				'generate_text',
				'search_web'
		]
	
	def normalize_domains( self, domains: Any ) -> List[ str ]:
		"""Normalize domains.

		Purpose:
				Normalizes user-supplied domain values into canonical, de-duplicated domain
				names suitable for provider web-search restrictions.

		Args:
				domains (Any): Optional domain restriction values for provider web-search tools.

		Returns:
				Normalized values suitable for the provider request.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			if domains is None:
				return [ ]
			
			if isinstance( domains, str ):
				parts = re.split( r'[\n,;]+', domains )
			elif isinstance( domains, (list, tuple, set) ):
				parts = [ str( item ) for item in domains if item is not None ]
			else:
				parts = [ str( domains ) ]
			
			values: List[ str ] = [ ]
			
			for entry in parts:
				value = str( entry ).strip( ).lower( )
				
				if not value:
					continue
				
				value = re.sub( r'^https?://', '', value )
				value = value.split( '/' )[ 0 ]
				value = re.sub( r':\d+$', '', value )
				value = value.lstrip( '.' )
				
				if value.startswith( 'www.' ):
					value = value[ 4: ]
				
				if not re.fullmatch( r'[a-z0-9][a-z0-9.-]*\.[a-z]{2,}', value ):
					raise ValueError( f'Invalid xAI web-search domain: {value}' )
				
				if value not in values:
					values.append( value )
			
			if len( values ) > 5:
				raise ValueError(
					'xAI web-search allowed domains are limited to five domains.'
				)
			
			return values
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'normalize_domains( self, domains: Any ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def supports_reasoning_effort( self, model: str ) -> bool:
		"""Supports reasoning effort.

		Purpose:
				Determines whether the selected provider model supports an explicit reasoning-
				effort request field.

		Args:
				model (str): Provider model identifier used for the request.

		Returns:
				Boolean indicator for the requested provider capability.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			throw_if( 'model', model )
			name = str( model ).strip( ).lower( )
			return name == 'grok-4.3'
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'supports_reasoning_effort( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def supports_reasoning_object( self, model: str ) -> bool:
		"""Supports reasoning object.

		Purpose:
				Determines whether the selected provider model supports the provider-specific
				reasoning object in the request payload.

		Args:
				model (str): Provider model identifier used for the request.

		Returns:
				Boolean indicator for the requested provider capability.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			throw_if( 'model', model )
			name = str( model ).strip( ).lower( )
			return name == 'grok-4.20-multi-agent'
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'supports_reasoning_object( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def is_reasoning_model( self, model: str ) -> bool:
		"""Is reasoning model.

		Purpose:
				Identifies whether the selected model is a reasoning-capable model based on its
				provider naming convention.

		Args:
				model (str): Provider model identifier used for the request.

		Returns:
				Boolean indicator for the requested provider capability.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			throw_if( 'model', model )
			name = str( model ).strip( ).lower( )
			return (
					'reasoning' in name
					or name.startswith( 'grok-4' )
					or name.startswith( 'grok-4.3' )
					or name.startswith( 'grok-4.20' )
			)
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'is_reasoning_model( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def build_instructions( self, system: str = None,
			response_format: str = None ) -> str | None:
		"""Build instructions.

		Purpose:
				Builds the final instruction text sent to the provider from system text,
				requested output format, and optional search context.

		Args:
				system (str): Optional system-level instruction text.
				response_format (str): Optional response-format or structured-output
				configuration.

		Returns:
				Provider-compatible request configuration produced from the supplied options.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			parts: List[ str ] = [ ]
			
			if system and str( system ).strip( ):
				parts.append( str( system ).strip( ) )
			
			if response_format and str( response_format ).strip( ).lower( ) == 'json':
				parts.append(
					'Return valid JSON only. Do not include markdown fences, prose, '
					'or commentary outside the JSON value.'
				)
			
			if parts:
				return '\n\n'.join( parts )
			
			return None
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = (
					'build_instructions( self, system: str | None=None, '
					'response_format: str | None=None ) -> str | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def build_tools( self, web_search: bool = False, search_domains: Any = None ) -> List[
		Dict[ str, Any ] ]:
		"""Build tools.

		Purpose:
				Builds provider tool declarations from web-search, grounding, file-search, and
				vector-store settings supported by the selected provider.

		Args:
				web_search (bool): Web search value used by the operation.
				search_domains (Any): Optional domain restriction values for provider web-search
				tools.

		Returns:
				Provider-compatible request configuration produced from the supplied options.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			tools: List[ Dict[ str, Any ] ] = [ ]
			
			if not web_search:
				return tools
			
			domains = self.normalize_domains( search_domains )
			tool: Dict[ str, Any ] = { 'type': 'web_search' }
			
			if domains:
				tool[ 'filters' ] = { 'allowed_domains': domains }
			
			tools.append( tool )
			return tools
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = (
					'build_tools( self, web_search: bool=False, '
					'search_domains: Any=None ) -> List[ Dict[ str, Any ] ]'
			)
			Logger( ).write( exception )
			raise exception
	
	def build_response_format( self, response_format: str = None ) -> Dict[
		                                                                  str, Any ] | None:
		"""Build response format.

		Purpose:
				Builds the provider response-format configuration for plain text, JSON, or
				schema-guided outputs.

		Args:
				response_format (str): Optional response-format or structured-output
				configuration.

		Returns:
				Provider-compatible request configuration produced from the supplied options.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			mode = str( response_format or '' ).strip( ).lower( )
			
			if not mode or mode == 'auto':
				return None
			
			if mode in [ 'json', 'json_object' ]:
				return { 'type': 'json_object' }
			
			if mode == 'text':
				return { 'type': 'text' }
			
			return None
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = (
					'build_response_format( self, response_format: str | None=None ) '
					'-> Dict[ str, Any ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def extract_output_text( self, response: Any ) -> str:
		"""Extract output text.

		Purpose:
				Extracts usable text from provider response shapes while handling common SDK
				object layouts and fallback representations.

		Args:
				response (Any): Response value used by the operation.

		Returns:
				Text extracted from the provider response.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			if response is None:
				return ''
			
			if hasattr( response, 'output_text' ) and response.output_text:
				return str( response.output_text )
			
			if hasattr( response, 'text' ) and response.text:
				return str( response.text )
			
			if isinstance( response, dict ):
				if response.get( 'output_text' ):
					return str( response.get( 'output_text' ) )
				
				if response.get( 'text' ):
					return str( response.get( 'text' ) )
				
				output = response.get( 'output', [ ] )
				if isinstance( output, list ):
					parts: List[ str ] = [ ]
					
					for item in output:
						if not isinstance( item, dict ):
							continue
						
						content = item.get( 'content', [ ] )
						if isinstance( content, list ):
							for block in content:
								if isinstance( block, dict ) and block.get( 'text' ):
									parts.append( str( block.get( 'text' ) ) )
					
					if parts:
						return '\n'.join( parts ).strip( )
			
			if hasattr( response, '__iter__' ) and not isinstance( response, (str, bytes, dict) ):
				parts: List[ str ] = [ ]
				
				for event in response:
					event_type = getattr( event, 'type', '' )
					
					if event_type == 'response.output_text.delta':
						delta = getattr( event, 'delta', '' )
						if delta:
							parts.append( str( delta ) )
					
					elif event_type == 'response.completed':
						final_response = getattr( event, 'response', None )
						if final_response is not None:
							text = self.extract_output_text( final_response )
							if text:
								return text
				
				if parts:
					return ''.join( parts )
			
			output = getattr( response, 'output', None )
			if output:
				parts: List[ str ] = [ ]
				
				for item in output:
					content = getattr( item, 'content', None )
					
					if content:
						for block in content:
							text = getattr( block, 'text', None )
							
							if text:
								parts.append( str( text ) )
				
				if parts:
					return '\n'.join( parts ).strip( )
			
			return str( response )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'extract_output_text( self, response: Any ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def create_response( self, payload: Dict[ str, Any ] ) -> Any:
		"""Create response.

		Purpose:
				Submits a prepared request payload to the xAI Responses API and stores the
				provider response for later inspection.

		Args:
				payload (Dict[str, Any]): Payload value used by the operation.

		Returns:
				Value produced by the operation.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			throw_if( 'payload', payload )
			
			if hasattr( self.client, 'responses' ) and hasattr( self.client.responses, 'create' ):
				return self.client.responses.create( **payload )
			
			if hasattr( self.client, 'chat' ) and hasattr( self.client.chat, 'create' ):
				messages = payload.get( 'input', [ ] )
				
				if isinstance( messages, str ):
					messages = [ { 'role': 'user', 'content': messages } ]
				
				chat_payload = {
						'model': payload.get( 'model' ),
						'messages': messages,
						'stream': payload.get( 'stream', False )
				}
				
				if payload.get( 'temperature' ) is not None:
					chat_payload[ 'temperature' ] = payload.get( 'temperature' )
				
				if payload.get( 'top_p' ) is not None:
					chat_payload[ 'top_p' ] = payload.get( 'top_p' )
				
				if payload.get( 'max_output_tokens' ) is not None:
					chat_payload[ 'max_tokens' ] = payload.get( 'max_output_tokens' )
				
				if payload.get( 'tools' ):
					chat_payload[ 'tools' ] = payload.get( 'tools' )
				
				if payload.get( 'tool_choice' ):
					chat_payload[ 'tool_choice' ] = payload.get( 'tool_choice' )
				
				if payload.get( 'stop' ):
					chat_payload[ 'stop' ] = payload.get( 'stop' )
				
				if payload.get( 'response_format' ):
					chat_payload[ 'response_format' ] = payload.get( 'response_format' )
				
				if payload.get( 'reasoning_effort' ):
					chat_payload[ 'reasoning_effort' ] = payload.get( 'reasoning_effort' )
				
				if payload.get( 'reasoning' ):
					chat_payload[ 'reasoning' ] = payload.get( 'reasoning' )
				
				return self.client.chat.create( **chat_payload )
			
			raise RuntimeError( 'The xAI client does not expose responses.create or chat.create.' )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'create_response( self, payload: Dict[ str, Any ] ) -> Any'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, query: str, model: str = 'grok-4-fast-reasoning',
			temperature: float = 0.7, max_tokens: int = 2048, top_p: float = 1.0,
			seed: int | None = None, system: str = None,
			response_format: str = None, reasoning_effort: str = None,
			web_search: bool = False, search_domains: Any = None,
			stop: List[ str ] = None, stream: bool = False, store: bool = True,
			parallel_tool_calls: bool = True, tool_choice: str = 'auto' ) -> str | None:
		"""Fetch.

		Purpose:
				Executes the provider-specific generation or retrieval workflow after validating
				input values and constructing the request payload required by the underlying
				provider API.

		Args:
				query (str): Input text submitted to the provider workflow.
				model (str): Provider model identifier used for the request.
				temperature (float): Sampling or generation control passed to the provider
				request.
				max_tokens (int): Sampling or generation control passed to the provider request.
				top_p (float): Sampling or generation control passed to the provider request.
				seed (int | None): Sampling or generation control passed to the provider
				request.
				system (str): Optional system-level instruction text.
				response_format (str): Optional response-format or structured-output
				configuration.
				reasoning_effort (str): Reasoning effort value used by the operation.
				web_search (bool): Web search value used by the operation.
				search_domains (Any): Optional domain restriction values for provider web-search
				tools.
				stop (List[str]): Stop value used by the operation.
				stream (bool): Stream value used by the operation.
				store (bool): Store value used by the operation.
				parallel_tool_calls (bool): Optional provider tool configuration.
				tool_choice (str): Optional provider tool configuration.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			throw_if( 'query', query )
			throw_if( 'model', model )
			
			self.query = str( query )
			self.model = str( model ).strip( )
			self.temperature = float( temperature )
			self.max_tokens = int( max_tokens )
			self.top_p = float( top_p )
			self.reasoning_effort = reasoning_effort if reasoning_effort else None
			self.stream = bool( stream )
			self.store = bool( store )
			self.web_search = bool( web_search )
			self.search_domains = self.normalize_domains( search_domains )
			self.parallel_tool_calls = bool( parallel_tool_calls )
			self.tool_choice = tool_choice or 'auto'
			self.system_instructions = self.build_instructions(
				system=system,
				response_format=response_format
			)
			self.tools = self.build_tools(
				web_search=self.web_search,
				search_domains=self.search_domains
			)
			
			input_messages: List[ Dict[ str, str ] ] = [ ]
			
			if self.system_instructions:
				input_messages.append(
					{
							'role': 'system',
							'content': self.system_instructions
					}
				)
			
			input_messages.append(
				{
						'role': 'user',
						'content': self.query
				}
			)
			
			self.params = {
					'model': self.model,
					'input': input_messages,
					'max_output_tokens': self.max_tokens,
					'stream': self.stream,
					'store': self.store,
					'parallel_tool_calls': self.parallel_tool_calls
			}
			
			if seed is not None:
				self.params[ 'seed' ] = int( seed )
			
			format_payload = self.build_response_format( response_format )
			if format_payload:
				self.params[ 'response_format' ] = format_payload
			
			if self.tools:
				self.params[ 'tools' ] = self.tools
				self.params[ 'tool_choice' ] = self.tool_choice
			
			is_reasoning = self.is_reasoning_model( self.model )
			
			if self.supports_reasoning_effort( self.model ) and self.reasoning_effort:
				self.params[ 'reasoning_effort' ] = self.reasoning_effort
			elif self.supports_reasoning_object( self.model ) and self.reasoning_effort:
				self.params[ 'reasoning' ] = { 'effort': self.reasoning_effort }
			
			if not is_reasoning:
				self.params[ 'temperature' ] = self.temperature
				self.params[ 'top_p' ] = self.top_p
				
				if stop:
					self.params[ 'stop' ] = [
							str( item )
							for item in stop
							if str( item ).strip( )
					]
			
			self.response = self.create_response( self.params )
			return self.extract_output_text( self.response )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = (
					'fetch( self, query: str, model: str="grok-4-fast-reasoning", '
					'temperature: float=0.7, max_tokens: int=2048, top_p: float=1.0, '
					'seed: int | None=None, system: str | None=None, '
					'response_format: str | None=None, reasoning_effort: str | None=None, '
					'web_search: bool=False, search_domains: Any=None, '
					'stop: List[ str ] | None=None, stream: bool=False, '
					'store: bool=True, parallel_tool_calls: bool=True, '
					'tool_choice: str="auto" ) -> str | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, query: str, model: str = 'grok-4-fast-reasoning',
			temperature: float = 0.7, max_tokens: int = 2048, top_p: float = 1.0,
			seed: int | None = None, system: str = None,
			response_format: str = None, reasoning_effort: str = None,
			web_search: bool = False, search_domains: Any = None,
			stop: List[ str ] = None, stream: bool = False, store: bool = True,
			parallel_tool_calls: bool = True, tool_choice: str = 'auto' ) -> str | None:
		"""Generate text.

		Purpose:
				Generates text output by delegating to the provider-specific fetch path while
				preserving a simplified call surface for text-generation workflows.

		Args:
				query (str): Input text submitted to the provider workflow.
				model (str): Provider model identifier used for the request.
				temperature (float): Sampling or generation control passed to the provider
				request.
				max_tokens (int): Sampling or generation control passed to the provider request.
				top_p (float): Sampling or generation control passed to the provider request.
				seed (int | None): Sampling or generation control passed to the provider
				request.
				system (str): Optional system-level instruction text.
				response_format (str): Optional response-format or structured-output
				configuration.
				reasoning_effort (str): Reasoning effort value used by the operation.
				web_search (bool): Web search value used by the operation.
				search_domains (Any): Optional domain restriction values for provider web-search
				tools.
				stop (List[str]): Stop value used by the operation.
				stream (bool): Stream value used by the operation.
				store (bool): Store value used by the operation.
				parallel_tool_calls (bool): Optional provider tool configuration.
				tool_choice (str): Optional provider tool configuration.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
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
				tool_choice=tool_choice
			)
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'generate_text( self, query: str, ... ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def search_web( self, query: str, model: str = 'grok-4-fast-reasoning',
			temperature: float = 0.7, max_tokens: int = 2048, top_p: float = 1.0,
			seed: int | None = None, system: str = None,
			response_format: str = None, reasoning_effort: str = None,
			search_domains: Any = None, stream: bool = False, store: bool = True,
			parallel_tool_calls: bool = True, tool_choice: str = 'auto' ) -> str | None:
		"""Search web.

		Purpose:
				Runs a provider-specific web-search-enabled generation workflow using the
				configured model and search controls.

		Args:
				query (str): Input text submitted to the provider workflow.
				model (str): Provider model identifier used for the request.
				temperature (float): Sampling or generation control passed to the provider
				request.
				max_tokens (int): Sampling or generation control passed to the provider request.
				top_p (float): Sampling or generation control passed to the provider request.
				seed (int | None): Sampling or generation control passed to the provider
				request.
				system (str): Optional system-level instruction text.
				response_format (str): Optional response-format or structured-output
				configuration.
				reasoning_effort (str): Reasoning effort value used by the operation.
				search_domains (Any): Optional domain restriction values for provider web-search
				tools.
				stream (bool): Stream value used by the operation.
				store (bool): Store value used by the operation.
				parallel_tool_calls (bool): Optional provider tool configuration.
				tool_choice (str): Optional provider tool configuration.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
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
				tool_choice=tool_choice
			)
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Grok'
			exception.method = 'search_web( self, query: str, ... ) -> str | None'
			Logger( ).write( exception )
			raise exception

class Gemini( Generator ):
	"""Gemini component.

	Purpose:
			Provides Google Gemini text generation and grounding support through Gemini-
			specific configuration building, thinking controls, tool construction, response
			extraction, and request execution.

	Attributes:
			api_key (Optional[str]): Runtime state or configuration value maintained by the
			component.
			client (Optional[Any]): Runtime state or configuration value maintained by the
			component.
			model (Optional[str]): Runtime state or configuration value maintained by the
			component.
			response (Optional[Any]): Runtime state or configuration value maintained by the
			component.
			query (Optional[str]): Runtime state or configuration value maintained by the
			component.
			params (Optional[Dict[str, Any]]): Runtime state or configuration value
			maintained by the component.
			temperature (Optional[float]): Runtime state or configuration value maintained
			by the component.
			max_tokens (Optional[int]): Runtime state or configuration value maintained by
			the component.
			top_p (Optional[float]): Runtime state or configuration value maintained by the
			component.
			top_k (Optional[int]): Runtime state or configuration value maintained by the
			component.
			candidate_count (Optional[int]): Runtime state or configuration value maintained
			by the component.
			seed (Optional[int]): Runtime state or configuration value maintained by the
			component.
			system_instructions (Optional[str]): Runtime state or configuration value
			maintained by the component.
			response_format (Optional[str]): Runtime state or configuration value maintained
			by the component.
			stop_sequences (Optional[List[str]]): Runtime state or configuration value
			maintained by the component.
			grounding (Optional[bool]): Runtime state or configuration value maintained by
			the component.
			search_domains (Optional[List[str]]): Runtime state or configuration value
			maintained by the component.
			reasoning (Optional[bool]): Runtime state or configuration value maintained by
			the component.
			thinking_level (Optional[str]): Runtime state or configuration value maintained
			by the component.
			thinking_budget (Optional[int]): Runtime state or configuration value maintained
			by the component.
			include_thoughts (Optional[bool]): Runtime state or configuration value
			maintained by the component.
			tools (Optional[List[Any]]): Runtime state or configuration value maintained by
			the component.
			config (Optional[Any]): Runtime state or configuration value maintained by the
			component.
		"""
	
	api_key: Optional[ str ]
	client: Optional[ Any ]
	model: Optional[ str ]
	response: Optional[ Any ]
	query: Optional[ str ]
	params: Optional[ Dict[ str, Any ] ]
	temperature: Optional[ float ]
	max_tokens: Optional[ int ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	candidate_count: Optional[ int ]
	seed: Optional[ int ]
	system_instructions: Optional[ str ]
	response_format: Optional[ str ]
	stop_sequences: Optional[ List[ str ] ]
	grounding: Optional[ bool ]
	search_domains: Optional[ List[ str ] ]
	reasoning: Optional[ bool ]
	thinking_level: Optional[ str ]
	thinking_budget: Optional[ int ]
	include_thoughts: Optional[ bool ]
	tools: Optional[ List[ Any ] ]
	config: Optional[ Any ]
	
	def __init__( self ) -> None:
		"""Initialize instance.

		Purpose:
				Initializes the object with default configuration, runtime state, and
				compatibility fields required by later method calls. The constructor performs
				local setup and does not execute a provider request.
			"""
		super( ).__init__( )
		self.api_key = cfg.GOOGLE_API_KEY
		self.client = genai.Client( api_key=self.api_key )
		self.model = 'gemini-2.5-flash'
		self.response = None
		self.query = None
		self.params = None
		self.temperature = 0.7
		self.max_tokens = 2048
		self.top_p = 1.0
		self.top_k = None
		self.candidate_count = 1
		self.seed = None
		self.system_instructions = None
		self.response_format = None
		self.stop_sequences = [ ]
		self.grounding = False
		self.search_domains = [ ]
		self.reasoning = False
		self.thinking_level = None
		self.thinking_budget = None
		self.include_thoughts = False
		self.tools = [ ]
		self.config = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.

		Purpose:
				Returns a stable ordering of public attributes and methods for interactive
				inspection, documentation surfaces, and UI member displays.

		Returns:
				Ordered public member names exposed by the object.
			"""
		return [
				'api_key',
				'client',
				'model',
				'response',
				'query',
				'params',
				'temperature',
				'max_tokens',
				'top_p',
				'top_k',
				'candidate_count',
				'seed',
				'system_instructions',
				'response_format',
				'stop_sequences',
				'grounding',
				'search_domains',
				'reasoning',
				'thinking_level',
				'thinking_budget',
				'include_thoughts',
				'tools',
				'config',
				'normalize_domains',
				'normalize_stop_sequences',
				'supports_thinking_level',
				'supports_thinking_budget',
				'build_system_instruction',
				'build_thinking_config',
				'build_tools',
				'build_config',
				'extract_text',
				'fetch',
				'generate_text',
				'search_web'
		]
	
	def normalize_domains( self, domains: Any ) -> List[ str ]:
		"""Normalize domains.

		Purpose:
				Normalizes user-supplied domain values into canonical, de-duplicated domain
				names suitable for provider web-search restrictions.

		Args:
				domains (Any): Optional domain restriction values for provider web-search tools.

		Returns:
				Normalized values suitable for the provider request.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			if domains is None:
				return [ ]
			
			if isinstance( domains, str ):
				parts = re.split( r'[\n,;]+', domains )
			elif isinstance( domains, (list, tuple, set) ):
				parts = [ str( item ) for item in domains if item is not None ]
			else:
				parts = [ str( domains ) ]
			
			values: List[ str ] = [ ]
			
			for entry in parts:
				value = str( entry ).strip( ).lower( )
				
				if not value:
					continue
				
				if not value.startswith( 'http://' ) and not value.startswith( 'https://' ):
					value = f'https://{value}'
				
				parsed = urllib.parse.urlparse( value )
				domain = (parsed.netloc or parsed.path or '').strip( ).lower( )
				domain = re.sub( r':\d+$', '', domain )
				domain = domain.lstrip( '.' )
				
				if domain.startswith( 'www.' ):
					domain = domain[ 4: ]
				
				if not re.fullmatch( r'[a-z0-9][a-z0-9.-]*\.[a-z]{2,}', domain ):
					raise ValueError( f'Invalid Gemini grounding domain: {domain}' )
				
				if domain not in values:
					values.append( domain )
			
			return values
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'normalize_domains( self, domains: Any ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def normalize_stop_sequences( self, stop_sequences: Any ) -> List[ str ]:
		"""Normalize stop sequences.

		Purpose:
				Normalizes stop-sequence input into a clean list of provider-compatible stop
				strings.

		Args:
				stop_sequences (Any): Stop sequences value used by the operation.

		Returns:
				Normalized values suitable for the provider request.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			if stop_sequences is None:
				return [ ]
			
			if isinstance( stop_sequences, str ):
				parts = stop_sequences.splitlines( )
			elif isinstance( stop_sequences, (list, tuple, set) ):
				parts = [ str( item ) for item in stop_sequences if item is not None ]
			else:
				parts = [ str( stop_sequences ) ]
			
			return [
					str( item ).strip( )
					for item in parts
					if str( item ).strip( )
			]
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = (
					'normalize_stop_sequences( self, stop_sequences: Any ) -> List[ str ]'
			)
			Logger( ).write( exception )
			raise exception
	
	def supports_thinking_level( self, model: str ) -> bool:
		"""Supports thinking level.

		Purpose:
				Determines whether the selected Gemini model supports an explicit thinking-level
				option.

		Args:
				model (str): Provider model identifier used for the request.

		Returns:
				Boolean indicator for the requested provider capability.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			throw_if( 'model', model )
			return str( model ).strip( ).lower( ).startswith( 'gemini-3' )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'supports_thinking_level( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def supports_thinking_budget( self, model: str ) -> bool:
		"""Supports thinking budget.

		Purpose:
				Determines whether the selected Gemini model supports an explicit thinking-
				budget option.

		Args:
				model (str): Provider model identifier used for the request.

		Returns:
				Boolean indicator for the requested provider capability.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			throw_if( 'model', model )
			return str( model ).strip( ).lower( ).startswith( 'gemini-2.5' )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'supports_thinking_budget( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def build_system_instruction( self, system: str = None, response_format: str = None,
			grounding: bool = False, search_domains: Any = None ) -> str | None:
		"""Build system instruction.

		Purpose:
				Builds the final Gemini system instruction text from system guidance, response-
				format requirements, and optional grounding context.

		Args:
				system (str): Optional system-level instruction text.
				response_format (str): Optional response-format or structured-output
				configuration.
				grounding (bool): Grounding value used by the operation.
				search_domains (Any): Optional domain restriction values for provider web-search
				tools.

		Returns:
				Provider-compatible request configuration produced from the supplied options.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			parts: List[ str ] = [ ]
			if system and str( system ).strip( ):
				parts.append( str( system ).strip( ) )
			
			if response_format and str( response_format ).strip( ).lower( ) == 'json':
				parts.append( 'Return valid JSON only. Do not include markdown fences, prose, '
				              'or commentary outside the JSON value.' )
			
			domains = self.normalize_domains( search_domains )
			if grounding and domains:
				parts.append( 'When using Google Search grounding, strongly prefer relevant '
				              f'sources from these domains when available: {", ".join( domains )}.' )
			
			if parts:
				return '\n\n'.join( parts )
			
			return None
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = (
					'build_system_instruction( self, system: str | None=None, '
					'response_format: str | None=None, grounding: bool=False, '
					'search_domains: Any=None ) -> str | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def build_thinking_config( self, model: str, reasoning: bool = False,
			thinking_level: str = None, thinking_budget: int | None = None,
			include_thoughts: bool = False ) -> Any:
		"""Build thinking config.

		Purpose:
				Builds Gemini thinking configuration from model capability, reasoning flags,
				thinking level, budget, and thought-inclusion controls.

		Args:
				model (str): Provider model identifier used for the request.
				reasoning (bool): Reasoning value used by the operation.
				thinking_level (str): Thinking level value used by the operation.
				thinking_budget (int | None): Thinking budget value used by the operation.
				include_thoughts (bool): Include thoughts value used by the operation.

		Returns:
				Provider-compatible request configuration produced from the supplied options.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			if not reasoning:
				return None
			
			thinking_data: Dict[ str, Any ] = { }
			
			if self.supports_thinking_level( model ):
				level = str( thinking_level or 'low' ).strip( ).lower( )
				
				if level not in [ 'minimal', 'low', 'medium', 'high' ]:
					level = 'low'
				
				thinking_data[ 'thinking_level' ] = level
			
			elif self.supports_thinking_budget( model ):
				if thinking_budget is not None:
					thinking_data[ 'thinking_budget' ] = int( thinking_budget )
				else:
					thinking_data[ 'thinking_budget' ] = -1
			
			else:
				return None
			
			if include_thoughts:
				thinking_data[ 'include_thoughts' ] = True
			
			if hasattr( types, 'ThinkingConfig' ):
				return types.ThinkingConfig( **thinking_data )
			
			return thinking_data
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = (
					'build_thinking_config( self, model: str, reasoning: bool=False, '
					'thinking_level: str | None=None, thinking_budget: int | None=None, '
					'include_thoughts: bool=False ) -> Any'
			)
			Logger( ).write( exception )
			raise exception
	
	def build_tools( self, grounding: bool = False ) -> List[ Any ]:
		"""Build tools.

		Purpose:
				Builds provider tool declarations from web-search, grounding, file-search, and
				vector-store settings supported by the selected provider.

		Args:
				grounding (bool): Grounding value used by the operation.

		Returns:
				Provider-compatible request configuration produced from the supplied options.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			if not grounding:
				return [ ]
			
			if hasattr( types, 'Tool' ) and hasattr( types, 'GoogleSearch' ):
				return [ types.Tool( google_search=types.GoogleSearch( ) ) ]
			
			return [ { 'google_search': { } } ]
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'build_tools( self, grounding: bool=False ) -> List[ Any ]'
			Logger( ).write( exception )
			raise exception
	
	def build_config( self, model: str, temperature: float = 0.7,
			max_tokens: int = 2048, top_p: float = 1.0, top_k: int | None = None,
			candidate_count: int = 1, seed: int | None = None,
			system: str = None, response_format: str = None,
			stop_sequences: Any = None, grounding: bool = False, search_domains: Any = None,
			reasoning: bool = False, thinking_level: str = None,
			thinking_budget: int | None = None, include_thoughts: bool = False,
			response_json_schema: Dict[ str, Any ] = None ) -> Any:
		"""Build config.

		Purpose:
				Builds the Gemini generation configuration from sampling, token, seed, response-
				format, safety, thinking, and tool settings.

		Args:
				model (str): Provider model identifier used for the request.
				temperature (float): Sampling or generation control passed to the provider
				request.
				max_tokens (int): Sampling or generation control passed to the provider request.
				top_p (float): Sampling or generation control passed to the provider request.
				top_k (int | None): Sampling or generation control passed to the provider
				request.
				candidate_count (int): Sampling or generation control passed to the provider
				request.
				seed (int | None): Sampling or generation control passed to the provider
				request.
				system (str): Optional system-level instruction text.
				response_format (str): Optional response-format or structured-output
				configuration.
				stop_sequences (Any): Stop sequences value used by the operation.
				grounding (bool): Grounding value used by the operation.
				search_domains (Any): Optional domain restriction values for provider web-search
				tools.
				reasoning (bool): Reasoning value used by the operation.
				thinking_level (str): Thinking level value used by the operation.
				thinking_budget (int | None): Thinking budget value used by the operation.
				include_thoughts (bool): Include thoughts value used by the operation.
				response_json_schema (Dict[str, Any]): Response json schema value used by the
				operation.

		Returns:
				Provider-compatible request configuration produced from the supplied options.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			config_data: Dict[ str, Any ] = {
					'temperature': float( temperature ),
					'max_output_tokens': int( max_tokens ),
					'top_p': float( top_p ),
					'candidate_count': int( candidate_count )
			}
			
			if top_k is not None and int( top_k ) > 0:
				config_data[ 'top_k' ] = int( top_k )
			
			if seed is not None:
				config_data[ 'seed' ] = int( seed )
			
			system_instruction = self.build_system_instruction(
				system=system,
				response_format=response_format,
				grounding=grounding,
				search_domains=search_domains
			)
			
			if system_instruction:
				config_data[ 'system_instruction' ] = system_instruction
			
			clean_stop = self.normalize_stop_sequences( stop_sequences )
			if clean_stop:
				config_data[ 'stop_sequences' ] = clean_stop
			
			if response_format and str( response_format ).strip( ).lower( ) == 'json':
				config_data[ 'response_mime_type' ] = 'application/json'
			
			if response_json_schema:
				config_data[ 'response_mime_type' ] = 'application/json'
				config_data[ 'response_json_schema' ] = response_json_schema
			
			tools_value = self.build_tools( grounding=grounding )
			if tools_value:
				config_data[ 'tools' ] = tools_value
			
			thinking_config = self.build_thinking_config(
				model=model,
				reasoning=reasoning,
				thinking_level=thinking_level,
				thinking_budget=thinking_budget,
				include_thoughts=include_thoughts
			)
			
			if thinking_config:
				config_data[ 'thinking_config' ] = thinking_config
			
			if hasattr( types, 'GenerateContentConfig' ):
				return types.GenerateContentConfig( **config_data )
			
			return config_data
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = (
					'build_config( self, model: str, temperature: float=0.7, '
					'max_tokens: int=2048, top_p: float=1.0, top_k: int | None=None, '
					'candidate_count: int=1, seed: int | None=None, '
					'system: str | None=None, response_format: str | None=None, '
					'stop_sequences: Any=None, grounding: bool=False, '
					'search_domains: Any=None, reasoning: bool=False, '
					'thinking_level: str | None=None, thinking_budget: int | None=None, '
					'include_thoughts: bool=False, '
					'response_json_schema: Dict[ str, Any ] | None=None ) -> Any'
			)
			Logger( ).write( exception )
			raise exception
	
	def extract_text( self, response: Any ) -> str:
		"""Extract text.

		Purpose:
				Extracts usable text from provider response shapes while handling common SDK
				object layouts and fallback representations.

		Args:
				response (Any): Response value used by the operation.

		Returns:
				Text extracted from the provider response.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			if response is None:
				return ''
			
			if hasattr( response, 'text' ) and response.text:
				return str( response.text )
			
			if isinstance( response, dict ):
				if response.get( 'text' ):
					return str( response.get( 'text' ) )
			
			candidates = getattr( response, 'candidates', None )
			if candidates:
				parts: List[ str ] = [ ]
				
				for candidate in candidates:
					content = getattr( candidate, 'content', None )
					candidate_parts = getattr( content, 'parts', None ) if content else None
					
					if not candidate_parts:
						continue
					
					for part in candidate_parts:
						text = getattr( part, 'text', None )
						
						if text:
							parts.append( str( text ) )
				
				if parts:
					return '\n'.join( parts ).strip( )
			
			return str( response )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'extract_text( self, response: Any ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, prompt: str, model: str = 'gemini-2.5-flash',
			temperature: float = 0.7, max_tokens: int = 2048, top_p: float = 1.0,
			top_k: int | None = None, candidate_count: int = 1,
			seed: int | None = None, system: str = None,
			response_format: str = None, stop_sequences: Any = None,
			grounding: bool = False, search_domains: Any = None,
			reasoning: bool = False, thinking_level: str = None,
			thinking_budget: int | None = None, include_thoughts: bool = False,
			response_json_schema: Dict[ str, Any ] = None ) -> str | None:
		"""Fetch.

		Purpose:
				Executes the provider-specific generation or retrieval workflow after validating
				input values and constructing the request payload required by the underlying
				provider API.

		Args:
				prompt (str): Input text submitted to the provider workflow.
				model (str): Provider model identifier used for the request.
				temperature (float): Sampling or generation control passed to the provider
				request.
				max_tokens (int): Sampling or generation control passed to the provider request.
				top_p (float): Sampling or generation control passed to the provider request.
				top_k (int | None): Sampling or generation control passed to the provider
				request.
				candidate_count (int): Sampling or generation control passed to the provider
				request.
				seed (int | None): Sampling or generation control passed to the provider
				request.
				system (str): Optional system-level instruction text.
				response_format (str): Optional response-format or structured-output
				configuration.
				stop_sequences (Any): Stop sequences value used by the operation.
				grounding (bool): Grounding value used by the operation.
				search_domains (Any): Optional domain restriction values for provider web-search
				tools.
				reasoning (bool): Reasoning value used by the operation.
				thinking_level (str): Thinking level value used by the operation.
				thinking_budget (int | None): Thinking budget value used by the operation.
				include_thoughts (bool): Include thoughts value used by the operation.
				response_json_schema (Dict[str, Any]): Response json schema value used by the
				operation.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			
			self.query = str( prompt )
			self.model = str( model ).strip( )
			self.temperature = float( temperature )
			self.max_tokens = int( max_tokens )
			self.top_p = float( top_p )
			self.top_k = int( top_k ) if top_k is not None else None
			self.candidate_count = int( candidate_count )
			self.seed = int( seed ) if seed is not None else None
			self.system_instructions = (str( system ).strip( )
			                            if system and str( system ).strip( )
			                            else None)
			self.response_format = response_format
			self.stop_sequences = self.normalize_stop_sequences( stop_sequences )
			self.grounding = bool( grounding )
			self.search_domains = self.normalize_domains( search_domains )
			self.reasoning = bool( reasoning )
			self.thinking_level = thinking_level
			self.thinking_budget = thinking_budget
			self.include_thoughts = bool( include_thoughts )
			self.config = self.build_config( model=self.model, temperature=self.temperature,
				max_tokens=self.max_tokens, top_p=self.top_p, top_k=self.top_k,
				candidate_count=self.candidate_count, seed=self.seed,
				system=self.system_instructions, response_format=self.response_format,
				stop_sequences=self.stop_sequences, grounding=self.grounding,
				search_domains=self.search_domains, reasoning=self.reasoning,
				thinking_level=self.thinking_level, thinking_budget=self.thinking_budget,
				include_thoughts=self.include_thoughts, response_json_schema=response_json_schema )
			self.params = {
					'model': self.model,
					'contents': self.query,
					'config': self.config
			}
			
			self.response = self.client.models.generate_content( **self.params )
			return self.extract_text( self.response )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'fetch( self, *args ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, prompt: str, model: str = 'gemini-2.5-flash',
			temperature: float = 0.7, max_tokens: int = 2048, top_p: float = 1.0,
			top_k: int | None = None, candidate_count: int = 1,
			seed: int | None = None, system: str = None,
			response_format: str = None, stop_sequences: Any = None,
			grounding: bool = False, search_domains: Any = None,
			reasoning: bool = False, thinking_level: str = None,
			thinking_budget: int | None = None, include_thoughts: bool = False,
			response_json_schema: Dict[ str, Any ] = None ) -> str | None:
		"""Generate text.

		Purpose:
				Generates text output by delegating to the provider-specific fetch path while
				preserving a simplified call surface for text-generation workflows.

		Args:
				prompt (str): Input text submitted to the provider workflow.
				model (str): Provider model identifier used for the request.
				temperature (float): Sampling or generation control passed to the provider
				request.
				max_tokens (int): Sampling or generation control passed to the provider request.
				top_p (float): Sampling or generation control passed to the provider request.
				top_k (int | None): Sampling or generation control passed to the provider
				request.
				candidate_count (int): Sampling or generation control passed to the provider
				request.
				seed (int | None): Sampling or generation control passed to the provider
				request.
				system (str): Optional system-level instruction text.
				response_format (str): Optional response-format or structured-output
				configuration.
				stop_sequences (Any): Stop sequences value used by the operation.
				grounding (bool): Grounding value used by the operation.
				search_domains (Any): Optional domain restriction values for provider web-search
				tools.
				reasoning (bool): Reasoning value used by the operation.
				thinking_level (str): Thinking level value used by the operation.
				thinking_budget (int | None): Thinking budget value used by the operation.
				include_thoughts (bool): Include thoughts value used by the operation.
				response_json_schema (Dict[str, Any]): Response json schema value used by the
				operation.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			return self.fetch( prompt=prompt, model=model, temperature=temperature,
				max_tokens=max_tokens, top_p=top_p, top_k=top_k,
				candidate_count=candidate_count, seed=seed, system=system,
				response_format=response_format, stop_sequences=stop_sequences,
				grounding=grounding, search_domains=search_domains, reasoning=reasoning,
				thinking_level=thinking_level, thinking_budget=thinking_budget,
				include_thoughts=include_thoughts, response_json_schema=response_json_schema )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'generate_text( self, prompt: str, ... ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def search_web( self, prompt: str, model: str = 'gemini-2.5-flash',
			temperature: float = 0.7, max_tokens: int = 2048, top_p: float = 1.0,
			top_k: int | None = None, candidate_count: int = 1,
			seed: int | None = None, system: str = None,
			response_format: str = None, stop_sequences: Any = None,
			search_domains: Any = None, reasoning: bool = False,
			thinking_level: str = None, thinking_budget: int | None = None,
			include_thoughts: bool = False,
			response_json_schema: Dict[ str, Any ] = None ) -> str | None:
		"""Search web.

		Purpose:
				Runs a provider-specific web-search-enabled generation workflow using the
				configured model and search controls.

		Args:
				prompt (str): Input text submitted to the provider workflow.
				model (str): Provider model identifier used for the request.
				temperature (float): Sampling or generation control passed to the provider
				request.
				max_tokens (int): Sampling or generation control passed to the provider request.
				top_p (float): Sampling or generation control passed to the provider request.
				top_k (int | None): Sampling or generation control passed to the provider
				request.
				candidate_count (int): Sampling or generation control passed to the provider
				request.
				seed (int | None): Sampling or generation control passed to the provider
				request.
				system (str): Optional system-level instruction text.
				response_format (str): Optional response-format or structured-output
				configuration.
				stop_sequences (Any): Stop sequences value used by the operation.
				search_domains (Any): Optional domain restriction values for provider web-search
				tools.
				reasoning (bool): Reasoning value used by the operation.
				thinking_level (str): Thinking level value used by the operation.
				thinking_budget (int | None): Thinking budget value used by the operation.
				include_thoughts (bool): Include thoughts value used by the operation.
				response_json_schema (Dict[str, Any]): Response json schema value used by the
				operation.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			return self.fetch( prompt=prompt, model=model, temperature=temperature,
				max_tokens=max_tokens, top_p=top_p, top_k=top_k, candidate_count=candidate_count,
				seed=seed, system=system, response_format=response_format,
				stop_sequences=stop_sequences, grounding=True, search_domains=search_domains,
				reasoning=reasoning, thinking_level=thinking_level, thinking_budget=thinking_budget,
				include_thoughts=include_thoughts, response_json_schema=response_json_schema )
		
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'generators'
			exception.cause = 'Gemini'
			exception.method = 'search_web( self, prompt: str, ... ) -> str | None'
			Logger( ).write( exception )
			raise exception

class Claude( Generator ):
	"""Claude component.

	Purpose:
			Provides Anthropic Claude text generation and search-oriented helper workflows
			through Claude-specific request construction, thinking support, domain
			normalization, and output extraction.

	Attributes:
			client (Optional[Claude]): Runtime state or configuration value maintained by
			the component.
			model (Optional[str]): Runtime state or configuration value maintained by the
			component.
			response (Optional[Any]): Runtime state or configuration value maintained by the
			component.
			api_key (Optional[str]): Runtime state or configuration value maintained by the
			component.
			messages (Optional[List[Dict[str, Any]]]): Runtime state or configuration value
			maintained by the component.
			params (Optional[Dict[str, Any]]): Runtime state or configuration value
			maintained by the component.
			temperature (Optional[float]): Runtime state or configuration value maintained
			by the component.
			max_tokens (Optional[int]): Runtime state or configuration value maintained by
			the component.
			top_p (Optional[float]): Runtime state or configuration value maintained by the
			component.
			top_k (Optional[int]): Runtime state or configuration value maintained by the
			component.
			thinking_budget (Optional[int]): Runtime state or configuration value maintained
			by the component.
			system_instructions (Optional[str]): Runtime state or configuration value
			maintained by the component.
			web_search (Optional[bool]): Runtime state or configuration value maintained by
			the component.
			search_domains (Optional[List[str]]): Runtime state or configuration value
			maintained by the component.
			blocked_domains (Optional[List[str]]): Runtime state or configuration value
			maintained by the component.
		"""
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
		"""Initialize instance.

		Purpose:
				Initializes the object with default configuration, runtime state, and
				compatibility fields required by later method calls. The constructor performs
				local setup and does not execute a provider request.
			"""
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
		"""Return member names.

		Purpose:
				Returns a stable ordering of public attributes and methods for interactive
				inspection, documentation surfaces, and UI member displays.

		Returns:
				Ordered public member names exposed by the object.
			"""
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
		"""Normalize domains.

		Purpose:
				Normalizes user-supplied domain values into canonical, de-duplicated domain
				names suitable for provider web-search restrictions.

		Args:
				domains (Any): Optional domain restriction values for provider web-search tools.

		Returns:
				Normalized values suitable for the provider request.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
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
			Logger( ).write( exception )
			raise exception
	
	def _supports_thinking( self, model: str ) -> bool:
		"""Supports thinking.

		Purpose:
				Determines whether the selected Claude model supports Claude thinking
				configuration.

		Args:
				model (str): Provider model identifier used for the request.

		Returns:
				Value produced by the operation.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			throw_if( 'model', model )
			_name = str( model ).strip( ).lower( )
			return _name.startswith( 'claude-' )
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'fetchers'
			exception.cause = 'Claude'
			exception.method = '_supports_thinking( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def _extract_text( self, response: Any ) -> str:
		"""Extract text.

		Purpose:
				Extracts usable text from provider response shapes while handling common SDK
				object layouts and fallback representations.

		Args:
				response (Any): Response value used by the operation.

		Returns:
				Text extracted from the provider response.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
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
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, query: str, model: str = 'claude-sonnet-4-6', temperature: float = 0.7,
			max_tokens: int = 2048, top_p: float = 1.0, top_k: int | None = None,
			system: str = None, stop_sequences: List[ str ] = None,
			thinking: bool = False, thinking_budget: int | None = None, web_search: bool = False,
			search_domains: Any = None, blocked_domains: Any = None ) -> str | None:
		"""Fetch.

		Purpose:
				Executes the provider-specific generation or retrieval workflow after validating
				input values and constructing the request payload required by the underlying
				provider API.

		Args:
				query (str): Input text submitted to the provider workflow.
				model (str): Provider model identifier used for the request.
				temperature (float): Sampling or generation control passed to the provider
				request.
				max_tokens (int): Sampling or generation control passed to the provider request.
				top_p (float): Sampling or generation control passed to the provider request.
				top_k (int | None): Sampling or generation control passed to the provider
				request.
				system (str): Optional system-level instruction text.
				stop_sequences (List[str]): Stop sequences value used by the operation.
				thinking (bool): Thinking value used by the operation.
				thinking_budget (int | None): Thinking budget value used by the operation.
				web_search (bool): Web search value used by the operation.
				search_domains (Any): Optional domain restriction values for provider web-search
				tools.
				blocked_domains (Any): Optional domain restriction values for provider web-
				search tools.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
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
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, query: str, model: str = 'claude-sonnet-4-6', temperature: float = 0.7,
			max_tokens: int = 2048, top_p: float = 1.0, top_k: int | None = None,
			system: str = None, stop_sequences: List[ str ] = None,
			thinking: bool = False, thinking_budget: int | None = None, web_search: bool = False,
			search_domains: Any = None, blocked_domains: Any = None ) -> str | None:
		"""Generate text.

		Purpose:
				Generates text output by delegating to the provider-specific fetch path while
				preserving a simplified call surface for text-generation workflows.

		Args:
				query (str): Input text submitted to the provider workflow.
				model (str): Provider model identifier used for the request.
				temperature (float): Sampling or generation control passed to the provider
				request.
				max_tokens (int): Sampling or generation control passed to the provider request.
				top_p (float): Sampling or generation control passed to the provider request.
				top_k (int | None): Sampling or generation control passed to the provider
				request.
				system (str): Optional system-level instruction text.
				stop_sequences (List[str]): Stop sequences value used by the operation.
				thinking (bool): Thinking value used by the operation.
				thinking_budget (int | None): Thinking budget value used by the operation.
				web_search (bool): Web search value used by the operation.
				search_domains (Any): Optional domain restriction values for provider web-search
				tools.
				blocked_domains (Any): Optional domain restriction values for provider web-
				search tools.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
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
			Logger( ).write( exception )
			raise exception
	
	def search_web( self, query: str, model: str = 'claude-sonnet-4-6', temperature: float = 0.7,
			max_tokens: int = 2048, top_p: float = 1.0, top_k: int | None = None,
			system: str = None, stop_sequences: List[ str ] = None,
			thinking: bool = False, thinking_budget: int | None = None,
			search_domains: Any = None, blocked_domains: Any = None ) -> str | None:
		"""Search web.

		Purpose:
				Runs a provider-specific web-search-enabled generation workflow using the
				configured model and search controls.

		Args:
				query (str): Input text submitted to the provider workflow.
				model (str): Provider model identifier used for the request.
				temperature (float): Sampling or generation control passed to the provider
				request.
				max_tokens (int): Sampling or generation control passed to the provider request.
				top_p (float): Sampling or generation control passed to the provider request.
				top_k (int | None): Sampling or generation control passed to the provider
				request.
				system (str): Optional system-level instruction text.
				stop_sequences (List[str]): Stop sequences value used by the operation.
				thinking (bool): Thinking value used by the operation.
				thinking_budget (int | None): Thinking budget value used by the operation.
				search_domains (Any): Optional domain restriction values for provider web-search
				tools.
				blocked_domains (Any): Optional domain restriction values for provider web-
				search tools.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
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
			Logger( ).write( exception )
			raise exception

class Mistral( Generator ):
	"""Mistral component.

	Purpose:
			Provides Mistral text generation and dynamic tool schema support through
			Mistral-specific request construction, response execution, and output
			extraction.

	Attributes:
			client (Optional[MistralAI]): Runtime state or configuration value maintained by
			the component.
			model (Optional[str]): Runtime state or configuration value maintained by the
			component.
			response (Optional[Any]): Runtime state or configuration value maintained by the
			component.
			api_key (Optional[str]): Runtime state or configuration value maintained by the
			component.
			query (Optional[str]): Runtime state or configuration value maintained by the
			component.
			params (Optional[Dict[str, Any]]): Runtime state or configuration value
			maintained by the component.
			temperature (Optional[float]): Runtime state or configuration value maintained
			by the component.
			max_tokens (Optional[int]): Runtime state or configuration value maintained by
			the component.
			top_p (Optional[float]): Runtime state or configuration value maintained by the
			component.
			messages (Optional[List[Dict[str, Any]]]): Runtime state or configuration value
			maintained by the component.
			system_instructions (Optional[str]): Runtime state or configuration value
			maintained by the component.
			seed (Optional[int]): Runtime state or configuration value maintained by the
			component.
			safe_prompt (Optional[bool]): Runtime state or configuration value maintained by
			the component.
		"""
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
		"""Initialize instance.

		Purpose:
				Initializes the object with default configuration, runtime state, and
				compatibility fields required by later method calls. The constructor performs
				local setup and does not execute a provider request.
			"""
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
		"""Return member names.

		Purpose:
				Returns a stable ordering of public attributes and methods for interactive
				inspection, documentation surfaces, and UI member displays.

		Returns:
				Ordered public member names exposed by the object.
			"""
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
		"""Extract text.

		Purpose:
				Extracts usable text from provider response shapes while handling common SDK
				object layouts and fallback representations.

		Args:
				response (Any): Response value used by the operation.

		Returns:
				Text extracted from the provider response.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
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
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, query: str, model: str = 'mistral-large-latest', temperature: float = 0.7,
			max_tokens: int = 1024, top_p: float = 1.0, seed: int | None = None,
			safe_mode: bool = False, system: str = None ) -> str | None:
		"""Fetch.

		Purpose:
				Executes the provider-specific generation or retrieval workflow after validating
				input values and constructing the request payload required by the underlying
				provider API.

		Args:
				query (str): Input text submitted to the provider workflow.
				model (str): Provider model identifier used for the request.
				temperature (float): Sampling or generation control passed to the provider
				request.
				max_tokens (int): Sampling or generation control passed to the provider request.
				top_p (float): Sampling or generation control passed to the provider request.
				seed (int | None): Sampling or generation control passed to the provider
				request.
				safe_mode (bool): Safe mode value used by the operation.
				system (str): Optional system-level instruction text.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
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
				self.messages.append( {
						'role': 'system',
						'content': self.system_instructions,
				} )
			
			self.messages.append( {
					'role': 'user',
					'content': self.query,
			} )
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
			exception.method = 'fetch( self, *args ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Create schema.

		Purpose:
				Constructs a dynamic tool schema definition from a function name, service name,
				description, parameter schema, and required-field list.

		Args:
				function (str): Function value used by the operation.
				tool (str): Optional provider tool configuration.
				description (str): Description value used by the operation.
				parameters (dict): Parameters value used by the operation.
				required (list[str]): Required value used by the operation.

		Returns:
				Value produced by the operation.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
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
			Logger( ).write( exception )
			raise exception

class Chat( Generator ):
	"""Chat component.

	Purpose:
			Provides the OpenAI chat and multimodal generation wrapper used by Foo. The
			class coordinates Responses API request construction, text output extraction,
			image analysis, document summarization, file search, web search, translation,
			transcription, and provider option helpers.

	Attributes:
			api_key (Optional[str]): Runtime state or configuration value maintained by the
			component.
			client (Optional[OpenAI]): Runtime state or configuration value maintained by
			the component.
			system_instructions (Optional[str]): Runtime state or configuration value
			maintained by the component.
			model (Optional[str]): Runtime state or configuration value maintained by the
			component.
			number (Optional[int]): Runtime state or configuration value maintained by the
			component.
			temperature (Optional[float]): Runtime state or configuration value maintained
			by the component.
			top_percent (Optional[float]): Runtime state or configuration value maintained
			by the component.
			frequency_penalty (Optional[float]): Runtime state or configuration value
			maintained by the component.
			presence_penalty (Optional[float]): Runtime state or configuration value
			maintained by the component.
			max_completion_tokens (Optional[int]): Runtime state or configuration value
			maintained by the component.
			store (Optional[bool]): Runtime state or configuration value maintained by the
			component.
			stream (Optional[bool]): Runtime state or configuration value maintained by the
			component.
			modalities (Optional[List[str]]): Runtime state or configuration value
			maintained by the component.
			stops (Optional[List[str]]): Runtime state or configuration value maintained by
			the component.
			response_format (Optional[str]): Runtime state or configuration value maintained
			by the component.
			reasoning_effort (Optional[str]): Runtime state or configuration value
			maintained by the component.
			input_text (Optional[str]): Runtime state or configuration value maintained by
			the component.
			id (Optional[str]): Runtime state or configuration value maintained by the
			component.
			vector_store_ids (Optional[List[str]]): Runtime state or configuration value
			maintained by the component.
			metadata (Optional[Dict[str, Any]]): Runtime state or configuration value
			maintained by the component.
			tools (Optional[List[Dict[str, Any]]]): Runtime state or configuration value
			maintained by the component.
			vector_stores (Optional[Dict[str, str]]): Runtime state or configuration value
			maintained by the component.
			web_search (Optional[bool]): Runtime state or configuration value maintained by
			the component.
			search_domains (Optional[List[str]]): Runtime state or configuration value
			maintained by the component.
			parallel_tool_calls (Optional[bool]): Runtime state or configuration value
			maintained by the component.
			tool_choice (Optional[str]): Runtime state or configuration value maintained by
			the component.
			request (Optional[Dict[str, Any]]): Runtime state or configuration value
			maintained by the component.
			response (Optional[Any]): Runtime state or configuration value maintained by the
			component.
			query (Optional[str]): Runtime state or configuration value maintained by the
			component.
			image_url (Optional[str]): Runtime state or configuration value maintained by
			the component.
			input (Optional[Any]): Runtime state or configuration value maintained by the
			component.
			messages (Optional[Any]): Runtime state or configuration value maintained by the
			component.
		"""
	
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	system_instructions: Optional[ str ]
	model: Optional[ str ]
	number: Optional[ int ]
	temperature: Optional[ float ]
	top_percent: Optional[ float ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	max_completion_tokens: Optional[ int ]
	store: Optional[ bool ]
	stream: Optional[ bool ]
	modalities: Optional[ List[ str ] ]
	stops: Optional[ List[ str ] ]
	response_format: Optional[ str ]
	reasoning_effort: Optional[ str ]
	input_text: Optional[ str ]
	id: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	metadata: Optional[ Dict[ str, Any ] ]
	tools: Optional[ List[ Dict[ str, Any ] ] ]
	vector_stores: Optional[ Dict[ str, str ] ]
	web_search: Optional[ bool ]
	search_domains: Optional[ List[ str ] ]
	parallel_tool_calls: Optional[ bool ]
	tool_choice: Optional[ str ]
	request: Optional[ Dict[ str, Any ] ]
	response: Optional[ Any ]
	query: Optional[ str ]
	image_url: Optional[ str ]
	input: Optional[ Any ]
	messages: Optional[ Any ]
	
	def __init__( self, num: int = 1, temp: float = 0.8, top: float = 0.9,
			freq: float = 0.0, pres: float = 0.0, iters: int = 10000,
			store: bool = True, stream: bool = True ) -> None:
		"""Initialize instance.

		Purpose:
				Initializes the object with default configuration, runtime state, and
				compatibility fields required by later method calls. The constructor performs
				local setup and does not execute a provider request.

		Args:
				num (int): Num value used by the operation.
				temp (float): Temp value used by the operation.
				top (float): Top value used by the operation.
				freq (float): Freq value used by the operation.
				pres (float): Pres value used by the operation.
				iters (int): Iters value used by the operation.
				store (bool): Store value used by the operation.
				stream (bool): Stream value used by the operation.
			"""
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
		self.vector_store_ids = [ 'vs_67e83bdf8abc81918bda0d6b39a19372' ]
		self.metadata = { }
		self.tools = [ ]
		self.vector_stores = { 'Code': 'vs_67e83bdf8abc81918bda0d6b39a19372' }
		self.web_search = False
		self.search_domains = [ ]
		self.parallel_tool_calls = True
		self.tool_choice = 'auto'
		self.request = None
		self.response = None
		self.query = None
		self.image_url = None
		self.input = None
		self.messages = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.

		Purpose:
				Returns a stable ordering of public attributes and methods for interactive
				inspection, documentation surfaces, and UI member displays.

		Returns:
				Ordered public member names exposed by the object.
			"""
		return [
				'api_key',
				'client',
				'system_instructions',
				'model',
				'number',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_completion_tokens',
				'store',
				'stream',
				'response_format',
				'reasoning_effort',
				'web_search',
				'search_domains',
				'parallel_tool_calls',
				'tool_choice',
				'tools',
				'vector_store_ids',
				'request',
				'response',
				'fetch',
				'generate_text',
				'generate_image',
				'analyze_image',
				'summarize_document',
				'search_web',
				'search_files',
				'translate',
				'transcribe',
				'get_format_options',
				'get_model_options',
				'get_effort_options',
				'get_data',
				'dump'
		]
	
	def normalize_domains( self, domains: Any ) -> List[ str ]:
		"""Normalize domains.

		Purpose:
				Normalizes user-supplied domain values into canonical, de-duplicated domain
				names suitable for provider web-search restrictions.

		Args:
				domains (Any): Optional domain restriction values for provider web-search tools.

		Returns:
				Normalized values suitable for the provider request.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			if domains is None:
				return [ ]
			
			if isinstance( domains, str ):
				parts = re.split( r'[\n,;]+', domains )
			elif isinstance( domains, (list, tuple, set) ):
				parts = [ str( item ) for item in domains if item is not None ]
			else:
				parts = [ str( domains ) ]
			
			values: List[ str ] = [ ]
			
			for entry in parts:
				value = str( entry ).strip( ).lower( )
				
				if not value:
					continue
				
				if not value.startswith( 'http://' ) and not value.startswith( 'https://' ):
					value = f'https://{value}'
				
				parsed = urllib.parse.urlparse( value )
				domain = (parsed.netloc or parsed.path or '').strip( ).lower( )
				domain = re.sub( r':\d+$', '', domain )
				domain = domain.lstrip( '.' )
				
				if domain.startswith( 'www.' ):
					domain = domain[ 4: ]
				
				if not domain:
					continue
				
				if not re.fullmatch( r'[a-z0-9][a-z0-9.-]*\.[a-z]{2,}', domain ):
					raise ValueError( f'Invalid domain: {domain}' )
				
				if domain not in values:
					values.append( domain )
			
			return values
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'normalize_domains( self, domains: Any ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def supports_reasoning( self, model: str ) -> bool:
		"""Supports reasoning.

		Purpose:
				Determines whether the selected model supports explicit reasoning configuration
				in the request payload.

		Args:
				model (str): Provider model identifier used for the request.

		Returns:
				Boolean indicator for the requested provider capability.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			throw_if( 'model', model )
			name = str( model ).strip( ).lower( )
			return name.startswith( 'gpt-5' ) or name.startswith( 'o' )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'supports_reasoning( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def build_instructions( self, system: str = None,
			response_format: str = None, web_search: bool = False,
			search_domains: Any = None ) -> str | None:
		"""Build instructions.

		Purpose:
				Builds the final instruction text sent to the provider from system text,
				requested output format, and optional search context.

		Args:
				system (str): Optional system-level instruction text.
				response_format (str): Optional response-format or structured-output
				configuration.
				web_search (bool): Web search value used by the operation.
				search_domains (Any): Optional domain restriction values for provider web-search
				tools.

		Returns:
				Provider-compatible request configuration produced from the supplied options.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			parts: List[ str ] = [ ]
			
			if system and str( system ).strip( ):
				parts.append( str( system ).strip( ) )
			
			if response_format and str( response_format ).strip( ).lower( ) == 'json':
				parts.append(
					'Return valid JSON only. Do not include markdown fences, prose, '
					'or commentary outside the JSON value.'
				)
			
			domains = self.normalize_domains( search_domains )
			if web_search and domains:
				parts.append(
					'When using web search, strongly prefer sources from the following '
					f'domains when they are relevant and available: {", ".join( domains )}.'
				)
			
			if parts:
				return '\n\n'.join( parts )
			
			return None
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = (
					'build_instructions( self, system: str | None=None, '
					'response_format: str | None=None, web_search: bool=False, '
					'search_domains: Any=None ) -> str | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def build_text_format( self, response_format: str | Dict[ str, Any ] = None,
			json_schema: Dict[ str, Any ] = None, schema_name: str = 'structured_response',
			schema_description: str = 'Structured JSON response.' ) -> Dict[ str, Any ] | None:
		"""Build text format.

		Purpose:
				Builds the OpenAI Responses API text-format configuration for plain text, JSON
				object output, or schema-constrained structured responses.

		Args:
				response_format (str | Dict[str, Any]): Optional response-format or structured-
				output configuration.
				json_schema (Dict[str, Any]): Optional response-format or structured-output
				configuration.
				schema_name (str): Schema name value used by the operation.
				schema_description (str): Schema description value used by the operation.

		Returns:
				Provider-compatible request configuration produced from the supplied options.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			if isinstance( response_format, dict ):
				return response_format
			
			mode = str( response_format or '' ).strip( ).lower( )
			
			if not mode or mode == 'auto':
				return None
			
			if mode == 'text':
				return { 'type': 'text' }
			
			if mode in [ 'json', 'json_object' ]:
				return { 'type': 'json_object' }
			
			if mode in [ 'json_schema', 'schema' ]:
				throw_if( 'json_schema', json_schema )
				return {
						'type': 'json_schema',
						'name': schema_name,
						'description': schema_description,
						'strict': True,
						'schema': json_schema
				}
			
			return None
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = (
					'build_text_format( self, response_format: str | Dict[ str, Any ] | None=None, '
					'json_schema: Dict[ str, Any ] | None=None, schema_name: str=structured_response, '
					'schema_description: str=Structured JSON response. ) -> Dict[ str, Any ] | None'
			)
			Logger( ).write( exception )
			raise exception
	
	def build_tools( self, web_search: bool = False, search_domains: Any = None,
			file_search: bool = False, vector_store_ids: List[ str ] = None,
			max_file_results: int = 20 ) -> List[ Dict[ str, Any ] ]:
		"""Build tools.

		Purpose:
				Builds provider tool declarations from web-search, grounding, file-search, and
				vector-store settings supported by the selected provider.

		Args:
				web_search (bool): Web search value used by the operation.
				search_domains (Any): Optional domain restriction values for provider web-search
				tools.
				file_search (bool): File search value used by the operation.
				vector_store_ids (List[str]): Vector store ids value used by the operation.
				max_file_results (int): Max file results value used by the operation.

		Returns:
				Provider-compatible request configuration produced from the supplied options.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			tools: List[ Dict[ str, Any ] ] = [ ]
			
			if web_search:
				tools.append( { 'type': 'web_search' } )
			
			if file_search:
				store_ids = vector_store_ids or self.vector_store_ids
				throw_if( 'vector_store_ids', store_ids )
				tools.append(
					{
							'type': 'file_search',
							'vector_store_ids': store_ids,
							'max_num_results': int( max_file_results )
					}
				)
			
			return tools
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = (
					'build_tools( self, web_search: bool=False, search_domains: Any=None, '
					'file_search: bool=False, vector_store_ids: List[ str ] | None=None, '
					'max_file_results: int=20 ) -> List[ Dict[ str, Any ] ]'
			)
			Logger( ).write( exception )
			raise exception
	
	def extract_output_text( self, response: Any ) -> str:
		"""Extract output text.

		Purpose:
				Extracts usable text from provider response shapes while handling common SDK
				object layouts and fallback representations.

		Args:
				response (Any): Response value used by the operation.

		Returns:
				Text extracted from the provider response.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			if response is None:
				return ''
			
			if hasattr( response, 'output_text' ) and response.output_text:
				return str( response.output_text )
			
			if isinstance( response, dict ):
				if response.get( 'output_text' ):
					return str( response.get( 'output_text' ) )
				if response.get( 'text' ):
					return str( response.get( 'text' ) )
			
			if hasattr( response, '__iter__' ) and not isinstance( response, (str, bytes, dict) ):
				parts: List[ str ] = [ ]
				
				for event in response:
					event_type = getattr( event, 'type', '' )
					
					if event_type == 'response.output_text.delta':
						delta = getattr( event, 'delta', '' )
						if delta:
							parts.append( str( delta ) )
					
					elif event_type == 'response.completed':
						final_response = getattr( event, 'response', None )
						if final_response is not None and hasattr( final_response, 'output_text' ):
							text = str( final_response.output_text or '' )
							if text:
								return text
				
				if parts:
					return ''.join( parts )
			
			output = getattr( response, 'output', None )
			if output:
				parts: List[ str ] = [ ]
				for item in output:
					content = getattr( item, 'content', None )
					if content:
						for block in content:
							text = getattr( block, 'text', None )
							if text:
								parts.append( str( text ) )
				
				if parts:
					return '\n'.join( parts ).strip( )
			
			return str( response )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'extract_output_text( self, response: Any ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def fetch( self, prompt: str, model: str = 'gpt-5-mini', temperature: float = 0.7,
			max_tokens: int = 1024, top_p: float = 1.0, seed: int | None = None,
			system: str = None, response_format: str | Dict[ str, Any ] = None,
			reasoning_effort: str = None, web_search: bool = False,
			search_domains: Any = None, store: bool = True, stream: bool = False,
			parallel_tool_calls: bool = True, tool_choice: str = 'auto',
			json_schema: Dict[ str, Any ] = None,
			schema_name: str = 'structured_response',
			schema_description: str = 'Structured JSON response.' ) -> str:
		"""Fetch.

		Purpose:
				Executes the provider-specific generation or retrieval workflow after validating
				input values and constructing the request payload required by the underlying
				provider API.

		Args:
				prompt (str): Input text submitted to the provider workflow.
				model (str): Provider model identifier used for the request.
				temperature (float): Sampling or generation control passed to the provider
				request.
				max_tokens (int): Sampling or generation control passed to the provider request.
				top_p (float): Sampling or generation control passed to the provider request.
				seed (int | None): Sampling or generation control passed to the provider
				request.
				system (str): Optional system-level instruction text.
				response_format (str | Dict[str, Any]): Optional response-format or structured-
				output configuration.
				reasoning_effort (str): Reasoning effort value used by the operation.
				web_search (bool): Web search value used by the operation.
				search_domains (Any): Optional domain restriction values for provider web-search
				tools.
				store (bool): Store value used by the operation.
				stream (bool): Stream value used by the operation.
				parallel_tool_calls (bool): Optional provider tool configuration.
				tool_choice (str): Optional provider tool configuration.
				json_schema (Dict[str, Any]): Optional response-format or structured-output
				configuration.
				schema_name (str): Schema name value used by the operation.
				schema_description (str): Schema description value used by the operation.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			
			self.query = prompt
			self.model = str( model ).strip( )
			self.temperature = float( temperature )
			self.max_completion_tokens = int( max_tokens )
			self.top_percent = float( top_p )
			self.store = bool( store )
			self.stream = bool( stream )
			self.web_search = bool( web_search )
			self.search_domains = self.normalize_domains( search_domains )
			self.parallel_tool_calls = bool( parallel_tool_calls )
			self.tool_choice = tool_choice or 'auto'
			self.response_format = (
					str( response_format ).strip( ).lower( )
					if isinstance( response_format, str )
					else response_format
			)
			self.reasoning_effort = reasoning_effort if reasoning_effort else None
			self.system_instructions = self.build_instructions(
				system=system,
				response_format=(
						response_format
						if isinstance( response_format, str )
						else None
				),
				web_search=self.web_search,
				search_domains=self.search_domains
			)
			self.tools = self.build_tools(
				web_search=self.web_search,
				search_domains=self.search_domains,
				file_search=False
			)
			
			self.request = {
					'model': self.model,
					'input': self.query,
					'max_output_tokens': self.max_completion_tokens,
					'store': self.store,
					'stream': self.stream,
					'parallel_tool_calls': self.parallel_tool_calls
			}
			
			text_format = self.build_text_format( response_format=response_format,
				json_schema=json_schema, schema_name=schema_name,
				schema_description=schema_description )
			
			if text_format:
				self.request[ 'text' ] = { 'format': text_format }
			
			if self.system_instructions:
				self.request[ 'instructions' ] = self.system_instructions
			
			if seed is not None:
				self.request[ 'seed' ] = int( seed )
			
			if self.tools:
				self.request[ 'tools' ] = self.tools
				self.request[ 'tool_choice' ] = self.tool_choice
			
			if self.supports_reasoning( self.model ) and self.reasoning_effort:
				self.request[ 'reasoning' ] = { 'effort': self.reasoning_effort }
			else:
				self.request[ 'temperature' ] = self.temperature
				self.request[ 'top_p' ] = self.top_percent
			
			self.response = self.client.responses.create( **self.request )
			return self.extract_output_text( self.response )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = (
					'fetch( self, prompt: str, model: str="gpt-5-mini", '
					'temperature: float=0.7, max_tokens: int=1024, top_p: float=1.0, '
					'seed: int | None=None, system: str | None=None, '
					'response_format: str | Dict[ str, Any ] | None=None, '
					'reasoning_effort: str | None=None, web_search: bool=False, '
					'search_domains: Any=None, store: bool=True, stream: bool=False, '
					'parallel_tool_calls: bool=True, tool_choice: str="auto", '
					'json_schema: Dict[ str, Any ] | None=None, '
					'schema_name: str="structured_response", '
					'schema_description: str="Structured JSON response." ) -> str'
			)
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, prompt: str, model: str = 'gpt-5-mini',
			temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 1.0,
			seed: int | None = None, system: str = None,
			response_format: str | Dict[ str, Any ] = None,
			reasoning_effort: str = None, web_search: bool = False,
			search_domains: Any = None, store: bool = True, stream: bool = False,
			parallel_tool_calls: bool = True, tool_choice: str = 'auto',
			json_schema: Dict[ str, Any ] = None ) -> str:
		"""Generate text.

		Purpose:
				Generates text output by delegating to the provider-specific fetch path while
				preserving a simplified call surface for text-generation workflows.

		Args:
				prompt (str): Input text submitted to the provider workflow.
				model (str): Provider model identifier used for the request.
				temperature (float): Sampling or generation control passed to the provider
				request.
				max_tokens (int): Sampling or generation control passed to the provider request.
				top_p (float): Sampling or generation control passed to the provider request.
				seed (int | None): Sampling or generation control passed to the provider
				request.
				system (str): Optional system-level instruction text.
				response_format (str | Dict[str, Any]): Optional response-format or structured-
				output configuration.
				reasoning_effort (str): Reasoning effort value used by the operation.
				web_search (bool): Web search value used by the operation.
				search_domains (Any): Optional domain restriction values for provider web-search
				tools.
				store (bool): Store value used by the operation.
				stream (bool): Stream value used by the operation.
				parallel_tool_calls (bool): Optional provider tool configuration.
				tool_choice (str): Optional provider tool configuration.
				json_schema (Dict[str, Any]): Optional response-format or structured-output
				configuration.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			return self.fetch( prompt=prompt, model=model, temperature=temperature,
				max_tokens=max_tokens, top_p=top_p, seed=seed, system=system,
				response_format=response_format, reasoning_effort=reasoning_effort,
				web_search=web_search, search_domains=search_domains, store=store,
				stream=stream, parallel_tool_calls=parallel_tool_calls,
				tool_choice=tool_choice, json_schema=json_schema )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str, ... ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def generate_image( self, prompt: str ) -> str:
		"""Generate image.

		Purpose:
				Generates an image from a prompt using the configured OpenAI image workflow and
				returns the generated image reference or payload.

		Args:
				prompt (str): Input text submitted to the provider workflow.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			throw_if( 'prompt', prompt )
			self.input_text = prompt
			self.response = self.client.images.generate(
				model='gpt-image-1',
				prompt=self.input_text,
				size='1024x1024'
			)
			
			if hasattr( self.response, 'data' ) and self.response.data:
				image = self.response.data[ 0 ]
				
				if hasattr( image, 'url' ) and image.url:
					return image.url
			
			return str( self.response )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'generate_image( self, prompt: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def analyze_image( self, prompt: str, url: str ) -> str:
		"""Analyze image.

		Purpose:
				Analyzes an image with a text prompt by building a multimodal request and
				returning the provider text response.

		Args:
				prompt (str): Input text submitted to the provider workflow.
				url (str): Url value used by the operation.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
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
									}
							]
					}
			]
			self.response = self.client.responses.create(
				model=self.model,
				input=self.input
			)
			return self.extract_output_text( self.response )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'analyze_image( self, prompt: str, url: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def summarize_document( self, prompt: str, path: str ) -> str:
		"""Summarize document.

		Purpose:
				Summarizes a document by uploading or referencing file content and submitting a
				document-aware provider request.

		Args:
				prompt (str): Input text submitted to the provider workflow.
				path (str): Path value used by the operation.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'path', path )
			file_path = Path( path )
			
			if not file_path.exists( ):
				raise FileNotFoundError( str( file_path ) )
			
			with file_path.open( 'rb' ) as stream:
				uploaded = self.client.files.create(
					file=stream,
					purpose='assistants'
				)
			
			self.messages = [
					{
							'role': 'user',
							'content': [
									{
											'type': 'input_text',
											'text': prompt
									},
									{
											'type': 'input_file',
											'file_id': uploaded.id
									}
							]
					}
			]
			self.response = self.client.responses.create( model=self.model, input=self.messages )
			return self.extract_output_text( self.response )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'summarize_document( self, prompt: str, path: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def search_web( self, prompt: str, model: str = 'gpt-5-mini',
			temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 1.0,
			seed: int | None = None, system: str = None,
			response_format: str | Dict[ str, Any ] = None,
			reasoning_effort: str = None, search_domains: Any = None,
			store: bool = True, stream: bool = False, parallel_tool_calls: bool = True,
			tool_choice: str = 'auto' ) -> str:
		"""Search web.

		Purpose:
				Runs a provider-specific web-search-enabled generation workflow using the
				configured model and search controls.

		Args:
				prompt (str): Input text submitted to the provider workflow.
				model (str): Provider model identifier used for the request.
				temperature (float): Sampling or generation control passed to the provider
				request.
				max_tokens (int): Sampling or generation control passed to the provider request.
				top_p (float): Sampling or generation control passed to the provider request.
				seed (int | None): Sampling or generation control passed to the provider
				request.
				system (str): Optional system-level instruction text.
				response_format (str | Dict[str, Any]): Optional response-format or structured-
				output configuration.
				reasoning_effort (str): Reasoning effort value used by the operation.
				search_domains (Any): Optional domain restriction values for provider web-search
				tools.
				store (bool): Store value used by the operation.
				stream (bool): Stream value used by the operation.
				parallel_tool_calls (bool): Optional provider tool configuration.
				tool_choice (str): Optional provider tool configuration.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			return self.fetch( prompt=prompt, model=model, temperature=temperature,
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
				tool_choice=tool_choice
			)
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'search_web( self, prompt: str, ... ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def search_files( self, prompt: str ) -> str:
		"""Search files.

		Purpose:
				Searches provider-managed files or vector stores with the supplied prompt and
				returns the generated text response.

		Args:
				prompt (str): Input text submitted to the provider workflow.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			throw_if( 'prompt', prompt )
			self.query = prompt
			self.tools = self.build_tools(
				web_search=False,
				file_search=True,
				vector_store_ids=self.vector_store_ids,
				max_file_results=20
			)
			self.request = {
					'model': self.model,
					'tools': self.tools,
					'input': prompt
			}
			self.response = self.client.responses.create( **self.request )
			return self.extract_output_text( self.response )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'search_files( self, prompt: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def translate( self, text: str ) -> str:
		"""Translate.

		Purpose:
				Translates the supplied text through the provider workflow configured for
				translation-style output.

		Args:
				text (str): Input text submitted to the provider workflow.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			throw_if( 'text', text )
			return self.fetch(
				prompt=f'Translate the following text faithfully and preserve meaning:\n\n{text}',
				model=self.model,
				temperature=0.2,
				max_tokens=self.max_completion_tokens,
				top_p=self.top_percent,
				system=self.system_instructions
			)
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'translate( self, text: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def transcribe( self, text: str ) -> str:
		"""Transcribe.

		Purpose:
				Transcribes the supplied text or audio-related input through the provider
				workflow configured for transcription-style output.

		Args:
				text (str): Input text submitted to the provider workflow.

		Returns:
				Provider response content produced by the requested workflow.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			throw_if( 'text', text )
			return text
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'transcribe( self, text: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def get_format_options( self ) -> List[ str ]:
		"""Get format options.

		Purpose:
				Returns the response-format options exposed to the Foo user interface.

		Returns:
				Configured values or option names used by the Foo interface.
			"""
		return [ 'auto', 'text', 'json', 'json_schema' ]
	
	def get_model_options( self ) -> List[ str ]:
		"""Get model options.

		Purpose:
				Returns the provider model options exposed to the Foo user interface.

		Returns:
				Configured values or option names used by the Foo interface.
			"""
		if hasattr( cfg, 'GPT_MODELS' ) and cfg.GPT_MODELS:
			return list( cfg.GPT_MODELS )
		
		return [
				'gpt-5.4',
				'gpt-5',
				'gpt-5-mini',
				'gpt-5-nano',
				'gpt-5.1',
				'gpt-5.2',
				'gpt-4.1'
		]
	
	def get_effort_options( self ) -> List[ str ]:
		"""Get effort options.

		Purpose:
				Returns the reasoning-effort options exposed to the Foo user interface.

		Returns:
				Configured values or option names used by the Foo interface.
			"""
		return [ 'minimal', 'low', 'medium', 'high' ]
	
	def get_data( self ) -> Dict[ str, Any ]:
		"""Get data.

		Purpose:
				Returns a structured snapshot of the wrapper state for diagnostics, display, or
				serialization.

		Returns:
				Configured values or option names used by the Foo interface.
			"""
		return {
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
				'request': self.request
		}
	
	def dump( self ) -> str:
		"""Dump.

		Purpose:
				Serializes the wrapper state into a JSON string for diagnostics, display, or
				persistence.

		Returns:
				Value produced by the operation.

		Raises:
				Error: Re-raised after the source exception is wrapped with structured Foo
				diagnostic metadata.
			"""
		try:
			return str( self.get_data( ) )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'generators'
			exception.cause = 'Chat'
			exception.method = 'dump( self ) -> str'
			Logger( ).write( exception )
			raise exception
