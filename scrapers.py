'''
  ******************************************************************************************
      Assembly:                Foo
      Filename:                scrapers.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="scrapers.py" company="Terry D. Eppler">

	     Foo is a python framework for web scraping information into ML pipelines.
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
    scrapers.py

    Purpose:
        Provides synchronous HTML scraping utilities for the Foo application. The module
        defines a small extractor base class and a requests-backed web extractor capable of
        retrieving pages, converting HTML to compact text, extracting common HTML element
        groups, and constructing tool schema dictionaries for model-facing workflows.
  </summary>
  ******************************************************************************************
'''
from typing import Optional, List, Pattern, Dict
from bs4 import BeautifulSoup
from requests import Response, HTTPError
from boogr import Error, Logger
import config as cfg
import re
import requests
from core import Result


def throw_if( name: str, value: object ) -> None:
	"""Throw if.

	Purpose:
	    Validates that a required argument contains a usable value so failures occur before provider, filesystem, or parsing work begins.

	Args:
	    name (str): Argument name included in validation error messages.
	    value (object): Candidate value to validate or normalize.

	Returns:
	    None: This method updates instance state or validates input and does not return a value.

	Raises:
	    ValueError: Raised when the method cannot satisfy its documented value requirement.
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	if isinstance( value, str ) and (not value.strip( )):
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	if isinstance( value, (list, tuple, dict, set) ) and len( value ) == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )


class Extractor( ):
	"""Extractor component.

	Purpose:
	    Defines the minimal HTML parsing state shared by concrete content extractors.

	Attributes:
	    raw_html (Optional[str]): Raw HTML retained by the extractor before transformation.
	    extracted_text (Optional[str]): Plain text produced from the current HTML source.
	    soup (Optional[BeautifulSoup]): BeautifulSoup document tree for the current HTML content.
	"""
	raw_html: Optional[ str ]
	extracted_text: Optional[ str ]
	soup: Optional[ BeautifulSoup ]
	
	def __init__( self ):
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    Any: Provider, loader, or normalized application value produced by the operation.
		"""
		self.raw_html = None
		self.extracted_text = None
		self.soup = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    List[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'raw_html', 'extract' ]

class WebExtractor( Extractor ):
	"""WebExtractor component.

	Purpose:
	    Fetches HTML synchronously and extracts plain text, semantic element groups, links, images, and tool schemas.

	Attributes:
	    soup (Optional[BeautifulSoup]): BeautifulSoup document tree for the current HTML content.
	    agents (Optional[str]): Configured user-agent string sent with web requests.
	    url (Optional[str]): Most recent endpoint or resource URL used by the instance.
	    html (Optional[str]): Raw HTML returned by the most recent web request.
	    re_tag (Optional[Pattern]): Compiled pattern used to remove residual HTML tags.
	    re_ws (Optional[Pattern]): Compiled pattern used to collapse repeated whitespace.
	    response (Optional[Response]): Most recent raw response returned by the provider client.
	    timeout (Any): Maximum request duration, in seconds, applied to provider calls.
	    headers (Any): HTTP headers sent with the current request.
	"""
	soup: Optional[ BeautifulSoup ]
	agents: Optional[ str ]
	url: Optional[ str ]
	html: Optional[ str ]
	re_tag: Optional[ Pattern ]
	re_ws: Optional[ Pattern ]
	response: Optional[ Response ]
	
	def __init__( self ) -> None:
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    None: This method updates instance state or validates input and does not return a value.
		"""
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
		"""Return visible member names.

		Purpose:
		    Returns the stable public-member ordering used by introspection, interactive tools, and generated documentation.

		Returns:
		    List[str]: Ordered public member names exposed by the instance.
		"""
		return [ 'agents', 'url', 'html', 'timeout', 'headers', 'fetch', 'html_to_text',
			'scrape_images', 'scrape_hyperlinks', 'scrape_images', 'scrape_hyperlinks',
			'scrape_blockquotes', 'scrape_sections', 'scrape_divisions', 'sracpe_headings',
			'scrape_tables', 'scrape_lists', 'scrape_paragraphse', ]
	
	def scrape( self, url: str, time: int = 10 ) -> Result | None:
		"""Scrape.

		Purpose:
		    Scrape using the class state and returns data required by the surrounding workflow.

		Args:
		    url (str): Absolute endpoint or resource URL.
		    time (int): Maximum request duration in seconds.

		Returns:
		    Result | None: Normalized Foo result for the completed provider request, or ``None`` when the selected path does not create one.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
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
			exception.module = 'scrapers'
			exception.cause = 'WebFetcher'
			exception.method = 'fetch( self, url: str, time: int=10  ) -> Result'
			Logger( ).write( exception )
			raise exception
	
	def html_to_text( self, html: str ) -> str:
		"""Html to text.

		Purpose:
		    Removes non-content HTML, strips remaining markup, and normalizes whitespace into text suitable for indexing or display.

		Args:
		    html (str): Raw HTML document content to parse or convert.

		Returns:
		    str: Normalized text produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
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
			exception.module = 'scrapers'
			exception.cause = 'WebFetchers'
			exception.method = 'html2text( )'
			Logger( ).write( exception )
			raise exception
	
	def scrape_paragraphs( self, uri: str ) -> List[ str ] | None:
		"""Scrape paragraphs.

		Purpose:
		    Fetches the target page and extracts non-empty paragraphs content for downstream indexing or analysis.

		Args:
		    uri (str): Fully qualified resource identifier for the target page.

		Returns:
		    List[str] | None: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
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
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_paragraphs( self, uri: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def scrape_lists( self, uri: str ) -> List[ str ] | None:
		"""Scrape lists.

		Purpose:
		    Fetches the target page and extracts non-empty lists content for downstream indexing or analysis.

		Args:
		    uri (str): Fully qualified resource identifier for the target page.

		Returns:
		    List[str] | None: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
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
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_lists( self, uri: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def scrape_tables( self, uri: str ) -> List[ str ] | None:
		"""Scrape tables.

		Purpose:
		    Fetches the target page and extracts non-empty tables content for downstream indexing or analysis.

		Args:
		    uri (str): Fully qualified resource identifier for the target page.

		Returns:
		    List[str] | None: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'uri', uri )
			self.response = requests.get( uri, timeout=10 )
			self.response.raise_for_status( )
			self.soup = BeautifulSoup( self.response.text, 'html.parser' )
			_results: List[ str ] = [ ]
			for table in self.soup.find_all( 'table' ):
				for row in table.find_all( 'tr' ):
					for cell in row.find_all( [ 'td', 'th' ] ):
						text = cell.get_text( ' ', strip=True )
						if text:
							_results.append( text )
			
			return _results
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_tables( self, uri: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def scrape_articles( self, uri: str ) -> List[ str ] | None:
		"""Scrape articles.

		Purpose:
		    Fetches the target page and extracts non-empty articles content for downstream indexing or analysis.

		Args:
		    uri (str): Fully qualified resource identifier for the target page.

		Returns:
		    List[str] | None: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'uri', uri )
			self.response = requests.get( uri, timeout=10 )
			self.response.raise_for_status( )
			self.soup = BeautifulSoup( self.response.text, 'html.parser' )
			blocks = [ art.get_text( " ", strip=True ) for art in self.soup.find_all( 'article' ) ]
			return [ b for b in blocks if b ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_articles( self, uri: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def scrape_headings( self, uri: str ) -> List[ str ] | None:
		"""Scrape headings.

		Purpose:
		    Fetches the target page and extracts non-empty headings content for downstream indexing or analysis.

		Args:
		    uri (str): Fully qualified resource identifier for the target page.

		Returns:
		    List[str] | None: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'uri', uri )
			self.response = requests.get( uri, timeout=10 )
			self.response.raise_for_status( )
			self.soup = BeautifulSoup( self.response.text, 'html.parser' )
			heading_tags = [ 'h1', 'h2', 'h3', 'h4', 'h5', 'h6' ]
			blocks = [ h.get_text( ' ', strip=True ) for h in self.soup.find_all( heading_tags ) ]
			return [ b for b in blocks if b ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_headings( self, uri: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def scrape_divisions( self, uri: str ) -> List[ str ] | None:
		"""Scrape divisions.

		Purpose:
		    Fetches the target page and extracts non-empty divisions content for downstream indexing or analysis.

		Args:
		    uri (str): Fully qualified resource identifier for the target page.

		Returns:
		    List[str] | None: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
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
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_divisions( self, uri: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def scrape_sections( self, uri: str ) -> List[ str ] | None:
		"""Scrape sections.

		Purpose:
		    Fetches the target page and extracts non-empty sections content for downstream indexing or analysis.

		Args:
		    uri (str): Fully qualified resource identifier for the target page.

		Returns:
		    List[str] | None: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'uri', uri )
			self.response = requests.get( uri, timeout=10 )
			self.response.raise_for_status( )
			self.soup = BeautifulSoup( self.response.text, 'html.parser' )
			blocks = [ sec.get_text( " ", strip=True ) for sec in self.soup.find_all( 'section' ) ]
			return [ b for b in blocks if b ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_sections( self, uri: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def scrape_blockquotes( self, uri: str ) -> List[ str ] | None:
		"""Scrape blockquotes.

		Purpose:
		    Fetches the target page and extracts non-empty blockquotes content for downstream indexing or analysis.

		Args:
		    uri (str): Fully qualified resource identifier for the target page.

		Returns:
		    List[str] | None: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
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
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_blockquotes( self, uri: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def scrape_hyperlinks( self, uri: str ) -> List[ str ] | None:
		"""Scrape hyperlinks.

		Purpose:
		    Fetches the target page and extracts non-empty hyperlinks content for downstream indexing or analysis.

		Args:
		    uri (str): Fully qualified resource identifier for the target page.

		Returns:
		    List[str] | None: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
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
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_hyperlinks( self, uri: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def scrape_images( self, uri: str ) -> List[ str ] | None:
		"""Scrape images.

		Purpose:
		    Fetches the target page and extracts non-empty images content for downstream indexing or analysis.

		Args:
		    uri (str): Fully qualified resource identifier for the target page.

		Returns:
		    List[str] | None: Ordered values or records produced by the operation.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
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
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_images( self, uri: str ) -> List[ str ] '
			Logger( ).write( exception )
			raise exception
	
	def create_schema( self, function: str, tool: str,
			description: str, parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None:
		"""Create schema.

		Purpose:
		    Builds a JSON-compatible function schema for model tool-calling and orchestration workflows.

		Args:
		    function (str): Function name exposed in the generated tool schema.
		    tool (str): Service or tool name referenced by the generated schema.
		    description (str): Human-readable explanation embedded in the generated schema.
		    parameters (dict): JSON Schema property definitions for the tool arguments.
		    required (list[str]): Argument names that callers must supply to the generated tool.

		Returns:
		    Dict[str, str] | None: Dictionary containing normalized provider data, configuration, metadata, or generated schema content.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
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
			exception.module = 'Foo'
			exception.cause = ''
			exception.method = ('create_schema( self, *args ) -> Dict[ str, str ]')
			Logger( ).write( exception )
			raise exception
	
