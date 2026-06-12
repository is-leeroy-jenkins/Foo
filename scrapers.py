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

def throw_if( name: str, value: object ):
	"""Validate a required value.

	Purpose:
		Checks whether a required argument has been supplied before a scraper workflow
		uses it. The helper raises a consistent validation error for missing values so
		callers fail early instead of passing invalid data into requests or parsing code.

	Args:
		name (str): Name of the argument being validated.
		value (object): Value to validate.

	Raises:
		ValueError: Raised when the value is ``None``.
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Extractor( ):
	"""Provide shared state for HTML extractors.

	Purpose:
		Defines the minimal state used by concrete extraction classes that transform raw
		HTML into plain text or structured element lists. The base class stores raw HTML,
		extracted text, and the active BeautifulSoup parser object without imposing a
		fetching implementation.

	Attributes:
		raw_html (Optional[str]): Raw HTML text held by the extractor.
		extracted_text (Optional[str]): Plain or structured text extracted from the source HTML.
		soup (Optional[BeautifulSoup]): BeautifulSoup parser object for the current document.
	"""
	raw_html: Optional[ str ]
	extracted_text: Optional[ str ]
	soup: Optional[ BeautifulSoup ]
	
	def __init__( self ):
		"""Initialize extractor state.

		Purpose:
			Initializes the base extractor with empty runtime fields. The constructor does
			not fetch, parse, or transform data; it simply prepares instance members used
			by concrete subclasses.
		"""
		self.raw_html = None
		self.extracted_text = None
		self.soup = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return visible member names.

		Purpose:
			Provides a stable ordering for tooling, documentation, REPL inspection, and
			UI surfaces that display the public extractor members.

		Returns:
			Ordered extractor member names.
		"""
		return [ 'raw_html', 'extract' ]

class WebExtractor( Extractor ):
	"""Fetch and extract content from web pages.

	Purpose:
		Provides a concrete synchronous web extractor built on ``requests`` and
		``BeautifulSoup``. The class retrieves HTML pages, converts HTML to compact
		plain text, extracts commonly useful element groups, and builds function schema
		dictionaries for downstream tool-calling workflows.

	Attributes:
		soup (Optional[BeautifulSoup]): Parser object for the current response HTML.
		agents (Optional[str]): User-agent string used for HTTP requests.
		url (Optional[str]): Last URL submitted to the extractor.
		html (Optional[str]): Last raw HTML string held by the extractor.
		re_tag (Optional[Pattern]): Compiled regular expression used to strip tags.
		re_ws (Optional[Pattern]): Compiled regular expression used to normalize whitespace.
		response (Optional[Response]): Last HTTP response object.
	"""
	soup: Optional[ BeautifulSoup ]
	agents: Optional[ str ]
	url: Optional[ str ]
	html: Optional[ str ]
	re_tag: Optional[ Pattern ]
	re_ws: Optional[ Pattern ]
	response: Optional[ Response ]
	
	def __init__( self ) -> None:
		"""Initialize the web extractor.

		Purpose:
			Initializes the extractor with request defaults, compiled regular expressions,
			blank response state, and the configured Foo user-agent header. The constructor
			prepares the instance for later scrape and extraction calls without performing
			network activity.
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
			Controls the member ordering presented by introspection and documentation
			surfaces. The returned list preserves the existing public names exposed by the
			class, including legacy spelling entries used by any external callers.

		Returns:
			Ordered attribute and method names for the web extractor.
		"""
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
	
	def scrape( self, url: str, time: int = 10 ) -> Result | None:
		"""Fetch a web page.

		Purpose:
			Performs an HTTP GET request against the supplied URL, stores request state on
			the extractor, checks the HTTP response for failure status codes, and returns a
			canonical Foo ``Result`` object for downstream scraping or writing workflows.

		Args:
			url (str): Absolute URL to fetch.
			time (int): Request timeout in seconds.

		Returns:
			Result object containing the response URL, status code, body text, encoding,
			and headers when the fetch succeeds.

		Raises:
			Error: Re-raised after the source exception is wrapped and logged.
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
		"""Convert HTML to plain text.

		Purpose:
			Removes script and style blocks, inserts simple line breaks around common block
			elements, strips remaining tags, and normalizes whitespace to produce compact
			plain text suitable for display, indexing, or basic downstream processing.

		Args:
			html (str): Raw HTML string to convert.

		Returns:
			Plain text extracted from the supplied HTML.

		Raises:
			Error: Re-raised after the source exception is wrapped and logged.
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
		"""Extract paragraph text.

		Purpose:
			Fetches the target HTML document and extracts readable text from all paragraph
			elements. Empty paragraph results are removed so callers receive only useful
			text blocks.

		Args:
			uri (str): Fully qualified URI of the target HTML document.

		Returns:
			Clean paragraph text entries.

		Raises:
			Error: Re-raised after the source exception is wrapped and logged.
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
		"""Extract list item text.

		Purpose:
			Fetches the target HTML document and extracts text from list item elements in
			ordered or unordered lists. Empty list item values are discarded before the
			results are returned.

		Args:
			uri (str): Fully qualified URI of the HTML page.

		Returns:
			Clean list item text segments.

		Raises:
			Error: Re-raised after the source exception is wrapped and logged.
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
		"""Extract table cell text.

		Purpose:
			Fetches the target HTML document and flattens every discovered table into a
			list of cell values. Both header and data cells are included so callers can
			reconstruct or inspect table content from a simple text sequence.

		Args:
			uri (str): URI of the HTML document.

		Returns:
			Table cell values with one entry for each non-empty ``td`` or ``th`` element.

		Raises:
			Error: Re-raised after the source exception is wrapped and logged.
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
		"""Extract article text.

		Purpose:
			Fetches the target HTML page and extracts consolidated readable text from each
			article element. Each article is returned as a single cleaned string so callers
			can treat article blocks as document-like units.

		Args:
			uri (str): URI of the HTML page.

		Returns:
			Article-level text blocks.

		Raises:
			Error: Re-raised after the source exception is wrapped and logged.
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
		"""Extract heading text.

		Purpose:
			Fetches the target HTML document and extracts visible text from heading tags
			``h1`` through ``h6``. The result is useful for outline extraction, page
			summarization, and quick content inspection.

		Args:
			uri (str): Fully qualified document URI.

		Returns:
			Clean heading strings.

		Raises:
			Error: Re-raised after the source exception is wrapped and logged.
		"""
		try:
			throw_if( 'uri', uri )
			self.response = requests.get( uri, timeout=10 )
			self.response.raise_for_status( )
			self.soup = BeautifulSoup( self.response.text, 'html.parser' )
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
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
			exception.method = 'scrape_headings( self, uri: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def scrape_divisions( self, uri: str ) -> List[ str ] | None:
		"""Extract division text.

		Purpose:
			Fetches the target HTML document and extracts cleaned text from ``div``
			elements. This method provides a broad extraction path for pages whose main
			content is organized through generic containers rather than semantic tags.

		Args:
			uri (str): URI of the HTML document.

		Returns:
			Clean division text blocks.

		Raises:
			Error: Re-raised after the source exception is wrapped and logged.
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
		"""Extract section text.

		Purpose:
			Fetches the target HTML document and extracts readable text from semantic
			``section`` elements. Each non-empty section is returned as a cleaned text
			block for downstream review or indexing.

		Args:
			uri (str): Fully qualified document URI.

		Returns:
			Clean section text blocks.

		Raises:
			Error: Re-raised after the source exception is wrapped and logged.
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
		"""Extract blockquote text.

		Purpose:
			Fetches the target HTML document and extracts readable text from blockquote
			elements. The method is useful for collecting quoted material, pull quotes,
			or cited excerpts that are marked up semantically in the source page.

		Args:
			uri (str): Document URI.

		Returns:
			Cleaned blockquote text entries.

		Raises:
			Error: Re-raised after the source exception is wrapped and logged.
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
		"""Extract hyperlink references.

		Purpose:
			Fetches the target HTML page and extracts ``href`` attribute values from anchor
			elements. The method returns the raw hyperlink values present in the document
			without resolving them against the page URL.

		Args:
			uri (str): URI of the web page.

		Returns:
			Hyperlink paths or URLs extracted from anchor elements.

		Raises:
			Error: Re-raised after the source exception is wrapped and logged.
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
		"""Extract image references.

		Purpose:
			Fetches the target HTML page and extracts ``src`` attribute values from image
			elements. The method returns the raw image references present in the document
			without resolving them against the page URL.

		Args:
			uri (str): Fully qualified HTML page URI.

		Returns:
			Image source values extracted from image elements.

		Raises:
			Error: Re-raised after the source exception is wrapped and logged.
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
		"""Create a tool schema dictionary.

		Purpose:
			Constructs a JSON-compatible schema dictionary for a dynamic tool definition.
			The schema includes the exposed function name, a tool-aware description, JSON
			schema properties, and required parameter names for downstream model or tool
			orchestration workflows.

		Args:
			function (str): Function name exposed to the model or caller.
			tool (str): Underlying system or service wrapped by the function.
			description (str): Description of the function behavior.
			parameters (dict): Mapping of parameter names to JSON-schema fragments.
			required (list[str]): Required parameter names. When ``None``, all parameter
				keys are treated as required.

		Returns:
			JSON-compatible dictionary defining the dynamic tool schema.

		Raises:
			ValueError: Raised when ``parameters`` is not a dictionary.
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
	
