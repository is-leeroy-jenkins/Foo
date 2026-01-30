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
    scrapers.py
  </summary>
  ******************************************************************************************
'''
from typing import Optional, List, Pattern, Dict
from bs4 import BeautifulSoup
from requests import Response, HTTPError
from boogr import Error, ErrorDialog
import config as cfg
import re
import requests
from core import Result

def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Extractor( ):
	"""

		Purpose:
		--------
		Abstract base for HTML → plain-text extraction.

	"""
	raw_html: Optional[ str ]
	extracted_text: Optional[ str ]
	soup: Optional[ BeautifulSoup ]
	
	def __init__( self ):
		self.raw_html = None
		self.extracted_text = None
		self.soup = None

	def __dir__( self ) -> List[ str ]:
		"""
		
			Purpose:
			----------
			Provide a stable ordering for tooling and REPL use.
			
		"""
		return [ 'raw_html', 'extract' ]
	
	
class WebExtractor( Extractor):
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
	
	def scrape( self, url: str, time: int = 10 ) -> Result | None:
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
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
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
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
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
					for cell in row.find_all( [ 'td',
					                            'th' ] ):
						text = cell.get_text( ' ', strip=True )
						if text:
							_results.append( text )
			
			return _results
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
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
			blocks = [ art.get_text( " ", strip=True ) for art in self.soup.find_all( 'article' ) ]
			return [ b for b in blocks if b ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
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
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
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
			self.response.raise_for_status( )
			self.soup = BeautifulSoup( self.response.text, 'html.parser' )
			blocks = [ sec.get_text( " ", strip=True ) for sec in self.soup.find_all( 'section' ) ]
			return [ b for b in blocks if b ]
		except Exception as exc:
			exception = Error( exc )
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
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
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
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
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
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
			exception.module = 'scrapers'
			exception.cause = 'WebExtractor'
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
			exception.method = ('create_schema( self, function: str, tool: str, description: str, '
			                    'parameters: dict, required: list[ str ] ) -> Dict[ str, str ]')
			error = ErrorDialog( exception )
			error.show( )
	
