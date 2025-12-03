from typing import Optional, List
from boogr import Error, ErrorDialog
from bs4 import BeautifulSoup
import html2text


def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class MarkdownConverter( ):
	"""

		Purpose:
		-----------
		Base class for HTML → Markdown conversion.
		
		Contract:
		-----------
		convert(html: str) -> str

	"""
	raw_html: Optional[ str ]
	parsed_text: Optional[ str ]

	def __dir__( self ) -> List[ str ]:
		"""
			
			Returns:
			-----------
			List[str]: attribute names followed by public methods.
			
		"""
		return [ 'raw_html', 'parsed_text', 'convert' ]

	def convert( self, html: str ) -> str | None:
		raise NotImplementedError( 'NOT IMPLEMENTED!' )

class Html2TextConverter( MarkdownConverter ):
	"""

		Purpose:
		-----------
		Uses the 'html2text' library for high-fidelity conversion when available.

	"""
	def __init__( self ) -> None:
		super( ).__init__( )
		self.raw_html = None
		self.parsed_text = None

	def __dir__( self ) -> List[ str ]:
		"""
		
			Purpose:
			-----------
			Provide ordering for Html2TextConverter's attributes and methods.
			
			Returns:
			-----------
			List[str]: attribute names followed by public methods.
			
		"""
		return [ 'raw_html',
		         'parsed_text',
		         'convert' ]


	def convert( self, html: str ) -> str | None:
		"""

			Purpose:
			-----------
			Convert HTML to Markdown using html2text.

			Parameters:
			-----------
			html (str): HTML fragment or full document.

			Returns:
			-----------
			str: Markdown representation.


		"""
		try:
			throw_if( 'htmel', html )
			self.raw_html = html
			h = html2text.HTML2Text( )
			h.body_width = 0
			h.ignore_links = False
			h.ignore_images = True
			h.skip_internal_links = False
			h.protect_links = True
			self.parsed_text = h.handle( self.raw_html ).strip( )
			return self.parsed_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'parsers'
			exception.cause = 'Html2TextConverter'
			exception.method = 'convert( self, html: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )


class FallbackConverter( MarkdownConverter ):
	"""

		Purpose:
		-----------
		Simple, dependency-light fallback that preserves headings, paragraphs,
		lists, and blockquotes from a parsed DOM.

	"""
	soup: Optional[ BeautifulSoup ]
	blocks: Optional[ List[ str ] ]
	raw_html: Optional[ str ]
	parsed_text: Optional[ str ]

	def __init__( self ):
		super( ).__init__( )
		self.blocks = [ ]
		self.raw_html = None
		self.parsed_text = None

	def __dir__( self ) -> List[ str ]:
		"""
			
			Returns:
			-----------
			List[str]: attribute names followed by public methods.
			
		"""
		return [ 'soup', 'blocks', 'raw_html', 'parsed_text', 'strip_noise', 'convert' ]

	def strip_noise( self, soup: BeautifulSoup ) -> None:
		try:
			throw_if( 'soup', soup )
			self.soup = soup
			for tag in self.soup(
					[ 'script', 'style', 'noscript', 'svg', 'canvas', 'iframe', 'form' ] ):
				tag.decompose( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'FallbackConverter'
			exception.method = 'strip_noise( self, soup: BeautifulSoup ) -> None'
			error = ErrorDialog( exception )
			error.show( )

	def convert( self, html: str ) -> str | None:
		"""

			Purpose:
			-----------
			Convert HTML to a basic Markdown string.

			Parameters:
			-----------
			html (str): HTML fragment or full document.

			Returns:
			-----------
			str: Markdown (simple).

		"""
		try:
			throw_if( 'html', html )
			self.raw_html = html
			self.soup = BeautifulSoup( self.raw_html, 'html.parser' )
			self.strip_noise( self.soup )
			body = self.soup.get_text
			if body is None:
				return self.soup.get_text( '\n', strip=True )
			tags = [ 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'blockquote', 'pre', 'code' ]
			for element in self.soup.find_all( tags ):
				txt = element.get_text( ' ', strip = True )
				if not txt:
					continue
					
				name = element.name.lower( )
				if name.startswith( 'h' ):
					level = int( name[ 1 ] ) if name[ 1: ].isdigit( ) else 2
					self.blocks.append( f'{"#" * level} {txt}' )
				elif name == 'li':
					self.blocks.append( f'- {txt}' )
				elif name == 'blockquote':
					quote = [ f'> {line}' for line in txt.splitlines( ) if line.strip( ) ]
					self.blocks.append( '\n'.join( quote ) )
				else:
					self.blocks.append( txt )
			if not self.blocks:
				return body.get_text( '\n', strip=True )
			self.parsed_text = '\n\n'.join( self.blocks )
			return self.parsed_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'SoupFallbackConverter'
			exception.method = 'convert( self, html: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )


