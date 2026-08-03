'''
	******************************************************************************************
	  Assembly:                Foo
	  Filename:                writers.py
	  Author:                  Terry D. Eppler
	  Created:                 05-31-2022

	  Last Modified By:        Terry D. Eppler
	  Last Modified On:        05-01-2025
	******************************************************************************************
	<copyright file="writers.py" company="Terry D. Eppler">

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
	writers.py

	Purpose:
		Provides Markdown writer utilities used by Foo workflows to persist extracted
		text and fetch results as documentation-friendly Markdown files. The module
		contains a base writer for plain text output and a specialized writer that
		serializes response metadata as YAML front matter followed by the response
		body.
	</summary>
	******************************************************************************************
'''
from pathlib import Path
from typing import Optional
from core import Result
from boogr import Error, Logger


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

class Writer( ):
	"""Writer component.

	Purpose:
	    Persists text payloads as UTF-8 Markdown files and retains the resulting output path.

	Attributes:
	    output_path (Optional[Path]): Directory that contains the generated Markdown output.
	    file_path (Optional[Path]): Resolved filesystem path of the current source or output file.
	    result (Optional[Result]): Most recent normalized Foo result produced by the instance.
	    body (Optional[str]): Text body retained for serialization by the writer.
	"""
	output_path: Optional[ Path ]
	file_path: Optional[ Path ]
	result: Optional[ Result ]
	body: Optional[ str ]
	
	def __init__( self ):
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    Any: Provider, loader, or normalized application value produced by the operation.
		"""
		self.output_path = None
		self.file_path = None
		self.result = None
		self.body = None
	
	def write( self, text: str, filename: str, directory: str = 'output' ) -> Path | None:
		"""Write.

		Purpose:
		    Persists the supplied content to the requested Markdown destination and returns the resolved output path.

		Args:
		    text (str): Text content supplied to the operation.
		    filename (str): Base filename used for the generated output file.
		    directory (str): Destination directory created when it does not already exist.

		Returns:
		    Path | None: Resolved path of the source or generated output.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'text', text )
			throw_if( 'file', filename )
			self.output_path = Path( directory )
			self.output_path.mkdir( parents=True, exist_ok=True )
			self.file_path = self.output_path / f'{filename}.md'
			self.file_path.write_text( text, encoding='utf-8' )
			return self.file_path
		except Exception as e:
			exc = Error( e )
			exc.module = 'writers'
			exc.cause = 'Writer'
			exc.method = 'write( self, text: str, filename: str, directory: str="output" ) -> Path '
			Logger( ).write( exc )
			raise exc

class MarkdownWriter( Writer ):
	"""MarkdownWriter component.

	Purpose:
	    Serializes a Foo Result as Markdown with YAML front matter containing source metadata.

	Attributes:
	    output_path (Any): Directory that contains the generated Markdown output.
	    file_path (Any): Resolved filesystem path of the current source or output file.
	    result (Any): Most recent normalized Foo result produced by the instance.
	    body (Any): Text body retained for serialization by the writer.
	"""
	def __init__( self ):
		"""Initialize the instance.

		Purpose:
		    Initializes instance state and provider or loader defaults required by subsequent operations.

		Returns:
		    Any: Provider, loader, or normalized application value produced by the operation.
		"""
		super( ).__init__( )
		self.output_path = None
		self.file_path = None
		self.result = None
		self.body = None
	
	def write( self, result: Result, path: str ) -> Path | None:
		"""Write.

		Purpose:
		    Persists the supplied content to the requested Markdown destination and returns the resolved output path.

		Args:
		    result (Result): Foo Result containing response metadata and body content.
		    path (str): Filesystem or resource path identifying the input or output.

		Returns:
		    Path | None: Resolved path of the source or generated output.

		Raises:
		    Error: Wraps the source exception with module, class, and method metadata, writes it to the application logger, and re-raises it.
		"""
		try:
			throw_if( 'result', result )
			throw_if( 'path', path )
			self.file_path = Path( path ).resolve( )
			self.result = result
			self.file_path.parent.mkdir( parents=True, exist_ok=True )
			front_matter = ('---\n'
			                + f'source_url: {self.result.url}\n'
			                + f'status_code: {self.result.status_code}\n'
			                + '---\n\n')
			
			body = self.result.text if self.result.text.endswith( '\n' ) else self.result.text + '\n'
			self.file_path.write_text( front_matter + body, encoding='utf-8' )
			return self.file_path
		except Exception as e:
			exception = Error( e )
			exception.module = 'writers'
			exception.cause = 'MarkdownWriter'
			exception.method = 'write( self, result: Result, path: str  ) -> Path'
			Logger( ).write( exception )
			raise exception
