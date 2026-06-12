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
	"""Validate a required argument.
	
	Purpose:
		Validates that a required argument contains a usable value before the writer
		workflow continues. This guard centralizes early validation so file-writing
		operations fail with consistent and readable error messages.
	
	Args:
		name (str): Name of the argument being validated.
		value (object): Argument value to validate.
	
	Raises:
		ValueError: Raised when the value is ``None`` or an empty string.
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be None.' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty.' )

class Writer( ):
	"""Write plain text content to Markdown files.

	Purpose:
		Provides the base Markdown-writing behavior used by Foo output workflows.
		The class creates the target output directory when needed, writes text using
		UTF-8 encoding, and stores the resolved output path on the instance for later
		inspection by callers or UI components.

	Attributes:
		output_path (Optional[Path]): Directory where the current output file is written.
		file_path (Optional[Path]): Path of the most recently written Markdown file.
		result (Optional[Result]): Optional result object retained by subclasses.
		body (Optional[str]): Optional body text retained by subclasses or callers.
	"""
	output_path: Optional[ Path ]
	file_path: Optional[ Path ]
	result: Optional[ Result ]
	body: Optional[ str ]
	
	def __init__( self ):
		"""Initialize the writer.

		Purpose:
			Initializes the writer with empty runtime state for output path, file path,
			result, and body content. The constructor performs only local assignment and
			does not touch the filesystem until ``write`` is called.
		"""
		self.output_path = None
		self.file_path = None
		self.result = None
		self.body = None
	
	def write( self, text: str, filename: str, directory: str = 'output' ) -> Path | None:
		"""Write text to a Markdown file.

		Purpose:
			Persists a text payload as a UTF-8 Markdown file under the requested output
			directory. The method validates required inputs, creates the output directory
			when needed, stores the final file path on the instance, and returns that path
			to the caller.

		Args:
			text (str): Text content to save.
			filename (str): Desired Markdown filename without the ``.md`` extension.
			directory (str): Directory where the Markdown file should be written.

		Returns:
			Path to the written Markdown file when the operation succeeds.

		Raises:
			Error: Re-raised after the source exception is wrapped and logged.
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
	"""Serialize fetch results as Markdown documents.

	Purpose:
		Extends the base writer to persist a ``Result`` object as a Markdown file with
		a compact YAML front matter block. The front matter captures source URL and
		status code metadata before writing the response text as the Markdown body.
	"""
	def __init__( self ):
		"""Initialize the Markdown result writer.

		Purpose:
			Initializes the inherited writer state and resets the fields used to track
			the target output file, source result, and generated body. No filesystem work
			is performed until ``write`` is called.
		"""
		super( ).__init__( )
		self.output_path = None
		self.file_path = None
		self.result = None
		self.body = None
	
	def write( self, result: Result, path: str ) -> Path | None:
		"""Write a result object to Markdown.

		Purpose:
			Writes a ``Result`` object to a Markdown file by creating parent directories,
			building a YAML front matter block from response metadata, appending the
			response text body, and returning the absolute output path.

		Args:
			result (Result): Result object containing response URL, status code, and text.
			path (str): Destination file path to write.

		Returns:
			Resolved path to the written Markdown file when the operation succeeds.

		Raises:
			Error: Re-raised after the source exception is wrapped and logged.
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
