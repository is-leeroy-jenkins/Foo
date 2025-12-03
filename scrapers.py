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
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Optional
from urllib.request import urlopen
from boogr import Error, ErrorDialog

def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Scraper( ):
	"""

		Purpose:
		--------
		Orchestrates the scraping process by combining fetch, parse, and write operations.

		Methods:
		-------
		scrape(url: str, file: str, dir: str = "output") -> Optional[str]:
		Executes the complete scrape and save workflow.

	"""
	soup: Optional[ BeautifulSoup ]
	url: Optional[ str ]
	raw_html: Optional[ str ]
	parsed_text: Optional[ str ]
	
	def __init__( self ) -> None:
		self.soup = None
		self.url = None
		self.raw_html = None
		self.parsed_text = None
	
	def scrape( self, url: str ) -> str | None:
		"""

			Purpose:
			---------
			Scrape a webpage and save the text content to a Markdown file.

			Parameters:
			-----------
			url (str): Target website URL.
			file (str): Desired Markdown filename (without extension).
			dir (str): Directory to save the file into.

			Returns:
			--------
			Optional[str]: Path to the saved file if successful, otherwise None.

		"""
		try:
			throw_if( 'url', url )
			self.url = url
			html = urlopen( url )
			self.raw_html = BeautifulSoup( html, 'html.parser' )
			return self.raw_html
		except Exception as e:
			exc = Error( e )
			exc.module = 'scrapers'
			exc.cause = 'Scraper'
			exc.method = 'scrape( self, url: str, file: str, dir: str=output ) -> str'
			err = ErrorDialog( exc )
			err.show( )