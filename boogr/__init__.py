'''
  ******************************************************************************************
      Assembly:                Gooey
      Filename:                init.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="init.py" company="Terry D. Eppler">

	     init.py is part of a data analysis tool integrating GenAI, Text Processing,
	     and Machine-Learning algorithms for federal analysts.
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
    init.py
  </summary>
  ******************************************************************************************
  '''
from __future__ import annotations

import io
import os
import traceback
from enum import Enum
import FreeSimpleGUI as sg
from sys import exc_info
from typing import List, Tuple, Optional
import matplotlib.figure
from .minion import App
from .enums import EXT, Client

matplotlib.use( 'TkAgg' )


class Error( Exception ):
	'''

        Purpose:
        ---------
		Class wrapping error used as the path argument for ErrorDialog class

        Constructor:
		----------
        Error( error: Exception, heading: str=None, cause: str=None,
                method: str=None, module: str=None )

    '''
	error: Optional[ Exception ]
	heading: Optional[ str ]
	cause: Optional[ str ]
	method: Optional[ str ]
	type: Optional[ BaseException ]
	trace: Optional[ str ]
	info: Optional[ str ]

	def __init__( self, error: Exception, heading: str=None, cause: str=None,
	              method: str=None, module: str=None ):
		super( ).__init__( )
		self.error = error
		self.heading = heading
		self.cause = cause
		self.method = method
		self.module = module
		self.type = exc_info( )[ 0 ]
		self.trace = traceback.format_exc( )
		self.info = str( exc_info( )[ 0 ] ) + ': \r\n \r\n' + traceback.format_exc( )

	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.info is not None:
				return self.info
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'message',
		         'cause',
		         'error',
		         'method',
		         'module',
		         'scaler',
		         'stack_trace',
		         'info' ]

class ButtonIcon( ):
	'''

        Constructor:
        -----------
		ButtonIcon( png: Enum )

        Pupose:
		---------
		Class representing form images

    '''

	def __init__( self, png: Enum ):
		self.name = png.name
		self.button = os.curdir + r'\boogr\resources\img\button'
		self.file_path = self.button + r'\\' + self.name + '.png'


	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.name is not None:
				return self.name


	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'button', 'name', 'file_path' ]

class TitleIcon( ):
	'''

	    Construcotr:
	    -----------
		TitleIcon( ico )

	    Purpose:
		--------
		Class used to define the TitleIcon used on the GUI

	'''

	def __init__( self, ico ):
		self.name = ico.name
		self.folder = os.curdir + r'\boogr\resources\ico'
		self.file_path = self.folder + r'\\' + self.name + r'.ico'


	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.file_path is not None:
			return self.file_path

	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'folder', 'name', 'authority_filepath' ]

class Dark(  ):
	'''

        Constructor:
		-----------
        Dark( )

        Pupose:
		-------
		Class representing the theme

    '''
	theme_background: Optional[ str ]
	theme_textcolor: Optional[ str ]
	element_forecolor: Optional[ str ]
	text_backcolor: Optional[ str ]
	text_forecolor: Optional[ str ]
	input_forecolor: Optional[ str ]
	input_backcolor: Optional[ str ]
	button_backcolor: Optional[ str ]
	button_forecolor: Optional[ str ]
	button_color: Optional[ Tuple[ str, str ] ]
	icon_path: Optional[ str ]
	theme_font: Optional[ Tuple[ str, int ] ]
	scrollbar_color: Optional[ str ]
	form_size: Optional[ Tuple[ int, int ] ]
	keep_on_top: Optional[ bool ]
	top_level: Optional[ bool ]
	resizeable: Optional[ bool ]
	context_menu: Optional[ List[ List[ str ] ] ]

	def __init__( self ):
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = os.curdir + r'\boogr\resources\ico\boogr.ico'
		self.theme_font = ( 'Roboto', 11 )
		self.scrollbar_color = '#755600'
		self.form_size = (400, 200)
		self.keep_on_top = True
		self.top_level = True
		self.resizable = True,
		self.context_menu = sg.MENU_RIGHT_CLICK_EDITME_VER_SETTINGS_EXIT
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Foo', os.curdir + r'\boogr\resources\theme' )
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size',
		         'theme_background',
		         'theme_textcolor',
		         'element_backcolor',
		         'element_forecolor',
		         'text_forecolor',
		         'text_backcolor',
		         'input_backcolor',
		         'input_forecolor',
		         'button_color',
		         'button_backcolor',
		         'button_forecolor',
		         'icon_path',
		         'theme_font',
		         'scrollbar_color'
		         'keep_on_top',
		         'top_level',
		         'resizeable',
		         'context_menu', ]

class FileDialog( Dark ):
	'''

	    Construcotr:
	    ------------
	    FileDialog( )

	    Purpose:
	    -------
	    Class that creates dialog to get path

	'''

	def __init__( self, extension=EXT.XLSX ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = os.curdir + r'\boogr\resources\ico\file_browse.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Foo', os.curdir + r'\boogr\resources\theme' )
		self.form_size = (500, 240)
		self.keep_on_top = True
		self.top_level = True
		self.resizable = True
		self.context_menu = sg.MENU_RIGHT_CLICK_EDITME_VER_SETTINGS_EXIT
		self.selected_item = None
		self.message = 'Grab File'
		self.extension = extension
		self.excel = (('Excel', '*.xlsx'),)
		self.csv = (('CSV', '*.csv'),)
		self.pdf = (('PDF', '*.pdf'),)
		self.sql = (('SQL', '*.sql',),)
		self.text = (('Text', '*.txt'),)
		self.access = (('Access', '*.accdb'),)
		self.sqlite = (('SQLite', '*.db'),)
		self.sqlserver = (('MSSQL', '*.mdf', '*.ldf', '*.sdf'),)

	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		return self.selected_item
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size',
		         'theme_background',
		         'theme_textcolor',
		         'element_backcolor',
		         'element_forecolor',
		         'text_forecolor',
		         'text_backcolor',
		         'input_backcolor',
		         'input_forecolor',
		         'button_color',
		         'button_backcolor',
		         'button_forecolor',
		         'icon_path',
		         'theme_font',
		         'scrollbar_color'
		         'keep_on_top',
		         'top_level',
		         'resizeable',
		         'selected_item',
		         'show',
		         'message',
		         'extension',
		         'excel',
		         'csv',
		         'pdf',
		         'sql',
		         'pages',
		         'access',
		         'sqlite',
		         'sqlserver' ]

	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_layout = [ [ sg.Text( ) ],
			            [ sg.Text( self.message, font = ( 'Roboto', 11 ) ) ],
			            [ sg.Text( ) ],
			            [ sg.Input( key='-PATH-' ), sg.FileBrowse( size=( 15, 1 ) ) ],
			            [ sg.Text( ) ],
			            [ sg.Text( ) ],
			            [ sg.OK( size=( 8, 1 ), ), sg.Cancel( size=( 10, 1 ) ) ] ]

			_window = sg.Window( ' File Search', _layout,
				font=self.theme_font,
				size=self.form_size,
				icon=self.icon_path,
				keep_on_top=self.keep_on_top )

			while True:
				_event, _values = _window.read( )
				if _event in ( sg.WIN_CLOSED, sg.WIN_X_EVENT, 'Cancel' ):
					break
				elif _event == 'OK':
					self.selected_item = _values[ '-PATH-' ]
					_window.close( )

			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'boogr'
			exception.cause = 'FileDialog'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )

class FolderDialog( Dark ):
	'''

		Purpose:
		----------
		Class defining dialog used to select a directory url

		Construcotr:
		-----------
		FolderDialog( )

	'''

	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = os.curdir + r'\boogr\resources\ico\folder_browse.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Foo', os.curdir + r'\boogr\resources\theme' )
		self.form_size = (500, 250)
		self.selected_item = None
		self.keep_on_top = True
		self.top_level = True
		self.resizable = True
		self.context_menu = sg.MENU_RIGHT_CLICK_EDITME_VER_SETTINGS_EXIT

	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.selected_item is not None:
			return self.selected_item

	def __dir__( self ) -> List[ str ] | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		return [ 'form_size',
		         'settings_path',
		         'original',
		         'theme_background',
		         'theme_textcolor',
		         'element_backcolor',
		         'element_forecolor',
		         'text_forecolor',
		         'text_backcolor',
		         'input_backcolor',
		         'input_forecolor',
		         'button_color',
		         'button_backcolor',
		         'button_forecolor',
		         'icon_path',
		         'theme_font',
		         'scrollbar_color',
		         'context_menu',
		         'selected_item',
		         'show',
		         'keep_on_top',
		         'top_level',
		         'resizeable' ]

	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_layout = [ [ sg.Text( ) ],
			            [ sg.Text( 'Folder Search' ) ],
			            [ sg.Text( ) ],
			            [ sg.Input( key='-PATH-' ), sg.FolderBrowse( size=( 15, 1 ) ) ],
			            [ sg.Text( size=( 100, 1 ) ) ],
			            [ sg.Text( size=( 100, 1 ) ) ],
			            [ sg.OK( size=( 8, 1 ) ), sg.Cancel( size=( 10, 1 ) ) ] ]

			_window = sg.Window( '  Gooey', _layout,
				font=self.theme_font,
				size=self.form_size,
				icon=self.icon_path,
				right_click_menu=self.context_menu,
				keep_on_top=self.keep_on_top,
				resizable=self.resizable,
				force_toplevel=self.top_level )

			while True:
				_event, _values = _window.read( )
				if _event in (sg.WIN_CLOSED, sg.WIN_X_EVENT, 'Cancel'):
					break
				elif _event == 'OK':
					self.selected_item = _values[ '-PATH-' ]
					sg.popup_ok( self.selected_item,
						title='Results',
						icon=self.icon_path,
						font=self.theme_font )

			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'boogr'
			exception.cause = 'FolderDialog'
			exception.method = 'show( self )'
			_error = ErrorDialog( exception )
			_error.show( )

class SaveFileDialog( Dark ):
	'''

	    Constructor:
	    ---------------
	    SaveFileDialog( url = '' ):

        Purpose:
        --------
        Class define object that provides a dialog to locate file destinations

    '''
	original: Optional[ str ]

	def __init__( self, path='' ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = os.curdir + r'\boogr\resources\ico\Save.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		self.file_name = None
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Foo', os.curdir + r'\boogr\resources\theme' )
		self.form_size = (550, 250)
		self.keep_on_top = True
		self.top_level = True
		self.resizable = True
		self.original = None

	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.file_name is not None:
			return self.file_name

	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size',
		         'settings_path',
		         'original',
		         'theme_background',
		         'theme_textcolor',
		         'element_backcolor',
		         'element_forecolor',
		         'text_forecolor',
		         'text_backcolor',
		         'input_backcolor',
		         'input_forecolor',
		         'button_color',
		         'button_backcolor',
		         'button_forecolor',
		         'icon_path',
		         'theme_font',
		         'scrollbar_color',
		         'original',
		         'file_name',
		         'show',
		         'keep_on_top',
		         'top_level',
		         'resizeable' ]

	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_username = os.environ.get( 'USERNAME' )
			_filename = sg.popup_get_file( 'Select Location / Enter File Name',
				title='  Gooey',
				font=self.theme_font,
				icon= self.icon_path,
				save_as=True,
				keep_on_top=self.keep_on_top )

			self.file_name = _filename

			if (self.original is not None and
					self.original != self.file_name and
					os.path.exists( self.original )):
				_src = io.open( self.original ).read( )
				_dest = io.open( _filename, 'w+' ).write( _src )
		except Exception as e:
			exception = Error( e )
			exception.module = 'boogr'
			exception.cause = 'SaveFileDialog'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )


class EmailDialog( Dark ):
	'''

	    Purpose:
	    --------
	    Class providing form used to send email messages.

	    Construcotr:
	    ------------
	    EmailDialog( sender: str=None, receiver: str=None,
			    subject: str=None, heading: str=None )


    '''

	def __init__( self, sender: str=None, receiver: list[ str ]=None,
	              subject: str=None, message: list[ str ]=None ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = os.curdir + r'\boogr\resources\ico\boogr.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Foo', os.curdir + r'\boogr\resources\theme' )
		self.image = os.curdir + r'\boogr\resources\img\app\web\outlook.png'
		self.form_size = (570, 550)
		self.keep_on_top = True
		self.top_level = True
		self.resizable = True
		self.context_menu = sg.MENU_RIGHT_CLICK_EDITME_VER_LOC_EXIT
		self.sender = sender
		self.receiver = receiver
		self.subject = subject
		self.message = message

	def __str__( self ) -> List[ str ] | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.message is not None:
			return self.message

	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'size',
		         'settings_path',
		         'theme_background',
		         'theme_textcolor',
		         'element_backcolor',
		         'element_forecolor',
		         'text_forecolor',
		         'text_backcolor',
		         'input_backcolor',
		         'input_forecolor',
		         'button_color',
		         'button_backcolor',
		         'button_forecolor',
		         'icon_path',
		         'theme_font',
		         'scrollbar_color',
		         'progressbar_color',
		         'sender',
		         'reciever',
		         'message',
		         'subject',
		         'others',
		         'password',
		         'username',
		         'show',
		         'keep_on_top',
		         'top_level',
		         'resizeable' ]

	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_btn = (20, 1)
			_input = (35, 1)
			_spc = (5, 1)
			_img = (50, 22)
			_clr = '#69B1EF'
			_layout = [ [ sg.Text( ' ', size = _spc ), ],
			            [ sg.Text( ' ', size = _spc ), ],
			            [ sg.Text( ' ', size = _spc ),
			              sg.Text( 'From:', size = _btn, text_color = _clr ),
			              sg.Input( key = '-EMAIL FROM-', size = _input ) ],
			            [ sg.Text( ' ', size = _spc ), sg.Text( 'To:', size = _btn,
				            text_color = _clr ),
			              sg.Input( key = '-EMAIL TO-', size = _input ) ],
			            [ sg.Text( ' ', size = _spc ),
			              sg.Text( 'Subject:', size = _btn, text_color = _clr ),
			              sg.Input( key = '-EMAIL SUBJECT-', size = _input ) ],
			            [ sg.Text( ' ', size = _spc ), sg.Text( ) ],
			            [ sg.Text( ' ', size = _spc ),
			              sg.Text( 'Username:', size = _btn, text_color = _clr ),
			              sg.Input( key = '-USER-', size = _input ) ],
			            [ sg.Text( ' ', size = _spc ),
			              sg.Text( 'Password:', size = _btn, text_color = _clr ),
			              sg.Input( password_char = '*', key = '-PASSWORD-', size = _input ) ],
			            [ sg.Text( ' ', size = _spc ) ],
			            [ sg.Text( ' ', size = _spc ),
			              sg.Multiline( 'Type your message here', size = (65, 10),
				              key = '-EMAIL TEXT-' ) ],
			            [ sg.Text( ' ', size = (100, 1) ) ],
			            [ sg.Text( ' ', size = _spc ), sg.Button( 'Send', size = _btn ),
			              sg.Text( ' ', size = _btn ), sg.Button( 'Cancel', size = _btn ) ] ]

			_window = sg.Window( '  Email', _layout,
				icon=self.icon_path,
				size=self.form_size,
				keep_on_top=self.keep_on_top )

			while True:  # Event Loop
				_event, _values = _window.read( )
				if _event in (sg.WIN_CLOSED, 'Cancel', 'Exit'):
					break
				if _event == 'Send':
					sg.popup_quick_message( 'Sending...this will take a moment...',
						background_color = 'red' )
			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'boogr'
			exception.cause = 'EmailDialog'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )

class MessageDialog( Dark ):
	'''

	    Purpose:
	    ---------
	    Class that provides form used to display informational messages

	    Construcotr:  MessageDialog( documents = '' )

    '''
	text: Optional[ str ]

	def __init__( self, text: str=None ):
		self.text = text
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = os.curdir + r'\boogr\resources\ico\boo.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Foo', os.curdir + r'\boogr\resources\theme' )
		self.form_size = (450, 250)
		self.keep_on_top = True
		self.top_level = True
		self.resizable = True

	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.text is not None:
			return self.text
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size',
		         'settings_path',
		         'original',
		         'theme_background',
		         'theme_textcolor',
		         'element_backcolor',
		         'element_forecolor',
		         'text_forecolor',
		         'text_backcolor',
		         'input_backcolor',
		         'input_forecolor',
		         'button_color',
		         'button_backcolor',
		         'button_forecolor',
		         'icon_path',
		         'theme_font',
		         'scrollbar_color',
		         'image',
		         'show',
		         'keep_on_top',
		         'top_level',
		         'resizeable' ]

	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_txtsz = (100, 1)
			_btnsz = (10, 1)
			_layout = [ [ sg.Text( size = _txtsz ) ],
			            [ sg.Text( size = _txtsz ) ],
			            [ sg.Text( size = (5, 1) ),
			              sg.Text( self.text,
				              font = ('Roboto', 11),
				              enable_events = True,
				              key = '-TEXT-',
				              text_color = '#69B1EF',
				              size = (80, 1) ) ],
			            [ sg.Text( size = _txtsz ) ],
			            [ sg.Text( size = _txtsz ) ],
			            [ sg.Text( size = _txtsz ) ],
			            [ sg.Text( size = (5, 1) ), sg.Ok( size = _btnsz ),
			              sg.Text( size = (15, 1) ), sg.Cancel( size = _btnsz ) ] ]

			_window = sg.Window( r' Message', _layout,
				icon=self.icon_path,
				font=self.theme_font,
				size=self.form_size,
				keep_on_top=True )

			while True:
				_event, _values = _window.read( )
				if _event in (sg.WIN_CLOSED, sg.WIN_X_EVENT, 'Ok', 'Cancel'):
					break

			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'boogr'
			exception.cause = 'MessageDialog'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )

class ErrorDialog( Dark ):
	'''

	    Construcotr:  ErrorDialog( error )

	    Purpose:  Class that displays excetption target_values that accepts
            a single, optional argument 'error' of scaler Error

    '''

	# Fields
	error: Exception = None
	heading: str = None
	module: str = None
	info: str = None
	cause: str = None
	method: str = None

	def __init__( self, error: Error ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = os.curdir + r'\boogr\resources\ico\error.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Foo', os.curdir + r'\boogr\resources\theme' )
		self.form_size = (500, 300)
		self.keep_on_top = True
		self.top_level = True
		self.resizable = True
		self.context_menu = sg.MENU_RIGHT_CLICK_EDITME_VER_LOC_EXIT
		self.error = error
		self.heading = error.heading
		self.module = error.module
		self.info = error.trace
		self.cause = error.cause
		self.method = error.method

	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if isinstance( self.info, str ):
			return self.info
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'size',
		         'settings_path',
		         'theme_background',
		         'theme_textcolor',
		         'element_backcolor',
		         'element_forecolor',
		         'text_forecolor',
		         'text_backcolor',
		         'input_backcolor',
		         'input_forecolor',
		         'button_color',
		         'button_backcolor',
		         'button_forecolor',
		         'icon_path',
		         'theme_font',
		         'scrollbar_color',
		         'progressbar_color',
		         'info',
		         'cause',
		         'method',
		         'error',
		         'heading',
		         'module',
		         'scaler',
		         'message',
		         'show', ]

	def show( self ) -> object:
		'''

            Purpose:
            --------


            Parameters:
            ----------


            Returns:
            ---------


		'''
		_msg = self.heading if isinstance( self.heading, str ) else None
		_info = f'Module:\t{self.module}\r\nClass:\t{self.cause}\r\n' \
		        f'Method:\t{self.method}\r\n \r\n{self.info}'
		_red = '#F70202'
		_font = ('Roboto', 10)
		_padsz = (3, 3)
		_layout = [ [ sg.Text( ) ],
		            [ sg.Text( f'{_msg}', size = (100, 1), key = '-MSG-', text_color = _red,
			            font = _font ) ],
		            [ sg.Text( size = (150, 1) ) ],
		            [ sg.Multiline( f'{_info}', key = '-INFO-', size = (80, 7), pad = _padsz ) ],
		            [ sg.Text( ) ],
		            [ sg.Text( size = (20, 1) ), sg.Cancel( size = (15, 1), key = '-CANCEL-' ),
		              sg.Text( size = (10, 1) ), sg.Ok( size = (15, 1), key = '-OK-' ) ] ]

		_window = sg.Window( r' Message', _layout,
			icon = self.icon_path,
			font = self.theme_font,
			size = self.form_size,
			keep_on_top=True )

		while True:
			_event, _values = _window.read( )
			if _event in (sg.WIN_CLOSED, sg.WIN_X_EVENT, 'Canel', '-OK-'):
				break

		_window.close( )

class InputDialog( Dark ):
	'''

	    Construcotr:  Input( prompt )

	    Purpose:  class that produces a contact path form

	'''
	# Fields
	theme_background: str = None
	response: str = None

	def __init__( self, question: str = None ):
		super( ).__init__( )
		self.question = question
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = os.curdir + r'\boogr\resources\ico\boogr.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Foo', os.curdir + r'\boogr\resources\theme' )
		self.form_size = (500, 250)
		self.selected_item = None
		self.keep_on_top = True
		self.top_level = True
		self.resizable = True
		self.context_menu = sg.MENU_RIGHT_CLICK_EDITME_VER_LOC_EXIT

	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.response is not None:
			return self.response
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size',
		         'settings_path',
		         'theme_background',
		         'theme_textcolor',
		         'element_backcolor',
		         'element_forecolor',
		         'text_forecolor',
		         'text_backcolor',
		         'input_backcolor',
		         'input_forecolor',
		         'button_color',
		         'button_backcolor',
		         'button_forecolor',
		         'icon_path',
		         'theme_font',
		         'scrollbar_color',
		         'input_text',
		         'show',
		         'keep_on_top',
		         'top_level',
		         'resizeable' ]

	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_layout = [ [ sg.Text( ) ],
			            [ sg.Text( self.question, font=('Roboto', 11 ) ) ],
			            [ sg.Text( ) ],
			            [ sg.Text( 'Enter:', size=(10, 2) ),
			              sg.InputText( key='-INPUT-', size=(40, 2) ) ],
			            [ sg.Text( size = (100, 1) ) ],
			            [ sg.Text( size = (100, 1) ) ],
			            [ sg.Text( size = (10, 1) ),
			              sg.Submit( size = (15, 1), key = '-SUBMIT-' ),
			              sg.Text( size = (5, 1) ),
			              sg.Cancel( size = (15, 1), key = '-CANCEL-' ) ] ]

			_window = sg.Window( '  Input', _layout,
				icon = self.icon_path,
				font = self.theme_font,
				size = self.form_size,
				keep_on_top=self.keep_on_top,
				resizable=self.resizable,
				force_toplevel=self.top_level )

			while True:
				_event, _values = _window.read( )
				if _event in (sg.WIN_X_EVENT, sg.WIN_CLOSED, '-CANCEL-', 'Exit'):
					break

				self.response = _values[ '-INPUT-' ]
				sg.popup( _event, _values, self.response,
					text_color = sg.theme_text_color( ),
					font = self.theme_font,
					icon = self.icon )

			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'boogr'
			exception.cause = 'InputDialog'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )

class SplashPanel( Dark ):
	'''

        Construcotr:  SplashPanel( )

        Purpose:  Class providing splash dialog behavior

	'''

	def __init__( self ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = os.curdir + r'\boogr\resources\ico\boogr.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Foo', os.curdir + r'\boogr\resources\theme' )
		self.image = os.curdir + r'\boogr\resources\img\gooey.png'
		self.form_size = (800, 600)
		self.timeout = 6000
		self.keep_on_top = True
		self.top_level = True
		self.resizable = True
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size',
		         'settings_path',
		         'theme_background',
		         'theme_textcolor',
		         'element_backcolor',
		         'element_forecolor',
		         'text_forecolor',
		         'text_backcolor',
		         'input_backcolor',
		         'input_forecolor',
		         'button_color',
		         'button_backcolor',
		         'button_forecolor',
		         'icon_path',
		         'theme_font',
		         'scrollbar_color',
		         'show',
		         'keep_on_top',
		         'top_level',
		         'resizeable' ]

	def show( self ) -> None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    None

		'''
		try:
			_img = self.image
			_imgsize = (500, 400)
			_line = (100, 2)
			_space = (15, 1)
			_layout = [ [ sg.Text( size = _space ), sg.Text( size = _line ) ],
			            [ sg.Text( size = _space ), sg.Text( size = _line ) ],
			            [ sg.Text( size = _space ),
			              sg.Image( filename = self.image, size = _imgsize ) ] ]
			_window = sg.Window( '  Gooey', _layout,
				no_titlebar = True,
				keep_on_top = True,
				grab_anywhere = True,
				size = self.form_size )
			while True:
				_event, _values = _window.read( timeout = self.timeout, close = True )
				if _event in (sg.WIN_CLOSED, 'Exit'):
					break
			_window.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'boogr'
			exception.cause = 'SplashPanel'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )

class Notification( Dark ):
	'''

        Purpose:
        ----------
        object providing form processing behavior

	'''

	def __init__( self, message: Optional[ str ] ):
		super( ).__init__( )
		sg.theme( 'DarkGrey15' )
		sg.theme_input_text_color( '#FFFFFF' )
		sg.theme_element_text_color( '#69B1EF' )
		sg.theme_text_color( '#69B1EF' )
		self.theme_background = sg.theme_background_color( )
		self.theme_textcolor = sg.theme_text_color( )
		self.element_forecolor = sg.theme_element_text_color( )
		self.element_backcolor = sg.theme_background_color( )
		self.text_backcolor = sg.theme_text_element_background_color( )
		self.text_forecolor = sg.theme_element_text_color( )
		self.input_forecolor = sg.theme_input_text_color( )
		self.input_backcolor = sg.theme_input_background_color( )
		self.button_backcolor = sg.theme_button_color_background( )
		self.button_forecolor = sg.theme_button_color_text( )
		self.button_color = sg.theme_button_color( )
		self.icon_path = os.curdir + r'\boogr\resources\ico\boogr.ico'
		self.theme_font = ('Roboto', 11)
		self.scrollbar_color = '#755600'
		sg.set_global_icon( icon=self.icon_path )
		sg.set_options( font=self.theme_font )
		sg.user_settings_save( 'Foo', os.curdir + r'\boogr\resources\theme' )
		self.form_size = (800, 600)
		self.success = b'iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAAA3NCSVQICAjb4U' \
		               b'/gAAAACXBIWXMAAAEKAAABCgEWpLzLAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5r' \
		               b'c2NhcGUub3Jnm+48GgAAAHJQTFRF////ZsxmbbZJYL9gZrtVar9VZsJcbMRYaM' \
		               b'ZVasFYaL9XbMFbasRZaMFZacRXa8NYasFaasJaasFZasJaasNZasNYasJYasJZ' \
		               b'asJZasJZasJZasJZasJYasJZasJZasJZasJZasJaasJZasJZasJZasJZ2IAizQ' \
		               b'AAACV0Uk5TAAUHCA8YGRobHSwtPEJJUVtghJeYrbDByNjZ2tvj6vLz9fb3/CyrN0oAAA' \
		               b'DnSURBVDjLjZPbWoUgFIQnbNPBIgNKiwwo5v1fsQvMvUXI5oqPf4DFOgCrhLKjC8GNV' \
		               b'gnsJY3nKm9kgTsduVHU3SU/TdxpOp15P7OiuV/PVzk5L3d0ExuachyaTWkAkLFtiBKAq' \
		               b'ZHPh/yuAYSv8R7XE0l6AVXnwBNJUsE2+GMOzWL8k3OEW7a/q5wOIS9e7t5qnGExvF5Bvl' \
		               b'c4w/LEM4Abt+d0S5BpAHD7seMcf7+ZHfclp10TlYZc2y2nOqc6OwruxUWx0rDjNJtyp6' \
		               b'HkUW4bJn0VWdf/a7nDpj1u++PBOR694+Ftj/8PKNdnDLn/V8YAAAAASUVORK5CYII='
		self.fail = b'iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAAA3NCSVQICAjb4U' \
		            b'/gAAAACXBIWXMAAADlAAAA5QGP5Zs8AAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm' \
		            b'+48GgAAAIpQTFRF////20lt30Bg30pg4FJc409g4FBe4E9f4U9f4U9g4U9f4E9g31Bf4E9f4E9f' \
		            b'4E9f4E9f4E9f4FFh4Vdm4lhn42Bv5GNx5W575nJ' \
		            b'/6HqH6HyI6YCM6YGM6YGN6oaR8Kev9MPI9cb' \
		            b'M9snO9s3R+Nfb+dzg+d/i++vt/O7v/fb3/vj5//z8//7' \
		            b'+////KofnuQAAABF0Uk5TAAcIGBktSY' \
		            b'SXmMHI2uPy8/XVqDFbAAAA8UlEQVQ4y4VT15LCMBBTQkgPYem9d9D' \
		            b'//x4P2I7vILN68kj2WtsAh' \
		            b'yDO8rKuyzyLA3wjSnvi0Eujf3KY9OUP+kno651CvlB0Gr1byQ9UXff' \
		            b'+py5SmRhhIS0oPj4SaUUC' \
		            b'AJHxP9+tLb/ezU0uEYDUsCc+l5' \
		            b'/T8smTIVMgsPXZkvepiMj0Tm5txQLENu7gSF7HIuMreRxYNkb' \
		            b'mHI0u5Hk4PJOXkSMz5I3nyY08HMjbpOFylF5WswdJPmYeVaL28968yNfGZ2r9gvqFalJNUy2UW' \
		            b'mq1Wa7di/3Kxl3tF1671YHRR04dWn3s9cXRV09f3vb1fwPD7z9j1WgeRgAAAABJRU5ErkJggg=='
		self.ninja = b'iVBORw0KGgoAAAANSUhEUgAAACAAAAAnCAYAAABuf0pMAAABhWlDQ1BJQ0MgUHJvZmlsZQA' \
		             b'AeJx9kT1Iw0AcxV9bS1WqDnYo4pChOlkQFRFcpIpFsFDaCq06mFz6BU0akhQXR8G14ODHYtXB' \
		             b'xVlXB1dBEPwAcXRyUnSREv+XFFrEeHDcj3f3HnfvAG' \
		             b'+jwhSjaxxQVFNPxWNCNrcqBF7hRz96E' \
		             b'MasyAwtkV7MwHV83cPD17soz3I/9+fok/MGAzwC8RzTdJN4g3h609Q47xOHWEmUic' \
		             b'+Jx3S6I' \
		             b'PEj1yWH3zgXbfbyzJCeSc0Th4iFYgdLHcxKukI8RRyRFZXyvVmHZc5bnJVKjbXuyV8YzKsr' \
		             b'aa7THEYcS0ggCQESaiijAhNRWlVSDKRoP+biH7L9SXJJ5CqDkWMBVSgQbT/4H/zu1ihMTjh' \
		             b'JwRjgf7GsjxEgsAs065b1fWxZzRPA9wxcqW1/tQHMfJJeb2uRI2BgG7i4bmvSHnC5A4SfNF' \
		             b'EXbclH01soAO9n9E05YPAW6F1zemvt4/QByFBXyzfAwSEwWqTsdZd3d3f29u+ZVn8/pE' \
		             b'Fyu/Q7rYsAAAbASURBVHicvZd/bJVXGcc/55z3vvdHuf3BbaFldGyDbQhSJsGNlSC66S' \
		             b'gM/hDYxhJLRIcsbs7IRBONiTEi0RmDJltUthlykegYCT+EyUKZcZBABGSzU34NKpcC7S' \
		             b'1tb2/f3h/v+57jH6Vd6S+gbXyS88853+d5vuf7nuc85xWMhVXWrgbWAAuBU8B24DUS8a5' \
		             b'buYpRJq4Bfg5UDbLaDLxMIr4N4P3tmyLBoB357uZdFWkncP6fJw9lRkUgWF7zW19F13ky' \
		             b'NCRmnKV5sabkaM38ioiBKs/39fZ9Z+Qfj4rf5S9tex7AGklyu/zJZYHcx+ssqwRlleCpK' \
		             b'L6wAZgQ8lk4XbGq5h7KxkfIZvPzUp0ZxhcV0NGZlasWz2hxDu5ueutGLDkSAoHcpbVCO2g' \
		             b'ZxlWFvckBHrrPJxyL8dKvz5DJ5ABwulyuJjs5eOwC44tC79ydPzu5B3/nClTWRkTq0CLI' \
		             b'o2UEgQYMLyyfzhe/MJei4jCHD5+gtfEqUkqUkgSDkt3vNXP6cisLKs8ejSn18i+KS8P' \
		             b'fa2/J3DGBSPbCHKE7bIRizlTBN55bwaxZDyKl4Oy58xw4cJz3/v4fFswIEw7ZHDp6gSMft' \
		             b'HDgfAGfKbdIvH1sabll1QOPAftu+xDGYjGSyaRdGJu5eO1Xl+x66qkVTJ02DcdxOH' \
		             b'GynncP/oMtf7nYiy8JaIqCgsspB+k7eIHxlNiae13FOq/hz1P0paNPNDVuvi0FtNbCGD' \
		             b'PbGLOxufHEJMuySKfT1NW9zxtbd3PoVIrualC9Pm2upM2FymiEq2mQOkdbPsh1YVFsVT7' \
		             b'9nO/th8Zbl2FrW9tdGF7yPO9bnueFHafr3N69e+/XydOUlpfhtLUjlaCwIISlJJ6vSTtZ' \
		             b'XNdn2oyZdF2/wjMb6zEotAxiRC/Jk8C8QRVQSpFMJudms7n1zU3JpzsdR9t2IB4KhTZXL' \
		             b'fhmTnWePL3ha0tFkeuSzuZZ9MTjZJINXEk6VEyIUFx+H/sPvEsm08Uv45fxVHSwNHOAH' \
		             b'w5QoOX69QVdXZmfdKQ6Pt/RmW4BXgVeq573SHMPpqB4+p5IwFv27JLZLP5cFRcbW3lz10' \
		             b'VOJKNUFki+vXwCD02PUXesiZ/taR1O4LabCDQ0/Hd5KtWx08lkEmBeAfF69byHM/29gh' \
		             b'O/NDWQ/fgEVmERQgESX0XJ2hWYO7taNvQS+PBf9YA46DjOW8aYP1Q/+og7nGekdF611J3' \
		             b'7kcEiEPhyHJlg5bDZBLqHoAN8h0R8Sy+BU6c+FEKK0OyqWQN2PJTZ5UsetPz2VwRmmVYF' \
		             b'ZAPlGARg6N9mlM4Q9FpM3irb4cnQ90nEGxiAGoEFK55caXmtO4wM4aoijLDwZLhf8mxL' \
		             b'wE/FtQz9Jn9lT0PftRE1o74mdWamMB7C70TKMDk1bgDGl6Fav3HHXwf1Hy0BLUOHDdKA' \
		             b'RvlpAn4aYfz+sPVD+Y/6EwDYFctqLL/9DV9FJ+Ws2JAwEvEBB3vUCgDkreI6hDJGDPtF5' \
		             b'w82OToClbUhAIGOCe3edQt045gRkJOfLaWytg5oobJ2o+U7VUaANC7K3KzyphfnA6RIx' \
		             b'M+NGQHbu75JYB4DCoAfuCq6ptpNpSf5DqABWFFdyOs/XsTKZQt5Xqf2DRVrRIcwPPHx1a5' \
		             b'VvNWTke4gxufu7HlmG03UKqLCZFBRi/uXzqX8nikEH5ieql2/bda1M/FE/1gjugdygbJ3' \
		             b'gm6L8e2wMAiMUFyxK7hmXPJWCQvcFOdyUTbc+wA76v7NgV8d18DDwAACIy7DgrJH610rNj' \
		             b'NvlfTOKZNDC4sVuascscvwIiGSGQPwdRLxNweLM4oqENdstwlLf9I6tAi0hgx7pnlN1Pg' \
		             b'dPckN8PZQUUZMQMvwTiMsZJ9Tb5AbVnvXUkV2IVNxeqaPkIh3jDmBrD1xixH2cWF8hPG1' \
		             b'1Ll222s/Dd5KVxWyy+ptzYeHizOqq1hOXlVoe6lPeaogLf2ujzwV9QM6rfLW+BttGYC' \
		             b'VJOI7h4oxqm6oL/+pIwvHAILli/Jg7JwVw9Jd9JQoQ9yAvZsYDYG+pnT2b9x48fZJDvD' \
		             b'B/4WAr8b9Pugm6T70pme6mUR82BfWmBHIXd2301WF9QE/jaVzH0njbwVm3spv1C+iHgu' \
		             b'WL1pjdObTvopkfBmqHq70+trYKFD5FSG99vW+jKBlKAysvV3XnlqRQBCwgQDdyki6f/b' \
		             b'kDVx/sobu1mfCpdVfllJszthT0J/8eu0CtpCI778VgUnAhEES3LZFYp99QQj5jFbRcC5' \
		             b'QKrUI9F3+KYn4j4YjAN07D3GzAoqbFRB98Kbf8PsM98bIAVl6HghD2P8Avm6w' \
		             b'ywIVvIgAAAAASUVORK5CYII='
		self.message = '\r\n ' + message if message is not None else '\r\nThe action you have performed has been successful!'

	def __str__( self ) -> str | None:
		'''

            Purpose:
            --------
			returns a string reprentation of the object

            Parameters:
            ----------
			self

            Returns:
            ---------
			str | None

		'''
		if self.message is not None:
			return self.message

	def __dir__( self ) -> List[ str ] | None:
		'''

		    Purpose:
		    --------
		    Creates a List[ str ] of type members

		    Parameters:
		    ----------
			self

		    Returns:
		    ---------
			List[ str ] | None

		'''
		return [ 'form_size',
		         'settings_path',
		         'theme_background',
		         'theme_textcolor',
		         'element_backcolor',
		         'element_forecolor',
		         'text_forecolor',
		         'text_backcolor',
		         'input_backcolor',
		         'input_forecolor',
		         'button_color',
		         'button_backcolor',
		         'button_forecolor',
		         'icon_path',
		         'theme_font',
		         'scrollbar_color',
		         'input_text',
		         'show' ]

	def show( self ) -> int | None:
		'''

		    Purpose:
		    --------
		    Method displays the control/form

		    Parameters:
		    ----------
		    self

		    Returns:
		    --------
		    int | None

		'''
		try:
			return sg.popup_notify( self.message,
				title = 'Notification',
				icon = self.ninja,
				display_duration_in_ms = 10000,
				fade_in_duration = 5000,
				alpha = 1 )

		except Exception as e:
			exception = Error( e )
			exception.module = 'boogr'
			exception.cause = 'Notification'
			exception.method = 'show( self )'
			error = ErrorDialog( exception )
			error.show( )
