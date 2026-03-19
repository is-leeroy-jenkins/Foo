'''
	******************************************************************************************
	  Assembly:                Foo
	  Filename:                aap.py
	  Author:                  Terry D. Eppler
	  Created:                 05-31-2022
	  Last Modified By:        Terry D. Eppler
	  Last Modified On:        08-25-2025
	******************************************************************************************
	<copyright file="app.py" company="Terry D. Eppler">
	
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
	
	 THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	 FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
	 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
	 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
	 ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	 DEALINGS IN THE SOFTWARE.
	
	 You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov
	
	</copyright>
	<summary>
	    app.py — the UI
	</summary>
	******************************************************************************************
'''
from __future__ import annotations
import altair
import inspect
from astroquery.simbad import Simbad
import base64

from bs4 import BeautifulSoup

import config as cfg
from collections import deque, Counter
import datetime as dt
import html as html_lib
import json
import numpy as np
import os
import re
import time
import types
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
from langchain_core.documents import Document
from lxml import etree
from loaders import (
	TextLoader,
	CsvLoader,
	PdfLoader,
	ExcelLoader,
	WordLoader,
	MarkdownLoader,
	HtmlLoader,
	JsonLoader,
	PowerPointLoader,
	WikiLoader,
	GithubLoader,
	WebLoader,
	ArXivLoader,
	XmlLoader,
	PubMedSearchLoader,
	OpenCityDocLoader,
	JupyterNotebookLoader,
	AwsFileLoader,
	OneDriveDocLoader,
	GoogleCloudFileLoader,
	GoogleSpeechToTextAudioLoader
)

from generators import Chat, Claude, Grok, Mistral, Gemini
from fetchers import (
	Wikipedia, TheNews, SatelliteCenter, WebFetcher,
	GoogleWeather, Grokipedia, OpenWeather, NavalObservatory,
	GoogleSearch, GoogleDrive, GoogleMaps, NearbyObjects, OpenScience,
	EarthObservatory, SpaceWeather, AstroCatalog, AstroQuery, StarMap,
	GovData, Congress, InternetArchive, StarChart, HistoricalWeather, GoogleGeocoding,
	USGSEarthquakes, USGSWaterData, USGSTheNationalMap, USGSScienceBase,
	AirNow, ClimateData, EoNet, EnviroFacts, TidesAndCurrents, UvIndex, PurpleAir,
	OpenAQ, Firms, CensusData, Socrata, HealthData, GlobalHealthData,
	UnitedNations, WorldPopulation, Wonder)

import nltk
from nltk import sent_tokenize
from nltk.corpus import stopwords, wordnet, words
from nltk.tokenize import word_tokenize
import plotly.graph_objects as px
import pandas as pd
from pandas import DataFrame
import streamlit as st
import scrapers
import sqlite3
from sqlite3 import Connection
from urllib.parse import urljoin, urlparse

try:
	import textstat
	
	TEXTSTAT_AVAILABLE = True
except ImportError:
	TEXTSTAT_AVAILABLE = False

# =====================================================================
# SESSION STATE DEFINITIONS
# =====================================================================

if 'mode' not in st.session_state or st.session_state[ 'mode' ] is None:
	st.session_state[ 'mode' ] = list( cfg.MODE_MAP.keys( ) )[ 0 ]
	
# -------- GENERATIVE AI VARIABLES --------------------

if 'model' not in st.session_state:
	st.session_state[ 'model' ] = ''

if 'max_tools' not in st.session_state:
	st.session_state[ 'max_tools' ] = 0

if 'max_tokens' not in st.session_state:
	st.session_state[ 'max_tokens' ] = 0

if 'temperature' not in st.session_state:
	st.session_state[ 'temperature' ] = 0.0

if 'top_percent' not in st.session_state:
	st.session_state[ 'top_percent' ] = 0.0

if 'frequency_penalty' not in st.session_state:
	st.session_state[ 'frequency_penalty' ] = 0.0

if 'presense_penalty' not in st.session_state:
	st.session_state[ 'presense_penalty' ] = 0.0

if 'background' not in st.session_state:
	st.session_state[ 'background' ] = False

if 'parallel_tools' not in st.session_state:
	st.session_state[ 'parallel_tools' ] = False

if 'store' not in st.session_state:
	st.session_state[ 'store' ] = False

if 'stream' not in st.session_state:
	st.session_state[ 'stream' ] = False

if 'response_format' not in st.session_state:
	st.session_state[ 'response_format' ] = ''

if 'tool_choice' not in st.session_state:
	st.session_state[ 'tool_choice' ] = ''

if 'reasoning' not in st.session_state:
	st.session_state[ 'reasoning' ] = ''

if 'stops' not in st.session_state:
	st.session_state[ 'stops' ] = [ ]

if 'include' not in st.session_state:
	st.session_state[ 'include' ] = [ ]

if 'input' not in st.session_state:
	st.session_state[ 'input' ] = [ ]

if 'tools' not in st.session_state:
	st.session_state[ 'tools' ] = [ ]

if 'messages' not in st.session_state:
	st.session_state[ 'messages' ] = [ ]

# ------------ LOADER VARIABLES -----------

if 'loader_results' not in st.session_state:
	st.session_state[ 'loader_results' ] = { }

if 'documents' not in st.session_state:
	st.session_state[ 'documents' ] = None

if 'tokens' not in st.session_state:
	st.session_state[ 'tokens' ] = None

if 'vocabulary' not in st.session_state:
	st.session_state[ 'vocabulary' ] = None

if 'raw_text' not in st.session_state:
	st.session_state[ 'raw_text' ] = ''

if 'processed_text' not in st.session_state:
	st.session_state[ 'processed_text' ] = ''

if 'token_counts' not in st.session_state:
	st.session_state[ 'token_counts' ] = None

if 'loader_path' not in st.session_state:
	st.session_state[ 'loader_path' ] = ''

if 'loader_text' not in st.session_state:
	st.session_state[ 'loader_text' ] = ''

if 'loader_files' not in st.session_state:
	st.session_state[ 'loader_files' ] = ''

# ------------- SCRAPPER VARIABLES --------------

if 'target_url' not in st.session_state:
	st.session_state[ 'target_url' ] = ''

for corpus in cfg.REQUIRED_CORPORA:
	try:
		nltk.data.find( f'corpora/{corpus}' )
	except LookupError:
		nltk.download( corpus )
	
# =====================================================================
# UTILITITES
# =====================================================================

def _filter_kwargs_for_callable( fn: Any, kwargs: dict[ str, Any ] ) -> dict[ str, Any ]:
	try:
		sig = inspect.signature( fn )
		accepted = set( sig.parameters.keys( ) )
		return { k: v for k, v in kwargs.items( ) if k in accepted }
	except Exception:
		return kwargs

def _invoke_provider( fetcher: Any, prompt: str, params: dict[ str, Any ] ) -> Any:
	if hasattr( fetcher, "fetch" ) and callable( getattr( fetcher, "fetch" ) ):
		fn = getattr( fetcher, "fetch" )
		safe = _filter_kwargs_for_callable( fn, params )
		try:
			return fn( prompt, **safe )
		except TypeError:
			safe2 = _filter_kwargs_for_callable( fn, { **safe, "query": prompt } )
			return fn( **safe2 )
	
	if hasattr( fetcher, "chat" ) and callable( getattr( fetcher, "chat" ) ):
		fn = getattr( fetcher, "chat" )
		safe = _filter_kwargs_for_callable( fn, params )
		try:
			return fn( prompt, **safe )
		except TypeError:
			safe2 = _filter_kwargs_for_callable( fn, { **safe,
			                                           "prompt": prompt } )
			return fn( **safe2 )
	
	if hasattr( fetcher, "invoke" ) and callable( getattr( fetcher, "invoke" ) ):
		fn = getattr( fetcher, "invoke" )
		safe = _filter_kwargs_for_callable( fn, params )
		try:
			return fn( prompt, **safe )
		except TypeError:
			safe2 = _filter_kwargs_for_callable( fn, { **safe,
			                                           "input": prompt } )
			return fn( **safe2 )
	
	raise RuntimeError(
		f"Provider '{type( fetcher ).__name__}' does not expose fetch/chat/invoke."
	)

def _render_output( container: Any, result: Any ) -> None:
	if result is None:
		container.info( "No response returned." )
		return
	
	if isinstance( result, list ) and result and isinstance( result[ 0 ], Document ):
		with container.container( ):
			for idx, doc in enumerate( result, start=1 ):
				with st.expander( f"Document {idx}", expanded=False ):
					st.text_area( '', value=(doc.page_content or ""), height=300 )
					if doc.metadata:
						st.json( doc.metadata )
		return
	
	container.text_area( "Response", value=str( result ), height=320 )

def _render_result_metadata( result: Dict[ str, Any ] ) -> None:
	'''
		Purpose:
		--------
		Render common request metadata in a compact, readable format.

		Parameters:
		-----------
		result (Dict[str, Any]):
			Fetcher result dictionary.

		Returns:
		--------
		None

	'''
	if not isinstance( result, dict ) or not result:
		return
	
	meta = {
			'Mode': result.get( 'mode', '' ),
			'URL': result.get( 'url', '' ),
			'Params': result.get( 'params', { } ),
	}
	
	st.markdown( '#### Request Metadata' )
	st.json( meta )

def _render_summary_kv( title: str, data: Dict[ str, Any ] ) -> None:
	'''
		Purpose:
		--------
		Render a dictionary of scalar values as a two-column summary table.

		Parameters:
		-----------
		title (str):
			Section title.

		data (Dict[str, Any]):
			Scalar dictionary to render.

		Returns:
		--------
		None

	'''
	if not isinstance( data, dict ) or not data:
		return
	
	rows: List[ Dict[ str, Any ] ] = [ ]
	
	for key, value in data.items( ):
		if isinstance( value, (str, int, float, bool) ) or value is None:
			rows.append(
				{
						'Field': key,
						'Value': value,
				}
			)
	
	if rows:
		st.markdown( title )
		st.dataframe(
			pd.DataFrame( rows ),
			use_container_width=True,
			hide_index=True )

def _render_rows_table( title: str, rows: List[ Dict[ str, Any ] ] ) -> None:
	'''
		Purpose:
		--------
		Render a list of dictionaries as a dataframe.

		Parameters:
		-----------
		title (str):
			Section title.

		rows (List[Dict[str, Any]]):
			Rows to render.

		Returns:
		--------
		None

	'''
	if not isinstance( rows, list ) or not rows:
		st.info( 'No rows returned.' )
		return
	
	df_rows = pd.DataFrame( rows )
	
	if df_rows.empty:
		st.info( 'No rows returned.' )
		return
	
	st.markdown( title )
	st.dataframe(
		df_rows,
		use_container_width=True,
		hide_index=True,
		height=320 )

def _render_xml_preview( title: str, xml_text: str, max_chars: int = 12000 ) -> None:
	'''
		Purpose:
		--------
		Render XML content in a readable preview area.

		Parameters:
		-----------
		title (str):
			Section title.

		xml_text (str):
			XML payload text.

		max_chars (int):
			Maximum preview length.

		Returns:
		--------
		None

	'''
	if not str( xml_text or '' ).strip( ):
		st.info( 'No XML content returned.' )
		return
	
	st.markdown( title )
	st.code( str( xml_text )[ :int( max_chars ) ], language='xml' )

def _render_html_preview( title: str, html_text: str, max_chars: int = 5000 ) -> None:
	'''
		Purpose:
		--------
		Render a human-friendly HTML summary using title and hyperlinks when possible.

		Parameters:
		-----------
		title (str):
			Section title.

		html_text (str):
			HTML payload text.

		max_chars (int):
			Maximum preview length for fallback output.

		Returns:
		--------
		None

	'''
	text = str( html_text or '' )
	if not text.strip( ):
		st.info( 'No HTML content returned.' )
		return
	
	st.markdown( title )
	
	try:
		soup = BeautifulSoup( text, 'html.parser' )
		
		page_title = ''
		if soup.title and soup.title.string:
			page_title = str( soup.title.string ).strip( )
		
		if page_title:
			st.markdown( f'**Title:** {page_title}' )
		
		links: List[ Dict[ str, str ] ] = [ ]
		for anchor in soup.find_all( 'a', href=True )[ :15 ]:
			label = str( anchor.get_text( ' ', strip=True ) or anchor.get( 'href', '' ) ).strip( )
			href = str( anchor.get( 'href', '' ) ).strip( )
			if label or href:
				links.append(
					{
							'Label': label,
							'Link': href,
					}
				)
		
		if links:
			st.dataframe(
				pd.DataFrame( links ),
				use_container_width=True,
				hide_index=True,
				height=260 )
		else:
			st.code( text[ :int( max_chars ) ], language='html' )
	except Exception:
		st.code( text[ :int( max_chars ) ], language='html' )

def _render_fallback_raw( result: Any ) -> None:
	'''
		Purpose:
		--------
		Render a raw result payload inside a collapsed expander.

		Parameters:
		-----------
		result (Any):
			Result payload to render.

		Returns:
		--------
		None

	'''
	with st.expander( 'Raw Result', expanded=False ):
		if isinstance( result, (dict, list) ):
			st.json( result )
		else:
			st.text_area( 'Output', value=str( result ), height=320 )
			
def _model_selector( key_prefix: str, label: str, options: list[ str ], default_model: str ) -> str:
	base_options = options[ : ]
	if "Custom..." not in base_options:
		base_options.append( "Custom..." )
	
	idx_default = base_options.index( default_model ) if default_model in base_options else 0
	
	selected = st.selectbox(
		label=label,
		options=base_options,
		index=idx_default,
		key=f"{key_prefix}_model_select",
	)
	
	if selected == "Custom...":
		return st.text_input(
			"Custom Model",
			value=default_model,
			key=f"{key_prefix}_model_custom",
		)
	
	return selected

# -------- Text Utilities ---------------------

def render_google_results( response ) -> str:
	try:
		data = response.json( )
	except Exception:
		return "Failed to decode response."
	
	items = data.get( "items", [ ] )
	if not items:
		return "No results returned."
	
	lines = [ ]
	for idx, item in enumerate( items, start=1 ):
		title = item.get( "title", "Untitled" )
		snippet = item.get( "snippet", "" )
		link = item.get( "link", "" )
		
		lines.append( f"{idx}. {title}" )
		if snippet:
			lines.append( snippet )
		if link:
			lines.append( link )
		lines.append( "" )
	
	return "\n".join( lines )

def style_subheaders( ) -> None:
	"""
	
		Purpose:
		_________
		Sets the style of subheaders in the main UI
		
	"""
	st.markdown(
		"""
		<style>
		div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
			color: rgb(0, 120, 252) !important;
		}
		</style>
		""",
		unsafe_allow_html=True, )

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

def normalize_text( text: str ) -> str:
	"""
		
		Purpose
		-------
		Normalize text by:
			• Converting to lowercase
			• Removing punctuation except sentence delimiters (. ! ?)
			• Ensuring clean sentence boundary spacing
			• Collapsing whitespace
	
		Parameters
		----------
		text: str
	
		Returns
		-------
		str
		
	"""
	if not text:
		return ""
	
	# Lowercase
	text = text.lower( )
	
	# Remove punctuation except . ! ?
	text = re.sub( r"[^\w\s\.\!\?]", '', text )
	
	# Ensure single space after sentence delimiters
	text = re.sub( r"([.!?])\s*", r"\1 ", text )
	
	# Normalize whitespace
	text = re.sub( r"\s+", " ", text ).strip( )
	
	return text

def chunk_text( text: str, max_tokens: int = 400 ) -> list[ str ]:
	"""
		
		Purpose
		-------
		Segment normalized text into chunks by:
			1. Sentence boundaries
			2. Fallback to token windowing if needed
	
		Parameters
		----------
		text: str
		max_tokens: int
	
		Returns
		-------
		list[str]
		
	"""
	if not text:
		return [ ]
	
	# Sentence-based segmentation
	sentences = re.split( r"(?<=[.!?])\s+", text )
	sentences = [ s.strip( ) for s in sentences if s.strip( ) ]
	
	if len( sentences ) > 1:
		return sentences
	
	# Fallback: token window segmentation
	words = text.split( )
	chunks = [ ]
	current_chunk = [ ]
	token_count = 0
	
	for word in words:
		current_chunk.append( word )
		token_count += 1
		
		if token_count >= max_tokens:
			chunks.append( " ".join( current_chunk ) )
			current_chunk = [ ]
			token_count = 0
	
	if current_chunk:
		chunks.append( " ".join( current_chunk ) )
	
	return chunks

def cosine_similarity( a: np.ndarray, b: np.ndarray ) -> float:
	denom = np.linalg.norm( a ) * np.linalg.norm( b )
	return float( np.dot( a, b ) / denom ) if denom else 0.0

def sanitize_markdown( text: str ) -> str:
	"""
	
		Purpose:
		_________
		
		
	"""
	# Remove bold markers
	text = re.sub( r"\*\*(.*?)\*\*", r"\1", text )
	# Optional: remove italics
	text = re.sub( r"\*(.*?)\*", r"\1", text )
	return text

def normalize( obj ):
	if obj is None or isinstance( obj, (str, int, float, bool) ):
		return obj
	
	if isinstance( obj, dict ):
		return { k: normalize( v ) for k, v in obj.items( ) }
	
	if isinstance( obj, (list, tuple, set) ):
		return [ normalize( v ) for v in obj ]
	if hasattr( obj, 'model_dump' ):
		try:
			return obj.model_dump( )
		except Exception:
			return str( obj )
	return str( obj )

def metric_with_tooltip( label: str, value: str, tooltip: str ):
	"""
		Renders a metric with a hover tooltip using a two-column layout.
		Left column = the metric itself
		Right column = hoverable ℹ️ icon
	"""
	col_metric, col_info = st.columns( [ 0.5, 0.5 ] )
	
	with col_metric:
		st.metric( label, value )
	
	with col_info:
		if label not in [ 'Characters', 'Tokens', 'Unique Tokens', 'Avg Length' ]:
			st.markdown(
				f"""
	            <span style="
	                cursor: help;
	                font-size: 0.85rem;
	                color:#888;
	                vertical-align: super;
	            " title="{tooltip}">ℹ️ </span>
	            """,
				unsafe_allow_html=True,
			)
			
# ----------  Database Utilities --------------

def initialize_database( ) -> None:
	"""
		Purpose:
		--------
		Ensure required SQLite tables exist and that the Prompts table contains the
		columns required by the prompt utilities and Prompt Engineering mode.

		Parameters:
		-----------
		None

		Returns:
		--------
		None
	"""
	Path( 'stores/sqlite' ).mkdir( parents=True, exist_ok=True )
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute(
			"""
            CREATE TABLE IF NOT EXISTS chat_history
            (
                id
                INTEGER
                PRIMARY
                KEY
                AUTOINCREMENT,
                role
                TEXT,
                content
                TEXT
            )
			"""
		)
		
		conn.execute(
			"""
            CREATE TABLE IF NOT EXISTS embeddings
            (
                id
                INTEGER
                PRIMARY
                KEY
                AUTOINCREMENT,
                chunk
                TEXT,
                vector
                BLOB
            )
			"""
		)
		conn.execute(
			"""
            CREATE TABLE IF NOT EXISTS Prompts
            (
                PromptsId INTEGER NOT  NULL PRIMARY KEY AUTOINCREMENT,
                Caption  TEXT,
                Name TEXT,
                Text TEXT,
                Version TEXT,
                ID TEXT
            )
			"""
		)
		
		prompt_columns = [ row[ 1 ] for row in
		                   conn.execute( 'PRAGMA table_info("Prompts");' ).fetchall( ) ]
		
		if 'Caption' not in prompt_columns:
			conn.execute( 'ALTER TABLE "Prompts" ADD COLUMN "Caption" TEXT;' )
		
		conn.commit( )

def create_connection( ) -> Connection:
	return sqlite3.connect( cfg.DB_PATH )

def list_tables( ) -> List[ str ]:
	with create_connection( ) as conn:
		_query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
		rows = conn.execute( _query ).fetchall( )
		return [ r[ 0 ] for r in rows ]

def create_schema( table: str ) -> List[ Tuple ]:
	with create_connection( ) as conn:
		return conn.execute( f'PRAGMA table_info("{table}");' ).fetchall( )

def read_table( table: str, limit: int = None, offset: int = 0 ) -> DataFrame:
	"""
	
		Purpose:
		--------
		Read a SQLite table into a pandas DataFrame using a normalized scalar-only path.
	
		Parameters:
		-----------
		table : str
			Table name.
		limit : int = None
			Optional row limit.
		offset : int = 0
			Optional row offset.
	
		Returns:
		--------
		DataFrame
			DataFrame of plain Python scalar values.
	
	"""
	if not table:
		return DataFrame( )
	
	query = f'SELECT * FROM "{table}"'
	if limit:
		query += f' LIMIT {int( limit )} OFFSET {int( offset )}'
	
	with create_connection( ) as conn:
		cur = conn.cursor( )
		cur.execute( query )
		
		raw_columns = [ d[ 0 ] for d in (cur.description or [ ]) ]
		rows = cur.fetchall( )
	
	seen: Dict[ str, int ] = { }
	columns: List[ str ] = [ ]
	
	for col in raw_columns:
		name = str( col )
		if name not in seen:
			seen[ name ] = 0
			columns.append( name )
		else:
			seen[ name ] += 1
			columns.append( f'{name}_{seen[ name ]}' )
	
	def _scalarize( value: Any ) -> Any:
		if value is None or isinstance( value, (str, int, float, bool) ):
			return value
		
		if isinstance( value, bytes ):
			try:
				return value.decode( 'utf-8' )
			except Exception:
				return value.hex( )
		
		if isinstance( value, (list, tuple, set, dict) ):
			try:
				return str( normalize( value ) )
			except Exception:
				return str( value )
		
		if hasattr( value, 'model_dump' ):
			try:
				return str( value.model_dump( ) )
			except Exception:
				return str( value )
		
		return str( value )
	
	normalized_rows: List[ Dict[ str, Any ] ] = [ ]
	for row in rows:
		record: Dict[ str, Any ] = { }
		for idx, col in enumerate( columns ):
			record[ col ] = _scalarize( row[ idx ] )
		normalized_rows.append( record )
	
	return DataFrame( normalized_rows, columns=columns )

def render_table( df: DataFrame ) -> None:
	"""
	
		Purpose:
		--------
		Render a DataFrame safely in Streamlit. Use the normal interactive dataframe
		first, and fall back to HTML rendering if Streamlit/PyArrow serialization fails.
	
		Parameters:
		-----------
		df : DataFrame
			The DataFrame to render.
	
		Returns:
		--------
		None
	
	"""
	if df is None:
		st.info( 'No data available.' )
		return
	
	try:
		st.data_editor( df, use_container_width=True )
		return
	except Exception:
		pass
	
	fallback_df = df.copy( )
	fallback_df = fallback_df.where( pd.notnull( fallback_df ), '' )
	
	for col in fallback_df.columns:
		fallback_df[ col ] = fallback_df[ col ].map(
			lambda x: x if isinstance( x, (str, int, float, bool) ) or x == '' else str( x ) )
	
	st.markdown( fallback_df.to_html( index=False, escape=True ), unsafe_allow_html=True )

def make_display_safe( df: DataFrame ) -> DataFrame:
	display_df = df.copy( )
	
	for col in display_df.columns:
		display_df[ col ] = display_df[ col ].map(
			lambda x: '' if x is None else str( x )
		)
	
	return display_df

def drop_table( table: str ) -> None:
	"""
		Purpose:
		--------
		Safely drop a table if it exists.
	
		Parameters:
		-----------
		table : str
			Table name.
	"""
	if not table:
		return
	
	with create_connection( ) as conn:
		conn.execute( f'DROP TABLE IF EXISTS "{table}";' )
		conn.commit( )

def create_index( table: str, column: str ) -> None:
	"""
		Purpose:
		--------
		Create a safe SQLite index on a specified table column.
	
		Handles:
			- Spaces in column names
			- Special characters
			- Reserved words
			- Duplicate index names
			- Validation against actual table schema
	
		Parameters:
		-----------
		table : str
			Table name.
		column : str
			Column name to index.
	"""
	if not table or not column:
		return
	
	# ------------------------------------------------------------------
	# Validate table exists
	# ------------------------------------------------------------------
	tables = list_tables( )
	if table not in tables:
		raise ValueError( 'Invalid table name.' )
	
	# ------------------------------------------------------------------
	# Validate column exists
	# ------------------------------------------------------------------
	schema = create_schema( table )
	valid_columns = [ col[ 1 ] for col in schema ]
	
	if column not in valid_columns:
		raise ValueError( 'Invalid column name.' )
	
	# ------------------------------------------------------------------
	# Sanitize index name (identifier only)
	# ------------------------------------------------------------------
	safe_index_name = re.sub( r"[^0-9a-zA-Z_]+", "_", f"idx_{table}_{column}" )
	
	# ------------------------------------------------------------------
	# Create index safely (quote identifiers)
	# ------------------------------------------------------------------
	sql = f'CREATE INDEX IF NOT EXISTS "{safe_index_name}" ON "{table}"("{column}");'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def apply_filters( df: DataFrame) -> DataFrame:
	st.subheader( 'Advanced Filters' )
	conditions = [ ]
	col1, col2, col3 = st.columns( 3 )
	column = col1.selectbox( 'Column', df.columns )
	operator = col2.selectbox( 'Operator', [ '=', '!=', '>', '<', '>=', '<=', 'contains' ] )
	value = col3.text_input( 'Value' )
	if value:
		if operator == '=':
			df = df[ df[ column ] == value ]
		elif operator == '!=':
			df = df[ df[ column ] != value ]
		elif operator == '>':
			df = df[ df[ column ].astype( float ) > float( value ) ]
		elif operator == '<':
			df = df[ df[ column ].astype( float ) < float( value ) ]
		elif operator == '>=':
			df = df[ df[ column ].astype( float ) >= float( value ) ]
		elif operator == '<=':
			df = df[ df[ column ].astype( float ) <= float( value ) ]
		elif operator == 'contains':
			df = df[ df[ column ].astype( str ).str.contains( value ) ]
	
	return df

def create_aggregation( df: DataFrame ):
	st.subheader( 'Aggregation Engine' )
	
	numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
	
	if not numeric_cols:
		st.info( 'No numeric columns available.' )
		return
	
	col = st.selectbox( 'Column', numeric_cols )
	agg = st.selectbox( 'Aggregation', [ 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'MEDIAN' ] )
	
	if agg == 'COUNT':
		result = df[ col ].count( )
	elif agg == 'SUM':
		result = df[ col ].sum( )
	elif agg == 'AVG':
		result = df[ col ].mean( )
	elif agg == 'MIN':
		result = df[ col ].min( )
	elif agg == 'MAX':
		result = df[ col ].max( )
	elif agg == 'MEDIAN':
		result = df[ col ].median( )
	
	st.metric( 'Result', result )

def create_visualization( df: DataFrame ) -> None:
	"""
	
		Purpose:
		--------
		Render data visualizations without passing pandas objects directly into
		Plotly/Narwhals.
		
		Parameters:
		-----------
		df : DataFrame
			The input DataFrame.
		
		Returns:
		--------
		None
		
	"""
	st.subheader( 'Visualization Engine' )
	
	if df is None or df.empty:
		st.info( 'No data available.' )
		return
	
	df_plot = df.copy( )
	
	for col in df_plot.columns:
		if df_plot[ col ].dtype == object:
			df_plot[ col ] = df_plot[ col ].map(
				lambda x: '' if x is None else str( x )
			)
	
	numeric_cols: List[ str ] = [ ]
	for col in df_plot.columns:
		series_num = pd.to_numeric( df_plot[ col ], errors='coerce' )
		if series_num.notna( ).any( ):
			numeric_cols.append( col )
	
	categorical_cols: List[ str ] = [ col for col in df_plot.columns if col not in numeric_cols ]
	
	chart = st.selectbox(
		'Chart Type',
		[ 'Histogram', 'Bar', 'Line', 'Scatter', 'Box', 'Pie', 'Correlation' ] )
	
	if chart == 'Histogram':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		col = st.selectbox( 'Column', numeric_cols )
		values = pd.to_numeric( df_plot[ col ], errors='coerce' ).dropna( ).tolist( )
		
		fig = px.Figure( data=[ px.Histogram( x=values ) ] )
		fig.update_layout( xaxis_title=col, yaxis_title='Count' )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Bar':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		x = st.selectbox( 'X', df_plot.columns )
		y = st.selectbox( 'Y', numeric_cols )
		
		x_values = df_plot[ x ].astype( str ).tolist( )
		y_values = pd.to_numeric( df_plot[ y ], errors='coerce' ).fillna( 0 ).tolist( )
		
		fig = px.Figure( data=[ px.Bar( x=x_values, y=y_values ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Line':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		x = st.selectbox( 'X', df_plot.columns )
		y = st.selectbox( 'Y', numeric_cols )
		
		x_values = df_plot[ x ].astype( str ).tolist( )
		y_values = pd.to_numeric( df_plot[ y ], errors='coerce' ).fillna( 0 ).tolist( )
		
		fig = px.Figure( data=[ px.Scatter( x=x_values, y=y_values, mode='lines' ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Scatter':
		if len( numeric_cols ) < 2:
			st.info( 'At least two numeric columns are required.' )
			return
		
		x = st.selectbox( 'X', numeric_cols, key='viz_scatter_x' )
		y = st.selectbox( 'Y', numeric_cols, key='viz_scatter_y' )
		
		x_series = pd.to_numeric( df_plot[ x ], errors='coerce' )
		y_series = pd.to_numeric( df_plot[ y ], errors='coerce' )
		mask = x_series.notna( ) & y_series.notna( )
		
		x_values = x_series[ mask ].tolist( )
		y_values = y_series[ mask ].tolist( )
		
		fig = px.Figure( data=[ px.Scatter( x=x_values, y=y_values, mode='markers' ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Box':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		col = st.selectbox( 'Column', numeric_cols, key='viz_box_col' )
		values = pd.to_numeric( df_plot[ col ], errors='coerce' ).dropna( ).tolist( )
		
		fig = px.Figure( data=[ px.Box( y=values, name=col ) ] )
		fig.update_layout( yaxis_title=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Pie':
		if not categorical_cols:
			st.info( 'No categorical columns available.' )
			return
		
		col = st.selectbox( 'Category Column', categorical_cols )
		counts = df_plot[ col ].astype( str ).value_counts( )
		
		fig = px.Figure(
			data=[ px.Pie( labels=counts.index.tolist( ), values=counts.values.tolist( ) ) ] )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Correlation':
		if len( numeric_cols ) < 2:
			st.info( 'At least two numeric columns are required.' )
			return
		
		corr_df = DataFrame( )
		for col in numeric_cols:
			corr_df[ col ] = pd.to_numeric( df_plot[ col ], errors='coerce' )
		
		corr = corr_df.corr( )
		
		fig = px.Figure(
			data=[ px.Heatmap(
				z=corr.values.tolist( ),
				x=corr.columns.tolist( ),
				y=corr.index.tolist( ) ) ] )
		st.plotly_chart( fig, use_container_width=True )

def convert_dataframe( table_name: str, df: DataFrame ):
	columns = [ ]
	for col in df.columns:
		sql_type = get_sqlite_type( df[ col ].dtype )
		safe_col = col.replace( ' ', '_' )
		columns.append( f'{safe_col} {sql_type}' )
	
	create_stmt = f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join( columns )});'
	
	with create_connection( ) as conn:
		conn.execute( create_stmt )
		conn.commit( )

def insert_data( table_name: str, df: DataFrame ):
	df = df.copy( )
	df.columns = [ c.replace( ' ', '_' ) for c in df.columns ]
	
	placeholders = ', '.join( [ '?' ] * len( df.columns ) )
	stmt = f'INSERT INTO {table_name} VALUES ({placeholders});'
	
	with create_connection( ) as conn:
		conn.executemany( stmt, df.values.tolist( ) )
		conn.commit( )

def get_sqlite_type( dtype ) -> str:
	"""
		Purpose:
		--------
		Map a pandas dtype to an appropriate SQLite column type.
	
		Parameters:
		-----------
		dtype : pandas dtype
			The dtype of a pandas Series.
	
		Returns:
		--------
		str
			SQLite column type.
	"""
	dtype_str = str( dtype ).lower( )
	
	# ------------------------------------------------------------------
	# Integer Types (including nullable Int64)
	# ------------------------------------------------------------------
	if 'int' in dtype_str:
		return 'INTEGER'
	
	# ------------------------------------------------------------------
	# Float Types
	# ------------------------------------------------------------------
	if 'float' in dtype_str:
		return 'REAL'
	
	# ------------------------------------------------------------------
	# Boolean
	# ------------------------------------------------------------------
	if 'bool' in dtype_str:
		return 'INTEGER'
	
	# ------------------------------------------------------------------
	# Datetime
	# ------------------------------------------------------------------
	if 'datetime' in dtype_str:
		return 'TEXT'
	
	# ------------------------------------------------------------------
	# Categorical
	# ------------------------------------------------------------------
	if 'category' in dtype_str:
		return 'TEXT'
	
	# ------------------------------------------------------------------
	# Default fallback
	# ------------------------------------------------------------------
	return 'TEXT'

def create_custom_table( table_name: str, columns: list ) -> None:
	"""
		Purpose:
		--------
		Create a custom SQLite table from column definitions.
	
		Parameters:
		-----------
		table_name : str
			Name of table.
	
		columns : list of dict
			[
				{
					"name": str,
					"type": str,
					"not_null": bool,
					"primary_key": bool,
					"auto_increment": bool
				}
			]
	"""
	if not table_name:
		raise ValueError( 'Table name required.' )
	
	# Validate identifier
	if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", table_name ):
		raise ValueError( 'Invalid table name.' )
	
	col_defs = [ ]
	
	for col in columns:
		col_name = col[ 'name' ]
		col_type = col[ 'type' ].upper( )
		
		if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", col_name ):
			raise ValueError( f"Invalid column name: {col_name}" )
		
		definition = f'"{col_name}" {col_type}'
		
		if col[ 'primary_key' ]:
			definition += ' PRIMARY KEY'
			if col[ 'auto_increment' ] and col_type == 'INTEGER':
				definition += ' AUTOINCREMENT'
		
		if col[ "not_null" ]:
			definition += " NOT NULL"
		
		col_defs.append( definition )
	
	sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join( col_defs )});'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def is_safe_query( query: str ) -> bool:
	"""
	
		Purpose:
		--------
		Determine whether a SQL query is read-only and safe to execute.
	
		Allows:
			SELECT
			WITH (CTE returning SELECT)
			EXPLAIN SELECT
			PRAGMA (read-only)
	
		Blocks:
			INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH,
			DETACH, VACUUM, REPLACE, TRIGGER, and multiple statements.
			
	"""
	if not query or not isinstance( query, str ):
		return False
	
	q = query.strip( ).lower( )
	
	# ------------------------------------------------------------------
	# Block multiple statements
	# ------------------------------------------------------------------
	if ';' in q[ :-1 ]:
		return False
	
	# ------------------------------------------------------------------
	# Remove SQL comments
	# ------------------------------------------------------------------
	q = re.sub( r"--.*?$", "", q, flags=re.MULTILINE )
	q = re.sub( r"/\*.*?\*/", "", q, flags=re.DOTALL )
	q = q.strip( )
	
	# ------------------------------------------------------------------
	# Allowed starting keywords
	# ------------------------------------------------------------------
	allowed_starts = ('select', 'with', 'explain', 'pragma')
	if not q.startswith( allowed_starts ):
		return False
	
	# ------------------------------------------------------------------
	# Block dangerous keywords anywhere
	# ------------------------------------------------------------------
	blocked_keywords = ('insert ', 'update ', 'delete ', 'drop ', 'alter ',
	                    'create ', 'attach ', 'detach ', 'vacuum ', 'replace ', 'trigger ')
	
	for keyword in blocked_keywords:
		if keyword in q:
			return False
	
	return True

def create_identifier( name: str ) -> str:
	"""
	
		Purpose:
		--------
		Sanitize a string into a safe SQLite identifier.
	
		- Replaces invalid characters with underscores
		- Ensures it starts with a letter or underscore
		- Prevents empty names
		
	"""
	if not name or not isinstance( name, str ):
		raise ValueError( 'Invalid Identifier.' )
	
	safe = re.sub( r'[^0-9a-zA-Z_]', '_', name.strip( ) )
	if not re.match( r'^[A-Za-z_]', safe ):
		safe = f'_{safe}'
	
	if not safe:
		raise ValueError( 'Invalid identifier after sanitization.' )
	
	return safe

def get_indexes( table: str ):
	with create_connection( ) as conn:
		rows = conn.execute( f'PRAGMA index_list("{table}");' ).fetchall( )
		return rows

def add_column( table: str, column: str, col_type: str ):
	column = create_identifier( column )
	col_type = col_type.upper( )
	
	with create_connection( ) as conn:
		conn.execute(
			f'ALTER TABLE "{table}" ADD COLUMN "{column}" {col_type};' )
		conn.commit( )

def rename_column( table_name: str, old_name: str, new_name: str ) -> None:
	"""
	
		Purpose:
		--------
		Rename a column within an existing SQLite table. Attempts native ALTER TABLE rename
		first; if it fails, falls back to a schema-safe rebuild preserving column order, data,
		and indexes.

		Parameters:
		-----------
		table_name : str
			Table containing the column.

		old_name : str
			Existing column name.

		new_name : str
			New column name.

		Returns:
		--------
		None
		
	"""
	if not table_name or not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute(
				f'ALTER TABLE "{table_name}" RENAME COLUMN "{old_name}" TO "{new_name}";'
			)
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(table_name,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(table_name,)
		).fetchall( )
		
		schema = conn.execute( f'PRAGMA table_info("{table_name}");' ).fetchall( )
		cols = [ r[ 1 ] for r in schema ]
		if old_name not in cols:
			raise ValueError( "Column not found." )
		
		mapped_cols = [ (new_name if c == old_name else c) for c in cols ]
		
		temp_table = f"{table_name}__rebuild_temp"
		
		col_defs: List[ str ] = [ ]
		pk_cols = [ r for r in schema if int( r[ 5 ] or 0 ) > 0 ]
		single_pk = len( pk_cols ) == 1
		
		for row in schema:
			col_name = row[ 1 ]
			col_type = row[ 2 ] or ''
			not_null = int( row[ 3 ] or 0 )
			default_value = row[ 4 ]
			pk = int( row[ 5 ] or 0 )
			
			out_name = new_name if col_name == old_name else col_name
			col_def = f'"{out_name}" {col_type}'.strip( )
			
			if not_null:
				col_def += ' NOT NULL'
			
			if default_value is not None:
				col_def += f' DEFAULT {default_value}'
			
			if single_pk and pk == 1:
				col_def += ' PRIMARY KEY'
			
			col_defs.append( col_def )
		
		new_create_sql = f'CREATE TABLE "{temp_table}" ({", ".join( col_defs )});'
		
		old_select = ", ".join( [ f'"{c}"' for c in cols ] )
		new_insert = ", ".join( [ f'"{c}"' for c in mapped_cols ] )
		
		conn.execute( "BEGIN" )
		conn.execute( new_create_sql )
		conn.execute(
			f'INSERT INTO "{temp_table}" ({new_insert}) SELECT {old_select} FROM "{table_name}";'
		)
		
		conn.execute( f'DROP TABLE "{table_name}";' )
		conn.execute( f'ALTER TABLE "{temp_table}" RENAME TO "{table_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'"{old_name}"', f'"{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )

def create_profile_table( table: str ):
	df = read_table( table )
	profile_rows = [ ]
	total_rows = len( df )
	for col in df.columns:
		series = df[ col ]
		null_count = series.isna( ).sum( )
		distinct_count = series.nunique( dropna=True )
		row = \
			{
					'column': col, 'dtype': str( series.dtype ),
					'null_%': round( (null_count / total_rows) * 100, 2 ) if total_rows else 0,
					'distinct_%': round( (
							                     distinct_count / total_rows) * 100, 2 ) if total_rows else 0,
			}
		
		if pd.api.types.is_numeric_dtype( series ):
			row[ 'min' ] = series.min( )
			row[ 'max' ] = series.max( )
			row[ 'mean' ] = series.mean( )
		else:
			row[ 'min' ] = None
			row[ 'max' ] = None
			row[ 'mean' ] = None
		
		profile_rows.append( row )
	
	return DataFrame( profile_rows )

def drop_column( table: str, column: str ):
	if not table or not column:
		raise ValueError( 'Table and column required.' )
	
	with create_connection( ) as conn:
		# ------------------------------------------------------------
		# Fetch original CREATE TABLE statement
		# ------------------------------------------------------------
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(table,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( 'Table definition not found.' )
		
		create_sql = row[ 0 ]
		
		# ------------------------------------------------------------
		# Extract column definitions
		# ------------------------------------------------------------
		open_paren = create_sql.find( "(" )
		close_paren = create_sql.rfind( ")" )
		
		if open_paren == -1 or close_paren == -1:
			raise ValueError( "Malformed CREATE TABLE statement." )
		
		inner = create_sql[ open_paren + 1: close_paren ]
		
		column_defs = [ c.strip( ) for c in inner.split( "," ) ]
		
		# Remove target column
		new_defs = [ ]
		for col_def in column_defs:
			col_name = col_def.split( )[ 0 ].strip( '"' )
			if col_name != column:
				new_defs.append( col_def )
		
		if len( new_defs ) == len( column_defs ):
			raise ValueError( "Column not found." )
		
		# ------------------------------------------------------------
		# Build new CREATE TABLE statement
		# ------------------------------------------------------------
		temp_table = f"{table}_rebuild_temp"
		
		new_create_sql = (
				f'CREATE TABLE "{temp_table}" ('
				+ ", ".join( new_defs )
				+ ");"
		)
		
		# ------------------------------------------------------------
		# Begin transaction
		# ------------------------------------------------------------
		conn.execute( "BEGIN" )
		
		conn.execute( new_create_sql )
		
		remaining_cols = [
				c.split( )[ 0 ].strip( '"' )
				for c in new_defs
		]
		
		col_list = ", ".join( [ f'"{c}"' for c in remaining_cols ] )
		
		conn.execute(
			f'INSERT INTO "{temp_table}" ({col_list}) '
			f'SELECT {col_list} FROM "{table}";'
		)
		
		# Preserve indexes
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(table,)
		).fetchall( )
		
		conn.execute( f'DROP TABLE "{table}";' )
		conn.execute(
			f'ALTER TABLE "{temp_table}" RENAME TO "{table}";'
		)
		
		# Recreate indexes
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if column not in idx_sql:
				conn.execute( idx_sql )
		
		conn.commit( )

def rename_table( old_name: str, new_name: str ) -> None:
	"""
	
		Purpose:
		--------
		Rename an existing SQLite table. Attempts native ALTER TABLE rename first; if it fails,
		falls back to a schema-safe rebuild using the original CREATE TABLE statement and
		preserves indexes.

		Parameters:
		-----------
		old_name : str
			Existing table name.

		new_name : str
			New table name.

		Returns:
		--------
		None
		
	"""
	if not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute( f'ALTER TABLE "{old_name}" RENAME TO "{new_name}";' )
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(old_name,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(old_name,)
		).fetchall( )
		
		open_paren = create_sql.find( "(" )
		if open_paren == -1:
			raise ValueError( "Malformed CREATE TABLE statement." )
		
		temp_name = f"{new_name}__rebuild_temp"
		
		conn.execute( "BEGIN" )
		conn.execute( f'CREATE TABLE "{temp_name}" {create_sql[ open_paren: ]}' )
		
		cols = [ r[ 1 ] for r in conn.execute( f'PRAGMA table_info("{old_name}");' ).fetchall( ) ]
		col_list = ", ".join( [ f'"{c}"' for c in cols ] )
		
		conn.execute(
			f'INSERT INTO "{temp_name}" ({col_list}) SELECT {col_list} FROM "{old_name}";'
		)
		
		conn.execute( f'DROP TABLE "{old_name}";' )
		conn.execute( f'ALTER TABLE "{temp_name}" RENAME TO "{new_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'ON "{old_name}"', f'ON "{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )

def clear_if_active( loader_name: str ) -> None:
	if st.session_state.active_loader == loader_name:
		st.session_state.documents = None
		st.session_state.active_loader = None
		st.session_state.tokens = None
		st.session_state.vocabulary = None
		st.session_state.token_counts = None
		st.session_state.chunks = None
		st.session_state.chunk_modes = None
		st.session_state.chunked_documents = None
		st.session_state.embeddings = None
		st.session_state.active_table = None
		st.session_state.df_frequency = None
		st.session_state.df_tables = None
		st.session_state.df_schema = None
		st.session_state.df_preview = None
		st.session_state.df_count = None
		st.session_state.df_chunks = None
		st.session_state.lines = None

# -------- Expander Utilities

def _promote_loader_documents( documents: List[ Document ] | None, active_loader: str ) -> int:
	'''
		Purpose:
		--------
		Promote loaded LangChain documents into the shared Loading-mode session state
		using one consistent contract across all loader expanders.

		Parameters:
		-----------
		documents (List[Document] | None):
			LangChain documents returned by a loader.

		active_loader (str):
			Canonical loader name to store in session state.

		Returns:
		--------
		int:
			Count of promoted documents.

	'''
	docs: List[ Document ] = list( documents or [ ] )
	
	st.session_state.documents = docs
	st.session_state.raw_documents = list( docs )
	st.session_state.raw_text = '\n\n'.join(
		doc.page_content for doc in docs
		if hasattr( doc, 'page_content' )
		and isinstance( doc.page_content, str )
		and doc.page_content.strip( )
	)
	st.session_state.processed_text = ''
	st.session_state.tokens = None
	st.session_state.vocabulary = None
	st.session_state.token_counts = None
	st.session_state.active_loader = active_loader
	return len( docs )

def _clear_loader_documents( loader_name: str ) -> int:
	'''
		Purpose:
		--------
		Clear the active loader state and rebuild shared text/cache state consistently.

		Parameters:
		-----------
		loader_name (str):
			Canonical loader name.

		Returns:
		--------
		int:
			Remaining document count after the clear operation.

	'''
	clear_if_active( loader_name )
	st.session_state.raw_text = _rebuild_raw_text_from_documents( ) or ''
	st.session_state.processed_text = ''
	st.session_state.tokens = None
	st.session_state.vocabulary = None
	st.session_state.token_counts = None
	
	remaining = st.session_state.get( 'documents' ) or [ ]
	return len( remaining )

# =========================================================================
# SESSION-STATE INITIALIZATION
# =========================================================================
for key, default in cfg.SESSION_STATE_DEFAULTS.items( ):
	if key not in st.session_state:
		st.session_state[ key ] = default

for corpus in cfg.REQUIRED_CORPORA:
	try:
		nltk.data.find( f'corpora/{corpus}' )
	except LookupError:
		nltk.download( corpus )
		
# =========================================================================
# APP SET-UP
# =========================================================================
st.set_page_config( page_title=cfg.APP_TITLE, layout='wide', page_icon=cfg.FAVICON )
style_subheaders( )
st.logo( cfg.LOGO, size='large' )
col_left, col_center, col_right = st.columns( [ 1, 2, 1 ], vertical_alignment='top' )

# ===========================================================================
# SIDEBAR
# ===========================================================================
modes = list( cfg.MODE_MAP.keys( ) )
with st.sidebar:
	st.divider( )
	
	# ------------- Modes -------------
	st.text( '🕹️ Mode' )
	mode = st.sidebar.radio( label='Mode', options=modes, label_visibility='collapsed' )
	if mode:
		st.session_state[ 'mode' ] = mode
	else:
		st.session_state[ 'mode' ] = 'Loaders'
	
	st.divider( )

	# ------------- API Keys -------------
	st.text( '💻 Configuration' )
	with st.expander( 'API Keys', expanded=False ):
		for attr in dir( cfg ):
			if attr.endswith( '_API_KEY' ) or attr.endswith( '_TOKEN' ):
				current = getattr( cfg, attr, "" ) or ""
				val = st.text_input( attr, value=current, type='password' )
				if val:
					os.environ[ attr ] = val

# =============================================================================
# DOCUMENT LOADING MODE
# =============================================================================
if mode == 'Loading':
	st.subheader( f'📤  Data Loading' )
	st.divider( )
	metrics_container = st.container( )
	tokens = st.session_state.get( 'tokens' )
	
	def render_metrics_panel( ):
		raw_text = st.session_state.get( 'raw_text' )
		processed_text = st.session_state.get( 'processed_text' )
		if isinstance( processed_text, str ) and processed_text.strip( ):
			text = processed_text
		elif isinstance( raw_text, str ) and raw_text.strip( ):
			text = raw_text
		else:
			st.info( 'Load a document to compute metrics.' )
			return
		
		# ----------------------------------------------
		# Tokenization (session-cached)
		# ----------------------------------------------
		if st.session_state.tokens is None:
			try:
				tokens = [ t.lower( ) for t in word_tokenize( text ) if t.isalpha( ) ]
			except LookupError:
				st.error(
					'NLTK resources missing.\n\n'
					'Run:\n'
					'`python -m nltk.downloader punkt stopwords`' )
				return
			
			if not tokens:
				st.warning( 'No valid alphabetic tokens found.' )
				return
			
			st.session_state.tokens = tokens
			st.session_state.vocabulary = set( tokens )
			st.session_state.token_counts = Counter( tokens )
		tokens = st.session_state.tokens
		vocabulary = st.session_state.vocabulary
		counts = st.session_state.token_counts
		
		# ----------------------------------------------
		# Metric calculations (derived only)
		# ----------------------------------------------
		char_count = len( text )
		token_count = len( tokens )
		vocab_size = len( vocabulary )
		hapax_count = sum( 1 for c in counts.values( ) if c == 1 )
		hapax_ratio = hapax_count / vocab_size if vocab_size else 0.0
		avg_word_len = sum( len( t ) for t in tokens ) / token_count
		ttr = vocab_size / token_count
		stopword_ratio = 0.0
		lexical_density = 0.0
		try:
			stop_words = set( stopwords.words( 'english' ) )
			stopword_ratio = sum( 1 for t in tokens if t in stop_words ) / token_count
			lexical_density = 1.0 - stopword_ratio
		except LookupError:
			pass
		
		# -------------------------------
		# Top Tokens
		# -------------------------------
		with st.expander( '🔤 Top Tokens', expanded=False ):
			top_tokens = counts.most_common( 10 )
			df_top = pd.DataFrame( top_tokens, columns=[ 'token', 'count' ] ).set_index( 'token' )
			st.bar_chart( df_top, color='#01438A' )
		
		# -------------------------------
		# Corpus Metrics
		# -------------------------------
		with st.expander( '📊 Corpus Metrics', expanded=False ):
			col1, col2, col3, col4 = st.columns( 4, border=True )
			with col1:
				metric_with_tooltip(
					'Characters',
					f'{char_count:,}',
					'Total number of characters in the selected text.'
				)
			with col2:
				metric_with_tooltip(
					'Tokens',
					f'{token_count:,}',
					'Token Count: total number of tokenized words after cleanup.'
				)
			with col3:
				metric_with_tooltip(
					'Unique Tokens',
					f'{vocab_size:,}',
					'Vocabulary Size: number of distinct word types in the text.'
				)
			with col4:
				metric_with_tooltip(
					'TTR',
					f'{ttr:.3f}',
					'Type–Token Ratio: unique words ÷ total words.'
				)
			
			col5, col6, col7, col8 = st.columns( 4, border=True )
			with col5:
				metric_with_tooltip(
					'Hapax Ratio',
					f'{hapax_ratio:.3f}',
					'Hapax Ratio: proportion of words that occur only once.'
				)
			with col6:
				metric_with_tooltip(
					'Avg Length',
					f'{avg_word_len:.2f}',
					'Average number of characters per token.'
				)
			with col7:
				metric_with_tooltip(
					'Stopword Ratio',
					f'{stopword_ratio:.2%}',
					'Percentage of stopwords in the text.'
				)
			with col8:
				metric_with_tooltip(
					'Lexical Density',
					f'{lexical_density:.2%}',
					'Proportion of content-bearing words.'
				)
		
		# -------------------------------
		# Readability
		# -------------------------------
		with st.expander( '📖 Readability', expanded=False ):
			if TEXTSTAT_AVAILABLE:
				r1, r2, r3, r4 = st.columns( 4, border=True )
				with r1:
					metric_with_tooltip(
						'Flesch Reading Ease',
						f'{textstat.flesch_reading_ease( text ):.1f}',
						'Higher scores indicate easier readability.'
					)
				with r2:
					metric_with_tooltip(
						'Flesch–Kincaid Grade',
						f'{textstat.flesch_kincaid_grade( text ):.1f}',
						'Estimated U.S. grade level required.'
					)
				with r3:
					metric_with_tooltip(
						'Gunning Fog',
						f'{textstat.gunning_fog( text ):.1f}',
						'Readability based on sentence length and complex words.'
					)
				with r4:
					metric_with_tooltip(
						'Coleman–Liau Index',
						f'{textstat.coleman_liau_index( text ):.1f}',
						'Readability based on characters and sentences.'
					)
			else:
				st.caption( 'Install `textstat` to enable readability metrics.' )
	
	# ------------------------------------------------------------------
	# SINGLE metrics
	# ------------------------------------------------------------------
	with metrics_container:
		render_metrics_panel( )
	
	# ------------------------------------------------------------------
	# Left Layout
	# ------------------------------------------------------------------
	left, right = st.columns( [ 1, 1.5 ] )
	with left:
		_loader_msg = st.session_state.pop( '_loader_status', None )
		if isinstance( _loader_msg, str ) and _loader_msg.strip( ):
			st.success( _loader_msg )
		
		def _rebuild_raw_text_from_documents( ) -> str | None:
			docs = st.session_state.get( "documents" ) or [ ]
			if not docs:
				return None
			text = "\n\n".join(
				d.page_content for d in docs
				if hasattr( d, "page_content" ) and isinstance( d.page_content, str ) and d.page_content.strip( )
			)
			return text if text.strip( ) else None
		
		# --------------------------- Text Loader
		with st.expander( label='Text Loader', icon='📄', expanded=False ):
			files = st.file_uploader( 'Upload TXT files', type=[ 'txt' ],
				accept_multiple_files=True, key='txt_upload' )
			
			# ------------------------------------------------------------------
			# Buttons: Load / Clear / Save (same placement + interaction model)
			# ------------------------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_txt = col_load.button( 'Load', key='txt_load' )
			clear_txt = col_clear.button( 'Clear', key='txt_clear' )
			can_save = (st.session_state.get( 'active_loader' ) == 'TextLoader'
			            and isinstance( st.session_state.get( 'raw_text' ), str )
			            and st.session_state.get( 'raw_text' ).strip( ))
			
			if can_save:
				col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
					file_name='text_loader_output.txt', mime='text/plain', key='txt_save' )
			else:
				col_save.button( 'Save', key='txt_save_disabled', disabled=True )
			
			# ------------------------------------------------------------------
			# Clear (unchanged behavior)
			# ------------------------------------------------------------------
			if clear_txt:
				clear_if_active( 'TextLoader' )
				st.info( 'Text Loader state cleared.' )
			
			# ------------------------------------------------------------------
			# Load (unchanged behavior)
			# ------------------------------------------------------------------
			if load_txt and files:
				documents = [ ]
				for f in files:
					text = f.read( ).decode( 'utf-8', errors='ignore' )
					documents.append( Document( page_content=text,
						metadata={
								'source': f.name,
								'loader': 'TextLoader' }, ) )
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = "\n\n".join( d.page_content for d in documents )
				st.session_state.active_loader = "TextLoader"
				st.success( f'Loaded {len( documents )} text document(s).' )
		
		# --------------------------- NLTK Loader
		with st.expander( label='Corpora Loader', icon='📚', expanded=False ):
			import nltk
			from nltk.corpus import (
				brown,
				gutenberg,
				reuters,
				webtext,
				inaugural,
				state_union,
			)
			
			st.markdown( '###### NLTK Corpora' )
			
			corpus_name = st.selectbox( 'Select corpus',
				[ 'Brown', 'Gutenberg', 'Reuters', 'WebText', 'Inaugural', 'State of the Union', ],
				key='nltk_corpus_name', )
			
			file_ids = [ ]
			try:
				if corpus_name == 'Brown':
					file_ids = brown.fileids( )
				elif corpus_name == 'Gutenberg':
					file_ids = gutenberg.fileids( )
				elif corpus_name == 'Reuters':
					file_ids = reuters.fileids( )
				elif corpus_name == 'WebText':
					file_ids = webtext.fileids( )
				elif corpus_name == 'Inaugural':
					file_ids = inaugural.fileids( )
				elif corpus_name == 'State of the Union':
					file_ids = state_union.fileids( )
			except LookupError:
				st.error(
					"NLTK corpus not found. Run:\n\n"
					"python -m nltk.downloader all\n\n"
					"or download individual corpora."
				)
			
			selected_files = st.multiselect( 'Select files (leave empty to load all)',
				options=file_ids, key='nltk_file_ids', )
			
			st.divider( )
			
			st.markdown( '###### Local Corpus' )
			
			local_corpus_dir = st.text_input( 'Local directory', placeholder='path/to/text/files',
				key='nltk_local_dir', )
			
			# ------------------------------------------------------------------
			# Load / Clear / Save controls
			# ------------------------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_nltk = col_load.button( 'Load', key='nltk_load' )
			clear_nltk = col_clear.button( 'Clear', key='nltk_clear' )
			
			_docs = st.session_state.get( 'documents' ) or [ ]
			_nltk_docs = [ d for d in _docs if d.metadata.get( 'loader' ) == 'NLTKLoader' ]
			_nltk_text = "\n\n".join( d.page_content for d in _nltk_docs )
			_export_name = f"nltk_{corpus_name.lower( ).replace( ' ', '_' )}.txt"
			
			col_save.download_button(
				'Save',
				data=_nltk_text,
				file_name=_export_name,
				mime='text/plain',
				disabled=not bool( _nltk_text.strip( ) ),
			)
			
			# ------------------------------------------------------------------
			# Clear
			# ------------------------------------------------------------------
			if clear_nltk and st.session_state.get( 'documents' ):
				st.session_state.documents = [ d for d in st.session_state.documents
				                               if d.metadata.get( 'loader' ) != 'NLTKLoader'
				                               ]
				
				st.session_state.raw_text = (
						"\n\n".join( d.page_content for d in st.session_state.documents )
						if st.session_state.documents else None)
				
				st.session_state.active_loader = None
				
				st.info( 'NLTKLoader documents removed.' )
			
			# ------------------------------------------------------------------
			# Load
			# ------------------------------------------------------------------
			if load_nltk:
				documents = [ ]
				
				# Built-in corpora
				if file_ids:
					files_to_load = selected_files or file_ids
					
					for fid in files_to_load:
						try:
							if corpus_name == 'Brown':
								text = ' '.join( brown.words( fid ) )
							elif corpus_name == 'Gutenberg':
								text = gutenberg.raw( fid )
							elif corpus_name == 'Reuters':
								text = reuters.raw( fid )
							elif corpus_name == 'WebText':
								text = webtext.raw( fid )
							elif corpus_name == 'Inaugural':
								text = inaugural.raw( fid )
							elif corpus_name == 'State of the Union':
								text = state_union.raw( fid )
							
							if text.strip( ):
								documents.append(
									Document(
										page_content=text,
										metadata={
												'loader': 'NLTKLoader',
												'corpus': corpus_name,
												'file_id': fid,
										},
									)
								)
						except Exception:
							continue
				
				# Local corpus
				if local_corpus_dir and os.path.isdir( local_corpus_dir ):
					for fname in os.listdir( local_corpus_dir ):
						path = os.path.join( local_corpus_dir, fname )
						if os.path.isfile( path ) and fname.lower( ).endswith( '.txt' ):
							with open( path, 'r', encoding='utf-8', errors='ignore' ) as f:
								text = f.read( )
							
							if text.strip( ):
								documents.append(
									Document(
										page_content=text,
										metadata={
												'loader': 'NLTKLoader',
												'source': path,
										},
									)
								)
				
				if documents:
					if st.session_state.get( 'documents' ):
						st.session_state.documents.extend( documents )
					else:
						st.session_state.documents = documents
						st.session_state.raw_documents = list( documents )
					
					st.session_state.raw_text = "\n\n".join(
						d.page_content for d in st.session_state.documents
					)
					
					st.session_state.processed_text = None
					st.session_state.active_loader = 'NLTKLoader'
					
					st.success( f'Loaded {len( documents )} document(s) from NLTK.' )
				else:
					st.warning( 'No documents were loaded.' )
		
		# --------------------------- CSV Loader
		with st.expander( label="CSV Loader", icon='📑', expanded=False ):
			csv_file = st.file_uploader( "Upload CSV", type=[ "csv" ],
				key="csv_upload", )
			
			delimiter = st.text_input( "Delimiter", value="\n\n", key="csv_delim", )
			quotechar = st.text_input( "Quote Character", value='"', key="csv_quote", )
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_csv = col_load.button( 'Load', key='csv_load' )
			clear_csv = col_clear.button( 'Clear', key='csv_clear' )
			
			can_save = (
					st.session_state.get( 'active_loader' ) == 'CsvLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button( 'Save', data=st.session_state.get( 'raw_text' ),
					file_name='csv_loader_output.txt', mime='text/plain', key='csv_save', )
			else:
				col_save.button( 'Save', key='csv_save_disabled', disabled=True )
			
			# --------------------------------------------------
			# Clear
			# --------------------------------------------------
			if clear_csv:
				clear_if_active( "CsvLoader" )
				st.session_state.raw_text = _rebuild_raw_text_from_documents( )
				st.session_state[ "_loader_status" ] = "CSV Loader state cleared."
				st.rerun( )
			
			# --------------------------------------------------
			# Load
			# --------------------------------------------------
			if load_csv and csv_file:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, csv_file.name )
					with open( path, "wb" ) as f:
						f.write( csv_file.read( ) )
					
					loader = CsvLoader( )
					documents = loader.load(
						path,
						columns=None,
						delimiter=delimiter,
						quotechar=quotechar,
					) or [ ]
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = "\n\n".join(
					d.page_content for d in documents
					if
					hasattr( d, "page_content" ) and isinstance( d.page_content, str ) and d.page_content.strip( )
				)
				st.session_state.processed_text = None
				st.session_state.active_loader = "CsvLoader"
				
				st.session_state[ "_loader_status" ] = f"Loaded {len( documents )} CSV document(s)."
				st.rerun( )
		
		# -------------------------- XML Loader Expander
		with st.expander( label='XML Loader', icon='🧬', expanded=False ):
			# ------------------------------------------------------------------
			# Session-backed loader instance
			# ------------------------------------------------------------------
			if 'xml_loader' not in st.session_state or st.session_state.xml_loader is None:
				st.session_state.xml_loader = XmlLoader( )
			
			loader = st.session_state.xml_loader
			
			xml_file = st.file_uploader(
				label='Select XML file',
				type=[ 'xml' ],
				accept_multiple_files=False,
				key='xml_file_uploader'
			)
			
			st.subheader( 'Semantic XML Loading (Unstructured)' )
			
			col1, col2 = st.columns( 2 )
			
			with col1:
				chunk_size = st.number_input(
					'Chunk Size',
					min_value=100,
					max_value=5000,
					value=1000,
					step=100
				)
			
			with col2:
				overlap_amount = st.number_input(
					'Chunk Overlap',
					min_value=0,
					max_value=1000,
					value=200,
					step=50
				)
			
			# --------------------------------------------------
			# Semantic Load
			# --------------------------------------------------
			if st.button( 'Load XML (Semantic)', use_container_width=True ):
				if xml_file is None:
					st.warning( 'Please select an XML file.' )
				else:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, xml_file.name )
						with open( path, 'wb' ) as f:
							f.write( xml_file.read( ) )
						
						with st.spinner( 'Loading XML via UnstructuredXMLLoader...' ):
							documents = loader.load( path )
					
					if documents:
						raw_text = '\n\n'.join(
							d.page_content
							for d in documents
							if hasattr( d, 'page_content' )
							and isinstance( d.page_content, str )
							and d.page_content.strip( )
						)
						
						st.session_state.documents = documents
						st.session_state.raw_documents = list( documents )
						st.session_state.raw_text = raw_text
						st.session_state.processed_text = None
						st.session_state.active_loader = 'XmlLoader'
						st.session_state[ 'xml_documents' ] = documents
						st.session_state[ 'xml_tree_loaded' ] = False
						st.session_state[ 'xml_xpath_results' ] = None
						st.session_state[ 'xml_namespaces' ] = None
						st.rerun( )
					else:
						st.warning( 'No extractable text found in XML.' )
			
			# --------------------------------------------------
			# Split Semantic Documents
			# --------------------------------------------------
			if st.button( 'Split Semantic Documents', use_container_width=True ):
				with st.spinner( 'Splitting documents...' ):
					split_docs = loader.split(
						size=int( chunk_size ),
						amount=int( overlap_amount )
					)
				
				if split_docs:
					st.session_state[ 'xml_split_documents' ] = split_docs
					st.success( f'Produced {len( split_docs )} document chunks.' )
			
			# ------------------------------------------------------------------
			# Structured XML Tree Loading
			# ------------------------------------------------------------------
			st.divider( )
			st.subheader( 'Structured XML Tree Loading (XPath)' )
			
			if st.button( 'Load XML Tree', use_container_width=True ):
				if xml_file is None:
					st.warning( 'Please select an XML file.' )
				else:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, xml_file.name )
						with open( path, 'wb' ) as f:
							f.write( xml_file.read( ) )
						
						with st.spinner( 'Parsing XML into ElementTree...' ):
							tree = loader.load_tree( path )
					
					if tree is not None:
						xml_text = etree.tostring(
							tree,
							pretty_print=True,
							encoding='unicode'
						)
						
						st.session_state.raw_text = xml_text
						st.session_state.processed_text = None
						st.session_state.active_loader = 'XmlLoader'
						st.session_state[ 'xml_tree_loaded' ] = True
						st.session_state[ 'xml_namespaces' ] = loader.xml_namespaces
						st.session_state[ 'xml_xpath_results' ] = None
						
						st.success( 'XML tree loaded successfully.' )
					else:
						st.warning( 'Failed to parse XML tree.' )
			
			# ------------------------------------------------------------------
			# XPath Query Interface
			# ------------------------------------------------------------------
			xml_loader = st.session_state.get( 'xml_loader' )
			
			if xml_loader is None:
				st.info( 'No loader initialized.' )
			elif not hasattr( xml_loader, 'xml_root' ):
				st.info( 'XML loader does not support XML tree operations.' )
			elif xml_loader.xml_root is None:
				st.info( 'XML loader initialized but no XML tree loaded.' )
			else:
				st.markdown( '**XPath Query**' )
				
				xpath_expr = st.text_input(
					'XPath Expression',
					value='//*',
					help='Use namespace prefixes if applicable.'
				)
				
				if st.button( 'Run XPath Query', use_container_width=True ):
					with st.spinner( 'Executing XPath...' ):
						elements = xml_loader.get_elements( xpath_expr )
					
					if elements is not None:
						st.session_state[ 'xml_xpath_results' ] = elements
						st.success( f'Returned {len( elements )} elements.' )
				
				if 'xml_xpath_results' in st.session_state and \
						st.session_state[ 'xml_xpath_results' ] is not None:
					preview_count = min(
						10,
						len( st.session_state[ 'xml_xpath_results' ] )
					)
					
					st.caption( f'Previewing first {preview_count} elements' )
					
					for el in st.session_state[ 'xml_xpath_results' ][ :preview_count ]:
						st.code(
							etree.tostring(
								el,
								pretty_print=True,
								encoding='unicode'
							),
							language='xml'
						)
			
			# ------------------------------------------------------------------
			# Debug / Introspection
			# ------------------------------------------------------------------
			with st.expander( "ℹ Loader State" ):
				xml_loader = st.session_state.get( 'xml_loader' )
				
				if xml_loader is None:
					st.info( "No loader initialized." )
				else:
					st.json(
						{
								"file_path": getattr( xml_loader, 'file_path', None ),
								"documents_loaded": getattr( xml_loader, 'documents', None ) is not None,
								"xml_tree_loaded": getattr( xml_loader, 'xml_tree', None ) is not None,
								"namespaces": getattr( xml_loader, 'xml_namespaces', None ),
								"chunk_size": getattr( xml_loader, 'chunk_size', None ),
								"overlap_amount": getattr( xml_loader, 'overlap_amount', None ),
						}
					)
		
		# --------------------------- PDF Loader
		with st.expander( label='PDF Loader', icon='📕', expanded=False ):
			pdf = st.file_uploader( 'Upload PDF', type=[ 'pdf' ], key='pdf_upload' )
			mode = st.selectbox( 'Mode', [ 'single', 'elements' ], key='pdf_mode' )
			extract = st.selectbox( 'Extract', [ 'plain', 'ocr' ], key='pdf_extract' )
			include = st.checkbox( 'Include Images', value=False, key='pdf_include' )
			fmt = st.selectbox( 'Format', [ 'markdown-img', 'text' ], key='pdf_fmt' )
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_pdf = col_load.button( 'Load', key='pdf_load' )
			clear_pdf = col_clear.button( 'Clear', key='pdf_clear' )
			
			can_save = (
					st.session_state.get( 'active_loader' ) == 'PdfLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='pdf_loader_output.txt',
					mime='text/plain',
					key='pdf_save',
				)
			else:
				col_save.button( 'Save', key='pdf_save_disabled', disabled=True )
			
			# --------------------------------------------------
			# Clear
			# --------------------------------------------------
			if clear_pdf:
				clear_if_active( "PdfLoader" )
				st.session_state.raw_text = _rebuild_raw_text_from_documents( )
				st.session_state[ "_loader_status" ] = "PDF Loader state cleared."
				st.rerun( )
			
			# --------------------------------------------------
			# Load
			# --------------------------------------------------
			if load_pdf and pdf:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, pdf.name )
					with open( path, "wb" ) as f:
						f.write( pdf.read( ) )
					
					loader = PdfLoader( )
					documents = loader.load(
						path,
						mode=mode,
						extract=extract,
						include=include,
						format=fmt,
					) or [ ]
				
				# Canonical promotion: loaded content == raw_text
				raw_text = "\n\n".join(
					d.page_content for d in documents
					if
					hasattr( d, "page_content" )
					and isinstance( d.page_content, str )
					and d.page_content.strip( )
				)
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = raw_text
				st.session_state.processed_text = raw_text
				st.session_state.active_loader = "PdfLoader"
				
				st.session_state[ "_loader_status" ] = \
					f"Loaded {len( documents )} PDF document(s)."
				st.rerun( )
		
		# --------------------------- Markdown Loader
		with st.expander( label='Markdown Loader', icon='🧾', expanded=False ):
			md = st.file_uploader(
				'Upload Markdown',
				type=[ 'md',
				       'markdown' ],
				key='md_upload',
			)
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_md = col_load.button(
				'Load',
				key='md_load',
			)
			
			clear_md = col_clear.button(
				'Clear',
				key='md_clear',
			)
			
			# Save enabled only when MarkdownLoader is active and raw_text exists
			can_save = (
					st.session_state.get( 'active_loader' ) == 'MarkdownLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='markdown_loader_output.txt',
					mime='text/plain',
					key='md_save',
				)
			else:
				col_save.button(
					'Save',
					key='md_save_disabled',
					disabled=True,
				)
			
			# --------------------------------------------------
			# Clear (UNCHANGED behavior)
			# --------------------------------------------------
			if clear_md:
				clear_if_active( 'MarkdownLoader' )
				st.info( "Markdown Loader state cleared." )
			
			# --------------------------------------------------
			# Load (UNCHANGED behavior)
			# --------------------------------------------------
			if load_md and md:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, md.name )
					with open( path, "wb" ) as f:
						f.write( md.read( ) )
					
					loader = MarkdownLoader( )
					documents = loader.load( path )
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = "\n\n".join( d.page_content for d in documents )
				st.session_state.active_loader = "MarkdownLoader"
				
				st.success( f"Loaded {len( documents )} Markdown document(s)." )
		
		# --------------------------- HTML Loader
		with st.expander( label='HTML Loader', icon='🌐', expanded=False ):
			html = st.file_uploader( 'Upload HTML', type=[ 'html', 'htm' ], key='html_upload' )
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_html = col_load.button( 'Load', key='html_load' )
			clear_html = col_clear.button( 'Clear', key='html_clear' )
			
			# Save enabled only when HtmlLoader is active and raw_text exists
			can_save = (
					st.session_state.get( 'active_loader' ) == 'HtmlLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='html_loader_output.txt',
					mime='text/plain',
					key='html_save',
				)
			else:
				col_save.button(
					'Save',
					key='html_save_disabled',
					disabled=True,
				)
			
			# --------------------------------------------------
			# Clear (UNCHANGED behavior)
			# --------------------------------------------------
			if clear_html:
				clear_if_active( "HtmlLoader" )
				st.info( "HTML Loader state cleared." )
			
			# --------------------------------------------------
			# Load (UNCHANGED behavior)
			# --------------------------------------------------
			if load_html and html:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, html.name )
					with open( path, "wb" ) as f:
						f.write( html.read( ) )
					
					loader = HtmlLoader( )
					documents = loader.load( path )
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = "\n\n".join( d.page_content for d in documents )
				st.session_state.active_loader = "HtmlLoader"
				st.success( f"Loaded {len( documents )} HTML document(s)." )
		
		# --------------------------- JSON Loader
		with st.expander( label='JSON Loader', icon='🧩', expanded=False ):
			js = st.file_uploader( 'Upload JSON', type=[ 'json' ], key='json_upload', )
			
			is_lines = st.checkbox( 'JSON Lines', value=False, key='json_lines', )
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_json = col_load.button( 'Load', key='json_load', )
			
			clear_json = col_clear.button( 'Clear', key='json_clear', )
			
			# Save enabled only when JsonLoader is active and raw_text exists
			can_save = (
					st.session_state.get( 'active_loader' ) == 'JsonLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='json_loader_output.txt',
					mime='text/plain',
					key='json_save',
				)
			else:
				col_save.button(
					'Save',
					key='json_save_disabled',
					disabled=True,
				)
			
			# --------------------------------------------------
			# Clear (UNCHANGED behavior)
			# --------------------------------------------------
			if clear_json:
				clear_if_active( 'JsonLoader' )
				st.info( 'JSON Loader state cleared.' )
			
			# --------------------------------------------------
			# Load (UNCHANGED behavior)
			# --------------------------------------------------
			if load_json and js:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, js.name )
					with open( path, 'wb' ) as f:
						f.write( js.read( ) )
					
					loader = JsonLoader( )
					documents = loader.load(
						path,
						is_text=True,
						is_lines=is_lines,
					)
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = "\n\n".join( d.page_content for d in documents )
				st.session_state.active_loader = "JsonLoader"
				st.success( f"Loaded {len( documents )} JSON document(s)." )
		
		# --------------------------- PowerPoint Loader
		with st.expander( '📽 Power Point Loader', expanded=False ):
			pptx = st.file_uploader(
				'Upload PPTX',
				type=[ 'pptx' ],
				key='pptx_upload',
			)
			
			mode = st.selectbox(
				'Mode',
				[ 'single',
				  'multiple' ],
				key='pptx_mode',
			)
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save (same row, same style)
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_pptx = col_load.button(
				'Load',
				key='pptx_load',
			)
			
			clear_pptx = col_clear.button(
				'Clear',
				key='pptx_clear',
			)
			
			# Save enabled only when PowerPointLoader is active and raw_text exists
			can_save = (
					st.session_state.get( 'active_loader' ) == 'PowerPointLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='powerpoint_loader_output.txt',
					mime='text/plain',
					key='pptx_save',
				)
			else:
				col_save.button(
					'Save',
					key='pptx_save_disabled',
					disabled=True,
				)
			
			# --------------------------------------------------
			# Clear (UNCHANGED behavior)
			# --------------------------------------------------
			if clear_pptx:
				clear_if_active( 'PowerPointLoader' )
				st.info( 'PowerPoint Loader state cleared.' )
			
			# --------------------------------------------------
			# Load (UNCHANGED behavior)
			# --------------------------------------------------
			if load_pptx and pptx:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, pptx.name )
					with open( path, "wb" ) as f:
						f.write( pptx.read( ) )
					
					loader = PowerPointLoader( )
					documents = (
							loader.load( path )
							if mode == "single"
							else loader.load_multiple( path )
					)
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = "\n\n".join( d.page_content for d in documents )
				st.session_state.active_loader = "PowerPointLoader"
				st.success( f"Loaded {len( documents )} PowerPoint document(s)." )
		
		# --------------------------- Excel Loader
		with st.expander( '📊 Excel Loader', expanded=False ):
			excel_file = st.file_uploader(
				'Upload Excel file',
				type=[ 'xlsx',
				       'xls' ],
				key='excel_upload',
			)
			
			sheet_name = st.text_input(
				'Sheet name (leave blank for all sheets)',
				key='excel_sheet',
			)
			
			table_prefix = st.text_input(
				'SQLite table prefix',
				value='excel',
				help='Each sheet will be written as <prefix>_<sheetname>',
				key='excel_table_prefix',
			)
			
			# --------------------------------------------------
			# Buttons: Load / Clear / Save
			# --------------------------------------------------
			col_load, col_clear, col_save = st.columns( 3 )
			load_excel = col_load.button( 'Load', key='excel_load' )
			clear_excel = col_clear.button( 'Clear', key='excel_clear' )
			
			can_save = (
					st.session_state.get( 'active_loader' ) == 'ExcelLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='excel_loader_output.txt',
					mime='text/plain',
					key='excel_save',
				)
			else:
				col_save.button(
					'Save',
					key='excel_save_disabled',
					disabled=True,
				)
			
			# --------------------------------------------------
			# Clear (remove only ExcelLoader documents)
			# --------------------------------------------------
			if clear_excel and st.session_state.get( 'documents' ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( 'loader' ) != 'ExcelLoader'
				]
				
				st.session_state.raw_text = (
						"\n\n".join(
							d.page_content
							for d in st.session_state.documents
							if isinstance( d.page_content, str )
							and d.page_content.strip( )
						)
						if st.session_state.documents else None
				)
				
				st.session_state.active_loader = None
				
				st.info( "ExcelLoader documents removed." )
			
			# --------------------------------------------------
			# Load + SQLite ingestion
			# --------------------------------------------------
			if load_excel and excel_file:
				sqlite_path = os.path.join( "stores", "sqlite", "data.db" )
				os.makedirs( os.path.dirname( sqlite_path ), exist_ok=True )
				
				with tempfile.TemporaryDirectory( ) as tmp:
					excel_path = os.path.join( tmp, excel_file.name )
					with open( excel_path, "wb" ) as f:
						f.write( excel_file.read( ) )
					
					if sheet_name.strip( ):
						dfs = {
								sheet_name: pd.read_excel(
									excel_path,
									sheet_name=sheet_name,
								)
						}
					else:
						dfs = pd.read_excel(
							excel_path,
							sheet_name=None,
						)
				
				conn = sqlite3.connect( sqlite_path )
				documents = [ ]
				
				for sheet, df in dfs.items( ):
					if df.empty:
						continue
					
					table_name = f"{table_prefix}_{sheet}".replace(
						" ", "_"
					).lower( )
					
					df.to_sql(
						table_name,
						conn,
						if_exists="replace",
						index=False,
					)
					
					text = df.to_csv( index=False )
					
					documents.append(
						Document(
							page_content=text,
							metadata={
									'loader': 'ExcelLoader',
									'source': excel_file.name,
									'sheet': sheet,
									'table': table_name,
									'sqlite_db': sqlite_path,
							},
						)
					)
				
				conn.close( )
				
				if documents:
					if st.session_state.get( 'documents' ):
						st.session_state.documents.extend( documents )
					else:
						st.session_state.documents = documents
						st.session_state.raw_documents = list( documents )
					
					st.session_state.raw_text = "\n\n".join(
						d.page_content
						for d in st.session_state.documents
						if isinstance( d.page_content, str )
						and d.page_content.strip( )
					)
					
					st.session_state.processed_text = None
					st.session_state.active_loader = 'ExcelLoader'
					
					st.success(
						f"Loaded {len( documents )} sheet(s) and stored in SQLite."
					)
				else:
					st.warning(
						"No data loaded (empty sheets or invalid selection)."
					)
		
		# --------------------------- arXiv Loader
		with st.expander( "🧠 ArXiv Loader", expanded=False ):
			arxiv_query = st.text_input(
				"Query",
				placeholder="e.g., transformer OR llm",
				key="arxiv_query",
			)
			
			arxiv_max_chars = st.number_input(
				"Max characters per document",
				min_value=250,
				max_value=100000,
				value=1000,
				step=250,
				key="arxiv_max_chars",
				help="Maximum characters read",
			)
			
			col_fetch, col_clear, col_save = st.columns( 3 )
			arxiv_fetch = col_fetch.button( "Load", key="arxiv_fetch" )  # label kept as Load button row convention
			arxiv_clear = col_clear.button( "Clear", key="arxiv_clear" )
			
			can_save = (
					st.session_state.get( "active_loader" ) == "ArXivLoader"
					and isinstance( st.session_state.get( "raw_text" ), str )
					and st.session_state.get( "raw_text" ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					"Save",
					data=st.session_state.get( "raw_text" ),
					file_name="arxiv_loader_output.txt",
					mime="text/plain",
					key="arxiv_save",
				)
			else:
				col_save.button( "Save", key="arxiv_save_disabled", disabled=True )
			
			if arxiv_clear and st.session_state.get( "documents" ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( "loader" ) != "ArXivLoader"
				]
				st.session_state.raw_text = _rebuild_raw_text_from_documents( )
				st.session_state[ "_loader_status" ] = "ArXivLoader documents removed."
				st.rerun( )
			
			if arxiv_fetch and arxiv_query:
				loader = ArXivLoader( )
				documents = loader.load(
					arxiv_query,
					max_chars=int( arxiv_max_chars ),
				) or [ ]
				
				for d in documents:
					d.metadata[ "loader" ] = "ArXivLoader"
					d.metadata[ "source" ] = arxiv_query
				
				if documents:
					if st.session_state.get( "documents" ):
						st.session_state.documents.extend( documents )
					else:
						st.session_state.documents = documents
						st.session_state.raw_documents = list( documents )
					
					st.session_state.raw_text = _rebuild_raw_text_from_documents( )
					st.session_state.active_loader = "ArXivLoader"
					
					st.session_state[
						"_loader_status" ] = f"Fetched {len( documents )} arXiv document(s)."
					st.rerun( )
		
		# --------------------------- Wikipedia Loader
		with st.expander( "📚 Wikipedia Loader", expanded=False ):
			wiki_query = st.text_input(
				"Query",
				placeholder="e.g., Natural language processing",
				key="wiki_query",
			)
			
			wiki_max_docs = st.number_input(
				"Max documents",
				min_value=1,
				max_value=250,
				value=25,
				step=1,
				key="wiki_max_docs",
				help="Maximum number of documents loaded",
			)
			
			wiki_max_chars = st.number_input(
				"Max characters per document",
				min_value=250,
				max_value=100000,
				value=4000,
				step=250,
				key="wiki_max_chars",
				help="Upper limit on the number of characters",
			)
			
			col_fetch, col_clear, col_save = st.columns( 3 )
			wiki_fetch = col_fetch.button( "Load", key="wiki_fetch" )
			wiki_clear = col_clear.button( "Clear", key="wiki_clear" )
			
			can_save = (
					st.session_state.get( "active_loader" ) == "WikiLoader"
					and isinstance( st.session_state.get( "raw_text" ), str )
					and st.session_state.get( "raw_text" ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					"Save",
					data=st.session_state.get( "raw_text" ),
					file_name="wiki_loader_output.txt",
					mime="text/plain",
					key="wiki_save",
				)
			else:
				col_save.button( "Save", key="wiki_save_disabled", disabled=True )
			
			if wiki_clear and st.session_state.get( "documents" ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( "loader" ) != "WikiLoader"
				]
				st.session_state.raw_text = _rebuild_raw_text_from_documents( )
				st.session_state[ "_loader_status" ] = "WikiLoader documents removed."
				st.rerun( )
			
			if wiki_fetch and wiki_query:
				loader = WikiLoader( )
				documents = loader.load(
					wiki_query,
					max_docs=int( wiki_max_docs ),
					max_chars=int( wiki_max_chars ),
				) or [ ]
				
				for d in documents:
					d.metadata[ "loader" ] = "WikiLoader"
					d.metadata[ "source" ] = wiki_query
				
				if documents:
					if st.session_state.get( "documents" ):
						st.session_state.documents.extend( documents )
					else:
						st.session_state.documents = documents
						st.session_state.raw_documents = list( documents )
					
					st.session_state.raw_text = _rebuild_raw_text_from_documents( )
					st.session_state.active_loader = "WikiLoader"
					
					st.session_state[
						"_loader_status" ] = f"Fetched {len( documents )} Wikipedia document(s)."
					st.rerun( )
		
		# --------------------------- GitHub Loader
		with st.expander( "🐙 GitHub Loader", expanded=False ):
			gh_url = st.text_input(
				"GitHub API URL",
				placeholder="https://api.github.com",
				value="https://api.github.com",
				key="gh_url",
				help="web url to a github repository",
			)
			
			gh_repo = st.text_input(
				"Repo (owner/name)",
				placeholder="openai/openai-python",
				key="gh_repo",
				help="Name of the repository",
			)
			
			gh_branch = st.text_input(
				"Branch",
				placeholder="main",
				value="main",
				key="gh_branch",
				help="The branch of the repository",
			)
			
			gh_filetype = st.text_input(
				"File type filter",
				value=".md",
				key="gh_filetype",
				help="Filtering by file type. Example: .py, .md, .txt",
			)
			
			col_fetch, col_clear, col_save = st.columns( 3 )
			gh_fetch = col_fetch.button( "Load", key="gh_fetch" )
			gh_clear = col_clear.button( "Clear", key="gh_clear" )
			
			can_save = (st.session_state.get( "active_loader" ) == "GithubLoader"
			            and isinstance( st.session_state.get( "raw_text" ), str )
			            and st.session_state.get( "raw_text" ).strip( ))
			
			if can_save:
				col_save.download_button(
					"Save",
					data=st.session_state.get( "raw_text" ),
					file_name="github_loader_output.txt",
					mime="text/plain",
					key="gh_save", )
			else:
				col_save.button( "Save", key="gh_save_disabled", disabled=True )
			
			if gh_clear and st.session_state.get( "documents" ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( "loader" ) != "GithubLoader" ]
				st.session_state.raw_text = _rebuild_raw_text_from_documents( )
				st.session_state[ "_loader_status" ] = "GithubLoader documents removed."
				st.rerun( )
			
			if gh_fetch and gh_repo and gh_branch:
				loader = GithubLoader( )
				documents = loader.load(
					gh_url,
					gh_repo,
					gh_branch,
					gh_filetype,
				) or [ ]
				
				for d in documents:
					d.metadata[ "loader" ] = "GithubLoader"
					d.metadata[ "source" ] = f"{gh_repo}@{gh_branch}"
				
				if documents:
					if st.session_state.get( "documents" ):
						st.session_state.documents.extend( documents )
					else:
						st.session_state.documents = documents
						st.session_state.raw_documents = list( documents )
					
					st.session_state.raw_text = _rebuild_raw_text_from_documents( )
					st.session_state.active_loader = "GithubLoader"
					
					st.session_state[
						"_loader_status" ] = f"Fetched {len( documents )} GitHub document(s)."
					st.rerun( )
		
		# --------------------------- Web Loader
		with st.expander( "🔗 Web Loader", expanded=False ):
			urls = st.text_area(
				"Enter one URL per line",
				placeholder="https://example.com\nhttps://another.com",
				key="web_urls", )
			
			col_fetch, col_clear, col_save = st.columns( 3 )
			load_web = col_fetch.button( "Load", key="web_fetch" )
			clear_web = col_clear.button( "Clear", key="web_clear" )
			can_save = (st.session_state.get( "active_loader" ) == "WebLoader"
			            and isinstance( st.session_state.get( "raw_text" ), str )
			            and st.session_state.get( "raw_text" ).strip( ))
			
			if can_save:
				col_save.download_button(
					"Save",
					data=st.session_state.get( "raw_text" ),
					file_name="web_loader_output.txt",
					mime="text/plain",
					key="web_save",
				)
			else:
				col_save.button( "Save", key="web_save_disabled", disabled=True )
			
			if clear_web and st.session_state.get( "documents" ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( "loader" ) != "WebLoader"
				]
				st.session_state.raw_text = _rebuild_raw_text_from_documents( )
				st.session_state[ "_loader_status" ] = "WebLoader documents removed."
				st.rerun( )
			
			if load_web and urls.strip( ):
				loader = WebLoader( recursive=False )
				new_docs = [ ]
				
				for url in [ u.strip( ) for u in urls.splitlines( ) if u.strip( ) ]:
					documents = loader.load( url ) or [ ]
					for d in documents:
						d.metadata[ "loader" ] = "WebLoader"
						d.metadata[ "source" ] = url
					new_docs.extend( documents )
				
				if new_docs:
					if st.session_state.get( "documents" ):
						st.session_state.documents.extend( new_docs )
					else:
						st.session_state.documents = new_docs
						st.session_state.raw_documents = list( new_docs )
					
					st.session_state.raw_text = _rebuild_raw_text_from_documents( )
					st.session_state.active_loader = "WebLoader"
					
					st.session_state[
						"_loader_status" ] = f"Fetched {len( new_docs )} web document(s)."
					st.rerun( )
		
		# --------------------------- Web Crawler
		with st.expander( "🕷️ Web Crawler", expanded=False ):
			start_url = st.text_input(
				"Start URL",
				placeholder="https://example.com",
				key="crawl_start_url",
			)
			
			max_depth = st.number_input(
				"Max crawl depth",
				min_value=1,
				max_value=5,
				value=2,
				step=1,
				key="crawl_depth",
			)
			
			stay_on_domain = st.checkbox(
				"Stay on starting domain",
				value=True,
				key="crawl_domain_lock",
			)
			
			col_run, col_clear, col_save = st.columns( 3 )
			run_crawl = col_run.button( "Load", key="crawl_run" )
			clear_crawl = col_clear.button( "Clear", key="crawl_clear" )
			
			can_save = (
					st.session_state.get( "active_loader" ) == "WebCrawler"
					and isinstance( st.session_state.get( "raw_text" ), str )
					and st.session_state.get( "raw_text" ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					"Save",
					data=st.session_state.get( "raw_text" ),
					file_name="web_crawler_output.txt",
					mime="text/plain",
					key="crawl_save",
				)
			else:
				col_save.button( "Save", key="crawl_save_disabled", disabled=True )
			
			if clear_crawl and st.session_state.get( "documents" ):
				st.session_state.documents = [
						d for d in st.session_state.documents
						if d.metadata.get( "loader" ) != "WebCrawler"
				]
				st.session_state.raw_text = _rebuild_raw_text_from_documents( )
				st.session_state[ "_loader_status" ] = "WebCrawler documents removed."
				st.rerun( )
			
			if run_crawl and start_url:
				loader = WebLoader(
					recursive=True,
					max_depth=max_depth,
					prevent_outside=stay_on_domain,
				)
				
				documents = loader.load( start_url ) or [ ]
				for d in documents:
					d.metadata[ "loader" ] = "WebCrawler"
					d.metadata[ "source" ] = start_url
				
				if documents:
					if st.session_state.get( "documents" ):
						st.session_state.documents.extend( documents )
					else:
						st.session_state.documents = documents
						st.session_state.raw_documents = list( documents )
					
					st.session_state.raw_text = _rebuild_raw_text_from_documents( )
					st.session_state.active_loader = "WebCrawler"
					st.session_state[
						"_loader_status" ] = f"Crawled {len( documents )} document(s)."
					st.rerun( )
	
	# ------------------------------------------------------------------
	# RIGHT COLUMN — Document Preview
	# ------------------------------------------------------------------
	with right:
		documents = st.session_state.documents
		if not documents:
			st.info( 'No documents loaded.' )
		else:
			st.caption( f'Active Loader: {st.session_state.active_loader}' )
			st.write( f'Documents: {len( documents )}' )
			for i, d in enumerate( documents[ :5 ] ):
				with st.expander( f'Document {i + 1}', expanded=True ):
					st.json( d.metadata )
					st.text_area( 'Content', d.page_content[ :5000 ],
						height=500, key=f'preview_doc_{i}' )

# =============================================================================
# SCRAPING MODE
# ==============================================================================
elif mode == 'Scraping':
	st.subheader( f'🕷️ Web Scraping' )
	st.divider( )
	
	if 'webscrape_clear_request' not in st.session_state:
		st.session_state[ 'webscrape_clear_request' ] = False
	
	if st.session_state.get( 'webscrape_clear_request', False ):
		st.session_state[ 'webfetcher_url' ] = ''
		st.session_state[ 'webscrape_results' ] = [ ]
		st.session_state[ 'webscrape_summary' ] = { }
		st.session_state[ 'webscrape_clear_request' ] = False
	
	def _clear_webscrape_state( ) -> None:
		st.session_state[ 'webscrape_clear_request' ] = True
	
	def _coerce_items( value: Any ) -> list[ str ]:
		if value is None:
			return [ ]
		if isinstance( value, list ):
			return [ str( item ) for item in value if item is not None ]
		return [ str( value ) ]
	
	def _extract_title_from_html( html: str ) -> str:
		try:
			if not isinstance( html, str ) or not html.strip( ):
				return ''
			
			match = re.search(
				r'<title[^>]*>(.*?)</title>',
				html,
				flags=re.IGNORECASE | re.DOTALL )
			
			if not match:
				return ''
			
			title = re.sub( r'\s+', ' ', match.group( 1 ) ).strip( )
			return html_lib.unescape( title )
		except Exception:
			return ''
	
	def _truncate_text( text: str, limit: int = 12000 ) -> str:
		if not isinstance( text, str ):
			return ''
		if len( text ) <= limit:
			return text
		return text[ : limit ] + '\n\n... [truncated]'
	
	def _normalize_url( base_url: str, href: str ) -> str:
		try:
			if not href or not isinstance( href, str ):
				return ''
			
			href = href.strip( )
			if not href:
				return ''
			
			absolute = urljoin( base_url, href )
			parsed = urlparse( absolute )
			if parsed.scheme not in ('http', 'https'):
				return ''
			
			normalized = parsed._replace( fragment='' )
			return normalized.geturl( )
		except Exception:
			return ''
	
	def _same_domain( left: str, right: str ) -> bool:
		try:
			left_host = (urlparse( left ).netloc or '').lower( )
			right_host = (urlparse( right ).netloc or '').lower( )
			return bool( left_host ) and left_host == right_host
		except Exception:
			return False
	
	def _extract_links_from_html( base_url: str, html: str ) -> list[ str ]:
		try:
			if not isinstance( html, str ) or not html.strip( ):
				return [ ]
			
			soup = BeautifulSoup( html, 'html.parser' )
			results: list[ str ] = [ ]
			seen: set[ str ] = set( )
			for tag in soup.find_all( 'a', href=True ):
				candidate = _normalize_url( base_url, tag.get( 'href', '' ) )
				if candidate and candidate not in seen:
					seen.add( candidate )
					results.append( candidate )
			
			return results
		except Exception:
			return [ ]
	
	def _scrape_single_page(
			url: str,
			include_title: bool,
			include_basic_text: bool,
			include_raw_html: bool,
			selected_methods: list[ str ] ) -> dict[ str, Any ]:
		page_result: dict[ str, Any ] = \
		{
				'url': url,
				'status_code': None,
				'encoding': None,
				'title': '',
				'plain_text': '',
				'raw_html': '',
				'links_discovered': [ ],
				'data': { },
				'errors': [ ],
		}
		
		fetcher = WebFetcher( )
		
		try:
			response = fetcher.fetch( url )
			if response is None:
				page_result[ 'errors' ].append( 'No response returned.' )
				return page_result
			
			page_result[ 'status_code' ] = getattr( response, 'status_code', None )
			page_result[ 'encoding' ] = getattr( response, 'encoding', None )
			raw_html = getattr( response, 'text', '' ) or ''
			page_result[ 'links_discovered' ] = _extract_links_from_html( url, raw_html )
			if include_title:
				page_result[ 'title' ] = _extract_title_from_html( raw_html )
			
			if include_basic_text:
				try:
					page_result[ 'plain_text' ] = fetcher.html_to_text( raw_html ) or ''
				except Exception as exc:
					page_result[ 'errors' ].append( f'Basic Text: {str( exc )}' )
			
			if include_raw_html:
				page_result[ 'raw_html' ] = raw_html
		
		except Exception as exc:
			page_result[ 'errors' ].append( f'Fetch: {str( exc )}' )
			return page_result
		
		REGISTRY: dict[ str, tuple[ str, callable ] ] = \
		{
				'scrape_headings': ('Headings', fetcher.scrape_headings),
				'scrape_paragraphs': ('Paragraphs', fetcher.scrape_paragraphs),
				'scrape_lists': ('Lists', fetcher.scrape_lists),
				'scrape_tables': ('Tables', fetcher.scrape_tables),
				'scrape_articles': ('Articles', fetcher.scrape_articles),
				'scrape_sections': ('Sections', fetcher.scrape_sections),
				'scrape_divisions': ('Divisions', fetcher.scrape_divisions),
				'scrape_blockquotes': ('Blockquotes', fetcher.scrape_blockquotes),
				'scrape_hyperlinks': ('Hyperlinks', fetcher.scrape_hyperlinks),
				'scrape_images': ('Images', fetcher.scrape_images),
		}
		
		for method_name in selected_methods:
			if method_name not in REGISTRY:
				continue
			
			label, method = REGISTRY[ method_name ]
			try:
				data = method( url )
				page_result[ 'data' ][ label ] = _coerce_items( data )
			except Exception as exc:
				page_result[ 'data' ][ label ] = [ ]
				page_result[ 'errors' ].append( f'{label}: {str( exc )}' )
		
		return page_result
	
	def _crawl_pages(
			seed_url: str,
			include_title: bool,
			include_basic_text: bool,
			include_raw_html: bool,
			selected_methods: list[ str ],
			recursive: bool,
			max_depth: int,
			max_pages: int,
			same_domain_only: bool ) -> tuple[ list[ dict[ str, Any ] ], dict[ str, Any ] ]:
		results: list[ dict[ str, Any ] ] = [ ]
		visited: set[ str ] = set( )
		enqueued: set[ str ] = set( )
		queue: deque[ tuple[ str, int ] ] = deque( )
		skipped_urls: list[ str ] = [ ]
		
		normalized_seed = _normalize_url( seed_url, seed_url )
		if not normalized_seed:
			raise ValueError( 'A valid absolute URL is required.' )
		
		queue.append( (normalized_seed, 0) )
		enqueued.add( normalized_seed )
		
		while queue and len( results ) < max_pages:
			current_url, depth = queue.popleft( )
			
			if current_url in visited:
				continue
			
			visited.add( current_url )
			
			page_result = _scrape_single_page(
				url=current_url,
				include_title=include_title,
				include_basic_text=include_basic_text,
				include_raw_html=include_raw_html,
				selected_methods=selected_methods )
			
			page_result[ 'depth' ] = depth
			results.append( page_result )
			
			if not recursive:
				continue
			
			if depth >= max_depth:
				continue
			
			discovered_links = page_result.get( 'links_discovered', [ ] ) or [ ]
			for next_url in discovered_links:
				if len( results ) + len( queue ) >= max_pages:
					break
				
				if not next_url or next_url in visited or next_url in enqueued:
					continue
				
				if same_domain_only and not _same_domain( normalized_seed, next_url ):
					skipped_urls.append( next_url )
					continue
				
				queue.append( (next_url, depth + 1) )
				enqueued.add( next_url )
		
		summary: dict[ str, Any ] = \
		{
				'mode': 'recursive' if recursive else 'single-page',
				'seed_url': normalized_seed,
				'pages_processed': len( results ),
				'pages_visited': len( visited ),
				'pages_skipped': len( skipped_urls ),
				'recursive_requested': bool( recursive ),
				'max_depth': int( max_depth ),
				'max_pages': int( max_pages ),
				'same_domain_only': bool( same_domain_only ),
				'visited_urls': list( visited ),
				'skipped_urls': skipped_urls,
		}
		
		return results, summary
	
	col_left, col_right = st.columns( [ 1, 2 ], border=True )
	
	with col_left:
		target_url = st.text_input(
			'Enter Target URL',
			placeholder='https://example.com',
			key='webfetcher_url' )
		
		st.markdown( '##### Core Output' )
		
		include_title = st.checkbox(
			'Page Title',
			value=True,
			key='wf_page_title' )
		
		include_basic_text = st.checkbox(
			'Basic Text',
			value=True,
			key='wf_basic_text' )
		
		include_raw_html = st.checkbox(
			'Raw HTML',
			value=False,
			key='wf_raw_html' )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		st.markdown( '##### Structured Extraction' )

		col1, col2 = st.columns( [ 0.5, 0.5 ] )
		
		REGISTRY_LABELS: dict[ str, str ] = \
		{
				'scrape_headings': 'Headings',
				'scrape_paragraphs': 'Paragraphs',
				'scrape_lists': 'Lists',
				'scrape_tables': 'Tables',
				'scrape_articles': 'Articles',
				'scrape_sections': 'Sections',
				'scrape_divisions': 'Divisions',
				'scrape_blockquotes': 'Blockquotes',
				'scrape_hyperlinks': 'Hyperlinks',
				'scrape_images': 'Images',
		}
		
		selected_methods: list[ str ] = [ ]
		
		_registry_items: list[ tuple[ str, str ] ] = list( REGISTRY_LABELS.items( ) )
		_col1_items: list[ tuple[ str, str ] ] = _registry_items[ :5 ]
		_col2_items: list[ tuple[ str, str ] ] = _registry_items[ 5: ]
		
		with col1:
			for method_name, label in _col1_items:
				if st.checkbox( label, key=f'wf_{method_name}' ):
					selected_methods.append( method_name )
		
		with col2:
			for method_name, label in _col2_items:
				if st.checkbox( label, key=f'wf_{method_name}' ):
					selected_methods.append( method_name )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		st.markdown( '##### Crawl Controls' )
		
		enable_recursive = st.checkbox(
			'Recursive Crawl',
			value=False,
			key='wf_recursive' )
		
		max_depth = st.number_input(
			'Max Depth',
			min_value=0,
			max_value=10,
			value=1,
			step=1,
			key='wf_max_depth',
			disabled=(not enable_recursive) )
		
		max_pages = st.number_input(
			'Max Pages',
			min_value=1,
			max_value=500,
			value=10,
			step=1,
			key='wf_max_pages' )
		
		same_domain_only = st.checkbox(
			'Same Domain Only',
			value=True,
			key='wf_same_domain_only',
			disabled=(not enable_recursive) )
		
		b1, b2 = st.columns( 2 )
		
		with b1:
			run_scraper = st.button( 'Run Scraper', key='webfetcher_run' )
		
		with b2:
			st.button( 'Clear', key='webfetcher_clear', on_click=_clear_webscrape_state )
	
	with col_right:
		if run_scraper:
			try:
				if not target_url or not target_url.strip( ):
					raise ValueError( 'A target URL is required.' )
				
				results, summary = _crawl_pages(
					seed_url=target_url.strip( ),
					include_title=include_title,
					include_basic_text=include_basic_text,
					include_raw_html=include_raw_html,
					selected_methods=selected_methods,
					recursive=bool( enable_recursive ),
					max_depth=int( max_depth ),
					max_pages=int( max_pages ),
					same_domain_only=bool( same_domain_only ) )
				
				st.session_state[ 'webscrape_results' ] = results
				st.session_state[ 'webscrape_summary' ] = summary
				st.rerun( )
			
			except Exception as exc:
				st.error( str( exc ) )
		
		summary = st.session_state.get( 'webscrape_summary', { } )
		results = st.session_state.get( 'webscrape_results', [ ] )
		
		if summary:
			st.subheader( 'Summary' )
			st.json( summary )
		
		if not results:
			st.info( 'No results.' )
		else:
			st.subheader( 'Results' )
			
			for idx, page in enumerate( results, start=1 ):
				title = page.get( 'title', '' ) or page.get( 'url', f'Page {idx}' )
				depth = page.get( 'depth', 0 )
				
				with st.expander( f'Page {idx} [Depth {depth}]: {title}', expanded=(idx == 1) ):
					meta_col1, meta_col2 = st.columns( 2 )
					
					with meta_col1:
						st.markdown( f"**URL:** {page.get( 'url', '' )}" )
						st.markdown( f"**Status Code:** {page.get( 'status_code', '' )}" )
						st.markdown( f"**Depth:** {page.get( 'depth', 0 )}" )
					
					with meta_col2:
						st.markdown( f"**Encoding:** {page.get( 'encoding', '' )}" )
						st.markdown( f"**Title:** {page.get( 'title', '' )}" )
					
					plain_text = page.get( 'plain_text', '' )
					if isinstance( plain_text, str ) and plain_text.strip( ):
						st.subheader( 'Basic Text' )
						st.text_area(
							label='',
							value=_truncate_text( plain_text, limit=12000 ),
							height=280,
							key=f'webscrape_plain_text_{idx}' )
					
					raw_html = page.get( 'raw_html', '' )
					if isinstance( raw_html, str ) and raw_html.strip( ):
						st.subheader( 'Raw HTML' )
						st.text_area(
							label='',
							value=_truncate_text( raw_html, limit=12000 ),
							height=240,
							key=f'webscrape_raw_html_{idx}' )
					
					discovered_links = page.get( 'links_discovered', [ ] ) or [ ]
					if discovered_links:
						st.subheader( 'Links Discovered' )
						for link_idx, link in enumerate( discovered_links, start=1 ):
							st.write( f'{link_idx}. {link}' )
					
					data = page.get( 'data', { } ) or { }
					for label, items in data.items( ):
						st.subheader( f'{label}' )
						
						if not items:
							st.info( 'No results returned.' )
							continue
						
						for item_idx, item in enumerate( items, start=1 ):
							st.write( f'{item_idx}. {item}' )
					
					errors = page.get( 'errors', [ ] ) or [ ]
					if errors:
						st.subheader( 'Errors' )
						for err in errors:
							st.error( err )
							
# ==============================================================================
# FETCHING MODE
# =============================================================================
elif mode == 'Retrieval':
	st.subheader( f'🏛️ Public Collections & Archives' )
	st.divider( )
	st.session_state.setdefault( "arxiv_input", "" )
	st.session_state.setdefault( "arxiv_results", [ ] )
	
	# -------- ArXiv
	with st.expander( label='ArXiv', expanded=False  ):
		if 'arxiv_results' not in st.session_state:
			st.session_state[ 'arxiv_results' ] = [ ]
		
		if 'arxiv_clear_request' not in st.session_state:
			st.session_state[ 'arxiv_clear_request' ] = False
		
		if st.session_state.get( 'arxiv_clear_request', False ):
			st.session_state[ 'arxiv_input' ] = ''
			st.session_state[ 'arxiv_results' ] = [ ]
			st.session_state[ 'arxiv_max_docs' ] = 5
			st.session_state[ 'arxiv_full_documents' ] = False
			st.session_state[ 'arxiv_include_metadata' ] = False
			st.session_state[ 'arxiv_clear_request' ] = False
		
		def _clear_arxiv_state( ) -> None:
			st.session_state[ 'arxiv_clear_request' ] = True
		
		col1, col2 = st.columns( 2, border=True )
		
		with col1:
			arxiv_input = st.text_area(
				'Query',
				height=80,
				key='arxiv_input',
				placeholder=(
						'Examples:\n'
						'What is the ImageBind model?\n'
						'2401.01234\n'
						'graph neural networks for molecular property prediction'
				), )
			
			c1, c2 = st.columns( 2 )
			with c1:
				arxiv_max_docs = st.number_input(
					'Max Docs',
					min_value=1,
					max_value=300,
					value=st.session_state.get( 'arxiv_max_docs', 5 ),
					step=1,
					key='arxiv_max_docs',
					help='Maximum number of ArXiv documents to retrieve.'
				)
			
			with c2:
				arxiv_full_documents = st.checkbox(
					'Full Documents',
					value=st.session_state.get( 'arxiv_full_documents', False ),
					key='arxiv_full_documents',
					help='When checked, retrieves fuller document text instead of lighter summary-based output.'
				)
			
			arxiv_include_metadata = st.checkbox(
				'Include All Metadata',
				value=st.session_state.get( 'arxiv_include_metadata', False ),
				key='arxiv_include_metadata',
				help='Include additional metadata fields when available.'
			)
			
			b1, b2 = st.columns( 2 )
			
			with b1:
				do_submit = st.button( 'Submit', key='arxiv_submit' )
			
			with b2:
				st.button( 'Clear', key='arxiv_clear', on_click=_clear_arxiv_state )
			
			if do_submit:
				try:
					queries = [ q.strip( ) for q in (arxiv_input or '').splitlines( ) if
					            q.strip( ) ]
					
					if not queries:
						st.warning( 'No input provided.' )
					else:
						from fetchers import ArXiv
						
						fetcher = ArXiv(
							max_documents=int( arxiv_max_docs ),
							full_documents=bool( arxiv_full_documents ),
							include_metadata=bool( arxiv_include_metadata ) )
						
						results: list[ Document ] = [ ]
						
						for q in queries:
							docs = fetcher.fetch(
								q,
								max_documents=int( arxiv_max_docs ),
								full_documents=bool( arxiv_full_documents ),
								include_metadata=bool( arxiv_include_metadata ) )
							
							if isinstance( docs, list ):
								results.extend( docs )
						
						st.session_state[ 'arxiv_results' ] = results
						st.rerun( )
				
				except Exception as exc:
					st.error( 'ArXiv request failed.' )
					st.exception( exc )
		
		with col2:
			st.markdown( 'Results' )
			
			results = st.session_state.get( 'arxiv_results', [ ] )
			
			if not results:
				st.text( 'No results.' )
			else:
				for idx, doc in enumerate( results, start=1 ):
					title = ''
					if isinstance( doc, Document ):
						title = str( doc.metadata.get( 'Title', '' ) ) if doc.metadata else ''
					label = f'Document {idx}' if not title else f'Document {idx}: {title}'
					
					with st.expander( label, expanded=False ):
						if isinstance( doc, Document ):
							if doc.metadata:
								meta_col1, meta_col2 = st.columns( 2 )
								
								with meta_col1:
									if 'Title' in doc.metadata:
										st.markdown( f"**Title:** {doc.metadata.get( 'Title', '' )}" )
									if 'Authors' in doc.metadata:
										st.markdown( f"**Authors:** {doc.metadata.get( 'Authors', '' )}" )
								
								with meta_col2:
									if 'Published' in doc.metadata:
										st.markdown( f"**Published:** {doc.metadata.get( 'Published', '' )}" )
									if 'Entry ID' in doc.metadata:
										st.markdown( f"**Entry ID:** {doc.metadata.get( 'Entry ID', '' )}" )
							
							st.text_area(
								'Content',
								value=doc.page_content or '',
								height=300,
								key=f'arxiv_doc_{idx}' )
							
							if doc.metadata:
								st.json( doc.metadata )
						else:
							st.write( doc )
				
	# -------- Google Drive
	with st.expander( label='Google Drive', expanded=False ):
		if 'googledrive_results' not in st.session_state:
			st.session_state[ 'googledrive_results' ] = [ ]
		
		if 'googledrive_clear_request' not in st.session_state:
			st.session_state[ 'googledrive_clear_request' ] = False
		
		if st.session_state.get( 'googledrive_clear_request', False ):
			st.session_state[ 'googledrive_query' ] = ''
			st.session_state[ 'googledrive_folder_id' ] = cfg.GOOGLE_DRIVE_FOLDER_ID or 'root'
			st.session_state[ 'googledrive_results_limit' ] = 10
			st.session_state[ 'googledrive_template' ] = 'gdrive-query'
			st.session_state[ 'googledrive_mode' ] = 'documents'
			st.session_state[ 'googledrive_mime_type' ] = ''
			st.session_state[ 'googledrive_results' ] = [ ]
			st.session_state[ 'googledrive_clear_request' ] = False
		
		def _clear_googledrive_state( ) -> None:
			st.session_state[ 'googledrive_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			gd_query = st.text_area(
				'Google Drive Query',
				height=90,
				help=cfg.GOOGLE_DRIVE,
				key='googledrive_query',
				placeholder=(
						'Examples:\n'
						'machine learning\n'
						'budget execution\n'
						'FY 2026 operating plan\n'
						'\n'
						'Use "*" only when the selected template supports folder-wide or mime-type retrieval.'
				),
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				gd_folder_id = st.text_input(
					'Folder ID',
					value=st.session_state.get( 'googledrive_folder_id', cfg.GOOGLE_DRIVE_FOLDER_ID or 'root' ),
					key='googledrive_folder_id',
					placeholder='root or a Google Drive folder id',
					help='Use "root" for your My Drive root, or provide a specific folder id.'
				)
			
			with c2:
				gd_results_limit = st.number_input(
					'Max Docs',
					min_value=1,
					max_value=100,
					value=int( st.session_state.get( 'googledrive_results_limit', 10 ) ),
					step=1,
					key='googledrive_results_limit',
				)
			
			c3, c4 = st.columns( 2 )
			
			with c3:
				gd_template = st.selectbox(
					'Template',
					options=[
							'gdrive-all-in-folder',
							'gdrive-query',
							'gdrive-by-name',
							'gdrive-query-in-folder',
							'gdrive-mime-type',
							'gdrive-mime-type-in-folder',
							'gdrive-query-with-mime-type',
							'gdrive-query-with-mime-type-and-folder',
					],
					index=1,
					key='googledrive_template',
					help='Select the Drive retrieval strategy.'
				)
			
			with c4:
				gd_mode = st.selectbox(
					'Mode',
					options=[ 'documents', 'snippets' ],
					index=0,
					key='googledrive_mode',
					help='Use snippets for short metadata-driven returns.'
				)
			
			gd_mime_type = st.selectbox(
				'MIME Type Filter',
				options=[
						'',
						'text/text',
						'text/plain',
						'text/html',
						'text/csv',
						'text/markdown',
						'image/png',
						'image/jpeg',
						'application/epub+zip',
						'application/pdf',
						'application/rtf',
						'application/vnd.google-apps.document',
						'application/vnd.google-apps.presentation',
						'application/vnd.google-apps.spreadsheet',
						'application/vnd.google.colaboratory',
						'application/vnd.openxmlformats-officedocument.presentationml.presentation',
						'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
				],
				index=0,
				key='googledrive_mime_type',
				help='Optional MIME type restriction.'
			)
			
			st.caption(
				'Expected auth: GOOGLE_ACCOUNT_FILE for credentials JSON. '
				'Optional: GOOGLE_DRIVE_TOKEN_PATH for token persistence.'
			)
			
			b1, b2 = st.columns( 2 )
			
			with b1:
				gd_submit = st.button( 'Submit', key='googledrive_submit' )
			
			with b2:
				st.button( 'Clear', key='googledrive_clear', on_click=_clear_googledrive_state )
			
			if gd_submit:
				try:
					from fetchers import GoogleDrive
					
					fetcher = GoogleDrive( )
					docs = fetcher.fetch(
						question=gd_query,
						folder_id=gd_folder_id or 'root',
						results=int( gd_results_limit ),
						template=gd_template,
						mime_type=gd_mime_type or None,
						mode=gd_mode,
					)
					
					st.session_state[ 'googledrive_results' ] = docs or [ ]
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Google Drive request failed.' )
					st.exception( exc )
		
		with col_right:
			st.markdown( 'Results' )
			
			results = st.session_state.get( 'googledrive_results', [ ] )
			
			if not results:
				st.text( 'No results.' )
			else:
				for idx, doc in enumerate( results, start=1 ):
					title = ''
					if isinstance( doc, Document ):
						title = str( doc.metadata.get( 'name', '' ) ) if doc.metadata else ''
					
					label = f'Document {idx}' if not title else f'Document {idx}: {title}'
					
					with st.expander( label, expanded=False ):
						if isinstance( doc, Document ):
							if doc.metadata:
								meta_col1, meta_col2 = st.columns( 2 )
								
								with meta_col1:
									if 'name' in doc.metadata:
										st.markdown( f"**Name:** {doc.metadata.get( 'name', '' )}" )
									if 'id' in doc.metadata:
										st.markdown( f"**ID:** {doc.metadata.get( 'id', '' )}" )
								
								with meta_col2:
									if 'mimeType' in doc.metadata:
										st.markdown( f"**MIME Type:** {doc.metadata.get( 'mimeType', '' )}" )
									if 'modifiedTime' in doc.metadata:
										st.markdown( f"**Modified:** {doc.metadata.get( 'modifiedTime', '' )}" )
							
							st.text_area(
								'Content',
								value=doc.page_content or '',
								height=300,
								key=f'googledrive_doc_{idx}'
							)
							
							if doc.metadata:
								st.json( doc.metadata )
						else:
							st.write( doc )
	
	# -------- Wikipedia
	with st.expander( label='Wikipedia', expanded=False ):
		if 'wikipedia_results' not in st.session_state:
			st.session_state[ 'wikipedia_results' ] = [ ]
		
		if 'wikipedia_clear_request' not in st.session_state:
			st.session_state[ 'wikipedia_clear_request' ] = False
		
		if st.session_state.get( 'wikipedia_clear_request', False ):
			st.session_state[ 'wikipedia_query' ] = ''
			st.session_state[ 'wikipedia_language' ] = 'en'
			st.session_state[ 'wikipedia_max_docs' ] = 5
			st.session_state[ 'wikipedia_include_metadata' ] = False
			st.session_state[ 'wikipedia_results' ] = [ ]
			st.session_state[ 'wikipedia_clear_request' ] = False
		
		def _clear_wikipedia_state( ) -> None:
			st.session_state[ 'wikipedia_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			wiki_query = st.text_area(
				'Wikipedia Query',
				height=90,
				key='wikipedia_query',
				help=cfg.WIKIPEDIA,
				placeholder=(
						'Examples:\n'
						'Alan Turing\n'
						'History of machine learning\n'
						'Python (programming language)\n'
						'Battle of Midway'
				),
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				wiki_language = st.text_input(
					'Language Code',
					value=st.session_state.get( 'wikipedia_language', 'en' ),
					key='wikipedia_language',
					placeholder='en',
					help='Wikipedia language code, e.g. en, fr, de, ja.'
				)
			
			with c2:
				wiki_max_docs = st.number_input(
					'Max Docs',
					min_value=1,
					max_value=300,
					value=int( st.session_state.get( 'wikipedia_max_docs', 5 ) ),
					step=1,
					key='wikipedia_max_docs',
					help='Maximum number of Wikipedia documents to retrieve.'
				)
			
			wiki_include_metadata = st.checkbox(
				'Include All Metadata',
				value=st.session_state.get( 'wikipedia_include_metadata', False ),
				key='wikipedia_include_metadata',
				help='Include additional metadata fields when available.'
			)
			
			st.caption(
				'No API key is required for Wikipedia retrieval. '
				'Optional only: LANGSMITH_API_KEY for tracing.'
			)
			
			b1, b2 = st.columns( 2 )
			
			with b1:
				wiki_submit = st.button( 'Submit', key='wikipedia_submit' )
			
			with b2:
				st.button( 'Clear', key='wikipedia_clear', on_click=_clear_wikipedia_state )
			
			if wiki_submit:
				try:
					queries = [ q.strip( ) for q in (wiki_query or '').splitlines( ) if q.strip( ) ]
					
					if not queries:
						st.warning( 'No input provided.' )
					else:
						from fetchers import Wikipedia
						
						fetcher = Wikipedia(
							language=wiki_language or 'en',
							max_documents=int( wiki_max_docs ),
							include_metadata=bool( wiki_include_metadata ) )
						
						results: list[ Document ] = [ ]
						
						for q in queries:
							docs = fetcher.fetch(
								q,
								language=wiki_language or 'en',
								max_documents=int( wiki_max_docs ),
								include_metadata=bool( wiki_include_metadata ) )
							
							if isinstance( docs, list ):
								results.extend( docs )
						
						st.session_state[ 'wikipedia_results' ] = results
						st.rerun( )
				
				except Exception as exc:
					st.error( 'Wikipedia request failed.' )
					st.exception( exc )
		
		with col_right:
			st.markdown( 'Results' )
			
			results = st.session_state.get( 'wikipedia_results', [ ] )
			
			if not results:
				st.text( 'No results.' )
			else:
				for idx, doc in enumerate( results, start=1 ):
					title = ''
					if isinstance( doc, Document ):
						title = str( doc.metadata.get( 'title', '' ) ) if doc.metadata else ''
					
					label = f'Document {idx}' if not title else f'Document {idx}: {title}'
					
					with st.expander( 'Search Results', expanded=False ):
						if isinstance( doc, Document ):
							if doc.metadata:
								meta_col1, meta_col2 = st.columns( 2 )
								
								with meta_col1:
									if 'title' in doc.metadata:
										st.markdown( f"**Title:** {doc.metadata.get( 'title', '' )}" )
									if 'source' in doc.metadata:
										st.markdown( f"**Source:** {doc.metadata.get( 'source', '' )}" )
								
								with meta_col2:
									if 'categories' in doc.metadata:
										st.markdown( f"**Categories:** {doc.metadata.get( 'categories', '' )}" )
									if 'pageid' in doc.metadata:
										st.markdown( f"**Page ID:** {doc.metadata.get( 'pageid', '' )}" )
							
							st.text_area(
								'Content',
								value=doc.page_content or '',
								height=300,
								key=f'wikipedia_doc_{idx}' )
							
							if doc.metadata:
								st.json( doc.metadata )
						else:
							st.write( doc )
	
	# -------- Google Search
	with st.expander( label='Google Search', expanded=False ):
		if 'googlesearch_results' not in st.session_state:
			st.session_state[ 'googlesearch_results' ] = { }
		
		if 'googlesearch_clear_request' not in st.session_state:
			st.session_state[ 'googlesearch_clear_request' ] = False
		
		if st.session_state.get( 'googlesearch_clear_request', False ):
			st.session_state[ 'googlesearch_query' ] = ''
			st.session_state[ 'googlesearch_num_results' ] = 10
			st.session_state[ 'googlesearch_start' ] = 1
			st.session_state[ 'googlesearch_exact_terms' ] = ''
			st.session_state[ 'googlesearch_exclude_terms' ] = ''
			st.session_state[ 'googlesearch_file_type' ] = ''
			st.session_state[ 'googlesearch_date_restrict' ] = ''
			st.session_state[ 'googlesearch_gl' ] = ''
			st.session_state[ 'googlesearch_lr' ] = ''
			st.session_state[ 'googlesearch_safe' ] = 'off'
			st.session_state[ 'googlesearch_search_type' ] = ''
			st.session_state[ 'googlesearch_site_search' ] = ''
			st.session_state[ 'googlesearch_site_search_filter' ] = ''
			st.session_state[ 'googlesearch_sort' ] = ''
			st.session_state[ 'googlesearch_img_size' ] = ''
			st.session_state[ 'googlesearch_img_type' ] = ''
			st.session_state[ 'googlesearch_img_color_type' ] = ''
			st.session_state[ 'googlesearch_img_dominant_color' ] = ''
			st.session_state[ 'googlesearch_api_key' ] = ''
			st.session_state[ 'googlesearch_cse_id' ] = ''
			st.session_state[ 'googlesearch_timeout' ] = 10
			st.session_state[ 'googlesearch_results' ] = { }
			st.session_state[ 'googlesearch_clear_request' ] = False
		
		def _clear_googlesearch_state( ) -> None:
			st.session_state[ 'googlesearch_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			google_query = st.text_area(
				'Query',
				height=90,
				key='googlesearch_query',
				placeholder=(
						'Examples:\n'
						'site:epa.gov budget execution\n'
						'OpenAI GPT-5 reasoning\n'
						'filetype:pdf appropriations law'
				),
			)
			
			c1, c2, c3 = st.columns( 3 )
			
			with c1:
				google_num_results = st.number_input(
					'Results / Request',
					min_value=1,
					max_value=10,
					value=int( st.session_state.get( 'googlesearch_num_results', 10 ) ),
					step=1,
					key='googlesearch_num_results',
					help='Google Custom Search returns up to 10 results per request.'
				)
			
			with c2:
				google_start = st.number_input(
					'Start Index',
					min_value=1,
					max_value=91,
					value=int( st.session_state.get( 'googlesearch_start', 1 ) ),
					step=1,
					key='googlesearch_start'
				)
			
			with c3:
				google_timeout = st.number_input(
					'Timeout',
					min_value=1,
					max_value=60,
					value=int( st.session_state.get( 'googlesearch_timeout', 10 ) ),
					step=1,
					key='googlesearch_timeout'
				)
			
			c4, c5 = st.columns( 2 )
			
			with c4:
				google_exact_terms = st.text_input(
					'Exact Terms',
					value=st.session_state.get( 'googlesearch_exact_terms', '' ),
					key='googlesearch_exact_terms'
				)
			
			with c5:
				google_exclude_terms = st.text_input(
					'Exclude Terms',
					value=st.session_state.get( 'googlesearch_exclude_terms', '' ),
					key='googlesearch_exclude_terms'
				)
			
			c6, c7, c8 = st.columns( 3 )
			
			with c6:
				google_file_type = st.text_input(
					'File Type',
					value=st.session_state.get( 'googlesearch_file_type', '' ),
					key='googlesearch_file_type',
					placeholder='pdf'
				)
			
			with c7:
				google_date_restrict = st.text_input(
					'Date Restrict',
					value=st.session_state.get( 'googlesearch_date_restrict', '' ),
					key='googlesearch_date_restrict',
					placeholder='d7, m1, y1'
				)
			
			with c8:
				google_safe = st.selectbox(
					'Safe Search',
					options=[ 'off', 'active' ],
					index=[ 'off', 'active' ].index(
						st.session_state.get( 'googlesearch_safe', 'off' )
					),
					key='googlesearch_safe'
				)
			
			c9, c10, c11 = st.columns( 3 )
			
			with c9:
				google_gl = st.text_input(
					'Country (gl)',
					value=st.session_state.get( 'googlesearch_gl', '' ),
					key='googlesearch_gl',
					placeholder='us'
				)
			
			with c10:
				google_lr = st.text_input(
					'Language Restrict (lr)',
					value=st.session_state.get( 'googlesearch_lr', '' ),
					key='googlesearch_lr',
					placeholder='lang_en'
				)
			
			with c11:
				google_search_type = st.selectbox(
					'Search Type',
					options=[ '', 'image' ],
					index=[ '', 'image' ].index(
						st.session_state.get( 'googlesearch_search_type', '' )
					),
					key='googlesearch_search_type'
				)
			
			c12, c13 = st.columns( 2 )
			
			with c12:
				google_site_search = st.text_input(
					'Site Search',
					value=st.session_state.get( 'googlesearch_site_search', '' ),
					key='googlesearch_site_search',
					placeholder='example.gov'
				)
			
			with c13:
				google_site_search_filter = st.selectbox(
					'Site Search Filter',
					options=[ '', 'i', 'e' ],
					index=[ '', 'i', 'e' ].index(
						st.session_state.get( 'googlesearch_site_search_filter', '' )
					),
					key='googlesearch_site_search_filter',
					help='i=include, e=exclude'
				)
			
			google_sort = st.text_input(
				'Sort',
				value=st.session_state.get( 'googlesearch_sort', '' ),
				key='googlesearch_sort',
				placeholder='date'
			)
			
			c14, c15 = st.columns( 2 )
			
			with c14:
				google_img_size = st.selectbox(
					'Image Size',
					options=[ '', 'icon', 'small', 'medium', 'large', 'xlarge', 'xxlarge', 'huge' ],
					index=[ '', 'icon', 'small', 'medium', 'large', 'xlarge', 'xxlarge',
					        'huge' ].index(
						st.session_state.get( 'googlesearch_img_size', '' )
					),
					key='googlesearch_img_size',
					disabled=(google_search_type != 'image')
				)
			
			with c15:
				google_img_type = st.selectbox(
					'Image Type',
					options=[ '', 'clipart', 'face', 'lineart', 'stock', 'photo', 'animated' ],
					index=[ '', 'clipart', 'face', 'lineart', 'stock', 'photo', 'animated' ].index(
						st.session_state.get( 'googlesearch_img_type', '' )
					),
					key='googlesearch_img_type',
					disabled=(google_search_type != 'image')
				)
			
			c16, c17 = st.columns( 2 )
			
			with c16:
				google_img_color_type = st.selectbox(
					'Image Color Type',
					options=[ '', 'color', 'gray', 'mono', 'trans' ],
					index=[ '', 'color', 'gray', 'mono', 'trans' ].index(
						st.session_state.get( 'googlesearch_img_color_type', '' )
					),
					key='googlesearch_img_color_type',
					disabled=(google_search_type != 'image')
				)
			
			with c17:
				google_img_dominant_color = st.selectbox(
					'Image Dominant Color',
					options=[ '', 'black', 'blue', 'brown', 'gray', 'green', 'orange',
					          'pink', 'purple', 'red', 'teal', 'white', 'yellow' ],
					index=[ '', 'black', 'blue', 'brown', 'gray', 'green', 'orange',
					        'pink', 'purple', 'red', 'teal', 'white', 'yellow' ].index(
						st.session_state.get( 'googlesearch_img_dominant_color', '' )
					),
					key='googlesearch_img_dominant_color',
					disabled=(google_search_type != 'image')
				)
			
			c18, c19 = st.columns( 2 )
			
			with c18:
				google_api_key = st.text_input(
					'API Key',
					value='',
					type='password',
					key='googlesearch_api_key',
					placeholder='Uses GOOGLE_API_KEY when left blank.'
				)
			
			with c19:
				google_cse_id = st.text_input(
					'CSE ID',
					value='',
					key='googlesearch_cse_id',
					placeholder='Uses GOOGLE_CSE_ID when left blank.'
				)
			
			st.caption(
				'Required keys: GOOGLE_API_KEY and GOOGLE_CSE_ID. '
				'Endpoint updated to customsearch.googleapis.com/customsearch/v1.'
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				google_submit = st.button( 'Submit', key='googlesearch_submit' )
			with b2:
				st.button( 'Clear', key='googlesearch_clear', on_click=_clear_googlesearch_state )
		
		with col_right:
			st.markdown( 'Results' )
			
			if google_submit:
				try:
					f = GoogleSearch( )
					result = f.fetch(
						keywords=google_query,
						results=int( google_num_results ),
						start=int( google_start ),
						exact_terms=google_exact_terms,
						exclude_terms=google_exclude_terms,
						file_type=google_file_type,
						date_restrict=google_date_restrict,
						gl=google_gl,
						lr=google_lr,
						safe=google_safe,
						search_type=google_search_type,
						site_search=google_site_search,
						site_search_filter=google_site_search_filter,
						sort=google_sort,
						img_size=google_img_size,
						img_type=google_img_type,
						img_color_type=google_img_color_type,
						img_dominant_color=google_img_dominant_color,
						time=int( google_timeout ),
						api_key=(google_api_key or None),
						cse_id=(google_cse_id or None)
					)
					
					st.session_state[ 'googlesearch_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Google Search request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'googlesearch_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				queries = result.get( 'queries', { } ) if isinstance( result, dict ) else { }
				search_info = result.get( 'searchInformation', { } ) if isinstance( result, dict ) else { }
				items = result.get( 'items', [ ] ) if isinstance( result, dict ) else [ ]
				
				if queries or search_info:
					st.markdown( '#### Search Metadata' )
					
					meta_summary: Dict[ str, Any ] = { }
					
					if isinstance( search_info, dict ):
						for key in [ 'searchTime', 'formattedSearchTime', 'totalResults',
						             'formattedTotalResults' ]:
							if key in search_info:
								meta_summary[ key ] = search_info.get( key )
					
					if isinstance( queries, dict ) and 'request' in queries:
						requests = queries.get( 'request', [ ] )
						if isinstance( requests, list ) and requests:
							request_item = requests[ 0 ]
							if isinstance( request_item, dict ):
								for key in [ 'searchTerms', 'count', 'startIndex', 'inputEncoding',
								             'outputEncoding' ]:
									if key in request_item:
										meta_summary[ key ] = request_item.get( key )
					
					if meta_summary:
						st.json( meta_summary )
					
					with st.expander( 'Raw Search Metadata', expanded=False ):
						st.json(
							{
									'queries': queries,
									'searchInformation': search_info,
							}
						)
				
				if not items:
					st.info( 'No results returned.' )
				else:
					for idx, item in enumerate( items, start=1 ):
						title = item.get( 'title', f'Result {idx}' )
						
						with st.container( border=True ):
							st.markdown( f'**{idx}. {title}**' )
							
							link_value = item.get( 'link', '' )
							display_link = item.get( 'displayLink', '' )
							snippet_value = item.get( 'snippet', '' )
							
							meta_parts: List[ str ] = [ ]
							if display_link:
								meta_parts.append( f'Domain: `{display_link}`' )
							
							pagemap = item.get( 'pagemap', { } ) if isinstance( item, dict ) else { }
							if isinstance( pagemap, dict ):
								if 'metatags' in pagemap:
									meta_parts.append( 'Has metatags' )
								if 'cse_image' in pagemap:
									meta_parts.append( 'Has image' )
							
							if meta_parts:
								st.caption( ' | '.join( meta_parts ) )
							
							if link_value:
								st.markdown( f'**Link:** {link_value}' )
							
							if snippet_value:
								st.write( str( snippet_value ) )
							
							image_url = ''
							if isinstance( pagemap, dict ):
								cse_images = pagemap.get( 'cse_image', [ ] )
								if isinstance( cse_images, list ) and cse_images:
									first_img = cse_images[ 0 ]
									if isinstance( first_img, dict ):
										image_url = first_img.get( 'src', '' )
							
							if image_url and google_search_type == 'image':
								try:
									st.image( image_url, use_container_width=True )
								except Exception:
									pass
							
							with st.expander( 'Raw Item', expanded=False ):
								st.json( item )
	
	# -------- Google Geocoding
	with st.expander( label='Geocoding', expanded=False ):
		if 'googlegeocoding_results' not in st.session_state:
			st.session_state[ 'googlegeocoding_results' ] = { }
		
		if 'googlegeocoding_clear_request' not in st.session_state:
			st.session_state[ 'googlegeocoding_clear_request' ] = False
		
		if st.session_state.get( 'googlegeocoding_clear_request', False ):
			st.session_state[ 'googlegeocoding_mode' ] = 'forward'
			st.session_state[ 'googlegeocoding_query' ] = ''
			st.session_state[ 'googlegeocoding_latitude' ] = 38.8895
			st.session_state[ 'googlegeocoding_longitude' ] = -77.0353
			st.session_state[ 'googlegeocoding_place_id' ] = ''
			st.session_state[ 'googlegeocoding_language' ] = 'en'
			st.session_state[ 'googlegeocoding_region' ] = ''
			st.session_state[ 'googlegeocoding_result_type' ] = ''
			st.session_state[ 'googlegeocoding_location_type' ] = ''
			st.session_state[ 'googlegeocoding_api_key' ] = ''
			st.session_state[ 'googlegeocoding_timeout' ] = 10
			st.session_state[ 'googlegeocoding_results' ] = { }
			st.session_state[ 'googlegeocoding_clear_request' ] = False
		
		def _clear_googlegeocoding_state( ) -> None:
			st.session_state[ 'googlegeocoding_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			googlegeocoding_mode = st.selectbox(
				'Mode',
				options=[ 'forward', 'reverse', 'place' ],
				index=[ 'forward', 'reverse', 'place' ].index(
					st.session_state.get( 'googlegeocoding_mode', 'forward' )
				),
				key='googlegeocoding_mode',
				help='forward = address search; reverse = lat/lng to address; place = place_id lookup.'
			)
			
			googlegeocoding_query = st.text_area(
				'Address Query',
				height=80,
				key='googlegeocoding_query',
				placeholder=(
						'Examples:\n'
						'1600 Amphitheatre Parkway, Mountain View, CA\n'
						'Arlington, VA\n'
						'10 Downing Street, London'
				),
				disabled=(googlegeocoding_mode != 'forward')
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				googlegeocoding_latitude = st.number_input(
					'Latitude',
					min_value=-90.0,
					max_value=90.0,
					value=float( st.session_state.get( 'googlegeocoding_latitude', 38.8895 ) ),
					step=0.0001,
					format='%.6f',
					key='googlegeocoding_latitude',
					disabled=(googlegeocoding_mode != 'reverse')
				)
			
			with c2:
				googlegeocoding_longitude = st.number_input(
					'Longitude',
					min_value=-180.0,
					max_value=180.0,
					value=float( st.session_state.get( 'googlegeocoding_longitude', -77.0353 ) ),
					step=0.0001,
					format='%.6f',
					key='googlegeocoding_longitude',
					disabled=(googlegeocoding_mode != 'reverse')
				)
			
			googlegeocoding_place_id = st.text_input(
				'Place ID',
				value=st.session_state.get( 'googlegeocoding_place_id', '' ),
				key='googlegeocoding_place_id',
				placeholder='ChIJ2eUgeAK6j4ARbn5u_wAGqWA',
				disabled=(googlegeocoding_mode != 'place')
			)
			
			c3, c4 = st.columns( 2 )
			
			with c3:
				googlegeocoding_language = st.text_input(
					'Language',
					value=st.session_state.get( 'googlegeocoding_language', 'en' ),
					key='googlegeocoding_language',
					placeholder='en'
				)
			
			with c4:
				googlegeocoding_region = st.text_input(
					'Region Bias',
					value=st.session_state.get( 'googlegeocoding_region', '' ),
					key='googlegeocoding_region',
					placeholder='us',
					disabled=(googlegeocoding_mode == 'reverse')
				)
			
			c5, c6 = st.columns( 2 )
			
			with c5:
				googlegeocoding_result_type = st.text_input(
					'Result Type',
					value=st.session_state.get( 'googlegeocoding_result_type', '' ),
					key='googlegeocoding_result_type',
					placeholder='street_address|premise',
					disabled=(googlegeocoding_mode != 'reverse')
				)
			
			with c6:
				googlegeocoding_location_type = st.text_input(
					'Location Type',
					value=st.session_state.get( 'googlegeocoding_location_type', '' ),
					key='googlegeocoding_location_type',
					placeholder='ROOFTOP|GEOMETRIC_CENTER',
					disabled=(googlegeocoding_mode != 'reverse')
				)
			
			c7, c8 = st.columns( 2 )
			
			with c7:
				googlegeocoding_api_key = st.text_input(
					'API Key',
					value='',
					type='password',
					key='googlegeocoding_api_key',
					placeholder='Uses GOOGLE_API_KEY when left blank.'
				)
			
			with c8:
				googlegeocoding_timeout = st.number_input(
					'Timeout',
					min_value=1,
					max_value=60,
					value=int( st.session_state.get( 'googlegeocoding_timeout', 10 ) ),
					step=1,
					key='googlegeocoding_timeout'
				)
			
			st.caption(
				'Google Geocoding requires billing plus a Google API key. '
				'Result filters apply to reverse geocoding only.'
			)
			
			b1, b2 = st.columns( 2 )
			
			with b1:
				googlegeocoding_submit = st.button(
					'Submit',
					key='googlegeocoding_submit',
					use_container_width=True
				)
			
			with b2:
				st.button(
					'Clear',
					key='googlegeocoding_clear',
					on_click=_clear_googlegeocoding_state,
					use_container_width=True
				)
		
		with col_right:
			st.markdown( 'Results' )
			
			if googlegeocoding_submit:
				try:
					f = GoogleGeocoding( )
					result = f.fetch(
						mode=str( googlegeocoding_mode ),
						query=str( googlegeocoding_query ),
						latitude=float( googlegeocoding_latitude ),
						longitude=float( googlegeocoding_longitude ),
						place_id=str( googlegeocoding_place_id ),
						language=str( googlegeocoding_language or 'en' ).strip( ),
						region=str( googlegeocoding_region or '' ).strip( ),
						result_type=str( googlegeocoding_result_type or '' ).strip( ),
						location_type=str( googlegeocoding_location_type or '' ).strip( ),
						time=int( googlegeocoding_timeout ),
						api_key=(googlegeocoding_api_key or None)
					)
					
					st.session_state[ 'googlegeocoding_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Google Geocoding request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'googlegeocoding_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				st.markdown( '#### Request Metadata' )
				st.json(
					{
							'mode': result.get( 'mode', '' ),
							'url': result.get( 'url', '' ),
							'params': result.get( 'params', { } ),
							'status': result.get( 'status', '' ),
					}
				)
				
				results_list = result.get( 'results', [ ] ) if isinstance( result, dict ) else [ ]
				
				if not results_list:
					st.info( 'No geocoding results returned.' )
				else:
					for idx, item in enumerate( results_list, start=1 ):
						formatted_address = item.get( 'formatted_address', f'Result {idx}' )
						place_id_value = item.get( 'place_id', '' )
						types_value = item.get( 'types', [ ] )
						
						geometry = item.get( 'geometry', { } ) if isinstance( item, dict ) else { }
						location = geometry.get( 'location', { } ) if isinstance( geometry, dict ) else { }
						
						with st.container( border=True ):
							st.markdown( f'**{idx}. {formatted_address}**' )
							
							meta_parts: List[ str ] = [ ]
							
							if place_id_value:
								meta_parts.append( f'Place ID: `{place_id_value}`' )
							
							if isinstance( types_value, list ) and types_value:
								meta_parts.append( f"Types: `{', '.join( types_value[ :4 ] )}`" )
							
							if meta_parts:
								st.caption( ' | '.join( meta_parts ) )
							
							if isinstance( location, dict ):
								lat_value = location.get( 'lat', '' )
								lng_value = location.get( 'lng', '' )
								if str( lat_value ).strip( ) or str( lng_value ).strip( ):
									st.write( f'Coordinates: {lat_value}, {lng_value}' )
							
							address_components = item.get( 'address_components', [ ] )
							if isinstance( address_components, list ) and address_components:
								component_rows: List[ Dict[ str, Any ] ] = [ ]
								for component in address_components:
									if isinstance( component, dict ):
										component_rows.append(
											{
													'long_name': component.get( 'long_name', '' ),
													'short_name': component.get( 'short_name', '' ),
													'types': ', '.join( component.get( 'types', [ ] ) )
											}
										)
								
								if component_rows:
									with st.expander( 'Address Components', expanded=False ):
										st.dataframe(
											pd.DataFrame( component_rows ),
											use_container_width=True,
											hide_index=True
										)
							
							with st.expander( 'Raw Item', expanded=False ):
								st.json( item )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- Naval Observatory
	with st.expander( label='US Naval Observatory', expanded=False ):
		if 'navalobservatory_results' not in st.session_state:
			st.session_state[ 'navalobservatory_results' ] = { }
		
		if 'navalobservatory_clear_request' not in st.session_state:
			st.session_state[ 'navalobservatory_clear_request' ] = False
		
		if st.session_state.get( 'navalobservatory_clear_request', False ):
			st.session_state[ 'navalobservatory_date' ] = dt.date.today( )
			st.session_state[ 'navalobservatory_time' ] = dt.time( 12, 0 )
			st.session_state[ 'navalobservatory_latitude' ] = 38.9072
			st.session_state[ 'navalobservatory_longitude' ] = -77.0369
			st.session_state[ 'navalobservatory_location_label' ] = ''
			st.session_state[ 'navalobservatory_timeout' ] = 20
			st.session_state[ 'navalobservatory_results' ] = { }
			st.session_state[ 'navalobservatory_clear_request' ] = False
		
		def _clear_navalobservatory_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the Naval Observatory expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'navalobservatory_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			naval_date = st.date_input(
				'Date',
				value=st.session_state.get(
					'navalobservatory_date',
					dt.date.today( )
				),
				key='navalobservatory_date',
				help='USNO date parameter in YYYY-MM-DD format.'
			)
			
			naval_time = st.time_input(
				'Time (UTC)',
				value=st.session_state.get(
					'navalobservatory_time',
					dt.time( 12, 0 )
				),
				key='navalobservatory_time',
				help='USNO time parameter in 24-hour format.'
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				naval_latitude = st.number_input(
					'Latitude',
					min_value=-90.0,
					max_value=90.0,
					value=float(
						st.session_state.get( 'navalobservatory_latitude', 38.9072 )
					),
					step=0.0001,
					format='%.6f',
					key='navalobservatory_latitude',
					help='Decimal degrees. North positive.'
				)
			
			with c2:
				naval_longitude = st.number_input(
					'Longitude',
					min_value=-180.0,
					max_value=180.0,
					value=float(
						st.session_state.get( 'navalobservatory_longitude', -77.0369 )
					),
					step=0.0001,
					format='%.6f',
					key='navalobservatory_longitude',
					help='Decimal degrees. East positive, west negative.'
				)
			
			naval_location_label = st.text_input(
				'Location Label',
				value=st.session_state.get( 'navalobservatory_location_label', '' ),
				key='navalobservatory_location_label',
				placeholder='Example: Washington, DC'
			)
			
			naval_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'navalobservatory_timeout', 20 ) ),
				step=1,
				key='navalobservatory_timeout'
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				naval_submit = st.button(
					'Submit',
					key='navalobservatory_submit',
					use_container_width=True
				)
			with b2:
				naval_clear = st.button(
					'Clear',
					key='navalobservatory_clear',
					on_click=_clear_navalobservatory_state,
					use_container_width=True
				)
		
		with col_right:
			naval_output = st.empty( )
		
		if naval_submit:
			try:
				f = NavalObservatory( )
				
				result = f.fetch(
					mode='celnav',
					date_value=naval_date.strftime( '%Y-%m-%d' ),
					time_value=naval_time.strftime( '%H:%M:%S' ),
					latitude=float( naval_latitude ),
					longitude=float( naval_longitude ),
					location_label=naval_location_label,
					time=int( naval_timeout )
				)
				
				st.session_state[ 'navalobservatory_results' ] = result or { }
				st.rerun( )
			
			except Exception as exc:
				st.error( str( exc ) )
				
		result = st.session_state.get( 'navalobservatory_results', { } )
		
		if not result:
			naval_output.text( 'No results.' )
		else:
			data = result.get( 'data', { } ) if isinstance( result, dict ) else { }
			params = result.get( 'params', { } ) if isinstance( result, dict ) else { }
			
			with col_right:
				st.markdown( '#### Request Metadata' )
				st.json(
					{
							'mode': result.get( 'mode', '' ),
							'url': result.get( 'url', '' ),
							'params': params,
							'location_label': result.get( 'location_label', '' ),
					}
				)
				
				if not data:
					st.info( 'No results returned.' )
				else:
					st.markdown( '#### Observation Summary' )
					
					c1, c2 = st.columns( 2 )
					
					with c1:
						if params.get( 'date', '' ):
							st.markdown( f"**Date:** {params.get( 'date', '' )}" )
						if params.get( 'time', '' ):
							st.markdown( f"**Time:** {params.get( 'time', '' )}" )
						if result.get( 'location_label', '' ):
							st.markdown(
								f"**Location Label:** {result.get( 'location_label', '' )}"
							)
					
					with c2:
						if params.get( 'coords', '' ):
							st.markdown( f"**Coordinates:** {params.get( 'coords', '' )}" )
					
					bodies: List[ Dict[ str, Any ] ] = [ ]
					
					if isinstance( data, dict ):
						for key in [ 'data', 'bodies', 'results', 'celestialBodies',
						             'celestial_bodies' ]:
							value = data.get( key, None )
							if isinstance( value, list ):
								bodies = [ item for item in value if isinstance( item, dict ) ]
								break
					
					if bodies:
						st.markdown( '#### Celestial Bodies' )
						df_bodies = pd.DataFrame( bodies )
						if not df_bodies.empty:
							st.dataframe( df_bodies, use_container_width=True, hide_index=True )
						else:
							st.info( 'No displayable celestial body rows were found.' )
					else:
						top_fields = { }
						
						if isinstance( data, dict ):
							for key in [
									'gha', 'dec', 'hc', 'zn', 'altitude', 'azimuth',
									'sunrise', 'sunset', 'moonrise', 'moonset'
							]:
								if key in data:
									top_fields[ key ] = data.get( key )
						
						if top_fields:
							st.markdown( '#### Key Values' )
							st.json( top_fields )
						else:
							st.markdown( '#### Result' )
							st.json( data )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- Open Science
	with st.expander( label='Open Science', expanded=False ):
		if 'openscience_results' not in st.session_state:
			st.session_state[ 'openscience_results' ] = { }
		
		if 'openscience_clear_request' not in st.session_state:
			st.session_state[ 'openscience_clear_request' ] = False
		
		if st.session_state.get( 'openscience_clear_request', False ):
			st.session_state[ 'openscience_mode' ] = 'dataset'
			st.session_state[ 'openscience_accession' ] = ''
			st.session_state[ 'openscience_query' ] = ''
			st.session_state[ 'openscience_format' ] = 'json'
			st.session_state[ 'openscience_timeout' ] = 20
			st.session_state[ 'openscience_results' ] = { }
			st.session_state[ 'openscience_clear_request' ] = False
		
		def _clear_openscience_state( ) -> None:
			st.session_state[ 'openscience_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			openscience_mode = st.selectbox(
				'Mode',
				options=[ 'dataset', 'metadata', 'assays', 'data' ],
				index=[ 'dataset', 'metadata', 'assays', 'data' ].index(
					st.session_state.get( 'openscience_mode', 'dataset' )
				),
				key='openscience_mode',
				help=(
						'dataset = fetch dataset metadata by accession; '
						'metadata/assays/data = query the corresponding OSDR API endpoint.'
				)
			)
			
			openscience_accession = st.text_input(
				'Dataset Accession',
				value=st.session_state.get( 'openscience_accession', '' ),
				key='openscience_accession',
				placeholder='Example: OSD-48'
			)
			
			openscience_query = st.text_area(
				'Query',
				value=st.session_state.get( 'openscience_query', '' ),
				height=120,
				key='openscience_query',
				placeholder=(
						'Example: (id.accession=OSD-48) '
						'OR study.characteristics.organism=Mus musculus'
				)
			)
			
			openscience_format = st.selectbox(
				'Format',
				options=[ 'json', 'csv', 'tsv', 'browser' ],
				index=[ 'json', 'csv', 'tsv', 'browser' ].index(
					st.session_state.get( 'openscience_format', 'json' )
				),
				key='openscience_format'
			)
			
			openscience_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'openscience_timeout', 20 ) ),
				step=1,
				key='openscience_timeout'
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				openscience_submit = st.button(
					'Submit',
					key='openscience_submit',
					use_container_width=True
				)
			with b2:
				st.button(
					'Clear',
					key='openscience_clear',
					on_click=_clear_openscience_state,
					use_container_width=True
				)
		
		with col_right:
			st.markdown( 'Results' )
			
			if openscience_submit:
				try:
					f = OpenScience( )
					
					result = f.fetch(
						mode=str( openscience_mode ),
						query=str( openscience_query ),
						accession=str( openscience_accession ),
						format_value=str( openscience_format ),
						time=int( openscience_timeout )
					)
					
					st.session_state[ 'openscience_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( str( exc ) )
			
			result = st.session_state.get( 'openscience_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				mode_value = result.get( 'mode', '' ) if isinstance( result, dict ) else ''
				data = result.get( 'data', { } ) if isinstance( result, dict ) else { }
				params = result.get( 'params', { } ) if isinstance( result, dict ) else { }
				
				st.markdown( '#### Request Metadata' )
				st.json(
					{
							'mode': mode_value,
							'url': result.get( 'url', '' ),
							'params': params,
					}
				)
				
				if mode_value == 'dataset' and isinstance( data, dict ) and data:
					title_value = (
							data.get( 'title' )
							or data.get( 'name' )
							or data.get( 'accession' )
							or params.get( 'accession', '' )
							or 'Dataset'
					)
					
					st.markdown( f'### {title_value}' )
					
					meta_fields: Dict[ str, Any ] = { }
					for key in [
							'accession',
							'identifier',
							'organism',
							'platform',
							'assay',
							'project',
							'study'
					]:
						if key in data:
							meta_fields[ key ] = data.get( key )
					
					if meta_fields:
						st.json( meta_fields )
					
					for key in [ 'summary', 'description', 'abstract' ]:
						if key in data and str( data.get( key ) ).strip( ):
							st.markdown( '#### Description' )
							st.write( str( data.get( key ) ) )
							break
					
					with st.expander( 'Raw Dataset JSON', expanded=False ):
						st.json( data )
				
				elif isinstance( data, list ) and data:
					df_os = pd.DataFrame( data )
					if not df_os.empty:
						st.markdown( f'#### Result Rows ({len( df_os )})' )
						st.dataframe( df_os, use_container_width=True, hide_index=True )
					else:
						st.json( data )
				
				elif isinstance( data, dict ) and data:
					table_candidates: List[ Dict[ str, Any ] ] = [ ]
					for key in [ 'results', 'items', 'rows', 'data' ]:
						value = data.get( key, None )
						if isinstance( value, list ) and value:
							table_candidates = [ item for item in value if
							                     isinstance( item, dict ) ]
							break
					
					if table_candidates:
						df_os = pd.DataFrame( table_candidates )
						if not df_os.empty:
							st.markdown( f'#### Result Rows ({len( df_os )})' )
							st.dataframe( df_os, use_container_width=True, hide_index=True )
						else:
							st.json( data )
					else:
						st.markdown( '#### Result' )
						st.json( data )
				
				elif data:
					st.text_area(
						'Output',
						value=str( data ),
						height=320
					)
				else:
					st.info( 'No results returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- Gov Info
	with st.expander( label='Gov Info', expanded=False ):
		if 'govinfo_results' not in st.session_state:
			st.session_state[ 'govinfo_results' ] = { }
		
		if 'govinfo_clear_request' not in st.session_state:
			st.session_state[ 'govinfo_clear_request' ] = False
		
		if st.session_state.get( 'govinfo_clear_request', False ):
			st.session_state[ 'govinfo_mode' ] = 'search'
			st.session_state[ 'govinfo_query' ] = ''
			st.session_state[ 'govinfo_page_size' ] = 10
			st.session_state[ 'govinfo_offset_mark' ] = '*'
			st.session_state[ 'govinfo_sort_field' ] = 'score'
			st.session_state[ 'govinfo_sort_order' ] = 'DESC'
			st.session_state[ 'govinfo_package_id' ] = ''
			st.session_state[ 'govinfo_collection' ] = ''
			st.session_state[ 'govinfo_start_date' ] = '2025-01-01T00:00:00Z'
			st.session_state[ 'govinfo_timeout' ] = 20
			st.session_state[ 'govinfo_results' ] = { }
			st.session_state[ 'govinfo_clear_request' ] = False
		
		def _clear_govinfo_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the Gov Info expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'govinfo_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			govinfo_mode = st.selectbox(
				'Mode',
				options=[ 'search', 'package_summary', 'collection' ],
				index=[ 'search', 'package_summary', 'collection' ].index(
					st.session_state.get( 'govinfo_mode', 'search' )
				),
				key='govinfo_mode',
				help=(
						'search = GovInfo Search Service; '
						'package_summary = package details by package ID; '
						'collection = browse a collection since an ISO timestamp.'
				)
			)
			
			govinfo_query = st.text_area(
				'Query',
				value=st.session_state.get( 'govinfo_query', '' ),
				height=120,
				key='govinfo_query',
				placeholder=(
						'Example: collection:BILLS AND congress:118 '
						'AND title:"appropriations"'
				)
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				govinfo_page_size = st.number_input(
					'Page Size',
					min_value=1,
					max_value=1000,
					value=int( st.session_state.get( 'govinfo_page_size', 10 ) ),
					step=1,
					key='govinfo_page_size'
				)
			
			with c2:
				govinfo_offset_mark = st.text_input(
					'Offset Mark',
					value=st.session_state.get( 'govinfo_offset_mark', '*' ),
					key='govinfo_offset_mark',
					placeholder='*'
				)
			
			c3, c4 = st.columns( 2 )
			
			with c3:
				govinfo_sort_field = st.selectbox(
					'Sort Field',
					options=[ 'score', 'lastModified' ],
					index=[ 'score', 'lastModified' ].index(
						st.session_state.get( 'govinfo_sort_field', 'score' )
					),
					key='govinfo_sort_field'
				)
			
			with c4:
				govinfo_sort_order = st.selectbox(
					'Sort Order',
					options=[ 'DESC', 'ASC' ],
					index=[ 'DESC', 'ASC' ].index(
						st.session_state.get( 'govinfo_sort_order', 'DESC' )
					),
					key='govinfo_sort_order'
				)
			
			govinfo_package_id = st.text_input(
				'Package ID',
				value=st.session_state.get( 'govinfo_package_id', '' ),
				key='govinfo_package_id',
				placeholder='Example: CREC-2018-10-10'
			)
			
			c5, c6 = st.columns( 2 )
			
			with c5:
				govinfo_collection = st.text_input(
					'Collection',
					value=st.session_state.get( 'govinfo_collection', '' ),
					key='govinfo_collection',
					placeholder='Example: CREC'
				)
			
			with c6:
				govinfo_start_date = st.text_input(
					'Start Date (ISO)',
					value=st.session_state.get(
						'govinfo_start_date',
						'2025-01-01T00:00:00Z'
					),
					key='govinfo_start_date',
					placeholder='YYYY-MM-DDTHH:MM:SSZ'
				)
			
			govinfo_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'govinfo_timeout', 20 ) ),
				step=1,
				key='govinfo_timeout'
			)
			
			b1, b2 = st.columns( 2 )
			
			with b1:
				govinfo_submit = st.button(
					'Submit',
					key='govinfo_submit',
					use_container_width=True
				)
			
			with b2:
				govinfo_clear = st.button(
					'Clear',
					key='govinfo_clear',
					on_click=_clear_govinfo_state,
					use_container_width=True
				)
		
		with col_right:
			govinfo_output = st.empty( )
		
		result = st.session_state.get( 'govinfo_results', { } )
		
		if govinfo_submit:
			try:
				f = GovData( )
				
				result = f.fetch(
					mode=str( govinfo_mode ),
					query=str( govinfo_query ),
					page_size=int( govinfo_page_size ),
					offset_mark=str( govinfo_offset_mark ),
					sort_field=str( govinfo_sort_field ),
					sort_order=str( govinfo_sort_order ),
					package_id=str( govinfo_package_id ),
					collection=str( govinfo_collection ),
					start_date=str( govinfo_start_date ),
					time=int( govinfo_timeout )
				)
				
				st.session_state[ 'govinfo_results' ] = result or { }
				st.rerun( )
			
			except Exception as exc:
				st.error( str( exc ) )
				
				result = st.session_state.get( 'govinfo_results', { } )
		
		if not result:
			govinfo_output.text( 'No results.' )
		else:
			mode_value = result.get( 'mode', '' ) if isinstance( result, dict ) else ''
			data = result.get( 'data', { } ) if isinstance( result, dict ) else { }
			params = result.get( 'params', { } ) if isinstance( result, dict ) else { }
			payload = result.get( 'payload', { } ) if isinstance( result, dict ) else { }
			
			with col_right:
				st.markdown( '#### Request Metadata' )
				st.json(
					{
							'mode': mode_value,
							'url': result.get( 'url', '' ),
							'params': params,
							'payload': payload,
					}
				)
				
				items: List[ Dict[ str, Any ] ] = [ ]
				
				if isinstance( data, dict ):
					for key in [ 'packages', 'results', 'items' ]:
						value = data.get( key, None )
						if isinstance( value, list ):
							items = [ item for item in value if isinstance( item, dict ) ]
							break
				elif isinstance( data, list ):
					items = [ item for item in data if isinstance( item, dict ) ]
				
				if mode_value == 'package_summary' and isinstance( data, dict ) and data:
					st.markdown( '#### Package Summary' )
					
					title_value = (
							data.get( 'title' )
							or data.get( 'packageTitle' )
							or data.get( 'packageId' )
							or 'Package'
					)
					
					st.markdown( f'### {title_value}' )
					
					meta_parts: List[ str ] = [ ]
					for key in [ 'packageId', 'collectionCode', 'lastModified', 'dateIssued' ]:
						if key in data and str( data.get( key ) ).strip( ):
							meta_parts.append( f'{key}: `{data.get( key )}`' )
					
					if meta_parts:
						st.caption( ' | '.join( meta_parts ) )
					
					for text_key in [ 'summary', 'description' ]:
						if text_key in data and str( data.get( text_key ) ).strip( ):
							st.write( str( data.get( text_key ) ) )
							break
					
					with st.expander( 'Raw Package JSON', expanded=False ):
						st.json( data )
				
				elif items:
					st.markdown( '#### Results' )
					
					for index, item in enumerate( items, start=1 ):
						title_value = (
								item.get( 'title' )
								or item.get( 'packageTitle' )
								or item.get( 'packageId' )
								or item.get( 'granuleId' )
								or f'Result {index}'
						)
						
						package_value = (
								item.get( 'packageId' )
								or item.get( 'granuleId' )
								or item.get( 'id' )
								or ''
						)
						
						collection_value = (
								item.get( 'collectionCode' )
								or item.get( 'collectionName' )
								or item.get( 'collection' )
								or ''
						)
						
						date_value = (
								item.get( 'lastModified' )
								or item.get( 'dateIssued' )
								or item.get( 'publishDate' )
								or ''
						)
						
						summary_value = (
								item.get( 'summary' )
								or item.get( 'description' )
								or item.get( 'snippet' )
								or ''
						)
						
						with st.container( border=True ):
							st.markdown( f'**{index}. {title_value}**' )
							
							meta_parts: List[ str ] = [ ]
							
							if package_value:
								meta_parts.append( f'ID: `{package_value}`' )
							
							if collection_value:
								meta_parts.append( f'Collection: `{collection_value}`' )
							
							if date_value:
								meta_parts.append( f'Date: `{date_value}`' )
							
							if meta_parts:
								st.caption( ' | '.join( meta_parts ) )
							
							if summary_value:
								st.write( str( summary_value ) )
							else:
								st.caption( 'No summary available.' )
							
							with st.expander( 'Raw Item', expanded=False ):
								st.json( item )
				
				elif isinstance( data, dict ) and data:
					st.markdown( '#### Result' )
					st.json( data )
				elif data:
					st.markdown( '#### Result' )
					st.text_area(
						'Output',
						value=str( data ),
						height=320
					)
				else:
					st.info( 'No results returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- Congress
	with st.expander( label='US Congress', expanded=False ):
		if 'congress_results' not in st.session_state:
			st.session_state[ 'congress_results' ] = { }
		
		if 'congress_clear_request' not in st.session_state:
			st.session_state[ 'congress_clear_request' ] = False
		
		if st.session_state.get( 'congress_clear_request', False ):
			st.session_state[ 'congress_mode' ] = 'congresses'
			st.session_state[ 'congress_number' ] = 119
			st.session_state[ 'congress_bill_type' ] = ''
			st.session_state[ 'congress_bill_number' ] = 0
			st.session_state[ 'congress_law_type' ] = ''
			st.session_state[ 'congress_law_number' ] = 0
			st.session_state[ 'congress_report_type' ] = ''
			st.session_state[ 'congress_report_number' ] = 0
			st.session_state[ 'congress_offset' ] = 0
			st.session_state[ 'congress_limit' ] = 20
			st.session_state[ 'congress_sort' ] = 'updateDate+desc'
			st.session_state[ 'congress_from_datetime' ] = ''
			st.session_state[ 'congress_to_datetime' ] = ''
			st.session_state[ 'congress_conference' ] = False
			st.session_state[ 'congress_timeout' ] = 20
			st.session_state[ 'congress_results' ] = { }
			st.session_state[ 'congress_clear_request' ] = False
		
		def _clear_congress_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the Congress expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'congress_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			congress_mode = st.selectbox(
				'Mode',
				options=[
						'congresses',
						'bills',
						'bill_detail',
						'laws',
						'law_detail',
						'reports',
						'report_detail'
				],
				index=[
						'congresses',
						'bills',
						'bill_detail',
						'laws',
						'law_detail',
						'reports',
						'report_detail'
				].index( st.session_state.get( 'congress_mode', 'congresses' ) ),
				key='congress_mode',
				help=(
						'Congress.gov is a structured endpoint API. '
						'Choose the specific operation you want to perform.'
				)
			)
			
			congress_number = st.number_input(
				'Congress Number',
				min_value=1,
				max_value=999,
				value=int( st.session_state.get( 'congress_number', 119 ) ),
				step=1,
				key='congress_number'
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				congress_bill_type = st.selectbox(
					'Bill Type',
					options=[ '', 'hr', 's', 'hjres', 'sjres', 'hconres', 'sconres', 'hres',
					          'sres' ],
					index=[ '', 'hr', 's', 'hjres', 'sjres', 'hconres', 'sconres', 'hres',
					        'sres' ].index(
						st.session_state.get( 'congress_bill_type', '' )
					),
					key='congress_bill_type'
				)
			
			with c2:
				congress_bill_number = st.number_input(
					'Bill Number',
					min_value=0,
					max_value=999999,
					value=int( st.session_state.get( 'congress_bill_number', 0 ) ),
					step=1,
					key='congress_bill_number'
				)
			
			c3, c4 = st.columns( 2 )
			
			with c3:
				congress_law_type = st.selectbox(
					'Law Type',
					options=[ '', 'pub', 'priv' ],
					index=[ '', 'pub', 'priv' ].index(
						st.session_state.get( 'congress_law_type', '' )
					),
					key='congress_law_type'
				)
			
			with c4:
				congress_law_number = st.number_input(
					'Law Number',
					min_value=0,
					max_value=999999,
					value=int( st.session_state.get( 'congress_law_number', 0 ) ),
					step=1,
					key='congress_law_number'
				)
			
			c5, c6 = st.columns( 2 )
			
			with c5:
				congress_report_type = st.selectbox(
					'Report Type',
					options=[ '', 'hrpt', 'srpt', 'erpt' ],
					index=[ '', 'hrpt', 'srpt', 'erpt' ].index(
						st.session_state.get( 'congress_report_type', '' )
					),
					key='congress_report_type'
				)
			
			with c6:
				congress_report_number = st.number_input(
					'Report Number',
					min_value=0,
					max_value=999999,
					value=int( st.session_state.get( 'congress_report_number', 0 ) ),
					step=1,
					key='congress_report_number'
				)
			
			c7, c8 = st.columns( 2 )
			
			with c7:
				congress_offset = st.number_input(
					'Offset',
					min_value=0,
					max_value=1000000,
					value=int( st.session_state.get( 'congress_offset', 0 ) ),
					step=1,
					key='congress_offset'
				)
			
			with c8:
				congress_limit = st.number_input(
					'Limit',
					min_value=1,
					max_value=250,
					value=int( st.session_state.get( 'congress_limit', 20 ) ),
					step=1,
					key='congress_limit'
				)
			
			congress_sort = st.selectbox(
				'Sort',
				options=[ 'updateDate+desc', 'updateDate+asc' ],
				index=[ 'updateDate+desc', 'updateDate+asc' ].index(
					st.session_state.get( 'congress_sort', 'updateDate+desc' )
				),
				key='congress_sort'
			)
			
			c9, c10 = st.columns( 2 )
			
			with c9:
				congress_from_datetime = st.text_input(
					'From DateTime (ISO)',
					value=st.session_state.get( 'congress_from_datetime', '' ),
					key='congress_from_datetime',
					placeholder='YYYY-MM-DDTHH:MM:SSZ'
				)
			
			with c10:
				congress_to_datetime = st.text_input(
					'To DateTime (ISO)',
					value=st.session_state.get( 'congress_to_datetime', '' ),
					key='congress_to_datetime',
					placeholder='YYYY-MM-DDTHH:MM:SSZ'
				)
			
			congress_conference = st.checkbox(
				'Conference Reports',
				value=bool( st.session_state.get( 'congress_conference', False ) ),
				key='congress_conference'
			)
			
			congress_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'congress_timeout', 20 ) ),
				step=1,
				key='congress_timeout'
			)
			
			b1, b2 = st.columns( 2 )
			
			with b1:
				congress_submit = st.button(
					'Submit',
					key='congress_submit',
					use_container_width=True
				)
			
			with b2:
				congress_clear = st.button(
					'Clear',
					key='congress_clear',
					on_click=_clear_congress_state,
					use_container_width=True
				)
		
		with col_right:
			congress_output = st.empty( )
		
		if congress_submit:
			try:
				f = Congress( )
				
				result = f.fetch(
					mode=str( congress_mode ),
					congress=int( congress_number ),
					bill_type=str( congress_bill_type ),
					bill_number=int( congress_bill_number ),
					law_type=str( congress_law_type ),
					law_number=int( congress_law_number ),
					report_type=str( congress_report_type ),
					report_number=int( congress_report_number ),
					offset=int( congress_offset ),
					limit=int( congress_limit ),
					sort=str( congress_sort ),
					from_date_time=str( congress_from_datetime ),
					to_date_time=str( congress_to_datetime ),
					conference=bool( congress_conference ),
					time=int( congress_timeout )
				)
				
				st.session_state[ 'congress_results' ] = result or { }
				st.rerun( )
			
			except Exception as exc:
				st.error( str( exc ) )
				
		result = st.session_state.get( 'congress_results', { } )
		
		if not result:
			congress_output.text( 'No results.' )
		else:
			mode_value = result.get( 'mode', '' ) if isinstance( result, dict ) else ''
			data = result.get( 'data', { } ) if isinstance( result, dict ) else { }
			params = result.get( 'params', { } ) if isinstance( result, dict ) else { }
			
			with col_right:
				st.markdown( '#### Request Metadata' )
				st.json(
					{
							'mode': mode_value,
							'url': result.get( 'url', '' ),
							'params': params,
					}
				)
				
				items: List[ Dict[ str, Any ] ] = [ ]
				
				if isinstance( data, dict ):
					for key in [
							'bills',
							'laws',
							'reports',
							'committeeReports',
							'congresses',
							'sessions',
							'results',
							'items'
					]:
						value = data.get( key, None )
						if isinstance( value, list ):
							items = [ item for item in value if isinstance( item, dict ) ]
							break
				elif isinstance( data, list ):
					items = [ item for item in data if isinstance( item, dict ) ]
				
				if items:
					st.markdown( '#### Results' )
					
					for index, item in enumerate( items, start=1 ):
						title_value = (
								item.get( 'title' )
								or item.get( 'name' )
								or item.get( 'number' )
								or item.get( 'billNumber' )
								or item.get( 'lawNumber' )
								or item.get( 'reportNumber' )
								or f'Result {index}'
						)
						
						id_parts: List[ str ] = [ ]
						for key in [
								'congress',
								'type',
								'number',
								'billType',
								'billNumber',
								'lawType',
								'lawNumber',
								'reportType',
								'reportNumber'
						]:
							if key in item and str( item.get( key ) ).strip( ):
								id_parts.append( f'{key}={item.get( key )}' )
						
						action_value = (
								item.get( 'latestAction' )
								or item.get( 'latestActionText' )
								or item.get( 'actionDate' )
								or ''
						)
						
						with st.container( border=True ):
							st.markdown( f'**{index}. {title_value}**' )
							
							if id_parts:
								st.caption( ' | '.join( id_parts ) )
							
							if isinstance( action_value, dict ):
								st.write( str( action_value ) )
							elif action_value:
								st.write( str( action_value ) )
							
							with st.expander( 'Raw Item', expanded=False ):
								st.json( item )
				
				elif isinstance( data, dict ) and data:
					st.markdown( '#### Detail' )
					
					title_value = (
							data.get( 'title' )
							or data.get( 'name' )
							or data.get( 'number' )
							or data.get( 'billNumber' )
							or data.get( 'lawNumber' )
							or data.get( 'reportNumber' )
							or 'Result'
					)
					
					st.markdown( f'### {title_value}' )
					
					summary_fields: Dict[ str, Any ] = { }
					for key in [
							'congress',
							'type',
							'number',
							'billType',
							'billNumber',
							'lawType',
							'lawNumber',
							'reportType',
							'reportNumber',
							'updateDate',
							'actionDate'
					]:
						if key in data:
							summary_fields[ key ] = data.get( key )
					
					if summary_fields:
						st.json( summary_fields )
					
					for key in [ 'summary', 'latestAction', 'description' ]:
						if key in data and str( data.get( key ) ).strip( ):
							st.markdown( f'#### {key}' )
							st.write( str( data.get( key ) ) )
					
					with st.expander( 'Raw Detail JSON', expanded=False ):
						st.json( data )
				
				elif data:
					st.markdown( '#### Result' )
					st.text_area(
						'Output',
						value=str( data ),
						height=320
					)
				else:
					st.info( 'No results returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- Internet Archive
	with st.expander( label='Internet Archive', expanded=False ):
		if 'internetarchive_results' not in st.session_state:
			st.session_state[ 'internetarchive_results' ] = { }
		
		if 'internetarchive_clear_request' not in st.session_state:
			st.session_state[ 'internetarchive_clear_request' ] = False
		
		if st.session_state.get( 'internetarchive_clear_request', False ):
			st.session_state[ 'internetarchive_query' ] = ''
			st.session_state[ 'internetarchive_rows' ] = 10
			st.session_state[ 'internetarchive_page' ] = 1
			st.session_state[ 'internetarchive_sort' ] = 'downloads desc'
			st.session_state[ 'internetarchive_media_type' ] = ''
			st.session_state[ 'internetarchive_collection' ] = ''
			st.session_state[ 'internetarchive_timeout' ] = 20
			st.session_state[ 'internetarchive_results' ] = { }
			st.session_state[ 'internetarchive_clear_request' ] = False
		
		def _clear_internetarchive_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the Internet Archive expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'internetarchive_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			ia_query = st.text_area(
				'Query',
				value=st.session_state.get( 'internetarchive_query', '' ),
				height=80,
				key='internetarchive_query',
				placeholder=(
						'Examples:\n'
						'climate change\n'
						'title:"appropriations" AND creator:"Congress"\n'
						'budget execution'
				)
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				ia_rows = st.number_input(
					'Rows',
					min_value=1,
					max_value=100,
					value=int( st.session_state.get( 'internetarchive_rows', 10 ) ),
					step=1,
					key='internetarchive_rows'
				)
			
			with c2:
				ia_page = st.number_input(
					'Page',
					min_value=1,
					max_value=100000,
					value=int( st.session_state.get( 'internetarchive_page', 1 ) ),
					step=1,
					key='internetarchive_page'
				)
			
			ia_sort = st.selectbox(
				'Sort',
				options=[
						'downloads desc',
						'downloads asc',
						'publicdate desc',
						'publicdate asc',
						'titleSorter asc',
						'titleSorter desc'
				],
				index=[
						'downloads desc',
						'downloads asc',
						'publicdate desc',
						'publicdate asc',
						'titleSorter asc',
						'titleSorter desc'
				].index(
					st.session_state.get( 'internetarchive_sort', 'downloads desc' )
				),
				key='internetarchive_sort'
			)
			
			c3, c4 = st.columns( 2 )
			
			with c3:
				ia_media_type = st.text_input(
					'Mediatype',
					value=st.session_state.get( 'internetarchive_media_type', '' ),
					key='internetarchive_media_type',
					placeholder='Example: texts'
				)
			
			with c4:
				ia_collection = st.text_input(
					'Collection',
					value=st.session_state.get( 'internetarchive_collection', '' ),
					key='internetarchive_collection',
					placeholder='Example: americana'
				)
			
			ia_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'internetarchive_timeout', 20 ) ),
				step=1,
				key='internetarchive_timeout'
			)
			
			b1, b2 = st.columns( 2 )
			
			with b1:
				ia_submit = st.button(
					'Submit',
					key='internetarchive_submit',
					use_container_width=True
				)
			
			with b2:
				ia_clear = st.button(
					'Clear',
					key='internetarchive_clear',
					on_click=_clear_internetarchive_state,
					use_container_width=True
				)
		
		with col_right:
			ia_output = st.empty( )
			
		result = st.session_state.get( 'internetarchive_results', { } )
		
		if ia_submit:
			try:
				f = InternetArchive( )
				
				result = f.fetch(
					keywords=str( ia_query ),
					rows=int( ia_rows ),
					page=int( ia_page ),
					sort=str( ia_sort ),
					media_type=str( ia_media_type ),
					collection=str( ia_collection ),
					time=int( ia_timeout )
				)
				
				st.session_state[ 'internetarchive_results' ] = result or { }
				st.rerun( )
			
			except Exception as exc:
				st.error( str( exc ) )
				
			result = st.session_state.get( 'internetarchive_results', { } )
		
		if not result:
			ia_output.text( 'No results.' )
		else:
			mode_value = result.get( 'mode', '' ) if isinstance( result, dict ) else ''
			data = result.get( 'data', { } ) if isinstance( result, dict ) else { }
			params = result.get( 'params', { } ) if isinstance( result, dict ) else { }
			
			with col_right:
				st.markdown( '#### Request Metadata' )
				st.json(
					{
							'mode': mode_value,
							'url': result.get( 'url', '' ),
							'params': params,
					}
				)
				
				docs: List[ Dict[ str, Any ] ] = [ ]
				num_found = None
				
				if isinstance( data, dict ):
					response_block = data.get( 'response', { } )
					
					if isinstance( response_block, dict ):
						num_found = response_block.get( 'numFound', None )
						value = response_block.get( 'docs', [ ] )
						if isinstance( value, list ):
							docs = [ item for item in value if isinstance( item, dict ) ]
				
				if num_found is not None:
					st.markdown( f'#### Search Results ({num_found})' )
				else:
					st.markdown( '#### Search Results' )
				
				if docs:
					for index, item in enumerate( docs, start=1 ):
						title_value = (
								item.get( 'title' )
								or item.get( 'identifier' )
								or f'Result {index}'
						)
						
						identifier_value = item.get( 'identifier', '' )
						mediatype_value = item.get( 'mediatype', '' )
						
						collection_value = ''
						collection_raw = item.get( 'collection', '' )
						if isinstance( collection_raw, list ) and collection_raw:
							collection_value = ', '.join( [ str( x ) for x in
							                                collection_raw[ :3 ] ] )
						elif collection_raw:
							collection_value = str( collection_raw )
						
						date_value = (
								item.get( 'publicdate' )
								or item.get( 'date' )
								or item.get( 'addeddate' )
								or ''
						)
						
						desc_value = item.get( 'description', '' )
						if isinstance( desc_value, list ):
							desc_value = ' '.join( [ str( x ) for x in desc_value[ :2 ] ] )
						
						with st.container( border=True ):
							st.markdown( f'**{index}. {title_value}**' )
							
							meta_parts: List[ str ] = [ ]
							
							if identifier_value:
								meta_parts.append( f'Identifier: `{identifier_value}`' )
							
							if mediatype_value:
								meta_parts.append( f'Mediatype: `{mediatype_value}`' )
							
							if collection_value:
								meta_parts.append( f'Collection: `{collection_value}`' )
							
							if date_value:
								meta_parts.append( f'Date: `{date_value}`' )
							
							if meta_parts:
								st.caption( ' | '.join( meta_parts ) )
							
							if desc_value:
								st.write( str( desc_value ) )
							else:
								st.caption( 'No description available.' )
							
							with st.expander( 'Raw Item', expanded=False ):
								st.json( item )
				
				elif isinstance( data, dict ) and data:
					st.markdown( '#### Result' )
					st.json( data )
				elif isinstance( data, list ) and data:
					df_ia = pd.DataFrame( data )
					if not df_ia.empty:
						st.dataframe( df_ia, use_container_width=True, hide_index=True )
					else:
						st.json( data )
				elif data:
					st.markdown( '#### Result' )
					st.text_area(
						'Output',
						value=str( data ),
						height=320
					)
				else:
					st.info( 'No results returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )

	# -------- Grokipedia
	with st.expander( label='Grokipedia', expanded=False ):
		if 'grokipedia_results' not in st.session_state:
			st.session_state[ 'grokipedia_results' ] = { }
		
		if 'grokipedia_clear_request' not in st.session_state:
			st.session_state[ 'grokipedia_clear_request' ] = False
		
		if 'grokipedia_auto_fetch_page' not in st.session_state:
			st.session_state[ 'grokipedia_auto_fetch_page' ] = False
		
		if st.session_state.get( 'grokipedia_clear_request', False ):
			st.session_state[ 'grokipedia_mode' ] = 'search'
			st.session_state[ 'grokipedia_query' ] = ''
			st.session_state[ 'grokipedia_page' ] = ''
			st.session_state[ 'grokipedia_limit' ] = 12
			st.session_state[ 'grokipedia_offset' ] = 0
			st.session_state[ 'grokipedia_include_content' ] = True
			st.session_state[ 'grokipedia_results' ] = { }
			st.session_state[ 'grokipedia_auto_fetch_page' ] = False
			st.session_state[ 'grokipedia_clear_request' ] = False
		
		def _clear_grokipedia_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the Grokipedia expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'grokipedia_clear_request' ] = True
		
		def _load_grokipedia_page( slug: str ) -> None:
			'''
				Purpose:
				--------
				Load a selected Grokipedia slug into the page controls and trigger
				an immediate page fetch on rerun.

				Parameters:
				-----------
				slug (str):
					The selected page slug.

				Returns:
				--------
				None
			'''
			st.session_state[ 'grokipedia_mode' ] = 'page'
			st.session_state[ 'grokipedia_page' ] = str( slug ).strip( )
			st.session_state[ 'grokipedia_include_content' ] = True
			st.session_state[ 'grokipedia_auto_fetch_page' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			grokipedia_mode = st.selectbox(
				'Mode',
				options=[ 'search', 'page' ],
				index=[ 'search', 'page' ].index(
					st.session_state.get( 'grokipedia_mode', 'search' )
				),
				key='grokipedia_mode',
				help='search = keyword search; page = fetch a specific page by slug.'
			)
			
			grokipedia_query = st.text_input(
				'Query',
				value=st.session_state.get( 'grokipedia_query', '' ),
				key='grokipedia_query',
				placeholder='Example: machine learning'
			)
			
			grokipedia_page = st.text_input(
				'Page Slug',
				value=st.session_state.get( 'grokipedia_page', '' ),
				key='grokipedia_page',
				placeholder='Example: United_Petroleum'
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				grokipedia_limit = st.number_input(
					'Limit',
					min_value=1,
					max_value=100,
					value=int( st.session_state.get( 'grokipedia_limit', 12 ) ),
					step=1,
					key='grokipedia_limit'
				)
			
			with c2:
				grokipedia_offset = st.number_input(
					'Offset',
					min_value=0,
					max_value=100000,
					value=int( st.session_state.get( 'grokipedia_offset', 0 ) ),
					step=1,
					key='grokipedia_offset'
				)
			
			grokipedia_include_content = st.checkbox(
				'Include Content',
				value=bool(
					st.session_state.get( 'grokipedia_include_content', True )
				),
				key='grokipedia_include_content'
			)
			
			b1, b2 = st.columns( 2 )
			
			with b1:
				grokipedia_submit = st.button(
					'Submit',
					key='grokipedia_submit',
					use_container_width=True
				)
			
			with b2:
				grokipedia_clear = st.button(
					'Clear',
					key='grokipedia_clear',
					on_click=_clear_grokipedia_state,
					use_container_width=True
				)
		
		with col_right:
			grokipedia_output = st.empty( )
		
		should_fetch_grokipedia = False
		
		if grokipedia_submit:
			should_fetch_grokipedia = True
		
		if st.session_state.get( 'grokipedia_auto_fetch_page', False ):
			should_fetch_grokipedia = True
		
		if should_fetch_grokipedia:
			try:
				f = Grokipedia( )
				
				result = f.fetch(
					mode=str( st.session_state.get( 'grokipedia_mode', 'search' ) ),
					query=str( st.session_state.get( 'grokipedia_query', '' ) ),
					page=str( st.session_state.get( 'grokipedia_page', '' ) ),
					limit=int( st.session_state.get( 'grokipedia_limit', 12 ) ),
					offset=int( st.session_state.get( 'grokipedia_offset', 0 ) ),
					include_content=bool(
						st.session_state.get( 'grokipedia_include_content', True )
					)
				)
				
				st.session_state[ 'grokipedia_results' ] = result or { }
				st.session_state[ 'grokipedia_auto_fetch_page' ] = False
				st.rerun( )
			
			except Exception as exc:
				st.session_state[ 'grokipedia_auto_fetch_page' ] = False
				st.error( str( exc ) )
		
		result = st.session_state.get( 'grokipedia_results', { } )
		
		if not result:
			grokipedia_output.text( 'No results.' )
		else:
			mode_value = result.get( 'mode', '' ) if isinstance( result, dict ) else ''
			data = result.get( 'data', { } ) if isinstance( result, dict ) else { }
			
			with col_right:
				st.markdown( '#### Request Metadata' )
				st.json(
					{
							'mode': result.get( 'mode', '' ),
							'url': result.get( 'url', '' ),
							'params': result.get( 'params', { } ),
							'api_key_configured': result.get( 'api_key_configured', False )
					}
				)
				
				if mode_value == 'search':
					st.markdown( '#### Search Results' )
					
					items: List[ Dict[ str, Any ] ] = [ ]
					
					if isinstance( data, list ):
						items = [ item for item in data if isinstance( item, dict ) ]
					elif isinstance( data, dict ):
						for key in [ 'results', 'items', 'pages', 'data' ]:
							value = data.get( key, None )
							if isinstance( value, list ):
								items = [ item for item in value if isinstance( item, dict ) ]
								break
					
					if not items:
						if data:
							st.info(
								'No structured search hits were detected. '
								'Showing raw output.'
							)
							st.json( data )
						else:
							st.info( 'No results returned.' )
					else:
						for index, item in enumerate( items, start=1 ):
							title_value = (
									item.get( 'title' )
									or item.get( 'name' )
									or item.get( 'slug' )
									or item.get( 'id' )
									or f'Result {index}'
							)
							
							slug_value = (
									item.get( 'slug' )
									or item.get( 'page' )
									or item.get( 'path' )
									or item.get( 'id' )
									or ''
							)
							
							summary_value = (
									item.get( 'summary' )
									or item.get( 'description' )
									or item.get( 'excerpt' )
									or item.get( 'snippet' )
									or ''
							)
							
							score_value = (
									item.get( 'score' )
									or item.get( 'rank' )
									or item.get( 'relevance' )
							)
							
							with st.container( border=True ):
								st.markdown( f'**{index}. {title_value}**' )
								
								meta_parts: List[ str ] = [ ]
								
								if slug_value:
									meta_parts.append( f'Slug: `{slug_value}`' )
								
								if score_value is not None and str( score_value ).strip( ):
									meta_parts.append( f'Score: `{score_value}`' )
								
								if meta_parts:
									st.caption( ' | '.join( meta_parts ) )
								
								if summary_value:
									st.write( str( summary_value ) )
								else:
									st.caption( 'No summary available.' )
								
								ca, cb = st.columns( 2 )
								
								with ca:
									if slug_value:
										if st.button(
												'Load Page',
												key=f'grokipedia_load_page_{index}_{slug_value}',
												use_container_width=True
										):
											_load_grokipedia_page( slug_value )
											st.rerun( )
								
								with cb:
									with st.expander( 'Raw Item', expanded=False ):
										st.json( item )
				
				elif mode_value == 'page':
					st.markdown( '#### Page Result' )
					
					page_item: Dict[ str, Any ] = data if isinstance( data, dict ) else { }
					
					if not page_item:
						if data:
							st.text_area(
								'Output',
								value=str( data ),
								height=320
							)
						else:
							st.info( 'No results returned.' )
					else:
						title_value = (
								page_item.get( 'title' )
								or page_item.get( 'name' )
								or page_item.get( 'slug' )
								or page_item.get( 'id' )
								or 'Untitled Page'
						)
						
						slug_value = (
								page_item.get( 'slug' )
								or page_item.get( 'page' )
								or page_item.get( 'path' )
								or page_item.get( 'id' )
								or ''
						)
						
						summary_value = (
								page_item.get( 'summary' )
								or page_item.get( 'description' )
								or page_item.get( 'excerpt' )
								or ''
						)
						
						content_value = (
								page_item.get( 'content' )
								or page_item.get( 'text' )
								or page_item.get( 'body' )
								or ''
						)
						
						st.markdown( f'### {title_value}' )
						
						if slug_value:
							st.caption( f'Slug: `{slug_value}`' )
						
						if summary_value:
							st.markdown( '#### Summary' )
							st.write( str( summary_value ) )
						
						if content_value:
							st.markdown( '#### Content' )
							st.text_area(
								'Page Content',
								value=str( content_value ),
								height=360
							)
						else:
							st.info( 'No page content was returned.' )
						
						with st.expander( 'Raw Page JSON', expanded=False ):
							st.json( page_item )
				
				else:
					st.markdown( '#### Result' )
					
					if isinstance( data, dict ) or isinstance( data, list ):
						st.json( data )
					elif data:
						st.text_area(
							'Output',
							value=str( data ),
							height=320
						)
					else:
						st.info( 'No results returned.' )
	
	# ------- Jupyter Notebook Loader
	with st.expander( label='Jupyter Notebook', icon='📓', expanded=False ):
		notebook_file = st.file_uploader(
			'Upload Notebook',
			type=[ 'ipynb' ],
			key='notebook_upload'
		)
		include_outputs = st.checkbox( 'Include Outputs', value=True, key='notebook_outputs' )
		max_output_length = st.number_input(
			'Max Output Length',
			min_value=1,
			max_value=20000,
			value=100,
			step=10,
			key='notebook_max_output_length'
		)
		remove_newline = st.checkbox(
			'Remove Newlines',
			value=False,
			key='notebook_remove_newline'
		)
		include_traceback = st.checkbox(
			'Include Traceback',
			value=False,
			key='notebook_traceback'
		)
		
		col_load, col_clear, col_save = st.columns( 3 )
		load_notebook = col_load.button( 'Load', key='notebook_load' )
		clear_notebook = col_clear.button( 'Clear', key='notebook_clear' )
		
		can_save = (
				st.session_state.get( 'active_loader' ) == 'JupyterNotebookLoader'
				and isinstance( st.session_state.get( 'raw_text' ), str )
				and st.session_state.get( 'raw_text' ).strip( )
		)
		
		if can_save:
			col_save.download_button(
				'Save',
				data=st.session_state.get( 'raw_text' ),
				file_name='notebook_loader_output.txt',
				mime='text/plain',
				key='notebook_save'
			)
		else:
			col_save.button( 'Save', key='notebook_save_disabled', disabled=True )
		
		if clear_notebook:
			remaining = _clear_loader_documents( 'JupyterNotebookLoader' )
			st.info( f'Jupyter Notebook Loader state cleared. Remaining documents: {remaining}.' )
		
		if load_notebook and notebook_file:
			try:
				with tempfile.TemporaryDirectory( ) as tmp:
					path = os.path.join( tmp, notebook_file.name )
					with open( path, 'wb' ) as f:
						f.write( notebook_file.read( ) )
					
					loader = JupyterNotebookLoader(
						path,
						include_outputs=include_outputs,
						max_output_length=int( max_output_length ),
						remove_newline=remove_newline,
						traceback=include_traceback
					)
					documents = loader.load( ) or [ ]
				
				count = _promote_loader_documents( documents, 'JupyterNotebookLoader' )
				st.success( f'Loaded {count} notebook document(s).' )
			except Exception as e:
				st.error( str( e ) )
		
		# --------  Google Cloud Storage File Loader
		with st.expander( label='Google Cloud File Loader', icon='☁️', expanded=False ):
			project_name = st.text_input( 'Project Name', key='gcs_project_name' )
			bucket = st.text_input( 'Bucket', key='gcs_bucket' )
			blob = st.text_input( 'Blob', key='gcs_blob' )
			
			col_load, col_clear, col_save = st.columns( 3 )
			load_gcs = col_load.button( 'Load', key='gcs_load' )
			clear_gcs = col_clear.button( 'Clear', key='gcs_clear' )
			
			can_save = (
					st.session_state.get( 'active_loader' ) == 'GoogleCloudStorageFileLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='gcs_loader_output.txt',
					mime='text/plain',
					key='gcs_save'
				)
			else:
				col_save.button( 'Save', key='gcs_save_disabled', disabled=True )
			
			if clear_gcs:
				clear_if_active( 'GoogleCloudStorageFileLoader' )
				st.session_state.raw_text = _rebuild_raw_text_from_documents( )
				st.session_state[ '_loader_status' ] = 'Google Cloud Storage File Loader state cleared.'
				st.rerun( )
			
			if load_gcs and project_name and bucket and blob:
				loader = GoogleCloudStorageFileLoader( )
				documents = loader.load(
					project_name=project_name,
					bucket=bucket,
					blob=blob
				) or [ ]
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = '\n\n'.join(
					d.page_content for d in documents
					if hasattr( d, 'page_content' ) and isinstance( d.page_content, str )
					and d.page_content.strip( )
				)
				st.session_state.processed_text = None
				st.session_state.tokens = None
				st.session_state.vocabulary = None
				st.session_state.token_counts = None
				st.session_state.active_loader = 'GoogleCloudStorageFileLoader'
				st.session_state[ '_loader_status' ] = (
						f'Loaded {len( documents )} GCS document(s).'
				)
				st.rerun( )
		
		# --------  Microsoft OneDrive Loader
		with st.expander( label='OneDrive Loader', icon='🪟', expanded=False ):
			drive_id = st.text_input( 'Drive ID', key='onedrive_drive_id' )
			folder_path = st.text_input( 'Folder Path (Optional)', key='onedrive_folder_path' )
			auth_with_token = st.checkbox(
				'Authenticate With Cached Token',
				value=True,
				key='onedrive_auth_with_token'
			)
			
			col_load, col_clear, col_save = st.columns( 3 )
			load_onedrive = col_load.button( 'Load', key='onedrive_load' )
			clear_onedrive = col_clear.button( 'Clear', key='onedrive_clear' )
			
			can_save = (
					st.session_state.get( 'active_loader' ) == 'MicrosoftOneDriveFileLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='onedrive_loader_output.txt',
					mime='text/plain',
					key='onedrive_save'
				)
			else:
				col_save.button( 'Save', key='onedrive_save_disabled', disabled=True )
			
			if clear_onedrive:
				clear_if_active( 'MicrosoftOneDriveFileLoader' )
				st.session_state.raw_text = _rebuild_raw_text_from_documents( )
				st.session_state[ '_loader_status' ] = 'Microsoft OneDrive Loader state cleared.'
				st.rerun( )
			
			if load_onedrive and drive_id:
				loader = MicrosoftOneDriveFileLoader( )
				documents = loader.load(
					drive_id=drive_id,
					folder_path=folder_path.strip( ) if folder_path else None,
					auth_with_token=auth_with_token
				) or [ ]
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = '\n\n'.join(
					d.page_content for d in documents
					if hasattr( d, 'page_content' ) and isinstance( d.page_content, str )
					and d.page_content.strip( )
				)
				st.session_state.processed_text = None
				st.session_state.tokens = None
				st.session_state.vocabulary = None
				st.session_state.token_counts = None
				st.session_state.active_loader = 'MicrosoftOneDriveFileLoader'
				st.session_state[ '_loader_status' ] = (
						f'Loaded {len( documents )} OneDrive document(s).'
				)
				st.rerun( )
		
		# --------- AWS S3 File Loader
		with st.expander( label='AWS File Loader', icon='🪣', expanded=False ):
			bucket = st.text_input( 'Bucket', key='s3_bucket' )
			key_name = st.text_input( 'Key', key='s3_key' )
			region_name = st.text_input( 'Region (Optional)', key='s3_region_name' )
			aws_access_key_id = st.text_input(
				'AWS Access Key ID (Optional)',
				type='password',
				key='s3_access_key'
			)
			aws_secret_access_key = st.text_input(
				'AWS Secret Access Key (Optional)',
				type='password',
				key='s3_secret_key'
			)
			aws_session_token = st.text_input(
				'AWS Session Token (Optional)',
				type='password',
				key='s3_session_token'
			)
			
			col_load, col_clear, col_save = st.columns( 3 )
			load_s3 = col_load.button( 'Load', key='s3_load' )
			clear_s3 = col_clear.button( 'Clear', key='s3_clear' )
			
			can_save = (
					st.session_state.get( 'active_loader' ) == 'AwsS3FileLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='s3_loader_output.txt',
					mime='text/plain',
					key='s3_save'
				)
			else:
				col_save.button( 'Save', key='s3_save_disabled', disabled=True )
			
			if clear_s3:
				clear_if_active( 'AwsS3FileLoader' )
				st.session_state.raw_text = _rebuild_raw_text_from_documents( )
				st.session_state[ '_loader_status' ] = 'AWS S3 File Loader state cleared.'
				st.rerun( )
			
			if load_s3 and bucket and key_name:
				loader = AwsS3FileLoader( )
				documents = loader.load(
					bucket=bucket,
					key=key_name,
					aws_access_key_id=aws_access_key_id.strip( ) or None,
					aws_secret_access_key=aws_secret_access_key.strip( ) or None,
					aws_session_token=aws_session_token.strip( ) or None,
					region_name=region_name.strip( ) or None
				) or [ ]
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = '\n\n'.join(
					d.page_content for d in documents
					if hasattr( d, 'page_content' ) and isinstance( d.page_content, str )
					and d.page_content.strip( )
				)
				st.session_state.processed_text = None
				st.session_state.tokens = None
				st.session_state.vocabulary = None
				st.session_state.token_counts = None
				st.session_state.active_loader = 'AwsS3FileLoader'
				st.session_state[ '_loader_status' ] = f'Loaded {len( documents )} S3 document(s).'
				st.rerun( )
		
		# -------- Google Speech-to-Text Loader
		with st.expander( label='Google Speech-to-Text', icon='🎙️', expanded=False ):
			project_id = st.text_input( 'Project ID', key='gstt_project_id' )
			audio_file = st.file_uploader(
				'Upload Audio File',
				type=[ 'wav', 'flac', 'mp3', 'm4a', 'ogg' ],
				key='gstt_audio_upload'
			)
			gcs_audio_uri = st.text_input(
				'GCS Audio URI (Optional)',
				placeholder='gs://bucket/path/audio.flac',
				key='gstt_gcs_uri'
			)
			language_code = st.text_input(
				'Language Code (Optional)',
				value='en-US',
				key='gstt_language_code'
			)
			
			col_load, col_clear, col_save = st.columns( 3 )
			load_gstt = col_load.button( 'Load', key='gstt_load' )
			clear_gstt = col_clear.button( 'Clear', key='gstt_clear' )
			
			can_save = (
					st.session_state.get( 'active_loader' ) == 'GoogleSpeechToTextAudioLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='google_speech_to_text_output.txt',
					mime='text/plain',
					key='gstt_save'
				)
			else:
				col_save.button( 'Save', key='gstt_save_disabled', disabled=True )
			
			if clear_gstt:
				clear_if_active( 'GoogleSpeechToTextAudioLoader' )
				st.session_state.raw_text = _rebuild_raw_text_from_documents( )
				st.session_state[ '_loader_status' ] = 'Google Speech-to-Text Loader state cleared.'
				st.rerun( )
			
			if load_gstt and project_id and (audio_file or gcs_audio_uri.strip( )):
				config: Dict[ str, Any ] | None = None
				if language_code.strip( ):
					config = { 'language_code': language_code.strip( ) }
				
				if gcs_audio_uri.strip( ):
					file_path = gcs_audio_uri.strip( )
					loader = GoogleSpeechToTextAudioLoader( )
					documents = loader.load(
						project_id=project_id,
						file_path=file_path,
						config=config
					) or [ ]
				else:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, audio_file.name )
						with open( path, 'wb' ) as f:
							f.write( audio_file.read( ) )
						
						loader = GoogleSpeechToTextAudioLoader( )
						documents = loader.load(
							project_id=project_id,
							file_path=path,
							config=config
						) or [ ]
				
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = '\n\n'.join(
					d.page_content for d in documents
					if hasattr( d, 'page_content' ) and isinstance( d.page_content, str )
					and d.page_content.strip( )
				)
				st.session_state.processed_text = None
				st.session_state.tokens = None
				st.session_state.vocabulary = None
				st.session_state.token_counts = None
				st.session_state.active_loader = 'GoogleSpeechToTextAudioLoader'
				st.session_state[ '_loader_status' ] = (
						f'Loaded {len( documents )} transcript document(s).'
				)
				st.rerun( )
	
	# ------- Google Cloud File Loader
	with st.expander( label='Google Cloud File', icon='☁️', expanded=False ):
		project_name = st.text_input( 'Project Name', key='gcs_project_name' )
		bucket = st.text_input( 'Bucket', key='gcs_bucket' )
		blob = st.text_input( 'Blob', key='gcs_blob' )
		
		col_load, col_clear, col_save = st.columns( 3 )
		load_gcs = col_load.button( 'Load', key='gcs_load' )
		clear_gcs = col_clear.button( 'Clear', key='gcs_clear' )
		
		can_save = (
				st.session_state.get( 'active_loader' ) == 'GoogleCloudFileLoader'
				and isinstance( st.session_state.get( 'raw_text' ), str )
				and st.session_state.get( 'raw_text' ).strip( )
		)
		
		if can_save:
			col_save.download_button(
				'Save',
				data=st.session_state.get( 'raw_text' ),
				file_name='gcs_loader_output.txt',
				mime='text/plain',
				key='gcs_save'
			)
		else:
			col_save.button( 'Save', key='gcs_save_disabled', disabled=True )
		
		if clear_gcs:
			remaining = _clear_loader_documents( 'GoogleCloudFileLoader' )
			st.info( f'Google Cloud Storage File Loader state cleared. Remaining documents: {remaining}.' )
		
		if load_gcs and project_name.strip( ) and bucket.strip( ) and blob.strip( ):
			try:
				loader = GoogleCloudFileLoader(
					project_name=project_name.strip( ),
					bucket=bucket.strip( ),
					blob=blob.strip( )
				)
				documents = loader.load( ) or [ ]
				count = _promote_loader_documents( documents, 'GoogleCloudFileLoader' )
				st.success( f'Loaded {count} Google Cloud Storage document(s).' )
			except Exception as e:
				st.error( str( e ) )
	
	# -------- AWS File Loader
	with st.expander( label='AWS S3 File', icon='🪣', expanded=False ):
		bucket = st.text_input( 'Bucket', key='s3_bucket' )
		key_name = st.text_input( 'Key', key='s3_key' )
		region_name = st.text_input( 'Region (Optional)', key='s3_region_name' )
		aws_access_key_id = st.text_input(
			'AWS Access Key ID (Optional)',
			type='password',
			key='s3_access_key'
		)
		aws_secret_access_key = st.text_input(
			'AWS Secret Access Key (Optional)',
			type='password',
			key='s3_secret_key'
		)
		aws_session_token = st.text_input(
			'AWS Session Token (Optional)',
			type='password',
			key='s3_session_token'
		)
		
		col_load, col_clear, col_save = st.columns( 3 )
		load_s3 = col_load.button( 'Load', key='s3_load' )
		clear_s3 = col_clear.button( 'Clear', key='s3_clear' )
		
		can_save = (
				st.session_state.get( 'active_loader' ) == 'AwsS3FileLoader'
				and isinstance( st.session_state.get( 'raw_text' ), str )
				and st.session_state.get( 'raw_text' ).strip( )
		)
		
		if can_save:
			col_save.download_button(
				'Save',
				data=st.session_state.get( 'raw_text' ),
				file_name='s3_loader_output.txt',
				mime='text/plain',
				key='s3_save'
			)
		else:
			col_save.button( 'Save', key='s3_save_disabled', disabled=True )
		
		if clear_s3:
			remaining = _clear_loader_documents( 'AwsFileLoader' )
			st.info( f'AWS S3 File Loader state cleared. Remaining documents: {remaining}.' )
		
		if load_s3 and bucket.strip( ) and key_name.strip( ):
			try:
				kwargs: Dict[ str, Any ] = {
						'bucket': bucket.strip( ),
						'key': key_name.strip( ),
				}
				if aws_access_key_id.strip( ):
					kwargs[ 'aws_access_key_id' ] = aws_access_key_id.strip( )
				if aws_secret_access_key.strip( ):
					kwargs[ 'aws_secret_access_key' ] = aws_secret_access_key.strip( )
				if aws_session_token.strip( ):
					kwargs[ 'aws_session_token' ] = aws_session_token.strip( )
				if region_name.strip( ):
					kwargs[ 'region_name' ] = region_name.strip( )
				
				loader = AwsFileLoader( **kwargs )
				documents = loader.load( ) or [ ]
				count = _promote_loader_documents( documents, 'AwsFileLoader' )
				st.success( f'Loaded {count} S3 document(s).' )
			except Exception as e:
				st.error( str( e ) )
				
	# ------ OneDrive Loader
	with st.expander( label='OneDrive', icon='🪟', expanded=False ):
		drive_id = st.text_input( 'Drive ID', key='onedrive_drive_id' )
		folder_path = st.text_input( 'Folder Path (Optional)', key='onedrive_folder_path' )
		auth_with_token = st.checkbox(
			'Authenticate With Cached Token',
			value=True,
			key='onedrive_auth_with_token'
		)
		
		col_load, col_clear, col_save = st.columns( 3 )
		load_onedrive = col_load.button( 'Load', key='onedrive_load' )
		clear_onedrive = col_clear.button( 'Clear', key='onedrive_clear' )
		
		can_save = (
				st.session_state.get( 'active_loader' ) == 'OneDriveDocLoader'
				and isinstance( st.session_state.get( 'raw_text' ), str )
				and st.session_state.get( 'raw_text' ).strip( )
		)
		
		if can_save:
			col_save.download_button(
				'Save',
				data=st.session_state.get( 'raw_text' ),
				file_name='onedrive_loader_output.txt',
				mime='text/plain',
				key='onedrive_save'
			)
		else:
			col_save.button( 'Save', key='onedrive_save_disabled', disabled=True )
		
		if clear_onedrive:
			remaining = _clear_loader_documents( 'OneDriveDocLoader' )
			st.info( f'Microsoft OneDrive Loader state cleared. Remaining documents: {remaining}.' )
		
		if load_onedrive and drive_id.strip( ):
			try:
				kwargs: Dict[ str, Any ] = {
						'drive_id': drive_id.strip( ),
						'auth_with_token': auth_with_token,
				}
				if folder_path.strip( ):
					kwargs[ 'folder_path' ] = folder_path.strip( )
				
				loader = OneDriveDocLoader( **kwargs )
				documents = loader.load( ) or [ ]
				count = _promote_loader_documents( documents, 'OneDriveDocLoader' )
				st.success( f'Loaded {count} OneDrive document(s).' )
			except Exception as e:
				st.error( str( e ) )
	
	# ------- Google Speech-to-Text Loader
	with st.expander( label='Google Speech-to-Text', icon='🎙️', expanded=False ):
		project_id = st.text_input( 'Project ID', key='gstt_project_id' )
		audio_file = st.file_uploader(
			'Upload Audio File',
			type=[ 'wav', 'flac', 'mp3', 'm4a', 'ogg' ],
			key='gstt_audio_upload'
		)
		gcs_audio_uri = st.text_input(
			'GCS Audio URI (Optional)',
			placeholder='gs://bucket/path/audio.flac',
			key='gstt_gcs_uri'
		)
		language_code = st.text_input(
			'Language Code (Optional)',
			value='en-US',
			key='gstt_language_code'
		)
		
		col_load, col_clear, col_save = st.columns( 3 )
		load_gstt = col_load.button( 'Load', key='gstt_load' )
		clear_gstt = col_clear.button( 'Clear', key='gstt_clear' )
		
		can_save = (
				st.session_state.get( 'active_loader' ) == 'GoogleSpeechToTextAudioLoader'
				and isinstance( st.session_state.get( 'raw_text' ), str )
				and st.session_state.get( 'raw_text' ).strip( )
		)
		
		if can_save:
			col_save.download_button(
				'Save',
				data=st.session_state.get( 'raw_text' ),
				file_name='google_speech_to_text_output.txt',
				mime='text/plain',
				key='gstt_save'
			)
		else:
			col_save.button( 'Save', key='gstt_save_disabled', disabled=True )
		
		if clear_gstt:
			remaining = _clear_loader_documents( 'GoogleSpeechToTextAudioLoader' )
			st.info( f'Google Speech-to-Text Loader state cleared. Remaining documents: {remaining}.' )
		
		if load_gstt and project_id.strip( ) and (audio_file or gcs_audio_uri.strip( )):
			try:
				config: Dict[ str, Any ] | None = None
				if language_code.strip( ):
					config = { 'language_code': language_code.strip( ) }
				
				if gcs_audio_uri.strip( ):
					file_path = gcs_audio_uri.strip( )
					loader = LangChainSpeechToTextLoader(
						project_id=project_id.strip( ),
						file_path=file_path,
						config=config
					)
					documents = loader.load( ) or [ ]
				else:
					with tempfile.TemporaryDirectory( ) as tmp:
						path = os.path.join( tmp, audio_file.name )
						with open( path, 'wb' ) as f:
							f.write( audio_file.read( ) )
						
						loader = LangChainSpeechToTextLoader(
							project_id=project_id.strip( ),
							file_path=path,
							config=config
						)
						documents = loader.load( ) or [ ]
				
				count = _promote_loader_documents( documents, 'GoogleSpeechToTextLoader' )
				st.success( f'Loaded {count} transcript document(s).' )
			except Exception as e:
				st.error( str( e ) )

	# -------- Amazon Bucket
	with st.expander( label='AWS S3 Bucket', icon='🗂️', expanded=False ):
		bucket_name = st.text_input( 'Bucket', key='s3_directory_bucket' )
		prefix = st.text_input( 'Prefix (Optional)', key='s3_directory_prefix' )
		region_name = st.text_input( 'Region (Optional)', key='s3_directory_region_name' )
		endpoint_url = st.text_input( 'Endpoint URL (Optional)', key='s3_directory_endpoint_url' )
		aws_access_key_id = st.text_input(
			'AWS Access Key ID (Optional)',
			type='password',
			key='s3_directory_access_key'
		)
		aws_secret_access_key = st.text_input(
			'AWS Secret Access Key (Optional)',
			type='password',
			key='s3_directory_secret_key'
		)
		aws_session_token = st.text_input(
			'AWS Session Token (Optional)',
			type='password',
			key='s3_directory_session_token'
		)
		
		col_load, col_clear, col_save = st.columns( 3 )
		load_amazon_bucket = col_load.button( 'Load', key='s3_directory_load' )
		clear_amazon_bucket = col_clear.button( 'Clear', key='s3_directory_clear' )
		
		can_save = (
				st.session_state.get( 'active_loader' ) == 'AmazonBucketLoader'
				and isinstance( st.session_state.get( 'raw_text' ), str )
				and st.session_state.get( 'raw_text' ).strip( )
		)
		
		if can_save:
			col_save.download_button(
				'Save',
				data=st.session_state.get( 'raw_text' ),
				file_name='amazon_bucket_loader_output.txt',
				mime='text/plain',
				key='s3_directory_save'
			)
		else:
			col_save.button( 'Save', key='s3_directory_save_disabled', disabled=True )
		
		if clear_amazon_bucket:
			remaining = _clear_loader_documents( 'AmazonBucketLoader' )
			st.info( f'Amazon Bucket Loader state cleared. Remaining documents: {remaining}.' )
		
		if load_amazon_bucket and bucket_name.strip( ):
			try:
				loader = AmazonBucketLoader( )
				documents = loader.load(
					bucket=bucket_name.strip( ),
					prefix=prefix.strip( ) or None,
					aws_access_key_id=aws_access_key_id.strip( ) or None,
					aws_secret_access_key=aws_secret_access_key.strip( ) or None,
					aws_session_token=aws_session_token.strip( ) or None,
					region_name=region_name.strip( ) or None,
					endpoint_url=endpoint_url.strip( ) or None
				) or [ ]
				count = _promote_loader_documents( documents, 'AmazonBucketLoader' )
				st.success( f'Loaded {count} Amazon bucket document(s).' )
			except Exception as e:
				st.error( str( e ) )
	
	# -------- Google Bucket Loader
	with st.expander( label='Google Cloud Bucket', icon='🪣', expanded=False ):
		project_name = st.text_input( 'Project Name', key='gcs_bucket_project_name' )
		bucket_name = st.text_input( 'Bucket', key='gcs_bucket_name' )
		
		col_load, col_clear, col_save = st.columns( 3 )
		load_google_bucket = col_load.button( 'Load', key='gcs_bucket_load' )
		clear_google_bucket = col_clear.button( 'Clear', key='gcs_bucket_clear' )
		
		can_save = (
				st.session_state.get( 'active_loader' ) == 'GoogleBucketLoader'
				and isinstance( st.session_state.get( 'raw_text' ), str )
				and st.session_state.get( 'raw_text' ).strip( )
		)
		
		if can_save:
			col_save.download_button(
				'Save',
				data=st.session_state.get( 'raw_text' ),
				file_name='google_bucket_loader_output.txt',
				mime='text/plain',
				key='gcs_bucket_save'
			)
		else:
			col_save.button( 'Save', key='gcs_bucket_save_disabled', disabled=True )
		
		if clear_google_bucket:
			remaining = _clear_loader_documents( 'GoogleBucketLoader' )
			st.info( f'Google Bucket Loader state cleared. Remaining documents: {remaining}.' )
		
		if load_google_bucket and project_name.strip( ) and bucket_name.strip( ):
			try:
				loader = GoogleBucketLoader( )
				documents = loader.load(
					project_name=project_name.strip( ),
					bucket=bucket_name.strip( )
				) or [ ]
				count = _promote_loader_documents( documents, 'GoogleBucketLoader' )
				st.success( f'Loaded {count} Google bucket document(s).' )
			except Exception as e:
				st.error( str( e ) )
			
# ==============================================================================
# GEOSPATIAL MODE
# ==============================================================================
elif mode == 'Geospatial':
	st.subheader( f'📡 Weather & Geospatial Data' )
	st.divider( )
	
	# -------- Google Maps
	with st.expander( label='Google Maps', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			gm_query = st.text_area( "Address", value='',
				height=40, key='googlemaps_query' )
			
			gm_radius = st.number_input( 'Radius (meters)', min_value=1,
				max_value=50000, value=5000, step=100, key='googlemaps_radius' )
			
			m1, m2 = st.columns( 2 )
			with m1:
				gm_submit = st.button( 'Submit', key='googlemaps_submit' )
			with m2:
				gm_clear = st.button( 'Clear', key='googlemaps_clear' )
		
		with col_right:
			gm_output = st.empty( )
			
			if gm_clear:
				st.session_state.update( { 'googlemaps_query': '', 'googlemaps_radius': 5000 } )
				st.rerun( )
			
			if gm_submit:
				try:
					gm = GoogleMaps( )
					loc = gm.geocode_location( gm_query )
					coords = f'{loc[ 0 ]}, {loc[ 1 ]}'
					gm_output.text_area( 'Coords', value=coords, height=300 )
				except Exception as exc:
					st.error( exc )
	
	# -------- Google Weather
	with st.expander( label='Google Weather', expanded=False ):
		if 'googleweather_results' not in st.session_state:
			st.session_state[ 'googleweather_results' ] = { }
		
		if 'googleweather_clear_request' not in st.session_state:
			st.session_state[ 'googleweather_clear_request' ] = False
		
		if st.session_state.get( 'googleweather_clear_request', False ):
			st.session_state[ 'googleweather_location' ] = ''
			st.session_state[ 'googleweather_mode' ] = 'current'
			st.session_state[ 'googleweather_units' ] = 'METRIC'
			st.session_state[ 'googleweather_language' ] = 'en'
			st.session_state[ 'googleweather_hours' ] = 24
			st.session_state[ 'googleweather_days' ] = 5
			st.session_state[ 'googleweather_timeout' ] = 10
			st.session_state[ 'googleweather_results' ] = { }
			st.session_state[ 'googleweather_clear_request' ] = False
		
		def _clear_googleweather_state( ) -> None:
			st.session_state[ 'googleweather_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			gw_location = st.text_area(
				'Location',
				height=70,
				key='googleweather_location',
				placeholder=(
						'Examples:\n'
						'Washington, DC\n'
						'1600 Pennsylvania Ave NW, Washington, DC\n'
						'Arlington, VA'
				),
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				gw_mode = st.selectbox(
					'Mode',
					options=[ 'current', 'hourly_forecast', 'daily_forecast', 'alerts' ],
					index=[ 'current', 'hourly_forecast', 'daily_forecast', 'alerts' ].index(
						st.session_state.get( 'googleweather_mode', 'current' ) ),
					key='googleweather_mode'
				)
			
			with c2:
				gw_units = st.selectbox(
					'Units',
					options=[ 'METRIC', 'IMPERIAL' ],
					index=[ 'METRIC', 'IMPERIAL' ].index(
						st.session_state.get( 'googleweather_units', 'METRIC' ) ),
					key='googleweather_units'
				)
			
			c3, c4, c5 = st.columns( 3 )
			
			with c3:
				gw_language = st.text_input(
					'Language',
					value=st.session_state.get( 'googleweather_language', 'en' ),
					key='googleweather_language',
					placeholder='en'
				)
			
			with c4:
				gw_hours = st.number_input(
					'Hours',
					min_value=1,
					max_value=240,
					value=int( st.session_state.get( 'googleweather_hours', 24 ) ),
					step=1,
					key='googleweather_hours',
					disabled=(gw_mode != 'hourly_forecast')
				)
			
			with c5:
				gw_days = st.number_input(
					'Days',
					min_value=1,
					max_value=10,
					value=int( st.session_state.get( 'googleweather_days', 5 ) ),
					step=1,
					key='googleweather_days',
					disabled=(gw_mode != 'daily_forecast')
				)
			
			gw_timeout = st.number_input(
				'Timeout',
				min_value=1,
				max_value=60,
				value=int( st.session_state.get( 'googleweather_timeout', 10 ) ),
				step=1,
				key='googleweather_timeout'
			)
			
			st.caption(
				'Required key: GOOGLE_WEATHER_API_KEY. '
				'Weather API must be enabled in your Google Cloud project.'
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				gw_submit = st.button( 'Submit', key='googleweather_submit' )
			with b2:
				st.button( 'Clear', key='googleweather_clear', on_click=_clear_googleweather_state )
		
		with col_right:
			st.markdown( 'Results' )
			
			if gw_submit:
				try:
					f = GoogleWeather( )
					
					if gw_mode == 'current':
						result = f.fetch_current(
							address=gw_location,
							units_system=gw_units,
							language_code=gw_language,
							time=int( gw_timeout ) )
					elif gw_mode == 'hourly_forecast':
						result = f.fetch_hourly_forecast(
							address=gw_location,
							hours=int( gw_hours ),
							units_system=gw_units,
							language_code=gw_language,
							time=int( gw_timeout ) )
					elif gw_mode == 'daily_forecast':
						result = f.fetch_daily_forecast(
							address=gw_location,
							days=int( gw_days ),
							units_system=gw_units,
							language_code=gw_language,
							time=int( gw_timeout ) )
					else:
						result = f.fetch_alerts(
							address=gw_location,
							language_code=gw_language,
							time=int( gw_timeout ) )
					
					st.session_state[ 'googleweather_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Google Weather request failed.' )
					st.exception( exc )
					
		result = st.session_state.get( 'googleweather_results', { } )
		
		if not result:
			st.text( 'No results.' )
		else:
			mode_value = result.get( 'mode', '' ) if isinstance( result, dict ) else ''
			data = result.get( 'data', { } ) if isinstance( result, dict ) else { }
			params = result.get( 'params', { } ) if isinstance( result, dict ) else { }
			
			st.markdown( '#### Request Metadata' )
			st.json(
				{
						'mode': mode_value,
						'url': result.get( 'url', '' ),
						'params': params,
				}
			)
			
			if isinstance( data, dict ) and data:
				current = (
						data.get( 'currentWeather' )
						or data.get( 'current_weather' )
						or data.get( 'currentConditions' )
						or data.get( 'current_conditions' )
						or { }
				)
				
				hourly = (
						data.get( 'hourlyForecasts' )
						or data.get( 'hourly_forecasts' )
						or data.get( 'hours' )
						or [ ]
				)
				
				daily = (
						data.get( 'dailyForecasts' )
						or data.get( 'daily_forecasts' )
						or data.get( 'days' )
						or [ ]
				)
				
				alerts = (
						data.get( 'weatherAlerts' )
						or data.get( 'alerts' )
						or [ ]
				)
				
				location_bits: List[ str ] = [ ]
				for key in [ 'address', 'resolvedAddress', 'location', 'displayName', 'name' ]:
					value = data.get( key, '' )
					if value:
						location_bits.append( str( value ) )
						break
				
				if location_bits:
					st.markdown( '#### Location' )
					st.write( location_bits[ 0 ] )
				
				if current:
					st.markdown( '#### Current Conditions' )
					
					c1, c2, c3 = st.columns( 3 )
					
					with c1:
						for key in [ 'temperature', 'temp', 'temperature_f', 'temperature_c' ]:
							if key in current:
								st.metric( 'Temperature', current.get( key ) )
								break
					
					with c2:
						for key in [ 'humidity', 'relativeHumidity', 'relative_humidity' ]:
							if key in current:
								st.metric( 'Humidity', current.get( key ) )
								break
					
					with c3:
						for key in [ 'weatherCondition', 'condition', 'description', 'icon' ]:
							if key in current:
								st.metric( 'Condition', current.get( key ) )
								break
					
					with st.expander( 'Current Conditions Detail', expanded=False ):
						st.json( current )
				
				if isinstance( hourly, list ) and hourly:
					st.markdown( '#### Hourly Forecast' )
					df_hourly = pd.DataFrame( hourly )
					if not df_hourly.empty:
						st.dataframe( df_hourly.head( 24 ), use_container_width=True, hide_index=True )
					else:
						st.info( 'Hourly forecast returned no displayable rows.' )
				
				if isinstance( daily, list ) and daily:
					st.markdown( '#### Daily Forecast' )
					df_daily = pd.DataFrame( daily )
					if not df_daily.empty:
						st.dataframe( df_daily, use_container_width=True, hide_index=True )
					else:
						st.info( 'Daily forecast returned no displayable rows.' )
				
				if isinstance( alerts, list ) and alerts:
					st.markdown( '#### Alerts' )
					for idx, alert in enumerate( alerts, start=1 ):
						with st.expander( f'Alert {idx}', expanded=False ):
							if isinstance( alert, dict ):
								title_value = (
										alert.get( 'headline' )
										or alert.get( 'title' )
										or alert.get( 'event' )
										or f'Alert {idx}'
								)
								st.markdown( f'**{title_value}**' )
								
								desc_value = (
										alert.get( 'description' )
										or alert.get( 'summary' )
										or ''
								)
								if desc_value:
									st.write( str( desc_value ) )
								
								st.json( alert )
							else:
								st.write( alert )
				
				if not current and not hourly and not daily and not alerts:
					st.markdown( '#### Result' )
					st.json( data )
			
			elif isinstance( data, list ) and data:
				st.markdown( '#### Results' )
				df_weather = pd.DataFrame( data )
				if not df_weather.empty:
					st.dataframe( df_weather, use_container_width=True, hide_index=True )
				else:
					st.json( data )
			else:
				st.info( 'No results returned.' )
			
			with st.expander( 'Raw Result', expanded=False ):
				st.json( result )
	
	# -------- Open Weather
	with st.expander( label='Open Weather', expanded=False ):
		if 'openweather_results' not in st.session_state:
			st.session_state[ 'openweather_results' ] = { }
		
		if 'openweather_clear_request' not in st.session_state:
			st.session_state[ 'openweather_clear_request' ] = False
		
		if st.session_state.get( 'openweather_clear_request', False ):
			st.session_state[ 'openweather_location' ] = ''
			st.session_state[ 'openweather_mode' ] = 'current'
			st.session_state[ 'openweather_timezone' ] = 'auto'
			st.session_state[ 'openweather_forecast_days' ] = 7
			st.session_state[ 'openweather_past_days' ] = 0
			st.session_state[ 'openweather_count' ] = 10
			st.session_state[ 'openweather_results' ] = { }
			st.session_state[ 'openweather_clear_request' ] = False
		
		def _clear_openweather_state( ) -> None:
			'''

				Purpose:
				--------
				Flag the Open Weather expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None

			'''
			st.session_state[ 'openweather_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			openweather_location = st.text_area(
				'Location',
				height=80,
				key='openweather_location',
				placeholder=(
						'Examples:\n'
						'Arlington, VA\n'
						'Paris, France\n'
						'90210'
				)
			)
			
			openweather_mode = st.selectbox(
				'Mode',
				options=[ 'current', 'hourly', 'daily' ],
				index=[ 'current', 'hourly', 'daily' ].index(
					st.session_state.get( 'openweather_mode', 'current' )
				),
				key='openweather_mode'
			)
			
			cfg_c1, cfg_c2 = st.columns( 2 )
			
			with cfg_c1:
				openweather_forecast_days = st.number_input(
					'Forecast Days',
					min_value=1,
					max_value=16,
					value=int( st.session_state.get( 'openweather_forecast_days', 7 ) ),
					step=1,
					key='openweather_forecast_days',
					disabled=(openweather_mode == 'current')
				)
			
			with cfg_c2:
				openweather_past_days = st.number_input(
					'Past Days',
					min_value=0,
					max_value=92,
					value=int( st.session_state.get( 'openweather_past_days', 0 ) ),
					step=1,
					key='openweather_past_days'
				)
			
			meta_c1, meta_c2 = st.columns( 2 )
			
			with meta_c1:
				openweather_timezone = st.text_input(
					'Timezone',
					value=st.session_state.get( 'openweather_timezone', 'auto' ),
					key='openweather_timezone',
					placeholder='auto'
				)
			
			with meta_c2:
				openweather_count = st.number_input(
					'Geocode Matches',
					min_value=1,
					max_value=20,
					value=int( st.session_state.get( 'openweather_count', 10 ) ),
					step=1,
					key='openweather_count'
				)
			
			btn_c1, btn_c2 = st.columns( 2 )
			
			with btn_c1:
				openweather_submit = st.button(
					'Submit',
					key='openweather_submit'
				)
			
			with btn_c2:
				openweather_clear = st.button(
					'Clear',
					key='openweather_clear',
					on_click=_clear_openweather_state
				)
		
		with col_right:
			if openweather_submit:
				try:
					f = OpenWeather( )
					result = f.fetch(
						location=str( openweather_location ),
						mode=str( openweather_mode ),
						zone=str( openweather_timezone or 'auto' ).strip( ),
						forecast_days=int( openweather_forecast_days ),
						past_days=int( openweather_past_days ),
						count=int( openweather_count )
					)
					
					st.session_state[ 'openweather_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Open Weather request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'openweather_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'location' in result:
						st.markdown( f"**Location:** {result.get( 'location', '' )}" )
					if 'latitude' in result:
						st.markdown( f"**Latitude:** {result.get( 'latitude', '' )}" )
				
				with meta_c2:
					if 'longitude' in result:
						st.markdown( f"**Longitude:** {result.get( 'longitude', '' )}" )
					if 'timezone' in result:
						st.markdown( f"**Timezone:** {result.get( 'timezone', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				geocoding = result.get( 'geocoding', { } ) or { }
				if geocoding:
					st.markdown( '#### Geocoding Result' )
					st.json( geocoding )
				
				params = result.get( 'params', { } ) or { }
				if params:
					st.markdown( '#### Request Parameters' )
					st.json( params )
				
				data = result.get( 'data', { } ) or { }
				if data:
					st.markdown( '#### Forecast Payload' )
					st.json( data )
				
				message = result.get( 'message', '' )
				if message:
					st.info( message )
	
	# -------- Historical Weather
	with st.expander( label='Historical Weather', expanded=False ):
		if 'historicalweather_results' not in st.session_state:
			st.session_state[ 'historicalweather_results' ] = { }
		
		if 'historicalweather_clear_request' not in st.session_state:
			st.session_state[ 'historicalweather_clear_request' ] = False
		
		if st.session_state.get( 'historicalweather_clear_request', False ):
			st.session_state[ 'historicalweather_location' ] = ''
			st.session_state[ 'historicalweather_date' ] = dt.date.today( ) - dt.timedelta( days=1 )
			st.session_state[ 'historicalweather_timezone' ] = 'auto'
			st.session_state[ 'historicalweather_count' ] = 10
			st.session_state[ 'historicalweather_results' ] = { }
			st.session_state[ 'historicalweather_clear_request' ] = False
		
		def _clear_historicalweather_state( ) -> None:
			'''

				Purpose:
				--------
				Flag the Historical Weather expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None

			'''
			st.session_state[ 'historicalweather_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			historicalweather_location = st.text_area(
				'Location',
				height=80,
				key='historicalweather_location',
				placeholder=(
						'Examples:\n'
						'Arlington, VA\n'
						'Tokyo, Japan\n'
						'90210'
				)
			)
			
			date_c1, date_c2 = st.columns( 2 )
			
			with date_c1:
				historicalweather_date = st.date_input(
					'Date',
					value=st.session_state.get(
						'historicalweather_date',
						dt.date.today( ) - dt.timedelta( days=1 )
					),
					key='historicalweather_date'
				)
			
			with date_c2:
				historicalweather_count = st.number_input(
					'Geocode Matches',
					min_value=1,
					max_value=20,
					value=int( st.session_state.get( 'historicalweather_count', 10 ) ),
					step=1,
					key='historicalweather_count'
				)
			
			historicalweather_timezone = st.text_input(
				'Timezone',
				value=st.session_state.get( 'historicalweather_timezone', 'auto' ),
				key='historicalweather_timezone',
				placeholder='auto'
			)
			
			btn_c1, btn_c2 = st.columns( 2 )
			
			with btn_c1:
				historicalweather_submit = st.button(
					'Submit',
					key='historicalweather_submit'
				)
			
			with btn_c2:
				historicalweather_clear = st.button(
					'Clear',
					key='historicalweather_clear',
					on_click=_clear_historicalweather_state
				)
		
		with col_right:
			if historicalweather_submit:
				try:
					f = HistoricalWeather( )
					result = f.fetch(
						location=str( historicalweather_location ),
						date=historicalweather_date,
						zone=str( historicalweather_timezone or 'auto' ).strip( ),
						count=int( historicalweather_count )
					)
					
					st.session_state[ 'historicalweather_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Historical Weather request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'historicalweather_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'location' in result:
						st.markdown( f"**Location:** {result.get( 'location', '' )}" )
					if 'latitude' in result:
						st.markdown( f"**Latitude:** {result.get( 'latitude', '' )}" )
					if 'longitude' in result:
						st.markdown( f"**Longitude:** {result.get( 'longitude', '' )}" )
				
				with meta_c2:
					if 'date' in result:
						st.markdown( f"**Date:** {result.get( 'date', '' )}" )
					if 'timezone' in result:
						st.markdown( f"**Timezone:** {result.get( 'timezone', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				geocoding = result.get( 'geocoding', { } ) or { }
				if geocoding:
					st.markdown( '#### Geocoding Result' )
					st.json( geocoding )
				
				params = result.get( 'params', { } ) or { }
				if params:
					st.markdown( '#### Request Parameters' )
					st.json( params )
				
				data = result.get( 'data', { } ) or { }
				if data:
					st.markdown( '#### Historical Weather Payload' )
					st.json( data )
				
				message = result.get( 'message', '' )
				if message:
					st.info( message )
	
	# -------- USGS Earthquakes
	with st.expander( label='USGS Earthquakes', expanded=False ):
		if 'usgsearthquakes_results' not in st.session_state:
			st.session_state[ 'usgsearthquakes_results' ] = { }
		
		if 'usgsearthquakes_clear_request' not in st.session_state:
			st.session_state[ 'usgsearthquakes_clear_request' ] = False
		
		if st.session_state.get( 'usgsearthquakes_clear_request', False ):
			st.session_state[ 'usgsearthquakes_mode' ] = 'feed'
			st.session_state[ 'usgsearthquakes_feed' ] = 'all_day.geojson'
			st.session_state[
				'usgsearthquakes_start_date' ] = dt.date.today( ) - dt.timedelta( days=7 )
			st.session_state[ 'usgsearthquakes_end_date' ] = dt.date.today( )
			st.session_state[ 'usgsearthquakes_min_magnitude' ] = 1.0
			st.session_state[ 'usgsearthquakes_max_magnitude' ] = 10.0
			st.session_state[ 'usgsearthquakes_limit' ] = 25
			st.session_state[ 'usgsearthquakes_order_by' ] = 'time'
			st.session_state[ 'usgsearthquakes_event_type' ] = 'earthquake'
			st.session_state[ 'usgsearthquakes_latitude' ] = ''
			st.session_state[ 'usgsearthquakes_longitude' ] = ''
			st.session_state[ 'usgsearthquakes_max_radius_km' ] = ''
			st.session_state[ 'usgsearthquakes_timeout' ] = 20
			st.session_state[ 'usgsearthquakes_results' ] = { }
			st.session_state[ 'usgsearthquakes_clear_request' ] = False
		
		def _clear_usgsearthquakes_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the USGS Earthquakes expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'usgsearthquakes_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			usgseq_mode = st.selectbox(
				'Mode',
				options=[ 'feed', 'search' ],
				index=[ 'feed', 'search' ].index(
					st.session_state.get( 'usgsearthquakes_mode', 'feed' )
				),
				key='usgsearthquakes_mode'
			)
			
			usgseq_feed = st.selectbox(
				'Feed',
				options=[
						'all_hour.geojson',
						'all_day.geojson',
						'all_week.geojson',
						'all_month.geojson',
						'significant_hour.geojson',
						'significant_day.geojson',
						'significant_week.geojson',
						'significant_month.geojson',
						'4.5_hour.geojson',
						'4.5_day.geojson',
						'4.5_week.geojson',
						'4.5_month.geojson'
				],
				index=[
						'all_hour.geojson',
						'all_day.geojson',
						'all_week.geojson',
						'all_month.geojson',
						'significant_hour.geojson',
						'significant_day.geojson',
						'significant_week.geojson',
						'significant_month.geojson',
						'4.5_hour.geojson',
						'4.5_day.geojson',
						'4.5_week.geojson',
						'4.5_month.geojson'
				].index(
					st.session_state.get( 'usgsearthquakes_feed', 'all_day.geojson' )
				),
				key='usgsearthquakes_feed',
				disabled=(usgseq_mode != 'feed')
			)
			
			date_c1, date_c2 = st.columns( 2 )
			
			with date_c1:
				usgseq_start_date = st.date_input(
					'Start Date',
					value=st.session_state.get(
						'usgsearthquakes_start_date',
						dt.date.today( ) - dt.timedelta( days=7 )
					),
					key='usgsearthquakes_start_date',
					disabled=(usgseq_mode != 'search')
				)
			
			with date_c2:
				usgseq_end_date = st.date_input(
					'End Date',
					value=st.session_state.get(
						'usgsearthquakes_end_date',
						dt.date.today( )
					),
					key='usgsearthquakes_end_date',
					disabled=(usgseq_mode != 'search')
				)
			
			mag_c1, mag_c2 = st.columns( 2 )
			
			with mag_c1:
				usgseq_min_magnitude = st.number_input(
					'Min Magnitude',
					min_value=0.0,
					max_value=10.0,
					value=float( st.session_state.get( 'usgsearthquakes_min_magnitude', 1.0 ) ),
					step=0.1,
					key='usgsearthquakes_min_magnitude',
					disabled=(usgseq_mode != 'search')
				)
			
			with mag_c2:
				usgseq_max_magnitude = st.number_input(
					'Max Magnitude',
					min_value=0.0,
					max_value=10.0,
					value=float( st.session_state.get( 'usgsearthquakes_max_magnitude', 10.0 ) ),
					step=0.1,
					key='usgsearthquakes_max_magnitude',
					disabled=(usgseq_mode != 'search')
				)
			
			opt_c1, opt_c2 = st.columns( 2 )
			
			with opt_c1:
				usgseq_limit = st.number_input(
					'Limit',
					min_value=1,
					max_value=200,
					value=int( st.session_state.get( 'usgsearthquakes_limit', 25 ) ),
					step=1,
					key='usgsearthquakes_limit',
					disabled=(usgseq_mode != 'search')
				)
			
			with opt_c2:
				usgseq_order_by = st.selectbox(
					'Order By',
					options=[ 'time', 'time-asc', 'magnitude', 'magnitude-asc' ],
					index=[ 'time', 'time-asc', 'magnitude', 'magnitude-asc' ].index(
						st.session_state.get( 'usgsearthquakes_order_by', 'time' )
					),
					key='usgsearthquakes_order_by',
					disabled=(usgseq_mode != 'search')
				)
			
			usgseq_event_type = st.text_input(
				'Event Type',
				value=st.session_state.get( 'usgsearthquakes_event_type', 'earthquake' ),
				key='usgsearthquakes_event_type',
				disabled=(usgseq_mode != 'search')
			)
			
			coord_c1, coord_c2 = st.columns( 2 )
			
			with coord_c1:
				usgseq_latitude = st.text_input(
					'Latitude (optional)',
					value=st.session_state.get( 'usgsearthquakes_latitude', '' ),
					key='usgsearthquakes_latitude',
					disabled=(usgseq_mode != 'search')
				)
			
			with coord_c2:
				usgseq_longitude = st.text_input(
					'Longitude (optional)',
					value=st.session_state.get( 'usgsearthquakes_longitude', '' ),
					key='usgsearthquakes_longitude',
					disabled=(usgseq_mode != 'search')
				)
			
			usgseq_max_radius_km = st.text_input(
				'Max Radius KM (optional)',
				value=st.session_state.get( 'usgsearthquakes_max_radius_km', '' ),
				key='usgsearthquakes_max_radius_km',
				disabled=(usgseq_mode != 'search')
			)
			
			usgseq_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'usgsearthquakes_timeout', 20 ) ),
				step=1,
				key='usgsearthquakes_timeout'
			)
			
			st.caption(
				'Feed mode is best for quick display. Search mode supports date, '
				'magnitude, and optional radial filtering.'
			)
			
			btn_c1, btn_c2 = st.columns( 2 )
			
			with btn_c1:
				usgseq_submit = st.button(
					'Submit',
					key='usgsearthquakes_submit'
				)
			
			with btn_c2:
				st.button(
					'Clear',
					key='usgsearthquakes_clear',
					on_click=_clear_usgsearthquakes_state
				)
		
		with col_right:
			if usgseq_submit:
				try:
					f = USGSEarthquakes( )
					
					latitude_value = None
					longitude_value = None
					max_radius_value = None
					
					if str( usgseq_latitude or '' ).strip( ):
						latitude_value = float( usgseq_latitude )
					
					if str( usgseq_longitude or '' ).strip( ):
						longitude_value = float( usgseq_longitude )
					
					if str( usgseq_max_radius_km or '' ).strip( ):
						max_radius_value = float( usgseq_max_radius_km )
					
					result = f.fetch(
						mode=str( usgseq_mode ),
						feed=str( usgseq_feed ),
						start_date=str( usgseq_start_date ),
						end_date=str( usgseq_end_date ),
						min_magnitude=float( usgseq_min_magnitude ),
						max_magnitude=float( usgseq_max_magnitude ),
						limit=int( usgseq_limit ),
						order_by=str( usgseq_order_by ),
						event_type=str( usgseq_event_type or 'earthquake' ),
						latitude=latitude_value,
						longitude=longitude_value,
						max_radius_km=max_radius_value,
						time=int( usgseq_timeout )
					)
					
					st.session_state[ 'usgsearthquakes_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'USGS Earthquakes request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'usgsearthquakes_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'feed' in result:
						st.markdown( f"**Feed:** {result.get( 'feed', '' )}" )
					if 'count' in result:
						st.markdown( f"**Events Returned:** {result.get( 'count', 0 )}" )
				
				with meta_c2:
					if 'title' in result:
						st.markdown( f"**Title:** {result.get( 'title', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '##### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						max_mag = summary.get( 'max_magnitude', None )
						st.metric(
							'Strongest Magnitude',
							'' if max_mag is None else str( max_mag )
						)
					
					with sum_c3:
						strongest_place = str( summary.get( 'strongest_place', '' ) or '' )
						if strongest_place:
							st.markdown( f"**Strongest Event:** {strongest_place}" )
						else:
							st.markdown( '**Strongest Event:** N/A' )
					
					most_recent = str( summary.get( 'most_recent', '' ) or '' )
					if most_recent:
						st.caption( f'Most recent event in current result set: {most_recent}' )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '##### Events' )
					df_eq = pd.DataFrame( rows )
					
					if not df_eq.empty:
						st.dataframe( df_eq, use_container_width=True, hide_index=True )
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str( item.get( 'Place', '' ) or f'Event {idx}' )
							magnitude = item.get( 'Magnitude', '' )
							time_value = str( item.get( 'Time', '' ) or '' )
							
							with st.expander(
									f'Event {idx}: M{magnitude} - {label}',
									expanded=False
							):
								detail_c1, detail_c2 = st.columns( 2 )
								
								with detail_c1:
									st.markdown( f"**Time:** {time_value}" )
									st.markdown(
										f"**Depth (km):** {item.get( 'Depth (km)', '' )}"
									)
									st.markdown(
										f"**Latitude:** {item.get( 'Latitude', '' )}"
									)
									st.markdown(
										f"**Longitude:** {item.get( 'Longitude', '' )}"
									)
								
								with detail_c2:
									st.markdown( f"**Status:** {item.get( 'Status', '' )}" )
									st.markdown( f"**Alert:** {item.get( 'Alert', '' )}" )
									st.markdown(
										f"**Tsunami:** {item.get( 'Tsunami', '' )}"
									)
									st.markdown(
										f"**Felt Reports:** {item.get( 'Felt Reports', '' )}"
									)
								
								url_value = str( item.get( 'URL', '' ) or '' )
								if url_value:
									st.markdown( f"**URL:** {url_value}" )
					else:
						st.info( 'No displayable earthquake rows were found.' )
				else:
					st.info( 'No earthquake events were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- Earth Observatory
	with st.expander( label='NASA Earth Observatory', expanded=False ):
		if 'earthobservatory_results' not in st.session_state:
			st.session_state[ 'earthobservatory_results' ] = { }
		
		if 'earthobservatory_clear_request' not in st.session_state:
			st.session_state[ 'earthobservatory_clear_request' ] = False
		
		if st.session_state.get( 'earthobservatory_clear_request', False ):
			st.session_state[ 'earthobservatory_mode' ] = 'events'
			st.session_state[ 'earthobservatory_status' ] = 'open'
			st.session_state[ 'earthobservatory_category' ] = ''
			st.session_state[ 'earthobservatory_source' ] = ''
			st.session_state[ 'earthobservatory_limit' ] = 20
			st.session_state[ 'earthobservatory_days' ] = 30
			st.session_state[ 'earthobservatory_start_date' ] = ''
			st.session_state[ 'earthobservatory_end_date' ] = ''
			st.session_state[ 'earthobservatory_timeout' ] = 20
			st.session_state[ 'earthobservatory_results' ] = { }
			st.session_state[ 'earthobservatory_clear_request' ] = False
		
		def _clear_earthobservatory_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the Earth Observatory expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'earthobservatory_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			earth_mode = st.selectbox(
				'Mode',
				options=[ 'events', 'categories', 'sources', 'layers' ],
				index=[ 'events', 'categories', 'sources', 'layers' ].index(
					st.session_state.get( 'earthobservatory_mode', 'events' )
				),
				key='earthobservatory_mode',
				help='Choose the current documented EONET v3 endpoint.'
			)
			
			earth_status = st.selectbox(
				'Status',
				options=[ 'open', 'closed', 'all' ],
				index=[ 'open', 'closed', 'all' ].index(
					st.session_state.get( 'earthobservatory_status', 'open' )
				),
				key='earthobservatory_status',
				disabled=(earth_mode != 'events')
			)
			
			earth_category = st.text_input(
				'Category',
				value=st.session_state.get( 'earthobservatory_category', '' ),
				key='earthobservatory_category',
				placeholder='Examples: wildfires, severe storms, volcanoes ',
				help='Used for events filtering and layers category path.',
				disabled=(earth_mode not in [ 'events', 'layers' ])
			)
			
			earth_source = st.text_input(
				'Source',
				value=st.session_state.get( 'earthobservatory_source', '' ),
				key='earthobservatory_source',
				placeholder=(
						'Examples: InciWeb, InciWeb, EO'
				),
				disabled=(earth_mode != 'events')
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				earth_limit = st.number_input(
					'Limit',
					min_value=1,
					max_value=500,
					value=int( st.session_state.get( 'earthobservatory_limit', 20 ) ),
					step=1,
					key='earthobservatory_limit',
					disabled=(earth_mode != 'events')
				)
			
			with c2:
				earth_days = st.number_input(
					'Days',
					min_value=1,
					max_value=3650,
					value=int( st.session_state.get( 'earthobservatory_days', 30 ) ),
					step=1,
					key='earthobservatory_days',
					disabled=(earth_mode != 'events')
				)
			
			d1, d2 = st.columns( 2 )
			
			with d1:
				earth_start_date = st.text_input(
					'Start Date',
					value=st.session_state.get( 'earthobservatory_start_date', '' ),
					key='earthobservatory_start_date',
					placeholder='2026-03-01',
					disabled=(earth_mode != 'events')
				)
			
			with d2:
				earth_end_date = st.text_input(
					'End Date',
					value=st.session_state.get( 'earthobservatory_end_date', '' ),
					key='earthobservatory_end_date',
					placeholder='2026-03-15',
					disabled=(earth_mode != 'events')
				)
			
			earth_timeout = st.number_input(
				'Timeout',
				min_value=1,
				max_value=120,
				value=int( st.session_state.get( 'earthobservatory_timeout', 20 ) ),
				step=1,
				key='earthobservatory_timeout'
			)
			
			st.caption(
				'Examples: use events + category=wildfires, status=open; '
				'use sources to list event source providers; '
				'use layers + category=wildfires to inspect imagery layers.'
			)
			
			b1, b2 = st.columns( 2 )
			
			with b1:
				earth_submit = st.button(
					'Submit',
					key='earthobservatory_submit'
				)
			
			with b2:
				st.button(
					'Clear',
					key='earthobservatory_clear',
					on_click=_clear_earthobservatory_state
				)
		
		with col_right:
			if earth_submit:
				try:
					f = EarthObservatory( )
					result = f.fetch(
						mode=earth_mode,
						status=earth_status,
						category=earth_category,
						source=earth_source,
						limit=int( earth_limit ),
						days=int( earth_days ),
						start_date=str( earth_start_date ),
						end_date=str( earth_end_date ),
						time=int( earth_timeout )
					)
					
					st.session_state[ 'earthobservatory_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Earth Observatory request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'earthobservatory_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					if result.get( 'params', { } ):
						st.markdown( f"**Parameters:** {len( result.get( 'params', { } ) )}" )
				
				if result.get( 'params', { } ):
					st.markdown( '#### Request Parameters' )
					st.json( result.get( 'params', { } ) )
				
				if result.get( 'events', [ ] ):
					st.markdown( '#### Events' )
					df_events = pd.DataFrame( result.get( 'events', [ ] ) )
					st.dataframe( df_events, use_container_width=True, hide_index=True )
				
				if result.get( 'categories', [ ] ):
					st.markdown( '#### Categories' )
					df_categories = pd.DataFrame( result.get( 'categories', [ ] ) )
					st.dataframe( df_categories, use_container_width=True, hide_index=True )
				
				if result.get( 'sources', [ ] ):
					st.markdown( '#### Sources' )
					df_sources = pd.DataFrame( result.get( 'sources', [ ] ) )
					st.dataframe( df_sources, use_container_width=True, hide_index=True )
				
				if result.get( 'layers', [ ] ):
					st.markdown( '#### Layers' )
					df_layers = pd.DataFrame( result.get( 'layers', [ ] ) )
					st.dataframe( df_layers, use_container_width=True, hide_index=True )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- USGS Water Data
	with st.expander( label='USGS Water Data', expanded=False ):
		if 'usgswaterdata_results' not in st.session_state:
			st.session_state[ 'usgswaterdata_results' ] = { }
		
		if 'usgswaterdata_clear_request' not in st.session_state:
			st.session_state[ 'usgswaterdata_clear_request' ] = False
		
		if st.session_state.get( 'usgswaterdata_clear_request', False ):
			st.session_state[ 'usgswaterdata_mode' ] = 'monitoring-locations'
			st.session_state[ 'usgswaterdata_monitoring_location_id' ] = ''
			st.session_state[ 'usgswaterdata_state_code' ] = ''
			st.session_state[ 'usgswaterdata_county_code' ] = ''
			st.session_state[ 'usgswaterdata_site_type' ] = ''
			st.session_state[ 'usgswaterdata_parameter_code' ] = ''
			st.session_state[ 'usgswaterdata_limit' ] = 25
			st.session_state[ 'usgswaterdata_timeout' ] = 20
			st.session_state[ 'usgswaterdata_results' ] = { }
			st.session_state[ 'usgswaterdata_clear_request' ] = False
		
		def _clear_usgswaterdata_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the USGS Water Data expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'usgswaterdata_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			usgswd_mode = st.selectbox(
				'Mode',
				options=[
						'monitoring-locations',
						'time-series-metadata',
						'latest-continuous',
						'latest-daily'
				],
				index=[
						'monitoring-locations',
						'time-series-metadata',
						'latest-continuous',
						'latest-daily'
				].index(
					st.session_state.get(
						'usgswaterdata_mode',
						'monitoring-locations'
					)
				),
				key='usgswaterdata_mode'
			)
			
			usgswd_monitoring_location_id = st.text_input(
				'Monitoring Location ID',
				value=st.session_state.get(
					'usgswaterdata_monitoring_location_id',
					''
				),
				key='usgswaterdata_monitoring_location_id',
				placeholder='Example: USGS-01491000'
			)
			
			meta_c1, meta_c2 = st.columns( 2 )
			
			with meta_c1:
				usgswd_state_code = st.text_input(
					'State Code',
					value=st.session_state.get( 'usgswaterdata_state_code', '' ),
					key='usgswaterdata_state_code',
					disabled=(usgswd_mode != 'monitoring-locations'),
					placeholder='Example: VA'
				)
			
			with meta_c2:
				usgswd_county_code = st.text_input(
					'County Code',
					value=st.session_state.get( 'usgswaterdata_county_code', '' ),
					key='usgswaterdata_county_code',
					disabled=(usgswd_mode != 'monitoring-locations'),
					placeholder='Optional'
				)
			
			usgswd_site_type = st.text_input(
				'Site Type',
				value=st.session_state.get( 'usgswaterdata_site_type', '' ),
				key='usgswaterdata_site_type',
				disabled=(usgswd_mode != 'monitoring-locations'),
				placeholder='Examples: ST, LK, GW'
			)
			
			usgswd_parameter_code = st.text_input(
				'Parameter Code',
				value=st.session_state.get( 'usgswaterdata_parameter_code', '' ),
				key='usgswaterdata_parameter_code',
				disabled=(usgswd_mode == 'monitoring-locations'),
				placeholder='Example: 00060'
			)
			
			usgswd_limit = st.number_input(
				'Limit',
				min_value=1,
				max_value=200,
				value=int( st.session_state.get( 'usgswaterdata_limit', 25 ) ),
				step=1,
				key='usgswaterdata_limit'
			)
			
			usgswd_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'usgswaterdata_timeout', 20 ) ),
				step=1,
				key='usgswaterdata_timeout'
			)
			
			st.caption(
				'Monitoring Locations supports site discovery. The other modes '
				'focus on parameter metadata and latest reported values.'
			)
			
			btn_c1, btn_c2 = st.columns( 2 )
			
			with btn_c1:
				usgswd_submit = st.button(
					'Submit',
					key='usgswaterdata_submit'
				)
			
			with btn_c2:
				st.button(
					'Clear',
					key='usgswaterdata_clear',
					on_click=_clear_usgswaterdata_state
				)
		
		with col_right:
			if usgswd_submit:
				try:
					f = USGSWaterData( )
					result = f.fetch(
						mode=str( usgswd_mode ),
						monitoring_location_id=str(
							usgswd_monitoring_location_id
						).strip( ),
						state_code=str( usgswd_state_code ).strip( ),
						county_code=str( usgswd_county_code ).strip( ),
						site_type=str( usgswd_site_type ).strip( ),
						parameter_code=str( usgswd_parameter_code ).strip( ),
						limit=int( usgswd_limit ),
						time=int( usgswd_timeout )
					)
					
					st.session_state[ 'usgswaterdata_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'USGS Water Data request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'usgswaterdata_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'monitoring_location_id' in params:
						st.markdown(
							f"**Monitoring Location ID:** "
							f"{params.get( 'monitoring_location_id', '' )}"
						)
					if 'parameter_code' in params:
						st.markdown(
							f"**Parameter Code:** {params.get( 'parameter_code', '' )}"
						)
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						first_site = str( summary.get( 'first_site', '' ) or '' )
						if first_site:
							st.markdown( f"**First Site:** {first_site}" )
						else:
							st.markdown( '**First Site:** N/A' )
					
					with sum_c3:
						first_parameter = str(
							summary.get( 'first_parameter', '' ) or ''
						)
						first_value = str( summary.get( 'first_value', '' ) or '' )
						
						if first_parameter and first_value:
							st.markdown(
								f"**Sample Value:** {first_parameter} = {first_value}"
							)
						elif first_parameter:
							st.markdown(
								f"**Sample Parameter:** {first_parameter}"
							)
						else:
							st.markdown( '**Sample Value:** N/A' )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### Results' )
					df_usgswd = pd.DataFrame( rows )
					
					if not df_usgswd.empty:
						st.dataframe(
							df_usgswd,
							use_container_width=True,
							hide_index=True
						)
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str(
								item.get( 'Name', '' ) or
								item.get( 'Monitoring Location ID', '' ) or
								f'Record {idx}'
							)
							
							with st.expander(
									f'Record {idx}: {label}',
									expanded=False
							):
								st.json( item )
					else:
						st.info( 'No displayable USGS Water Data rows were found.' )
				else:
					st.info( 'No water data records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- USGS The National Map
	with st.expander( label='The National Map', expanded=False ):
		if 'usgstnm_results' not in st.session_state:
			st.session_state[ 'usgstnm_results' ] = { }
		
		if 'usgstnm_clear_request' not in st.session_state:
			st.session_state[ 'usgstnm_clear_request' ] = False
		
		if st.session_state.get( 'usgstnm_clear_request', False ):
			st.session_state[ 'usgstnm_mode' ] = 'products'
			st.session_state[ 'usgstnm_dataset' ] = ''
			st.session_state[ 'usgstnm_q' ] = ''
			st.session_state[ 'usgstnm_bbox' ] = ''
			st.session_state[ 'usgstnm_prod_formats' ] = ''
			st.session_state[ 'usgstnm_max_items' ] = 25
			st.session_state[ 'usgstnm_offset' ] = 0
			st.session_state[ 'usgstnm_timeout' ] = 20
			st.session_state[ 'usgstnm_results' ] = { }
			st.session_state[ 'usgstnm_clear_request' ] = False
		
		def _clear_usgstnm_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the USGS The National Map expander state for reset on the next
				rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'usgstnm_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			usgstnm_mode = st.selectbox(
				'Mode',
				options=[ 'products', 'datasets' ],
				index=[ 'products', 'datasets' ].index(
					st.session_state.get( 'usgstnm_mode', 'products' )
				),
				key='usgstnm_mode'
			)
			
			usgstnm_dataset = st.text_input(
				'Dataset',
				value=st.session_state.get( 'usgstnm_dataset', '' ),
				key='usgstnm_dataset',
				disabled=(usgstnm_mode != 'products'),
				placeholder='Example: Digital Elevation Model (DEM) 1 meter'
			)
			
			usgstnm_q = st.text_input(
				'Search Text',
				value=st.session_state.get( 'usgstnm_q', '' ),
				key='usgstnm_q',
				disabled=(usgstnm_mode != 'products'),
				placeholder='Optional keyword search'
			)
			
			usgstnm_bbox = st.text_input(
				'Bounding Box',
				value=st.session_state.get( 'usgstnm_bbox', '' ),
				key='usgstnm_bbox',
				disabled=(usgstnm_mode != 'products'),
				placeholder='minx,miny,maxx,maxy'
			)
			
			usgstnm_prod_formats = st.text_input(
				'Product Formats',
				value=st.session_state.get( 'usgstnm_prod_formats', '' ),
				key='usgstnm_prod_formats',
				disabled=(usgstnm_mode != 'products'),
				placeholder='Examples: GeoTIFF, IMG, LAS, LAZ'
			)
			
			page_c1, page_c2 = st.columns( 2 )
			
			with page_c1:
				usgstnm_max_items = st.number_input(
					'Max Items',
					min_value=1,
					max_value=500,
					value=int( st.session_state.get( 'usgstnm_max_items', 25 ) ),
					step=1,
					key='usgstnm_max_items',
					disabled=(usgstnm_mode != 'products')
				)
			
			with page_c2:
				usgstnm_offset = st.number_input(
					'Offset',
					min_value=0,
					max_value=10000,
					value=int( st.session_state.get( 'usgstnm_offset', 0 ) ),
					step=1,
					key='usgstnm_offset',
					disabled=(usgstnm_mode != 'products')
				)
			
			usgstnm_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'usgstnm_timeout', 20 ) ),
				step=1,
				key='usgstnm_timeout'
			)
			
			st.caption(
				'Products mode searches downloadable TNM records. Datasets mode '
				'returns the dataset catalog for discovery.'
			)
			
			btn_c1, btn_c2 = st.columns( 2 )
			
			with btn_c1:
				usgstnm_submit = st.button(
					'Submit',
					key='usgstnm_submit'
				)
			
			with btn_c2:
				st.button(
					'Clear',
					key='usgstnm_clear',
					on_click=_clear_usgstnm_state
				)
		
		with col_right:
			if usgstnm_submit:
				try:
					f = USGSTheNationalMap( )
					result = f.fetch(
						mode=str( usgstnm_mode ),
						dataset=str( usgstnm_dataset ).strip( ),
						q=str( usgstnm_q ).strip( ),
						bbox=str( usgstnm_bbox ).strip( ),
						prod_formats=str( usgstnm_prod_formats ).strip( ),
						max_items=int( usgstnm_max_items ),
						offset=int( usgstnm_offset ),
						time=int( usgstnm_timeout )
					)
					
					st.session_state[ 'usgstnm_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'USGS The National Map request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'usgstnm_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'datasets' in params:
						st.markdown(
							f"**Dataset Filter:** {params.get( 'datasets', '' )}"
						)
					if 'bbox' in params:
						st.markdown(
							f"**Bounding Box:** {params.get( 'bbox', '' )}"
						)
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						first_title = str( summary.get( 'first_title', '' ) or '' )
						if first_title:
							st.markdown( f"**First Result:** {first_title}" )
						else:
							st.markdown( '**First Result:** N/A' )
					
					with sum_c3:
						first_dataset = str( summary.get( 'first_dataset', '' ) or '' )
						if first_dataset:
							st.markdown( f"**Dataset:** {first_dataset}" )
						else:
							st.markdown( '**Dataset:** N/A' )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### Results' )
					df_usgstnm = pd.DataFrame( rows )
					
					if not df_usgstnm.empty:
						st.dataframe(
							df_usgstnm,
							use_container_width=True,
							hide_index=True
						)
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str(
								item.get( 'Title', '' ) or
								item.get( 'Name', '' ) or
								f'Record {idx}'
							)
							
							with st.expander(
									f'Record {idx}: {label}',
									expanded=False
							):
								left_c, right_c = st.columns( 2 )
								
								with left_c:
									if 'Dataset' in item:
										st.markdown(
											f"**Dataset:** {item.get( 'Dataset', '' )}"
										)
									if 'Format' in item:
										st.markdown(
											f"**Format:** {item.get( 'Format', '' )}"
										)
									if 'Publication Date' in item:
										st.markdown(
											f"**Publication Date:** "
											f"{item.get( 'Publication Date', '' )}"
										)
								
								with right_c:
									if 'Bounding Box' in item:
										st.markdown(
											f"**Bounding Box:** "
											f"{item.get( 'Bounding Box', '' )}"
										)
									if 'Download URL' in item:
										st.markdown(
											f"**Download URL:** "
											f"{item.get( 'Download URL', '' )}"
										)
									if 'Metadata URL' in item:
										st.markdown(
											f"**Metadata URL:** "
											f"{item.get( 'Metadata URL', '' )}"
										)
								
								if 'Description' in item and item.get( 'Description', '' ):
									st.markdown(
										f"**Description:** {item.get( 'Description', '' )}"
									)
					else:
						st.info( 'No displayable TNM rows were found.' )
				else:
					st.info( 'No TNM records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- USGS ScienceBase
	with st.expander( label='USGS Science Base', expanded=False ):
		if 'usgssb_results' not in st.session_state:
			st.session_state[ 'usgssb_results' ] = { }
		
		if 'usgssb_clear_request' not in st.session_state:
			st.session_state[ 'usgssb_clear_request' ] = False
		
		if st.session_state.get( 'usgssb_clear_request', False ):
			st.session_state[ 'usgssb_mode' ] = 'items'
			st.session_state[ 'usgssb_q' ] = ''
			st.session_state[ 'usgssb_item_id' ] = ''
			st.session_state[ 'usgssb_max_items' ] = 25
			st.session_state[ 'usgssb_offset' ] = 0
			st.session_state[ 'usgssb_fields' ] = ''
			st.session_state[ 'usgssb_timeout' ] = 20
			st.session_state[ 'usgssb_results' ] = { }
			st.session_state[ 'usgssb_clear_request' ] = False
		
		def _clear_usgssb_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the USGS ScienceBase expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'usgssb_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		with col_left:
			usgssb_mode = st.selectbox(
				'Mode',
				options=[ 'items', 'item' ],
				index=[ 'items', 'item' ].index(
					st.session_state.get( 'usgssb_mode', 'items' )
				),
				key='usgssb_mode'
			)
			
			usgssb_q = st.text_input(
				'Search Query',
				value=st.session_state.get( 'usgssb_q', '' ),
				key='usgssb_q',
				disabled=(usgssb_mode != 'items'),
				placeholder='Optional keyword search'
			)
			
			usgssb_item_id = st.text_input(
				'Item ID',
				value=st.session_state.get( 'usgssb_item_id', '' ),
				key='usgssb_item_id',
				disabled=(usgssb_mode != 'item'),
				placeholder='ScienceBase item identifier'
			)
			
			page_c1, page_c2 = st.columns( 2 )
			
			with page_c1:
				usgssb_max_items = st.number_input(
					'Max Items',
					min_value=1,
					max_value=500,
					value=int( st.session_state.get( 'usgssb_max_items', 25 ) ),
					step=1,
					key='usgssb_max_items',
					disabled=(usgssb_mode != 'items')
				)
			
			with page_c2:
				usgssb_offset = st.number_input(
					'Offset',
					min_value=0,
					max_value=10000,
					value=int( st.session_state.get( 'usgssb_offset', 0 ) ),
					step=1,
					key='usgssb_offset',
					disabled=(usgssb_mode != 'items')
				)
			
			usgssb_fields = st.text_input(
				'Fields',
				value=st.session_state.get( 'usgssb_fields', '' ),
				key='usgssb_fields',
				disabled=(usgssb_mode != 'items'),
				placeholder='Optional fields selector'
			)
			
			usgssb_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'usgssb_timeout', 20 ) ),
				step=1,
				key='usgssb_timeout'
			)
			
			st.caption(
				'Items mode performs catalog discovery. Item mode retrieves a single '
				'ScienceBase record by identifier.'
			)
			
			btn_c1, btn_c2 = st.columns( 2 )
			
			with btn_c1:
				usgssb_submit = st.button(
					'Submit',
					key='usgssb_submit'
				)
			
			with btn_c2:
				st.button(
					'Clear',
					key='usgssb_clear',
					on_click=_clear_usgssb_state
				)
		
		with col_right:
			if usgssb_submit:
				try:
					f = USGSScienceBase( )
					result = f.fetch(
						mode=str( usgssb_mode ),
						q=str( usgssb_q ).strip( ),
						item_id=str( usgssb_item_id ).strip( ),
						max_items=int( usgssb_max_items ),
						offset=int( usgssb_offset ),
						fields=str( usgssb_fields ).strip( ),
						time=int( usgssb_timeout )
					)
					
					st.session_state[ 'usgssb_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'USGS ScienceBase request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'usgssb_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'q' in params:
						st.markdown(
							f"**Search Query:** {params.get( 'q', '' )}"
						)
					if 'fields' in params:
						st.markdown(
							f"**Fields:** {params.get( 'fields', '' )}"
						)
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						first_title = str( summary.get( 'first_title', '' ) or '' )
						if first_title:
							st.markdown( f"**First Result:** {first_title}" )
						else:
							st.markdown( '**First Result:** N/A' )
					
					with sum_c3:
						st.metric(
							'Spatial Records',
							int( summary.get( 'spatial_count', 0 ) or 0 )
						)
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### Results' )
					df_usgssb = pd.DataFrame( rows )
					
					if not df_usgssb.empty:
						st.dataframe(
							df_usgssb,
							use_container_width=True,
							hide_index=True
						)
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str(
								item.get( 'Title', '' ) or
								item.get( 'Id', '' ) or
								f'Record {idx}'
							)
							
							with st.expander(
									f'Record {idx}: {label}',
									expanded=False
							):
								left_c, right_c = st.columns( 2 )
								
								with left_c:
									if 'Type' in item:
										st.markdown(
											f"**Type:** {item.get( 'Type', '' )}"
										)
									if 'Updated' in item:
										st.markdown(
											f"**Updated:** {item.get( 'Updated', '' )}"
										)
									if 'Has Spatial Metadata' in item:
										st.markdown(
											f"**Has Spatial Metadata:** "
											f"{item.get( 'Has Spatial Metadata', '' )}"
										)
								
								with right_c:
									if 'File Count' in item:
										st.markdown(
											f"**File Count:** {item.get( 'File Count', '' )}"
										)
									if 'Web Link Count' in item:
										st.markdown(
											f"**Web Link Count:** "
											f"{item.get( 'Web Link Count', '' )}"
										)
									if 'Contact Count' in item:
										st.markdown(
											f"**Contact Count:** "
											f"{item.get( 'Contact Count', '' )}"
										)
								
								if 'Summary' in item and item.get( 'Summary', '' ):
									st.markdown(
										f"**Summary:** {item.get( 'Summary', '' )}"
									)
					else:
						st.info( 'No displayable ScienceBase rows were found.' )
				else:
					st.info( 'No ScienceBase records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )

# ==============================================================================
# ENVIRONMENTAL MODE
# ==============================================================================
elif mode == 'Environmental':
	st.subheader( f'🌍 Environmental Data' )
	st.divider( )
	
	# -------- AirNow
	with st.expander( label='Air Now', expanded=False ):
		if 'airnow_results' not in st.session_state:
			st.session_state[ 'airnow_results' ] = { }
		
		if 'airnow_clear_request' not in st.session_state:
			st.session_state[ 'airnow_clear_request' ] = False
		
		if st.session_state.get( 'airnow_clear_request', False ):
			st.session_state[ 'airnow_mode' ] = 'current-zip'
			st.session_state[ 'airnow_zip_code' ] = ''
			st.session_state[ 'airnow_latitude' ] = ''
			st.session_state[ 'airnow_longitude' ] = ''
			st.session_state[ 'airnow_date' ] = dt.date.today( )
			st.session_state[ 'airnow_distance' ] = 25
			st.session_state[ 'airnow_timeout' ] = 20
			st.session_state[ 'airnow_results' ] = { }
			st.session_state[ 'airnow_clear_request' ] = False
		
		def _clear_airnow_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the AirNow expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'airnow_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			airnow_mode = st.selectbox(
				'Mode',
				options=[
						'current-zip',
						'current-latlon',
						'forecast-zip',
						'forecast-latlon'
				],
				index=[
						'current-zip',
						'current-latlon',
						'forecast-zip',
						'forecast-latlon'
				].index(
					st.session_state.get( 'airnow_mode', 'current-zip' )
				),
				key='airnow_mode'
			)
			
			airnow_zip_code = st.text_input(
				'Zip Code',
				value=st.session_state.get( 'airnow_zip_code', '' ),
				key='airnow_zip_code',
				disabled=(airnow_mode not in [ 'current-zip', 'forecast-zip' ]),
				placeholder='Example: 22201'
			)
			
			coord_c1, coord_c2 = st.columns( 2 )
			
			with coord_c1:
				airnow_latitude = st.text_input(
					'Latitude',
					value=st.session_state.get( 'airnow_latitude', '' ),
					key='airnow_latitude',
					disabled=(airnow_mode not in [ 'current-latlon', 'forecast-latlon' ]),
					placeholder='Example: 38.8816'
				)
			
			with coord_c2:
				airnow_longitude = st.text_input(
					'Longitude',
					value=st.session_state.get( 'airnow_longitude', '' ),
					key='airnow_longitude',
					disabled=(airnow_mode not in [ 'current-latlon', 'forecast-latlon' ]),
					placeholder='Example: -77.0910'
				)
			
			airnow_date = st.date_input(
				'Forecast Date',
				value=st.session_state.get( 'airnow_date', dt.date.today( ) ),
				key='airnow_date',
				disabled=(airnow_mode not in [ 'forecast-zip', 'forecast-latlon' ])
			)
			
			airnow_distance = st.number_input(
				'Distance (miles)',
				min_value=0,
				max_value=500,
				value=int( st.session_state.get( 'airnow_distance', 25 ) ),
				step=1,
				key='airnow_distance'
			)
			
			airnow_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'airnow_timeout', 20 ) ),
				step=1,
				key='airnow_timeout'
			)
			
			st.caption(
				'AirNow supports current observations and forecasts by Zip code or '
				'latitude/longitude.'
			)
			
			btn_c1, btn_c2 = st.columns( 2 )
			
			with btn_c1:
				airnow_submit = st.button(
					'Submit',
					key='airnow_submit'
				)
			
			with btn_c2:
				st.button(
					'Clear',
					key='airnow_clear',
					on_click=_clear_airnow_state
				)
		
		with col_right:
			if airnow_submit:
				try:
					f = AirNow( )
					
					latitude_value = None
					longitude_value = None
					
					if str( airnow_latitude or '' ).strip( ):
						latitude_value = float( airnow_latitude )
					
					if str( airnow_longitude or '' ).strip( ):
						longitude_value = float( airnow_longitude )
					
					result = f.fetch(
						mode=str( airnow_mode ),
						zip_code=str( airnow_zip_code ).strip( ),
						latitude=latitude_value,
						longitude=longitude_value,
						date=str( airnow_date ),
						distance=int( airnow_distance ),
						time=int( airnow_timeout )
					)
					
					st.session_state[ 'airnow_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'AirNow request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'airnow_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'zipCode' in params:
						st.markdown( f"**Zip Code:** {params.get( 'zipCode', '' )}" )
					if 'distance' in params:
						st.markdown( f"**Distance:** {params.get( 'distance', '' )} miles" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						max_aqi = summary.get( 'max_aqi', None )
						st.metric(
							'Peak AQI',
							'' if max_aqi is None else str( max_aqi )
						)
					
					with sum_c3:
						top_category = str( summary.get( 'top_category', '' ) or '' )
						if top_category:
							st.markdown( f"**Category:** {top_category}" )
						else:
							st.markdown( '**Category:** N/A' )
					
					reporting_area = str( summary.get( 'reporting_area', '' ) or '' )
					dominant_parameter = str(
						summary.get( 'dominant_parameter', '' ) or ''
					)
					
					if reporting_area:
						st.markdown( f"**Reporting Area:** {reporting_area}" )
					if dominant_parameter:
						st.markdown(
							f"**Dominant Parameter in Result Set:** {dominant_parameter}"
						)
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### Air Quality Results' )
					df_airnow = pd.DataFrame( rows )
					
					if not df_airnow.empty:
						st.dataframe(
							df_airnow,
							use_container_width=True,
							hide_index=True
						)
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str(
								item.get( 'Reporting Area', '' ) or f'Record {idx}'
							)
							aqi = item.get( 'AQI', '' )
							parameter = str( item.get( 'Parameter Name', '' ) or '' )
							
							with st.expander(
									f'Record {idx}: AQI {aqi} - {label} - {parameter}',
									expanded=False
							):
								left_c, right_c = st.columns( 2 )
								
								with left_c:
									st.markdown(
										f"**Date Observed:** "
										f"{item.get( 'Date Observed', '' )}"
									)
									st.markdown(
										f"**Hour Observed:** "
										f"{item.get( 'Hour Observed', '' )}"
									)
									st.markdown(
										f"**Reporting Area:** "
										f"{item.get( 'Reporting Area', '' )}"
									)
									st.markdown(
										f"**State Code:** {item.get( 'State Code', '' )}"
									)
								
								with right_c:
									st.markdown(
										f"**Parameter Name:** "
										f"{item.get( 'Parameter Name', '' )}"
									)
									st.markdown( f"**AQI:** {item.get( 'AQI', '' )}" )
									st.markdown(
										f"**Category:** {item.get( 'Category', '' )}"
									)
									st.markdown(
										f"**Action Day:** {item.get( 'Action Day', '' )}"
									)
								
								discussion = str( item.get( 'Discussion', '' ) or '' )
								if discussion:
									st.markdown( f"**Discussion:** {discussion}" )
					else:
						st.info( 'No displayable AirNow rows were found.' )
				else:
					st.info( 'No AirNow records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- NOAA Climate Data
	with st.expander( label='NOAA Climate Data', expanded=False ):
		if 'climatedata_results' not in st.session_state:
			st.session_state[ 'climatedata_results' ] = { }
		
		if 'climatedata_clear_request' not in st.session_state:
			st.session_state[ 'climatedata_clear_request' ] = False
		
		if st.session_state.get( 'climatedata_clear_request', False ):
			st.session_state[ 'climatedata_mode' ] = 'datasets'
			st.session_state[ 'climatedata_keyword' ] = ''
			st.session_state[ 'climatedata_dataset' ] = 'daily-summaries'
			st.session_state[
				'climatedata_start_date' ] = dt.date.today( ) - dt.timedelta( days=30 )
			st.session_state[ 'climatedata_end_date' ] = dt.date.today( )
			st.session_state[ 'climatedata_stations' ] = ''
			st.session_state[ 'climatedata_data_types' ] = ''
			st.session_state[ 'climatedata_limit' ] = 25
			st.session_state[ 'climatedata_offset' ] = 0
			st.session_state[ 'climatedata_timeout' ] = 20
			st.session_state[ 'climatedata_results' ] = { }
			st.session_state[ 'climatedata_clear_request' ] = False
		
		def _clear_climatedata_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the NOAA Climate Data expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'climatedata_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			climatedata_mode = st.selectbox(
				'Mode',
				options=[ 'datasets', 'data' ],
				index=[ 'datasets', 'data' ].index(
					st.session_state.get( 'climatedata_mode', 'datasets' )
				),
				key='climatedata_mode'
			)
			
			climatedata_keyword = st.text_input(
				'Keyword',
				value=st.session_state.get( 'climatedata_keyword', '' ),
				key='climatedata_keyword',
				disabled=(climatedata_mode != 'datasets'),
				placeholder='Optional dataset discovery keyword'
			)
			
			climatedata_dataset = st.text_input(
				'Dataset',
				value=st.session_state.get( 'climatedata_dataset', 'daily-summaries' ),
				key='climatedata_dataset',
				disabled=(climatedata_mode != 'data'),
				placeholder='Example: daily-summaries'
			)
			
			date_c1, date_c2 = st.columns( 2 )
			
			with date_c1:
				climatedata_start_date = st.date_input(
					'Start Date',
					value=st.session_state.get(
						'climatedata_start_date',
						dt.date.today( ) - dt.timedelta( days=30 )
					),
					key='climatedata_start_date'
				)
			
			with date_c2:
				climatedata_end_date = st.date_input(
					'End Date',
					value=st.session_state.get(
						'climatedata_end_date',
						dt.date.today( )
					),
					key='climatedata_end_date'
				)
			
			climatedata_stations = st.text_input(
				'Stations',
				value=st.session_state.get( 'climatedata_stations', '' ),
				key='climatedata_stations',
				disabled=(climatedata_mode != 'data'),
				placeholder='Comma-separated station IDs'
			)
			
			climatedata_data_types = st.text_input(
				'Data Types',
				value=st.session_state.get( 'climatedata_data_types', '' ),
				key='climatedata_data_types',
				disabled=(climatedata_mode != 'data'),
				placeholder='Comma-separated datatype IDs'
			)
			
			page_c1, page_c2 = st.columns( 2 )
			
			with page_c1:
				climatedata_limit = st.number_input(
					'Limit',
					min_value=1,
					max_value=500,
					value=int( st.session_state.get( 'climatedata_limit', 25 ) ),
					step=1,
					key='climatedata_limit'
				)
			
			with page_c2:
				climatedata_offset = st.number_input(
					'Offset',
					min_value=0,
					max_value=10000,
					value=int( st.session_state.get( 'climatedata_offset', 0 ) ),
					step=1,
					key='climatedata_offset',
					disabled=(climatedata_mode != 'datasets')
				)
			
			climatedata_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'climatedata_timeout', 20 ) ),
				step=1,
				key='climatedata_timeout'
			)
			
			st.caption(
				'Datasets mode discovers NOAA climate datasets. Data mode retrieves '
				'subsetted climate records from a selected dataset.'
			)
			
			btn_c1, btn_c2 = st.columns( 2 )
			
			with btn_c1:
				climatedata_submit = st.button(
					'Submit',
					key='climatedata_submit'
				)
			
			with btn_c2:
				st.button(
					'Clear',
					key='climatedata_clear',
					on_click=_clear_climatedata_state
				)
		
		with col_right:
			if climatedata_submit:
				try:
					f = ClimateData( )
					result = f.fetch(
						mode=str( climatedata_mode ),
						keyword=str( climatedata_keyword ).strip( ),
						dataset=str( climatedata_dataset ).strip( ),
						start_date=str( climatedata_start_date ),
						end_date=str( climatedata_end_date ),
						stations=str( climatedata_stations ).strip( ),
						data_types=str( climatedata_data_types ).strip( ),
						limit=int( climatedata_limit ),
						offset=int( climatedata_offset ),
						time=int( climatedata_timeout )
					)
					
					st.session_state[ 'climatedata_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'NOAA Climate Data request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'climatedata_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'dataset' in params:
						st.markdown( f"**Dataset:** {params.get( 'dataset', '' )}" )
					if 'stations' in params:
						st.markdown( f"**Stations:** {params.get( 'stations', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						first_title = str( summary.get( 'first_title', '' ) or '' )
						if first_title:
							st.markdown( f"**First Result:** {first_title}" )
						else:
							st.markdown( '**First Result:** N/A' )
					
					with sum_c3:
						first_dataset = str( summary.get( 'first_dataset', '' ) or '' )
						if first_dataset:
							st.markdown( f"**Dataset/Type:** {first_dataset}" )
						else:
							st.markdown( '**Dataset/Type:** N/A' )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### Climate Results' )
					df_climatedata = pd.DataFrame( rows )
					
					if not df_climatedata.empty:
						st.dataframe(
							df_climatedata,
							use_container_width=True,
							hide_index=True
						)
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str(
								item.get( 'Title', '' ) or
								item.get( 'Station', '' ) or
								item.get( 'Date', '' ) or
								f'Record {idx}'
							)
							
							with st.expander(
									f'Record {idx}: {label}',
									expanded=False
							):
								st.json( item )
					else:
						st.info( 'No displayable climate data rows were found.' )
				else:
					st.info( 'No climate records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- NASA EONET
	with st.expander( label='NASA EONET', expanded=False ):
		if 'eonet_results' not in st.session_state:
			st.session_state[ 'eonet_results' ] = { }
		
		if 'eonet_clear_request' not in st.session_state:
			st.session_state[ 'eonet_clear_request' ] = False
		
		if st.session_state.get( 'eonet_clear_request', False ):
			st.session_state[ 'eonet_mode' ] = 'events'
			st.session_state[ 'eonet_source' ] = ''
			st.session_state[ 'eonet_category' ] = ''
			st.session_state[ 'eonet_status' ] = 'open'
			st.session_state[ 'eonet_limit' ] = 25
			st.session_state[ 'eonet_days' ] = 30
			st.session_state[ 'eonet_start_date' ] = ''
			st.session_state[ 'eonet_end_date' ] = ''
			st.session_state[ 'eonet_bbox' ] = ''
			st.session_state[ 'eonet_timeout' ] = 20
			st.session_state[ 'eonet_results' ] = { }
			st.session_state[ 'eonet_clear_request' ] = False
		
		def _clear_eonet_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the NASA EONET expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'eonet_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			eonet_mode = st.selectbox(
				'Mode',
				options=[ 'events', 'categories' ],
				index=[ 'events', 'categories' ].index(
					st.session_state.get( 'eonet_mode', 'events' )
				),
				key='eonet_mode'
			)
			
			eonet_source = st.text_input(
				'Source',
				value=st.session_state.get( 'eonet_source', '' ),
				key='eonet_source',
				disabled=(eonet_mode != 'events'),
				placeholder='Optional source ID or comma-separated source IDs'
			)
			
			eonet_category = st.text_input(
				'Category',
				value=st.session_state.get( 'eonet_category', '' ),
				key='eonet_category',
				disabled=(eonet_mode != 'events'),
				placeholder='Optional category ID or comma-separated category IDs'
			)
			
			eonet_status = st.selectbox(
				'Status',
				options=[ 'open', 'closed', 'all' ],
				index=[ 'open', 'closed', 'all' ].index(
					st.session_state.get( 'eonet_status', 'open' )
				),
				key='eonet_status',
				disabled=(eonet_mode != 'events')
			)
			
			filter_c1, filter_c2 = st.columns( 2 )
			
			with filter_c1:
				eonet_limit = st.number_input(
					'Limit',
					min_value=1,
					max_value=500,
					value=int( st.session_state.get( 'eonet_limit', 25 ) ),
					step=1,
					key='eonet_limit',
					disabled=(eonet_mode != 'events')
				)
			
			with filter_c2:
				eonet_days = st.number_input(
					'Days',
					min_value=1,
					max_value=3650,
					value=int( st.session_state.get( 'eonet_days', 30 ) ),
					step=1,
					key='eonet_days',
					disabled=(eonet_mode != 'events')
				)
			
			date_c1, date_c2 = st.columns( 2 )
			
			with date_c1:
				eonet_start_date = st.text_input(
					'Start Date',
					value=st.session_state.get( 'eonet_start_date', '' ),
					key='eonet_start_date',
					disabled=(eonet_mode != 'events'),
					placeholder='YYYY-MM-DD'
				)
			
			with date_c2:
				eonet_end_date = st.text_input(
					'End Date',
					value=st.session_state.get( 'eonet_end_date', '' ),
					key='eonet_end_date',
					disabled=(eonet_mode != 'events'),
					placeholder='YYYY-MM-DD'
				)
			
			eonet_bbox = st.text_input(
				'Bounding Box',
				value=st.session_state.get( 'eonet_bbox', '' ),
				key='eonet_bbox',
				disabled=(eonet_mode != 'events'),
				placeholder='min_lon,max_lat,max_lon,min_lat'
			)
			
			eonet_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'eonet_timeout', 20 ) ),
				step=1,
				key='eonet_timeout'
			)
			
			st.caption(
				'Events mode retrieves natural events. Categories mode lists the '
				'available EONET event categories.'
			)
			
			btn_c1, btn_c2 = st.columns( 2 )
			
			with btn_c1:
				eonet_submit = st.button(
					'Submit',
					key='eonet_submit'
				)
			
			with btn_c2:
				st.button(
					'Clear',
					key='eonet_clear',
					on_click=_clear_eonet_state
				)
		
		with col_right:
			if eonet_submit:
				try:
					f = EoNet( )
					result = f.fetch(
						mode=str( eonet_mode ),
						source=str( eonet_source ).strip( ),
						category=str( eonet_category ).strip( ),
						status=str( eonet_status ).strip( ),
						limit=int( eonet_limit ),
						days=int( eonet_days ),
						start_date=str( eonet_start_date ).strip( ),
						end_date=str( eonet_end_date ).strip( ),
						bbox=str( eonet_bbox ).strip( ),
						time=int( eonet_timeout )
					)
					
					st.session_state[ 'eonet_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'NASA EONET request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'eonet_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'category' in params:
						st.markdown( f"**Category Filter:** {params.get( 'category', '' )}" )
					if 'status' in params:
						st.markdown( f"**Status:** {params.get( 'status', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						st.metric(
							'Open Records',
							int( summary.get( 'open_count', 0 ) or 0 )
						)
					
					with sum_c3:
						first_categories = str(
							summary.get( 'first_categories', '' ) or ''
						)
						if first_categories:
							st.markdown( f"**First Categories:** {first_categories}" )
						else:
							st.markdown( '**First Categories:** N/A' )
					
					first_title = str( summary.get( 'first_title', '' ) or '' )
					if first_title:
						st.markdown( f"**First Result:** {first_title}" )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### EONET Results' )
					df_eonet = pd.DataFrame( rows )
					
					if not df_eonet.empty:
						st.dataframe(
							df_eonet,
							use_container_width=True,
							hide_index=True
						)
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str(
								item.get( 'Title', '' ) or
								f'Record {idx}'
							)
							
							with st.expander(
									f'Record {idx}: {label}',
									expanded=False
							):
								left_c, right_c = st.columns( 2 )
								
								with left_c:
									if 'Status' in item:
										st.markdown( f"**Status:** {item.get( 'Status', '' )}" )
									if 'Categories' in item:
										st.markdown(
											f"**Categories:** {item.get( 'Categories', '' )}"
										)
									if 'Sources' in item:
										st.markdown(
											f"**Sources:** {item.get( 'Sources', '' )}"
										)
								
								with right_c:
									if 'Geometry Count' in item:
										st.markdown(
											f"**Geometry Count:** "
											f"{item.get( 'Geometry Count', '' )}"
										)
									if 'Last Geometry Date' in item:
										st.markdown(
											f"**Last Geometry Date:** "
											f"{item.get( 'Last Geometry Date', '' )}"
										)
									if 'Link' in item:
										st.markdown( f"**Link:** {item.get( 'Link', '' )}" )
								
								description = str( item.get( 'Description', '' ) or '' )
								if description:
									st.markdown( f"**Description:** {description}" )
					else:
						st.info( 'No displayable EONET rows were found.' )
				else:
					st.info( 'No EONET records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- EPA Envirofacts
	with st.expander( label='EPA Envirofacts', expanded=False ):
		if 'envirofacts_results' not in st.session_state:
			st.session_state[ 'envirofacts_results' ] = { }
		
		if 'envirofacts_clear_request' not in st.session_state:
			st.session_state[ 'envirofacts_clear_request' ] = False
		
		if st.session_state.get( 'envirofacts_clear_request', False ):
			st.session_state[ 'envirofacts_table_name' ] = 'TRI_FACILITY'
			st.session_state[ 'envirofacts_state_code' ] = ''
			st.session_state[ 'envirofacts_facility_name' ] = ''
			st.session_state[ 'envirofacts_limit' ] = 25
			st.session_state[ 'envirofacts_timeout' ] = 20
			st.session_state[ 'envirofacts_results' ] = { }
			st.session_state[ 'envirofacts_clear_request' ] = False
		
		def _clear_envirofacts_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the Envirofacts expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'envirofacts_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			envirofacts_table_name = st.selectbox(
				'Table',
				options=[
						'TRI_FACILITY',
						'TRI_RELEASE',
						'EF_W_EMISSIONS_SOURCE_GHG'
				],
				index=[
						'TRI_FACILITY',
						'TRI_RELEASE',
						'EF_W_EMISSIONS_SOURCE_GHG'
				].index(
					st.session_state.get( 'envirofacts_table_name', 'TRI_FACILITY' )
				),
				key='envirofacts_table_name'
			)
			
			envirofacts_state_code = st.text_input(
				'State Code',
				value=st.session_state.get( 'envirofacts_state_code', '' ),
				key='envirofacts_state_code',
				placeholder='Example: VA'
			)
			
			envirofacts_facility_name = st.text_input(
				'Facility Name Prefix',
				value=st.session_state.get( 'envirofacts_facility_name', '' ),
				key='envirofacts_facility_name',
				placeholder='Optional facility-name prefix'
			)
			
			envirofacts_limit = st.number_input(
				'Limit',
				min_value=1,
				max_value=500,
				value=int( st.session_state.get( 'envirofacts_limit', 25 ) ),
				step=1,
				key='envirofacts_limit'
			)
			
			envirofacts_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'envirofacts_timeout', 20 ) ),
				step=1,
				key='envirofacts_timeout'
			)
			
			st.caption(
				'This first-pass Envirofacts wrapper focuses on a constrained set of '
				'common tables so the output remains human-readable and easy to use.'
			)
			
			btn_c1, btn_c2 = st.columns( 2 )
			
			with btn_c1:
				envirofacts_submit = st.button(
					'Submit',
					key='envirofacts_submit'
				)
			
			with btn_c2:
				st.button(
					'Clear',
					key='envirofacts_clear',
					on_click=_clear_envirofacts_state
				)
		
		with col_right:
			if envirofacts_submit:
				try:
					f = EnviroFacts( )
					result = f.fetch(
						table_name=str( envirofacts_table_name ).strip( ),
						state_code=str( envirofacts_state_code ).strip( ),
						facility_name=str( envirofacts_facility_name ).strip( ),
						limit=int( envirofacts_limit ),
						time=int( envirofacts_timeout )
					)
					
					st.session_state[ 'envirofacts_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Envirofacts request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'envirofacts_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'table_name' in result:
						st.markdown( f"**Table:** {result.get( 'table_name', '' )}" )
				
				with meta_c2:
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						first_facility = str(
							summary.get( 'first_facility', '' ) or ''
						)
						if first_facility:
							st.markdown( f"**First Facility:** {first_facility}" )
						else:
							st.markdown( '**First Facility:** N/A' )
					
					with sum_c3:
						first_state = str( summary.get( 'first_state', '' ) or '' )
						if first_state:
							st.markdown( f"**First State:** {first_state}" )
						else:
							st.markdown( '**First State:** N/A' )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### Envirofacts Results' )
					df_envirofacts = pd.DataFrame( rows )
					
					if not df_envirofacts.empty:
						st.dataframe(
							df_envirofacts,
							use_container_width=True,
							hide_index=True
						)
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str(
								item.get( 'Facility Name', '' ) or
								item.get( 'Primary Name', '' ) or
								item.get( 'Name', '' ) or
								f'Record {idx}'
							)
							
							with st.expander(
									f'Record {idx}: {label}',
									expanded=False
							):
								st.json( item )
					else:
						st.info( 'No displayable Envirofacts rows were found.' )
				else:
					st.info( 'No Envirofacts records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- NOAA Tides & Currents
	with st.expander( label='NOAA Tides & Currents', expanded=False ):
		if 'tidesandcurrents_results' not in st.session_state:
			st.session_state[ 'tidesandcurrents_results' ] = { }
		
		if 'tidesandcurrents_clear_request' not in st.session_state:
			st.session_state[ 'tidesandcurrents_clear_request' ] = False
		
		if st.session_state.get( 'tidesandcurrents_clear_request', False ):
			st.session_state[ 'tidesandcurrents_mode' ] = 'water-level'
			st.session_state[ 'tidesandcurrents_station_id' ] = ''
			st.session_state[
				'tidesandcurrents_begin_date' ] = dt.date.today( ) - dt.timedelta( days=1 )
			st.session_state[ 'tidesandcurrents_end_date' ] = dt.date.today( )
			st.session_state[ 'tidesandcurrents_datum' ] = 'MLLW'
			st.session_state[ 'tidesandcurrents_units' ] = 'metric'
			st.session_state[ 'tidesandcurrents_time_zone' ] = 'gmt'
			st.session_state[ 'tidesandcurrents_interval' ] = 'hilo'
			st.session_state[ 'tidesandcurrents_timeout' ] = 20
			st.session_state[ 'tidesandcurrents_results' ] = { }
			st.session_state[ 'tidesandcurrents_clear_request' ] = False
		
		def _clear_tidesandcurrents_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the NOAA Tides & Currents expander state for reset on the next
				rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'tidesandcurrents_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			tac_mode = st.selectbox(
				'Mode',
				options=[ 'station', 'water-level', 'tide-predictions' ],
				index=[ 'station', 'water-level', 'tide-predictions' ].index(
					st.session_state.get( 'tidesandcurrents_mode', 'water-level' )
				),
				key='tidesandcurrents_mode'
			)
			
			tac_station_id = st.text_input(
				'Station ID',
				value=st.session_state.get( 'tidesandcurrents_station_id', '' ),
				key='tidesandcurrents_station_id',
				placeholder='Example: 8724580'
			)
			
			date_c1, date_c2 = st.columns( 2 )
			
			with date_c1:
				tac_begin_date = st.date_input(
					'Begin Date',
					value=st.session_state.get(
						'tidesandcurrents_begin_date',
						dt.date.today( ) - dt.timedelta( days=1 )
					),
					key='tidesandcurrents_begin_date',
					disabled=(tac_mode == 'station')
				)
			
			with date_c2:
				tac_end_date = st.date_input(
					'End Date',
					value=st.session_state.get(
						'tidesandcurrents_end_date',
						dt.date.today( )
					),
					key='tidesandcurrents_end_date',
					disabled=(tac_mode == 'station')
				)
			
			opt_c1, opt_c2 = st.columns( 2 )
			
			with opt_c1:
				tac_datum = st.text_input(
					'Datum',
					value=st.session_state.get( 'tidesandcurrents_datum', 'MLLW' ),
					key='tidesandcurrents_datum',
					disabled=(tac_mode == 'station')
				)
			
			with opt_c2:
				tac_units = st.selectbox(
					'Units',
					options=[ 'metric', 'english' ],
					index=[ 'metric', 'english' ].index(
						st.session_state.get( 'tidesandcurrents_units', 'metric' )
					),
					key='tidesandcurrents_units',
					disabled=(tac_mode == 'station')
				)
			
			opt_c3, opt_c4 = st.columns( 2 )
			
			with opt_c3:
				tac_time_zone = st.selectbox(
					'Time Zone',
					options=[ 'gmt', 'lst', 'lst_ldt' ],
					index=[ 'gmt', 'lst', 'lst_ldt' ].index(
						st.session_state.get( 'tidesandcurrents_time_zone', 'gmt' )
					),
					key='tidesandcurrents_time_zone',
					disabled=(tac_mode == 'station')
				)
			
			with opt_c4:
				tac_interval = st.selectbox(
					'Prediction Interval',
					options=[ 'hilo', 'h' ],
					index=[ 'hilo', 'h' ].index(
						st.session_state.get( 'tidesandcurrents_interval', 'hilo' )
					),
					key='tidesandcurrents_interval',
					disabled=(tac_mode != 'tide-predictions')
				)
			
			tac_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'tidesandcurrents_timeout', 20 ) ),
				step=1,
				key='tidesandcurrents_timeout'
			)
			
			st.caption(
				'Station mode returns metadata. Water-level mode returns observations. '
				'Tide-predictions mode returns predicted tides.'
			)
			
			btn_c1, btn_c2 = st.columns( 2 )
			
			with btn_c1:
				tac_submit = st.button(
					'Submit',
					key='tidesandcurrents_submit'
				)
			
			with btn_c2:
				st.button(
					'Clear',
					key='tidesandcurrents_clear',
					on_click=_clear_tidesandcurrents_state
				)
		
		with col_right:
			if tac_submit:
				try:
					f = TidesAndCurrents( )
					result = f.fetch(
						mode=str( tac_mode ),
						station_id=str( tac_station_id ).strip( ),
						begin_date=dt.datetime.strftime( tac_begin_date, '%Y%m%d' ),
						end_date=dt.datetime.strftime( tac_end_date, '%Y%m%d' ),
						datum=str( tac_datum ).strip( ),
						units=str( tac_units ).strip( ),
						time_zone=str( tac_time_zone ).strip( ),
						interval=str( tac_interval ).strip( ),
						time=int( tac_timeout )
					)
					
					st.session_state[ 'tidesandcurrents_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'NOAA Tides & Currents request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'tidesandcurrents_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'station_id' in result:
						st.markdown( f"**Station ID:** {result.get( 'station_id', '' )}" )
				
				with meta_c2:
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						first_station = str( summary.get( 'first_station', '' ) or '' )
						if first_station:
							st.markdown( f"**Station:** {first_station}" )
						else:
							st.markdown( '**Station:** N/A' )
					
					with sum_c3:
						first_value = str( summary.get( 'first_value', '' ) or '' )
						if first_value:
							st.markdown( f"**Sample Value:** {first_value}" )
						else:
							st.markdown( '**Sample Value:** N/A' )
					
					first_time = str( summary.get( 'first_time', '' ) or '' )
					if first_time:
						st.markdown( f"**First Time:** {first_time}" )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### Tides & Currents Results' )
					df_tac = pd.DataFrame( rows )
					
					if not df_tac.empty:
						st.dataframe(
							df_tac,
							use_container_width=True,
							hide_index=True
						)
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str(
								item.get( 'Name', '' ) or
								item.get( 'Time', '' ) or
								item.get( 'T', '' ) or
								f'Record {idx}'
							)
							
							with st.expander(
									f'Record {idx}: {label}',
									expanded=False
							):
								st.json( item )
					else:
						st.info( 'No displayable Tides & Currents rows were found.' )
				else:
					st.info( 'No Tides & Currents records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- EPA UV Index
	with st.expander( label='EPA UV Index', expanded=False ):
		if 'uvindex_results' not in st.session_state:
			st.session_state[ 'uvindex_results' ] = { }
		
		if 'uvindex_clear_request' not in st.session_state:
			st.session_state[ 'uvindex_clear_request' ] = False
		
		if st.session_state.get( 'uvindex_clear_request', False ):
			st.session_state[ 'uvindex_mode' ] = 'daily-zip'
			st.session_state[ 'uvindex_zip_code' ] = ''
			st.session_state[ 'uvindex_city' ] = ''
			st.session_state[ 'uvindex_state' ] = ''
			st.session_state[ 'uvindex_timeout' ] = 20
			st.session_state[ 'uvindex_results' ] = { }
			st.session_state[ 'uvindex_clear_request' ] = False
		
		def _clear_uvindex_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the EPA UV Index expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'uvindex_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			uvindex_mode = st.selectbox(
				'Mode',
				options=[
						'daily-zip',
						'daily-city-state',
						'hourly-zip',
						'hourly-city-state'
				],
				index=[
						'daily-zip',
						'daily-city-state',
						'hourly-zip',
						'hourly-city-state'
				].index(
					st.session_state.get( 'uvindex_mode', 'daily-zip' )
				),
				key='uvindex_mode'
			)
			
			uvindex_zip_code = st.text_input(
				'Zip Code',
				value=st.session_state.get( 'uvindex_zip_code', '' ),
				key='uvindex_zip_code',
				disabled=(uvindex_mode not in [ 'daily-zip', 'hourly-zip' ]),
				placeholder='Example: 22201'
			)
			
			city_c1, city_c2 = st.columns( 2 )
			
			with city_c1:
				uvindex_city = st.text_input(
					'City',
					value=st.session_state.get( 'uvindex_city', '' ),
					key='uvindex_city',
					disabled=(uvindex_mode not in [ 'daily-city-state', 'hourly-city-state' ]),
					placeholder='Example: Arlington'
				)
			
			with city_c2:
				uvindex_state = st.text_input(
					'State',
					value=st.session_state.get( 'uvindex_state', '' ),
					key='uvindex_state',
					disabled=(uvindex_mode not in [ 'daily-city-state', 'hourly-city-state' ]),
					placeholder='Example: VA'
				)
			
			uvindex_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'uvindex_timeout', 20 ) ),
				step=1,
				key='uvindex_timeout'
			)
			
			st.caption(
				'UV Index forecasts can be retrieved hourly or daily, by ZIP code or '
				'by city and state.'
			)
			
			btn_c1, btn_c2 = st.columns( 2 )
			
			with btn_c1:
				uvindex_submit = st.button(
					'Submit',
					key='uvindex_submit'
				)
			
			with btn_c2:
				st.button(
					'Clear',
					key='uvindex_clear',
					on_click=_clear_uvindex_state
				)
		
		with col_right:
			if uvindex_submit:
				try:
					f = UvIndex( )
					result = f.fetch(
						mode=str( uvindex_mode ),
						zip_code=str( uvindex_zip_code ).strip( ),
						city=str( uvindex_city ).strip( ),
						state=str( uvindex_state ).strip( ),
						time=int( uvindex_timeout )
					)
					
					st.session_state[ 'uvindex_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'EPA UV Index request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'uvindex_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'zip_code' in params:
						st.markdown( f"**Zip Code:** {params.get( 'zip_code', '' )}" )
					if 'city' in params:
						st.markdown(
							f"**City/State:** {params.get( 'city', '' )}, "
							f"{params.get( 'state', '' )}"
						)
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						max_uv = summary.get( 'max_uv', None )
						st.metric(
							'Peak UV',
							'' if max_uv is None else str( max_uv )
						)
					
					with sum_c3:
						first_alert = str( summary.get( 'first_alert', '' ) or '' )
						if first_alert:
							st.markdown( f"**Alert:** {first_alert}" )
						else:
							st.markdown( '**Alert:** N/A' )
					
					first_location = str( summary.get( 'first_location', '' ) or '' )
					if first_location:
						st.markdown( f"**Location:** {first_location}" )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### UV Index Results' )
					df_uvindex = pd.DataFrame( rows )
					
					if not df_uvindex.empty:
						st.dataframe(
							df_uvindex,
							use_container_width=True,
							hide_index=True
						)
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str(
								item.get( 'City', '' ) or
								item.get( 'Zip', '' ) or
								f'Record {idx}'
							)
							
							with st.expander(
									f'Record {idx}: {label}',
									expanded=False
							):
								st.json( item )
					else:
						st.info( 'No displayable UV Index rows were found.' )
				else:
					st.info( 'No UV Index records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- PurpleAir
	with st.expander( label='Purple Air', expanded=False ):
		if 'purpleair_results' not in st.session_state:
			st.session_state[ 'purpleair_results' ] = { }
		
		if 'purpleair_clear_request' not in st.session_state:
			st.session_state[ 'purpleair_clear_request' ] = False
		
		if st.session_state.get( 'purpleair_clear_request', False ):
			st.session_state[ 'purpleair_mode' ] = 'sensors'
			st.session_state[ 'purpleair_sensor_index' ] = ''
			st.session_state[ 'purpleair_nwlng' ] = ''
			st.session_state[ 'purpleair_nwlat' ] = ''
			st.session_state[ 'purpleair_selng' ] = ''
			st.session_state[ 'purpleair_selat' ] = ''
			st.session_state[ 'purpleair_location_type' ] = 0
			st.session_state[ 'purpleair_max_age' ] = 0
			st.session_state[ 'purpleair_modified_since' ] = 0
			st.session_state[ 'purpleair_timeout' ] = 20
			st.session_state[ 'purpleair_results' ] = { }
			st.session_state[ 'purpleair_clear_request' ] = False
		
		def _clear_purpleair_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the PurpleAir expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'purpleair_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			purpleair_mode = st.selectbox(
				'Mode',
				options=[ 'sensors', 'sensor' ],
				index=[ 'sensors', 'sensor' ].index(
					st.session_state.get( 'purpleair_mode', 'sensors' )
				),
				key='purpleair_mode'
			)
			
			purpleair_sensor_index = st.text_input(
				'Sensor Index',
				value=st.session_state.get( 'purpleair_sensor_index', '' ),
				key='purpleair_sensor_index',
				disabled=(purpleair_mode != 'sensor'),
				placeholder='Example: 78307'
			)
			
			bbox_c1, bbox_c2 = st.columns( 2 )
			
			with bbox_c1:
				purpleair_nwlng = st.text_input(
					'NW Longitude',
					value=st.session_state.get( 'purpleair_nwlng', '' ),
					key='purpleair_nwlng',
					disabled=(purpleair_mode != 'sensors')
				)
			
			with bbox_c2:
				purpleair_nwlat = st.text_input(
					'NW Latitude',
					value=st.session_state.get( 'purpleair_nwlat', '' ),
					key='purpleair_nwlat',
					disabled=(purpleair_mode != 'sensors')
				)
			
			bbox_c3, bbox_c4 = st.columns( 2 )
			
			with bbox_c3:
				purpleair_selng = st.text_input(
					'SE Longitude',
					value=st.session_state.get( 'purpleair_selng', '' ),
					key='purpleair_selng',
					disabled=(purpleair_mode != 'sensors')
				)
			
			with bbox_c4:
				purpleair_selat = st.text_input(
					'SE Latitude',
					value=st.session_state.get( 'purpleair_selat', '' ),
					key='purpleair_selat',
					disabled=(purpleair_mode != 'sensors')
				)
			
			opt_c1, opt_c2 = st.columns( 2 )
			
			with opt_c1:
				purpleair_location_type = st.number_input(
					'Location Type',
					min_value=0,
					max_value=10,
					value=int( st.session_state.get( 'purpleair_location_type', 0 ) ),
					step=1,
					key='purpleair_location_type',
					disabled=(purpleair_mode != 'sensors')
				)
			
			with opt_c2:
				purpleair_max_age = st.number_input(
					'Max Age',
					min_value=0,
					max_value=1000000,
					value=int( st.session_state.get( 'purpleair_max_age', 0 ) ),
					step=1,
					key='purpleair_max_age',
					disabled=(purpleair_mode != 'sensors')
				)
			
			purpleair_modified_since = st.number_input(
				'Modified Since',
				min_value=0,
				max_value=2147483647,
				value=int( st.session_state.get( 'purpleair_modified_since', 0 ) ),
				step=1,
				key='purpleair_modified_since',
				disabled=(purpleair_mode != 'sensors')
			)
			
			purpleair_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'purpleair_timeout', 20 ) ),
				step=1,
				key='purpleair_timeout'
			)
			
			st.caption(
				'PurpleAir requires an API key and uses a points system. This wrapper '
				'uses narrow field selection to keep calls focused and readable.'
			)
			
			btn_c1, btn_c2 = st.columns( 2 )
			
			with btn_c1:
				purpleair_submit = st.button(
					'Submit',
					key='purpleair_submit'
				)
			
			with btn_c2:
				st.button(
					'Clear',
					key='purpleair_clear',
					on_click=_clear_purpleair_state
				)
		
		with col_right:
			if purpleair_submit:
				try:
					f = PurpleAir( )
					result = f.fetch(
						mode=str( purpleair_mode ),
						sensor_index=None if not str( purpleair_sensor_index ).strip( ) else int( purpleair_sensor_index ),
						nwlng=None if not str( purpleair_nwlng ).strip( ) else float( purpleair_nwlng ),
						nwlat=None if not str( purpleair_nwlat ).strip( ) else float( purpleair_nwlat ),
						selng=None if not str( purpleair_selng ).strip( ) else float( purpleair_selng ),
						selat=None if not str( purpleair_selat ).strip( ) else float( purpleair_selat ),
						location_type=int( purpleair_location_type ),
						max_age=int( purpleair_max_age ),
						modified_since=int( purpleair_modified_since ),
						time=int( purpleair_timeout )
					)
					
					st.session_state[ 'purpleair_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'PurpleAir request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'purpleair_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'sensor_index' in params:
						st.markdown(
							f"**Sensor Index:** {params.get( 'sensor_index', '' )}"
						)
					if 'fields' in params:
						st.markdown(
							f"**Fields:** {params.get( 'fields', '' )}"
						)
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						max_pm25 = summary.get( 'max_pm25', None )
						st.metric(
							'Peak PM2.5',
							'' if max_pm25 is None else str( max_pm25 )
						)
					
					with sum_c3:
						first_name = str( summary.get( 'first_name', '' ) or '' )
						if first_name:
							st.markdown( f"**First Sensor:** {first_name}" )
						else:
							st.markdown( '**First Sensor:** N/A' )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### PurpleAir Results' )
					df_purpleair = pd.DataFrame( rows )
					
					if not df_purpleair.empty:
						st.dataframe(
							df_purpleair,
							use_container_width=True,
							hide_index=True
						)
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str(
								item.get( 'Name', '' ) or
								item.get( 'Sensor Index', '' ) or
								f'Record {idx}'
							)
							
							with st.expander(
									f'Record {idx}: {label}',
									expanded=False
							):
								st.json( item )
					else:
						st.info( 'No displayable PurpleAir rows were found.' )
				else:
					st.info( 'No PurpleAir records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- OpenAQ
	with st.expander( label='Open Air Quality', expanded=False ):
		if 'openaq_results' not in st.session_state:
			st.session_state[ 'openaq_results' ] = { }
		
		if 'openaq_clear_request' not in st.session_state:
			st.session_state[ 'openaq_clear_request' ] = False
		
		if st.session_state.get( 'openaq_clear_request', False ):
			st.session_state[ 'openaq_mode' ] = 'locations'
			st.session_state[ 'openaq_location_id' ] = ''
			st.session_state[ 'openaq_country_id' ] = ''
			st.session_state[ 'openaq_coordinates' ] = ''
			st.session_state[ 'openaq_radius' ] = 25000
			st.session_state[ 'openaq_providers_id' ] = ''
			st.session_state[ 'openaq_parameters_id' ] = ''
			st.session_state[ 'openaq_limit' ] = 25
			st.session_state[ 'openaq_page' ] = 1
			st.session_state[ 'openaq_timeout' ] = 20
			st.session_state[ 'openaq_results' ] = { }
			st.session_state[ 'openaq_clear_request' ] = False
		
		def _clear_openaq_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the OpenAQ expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'openaq_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			openaq_mode = st.selectbox(
				'Mode',
				options=[ 'locations', 'latest' ],
				index=[ 'locations', 'latest' ].index(
					st.session_state.get( 'openaq_mode', 'locations' )
				),
				key='openaq_mode'
			)
			
			openaq_location_id = st.text_input(
				'Location ID',
				value=st.session_state.get( 'openaq_location_id', '' ),
				key='openaq_location_id',
				disabled=(openaq_mode != 'latest'),
				placeholder='Example: 8118'
			)
			
			openaq_country_id = st.text_input(
				'Country ID',
				value=st.session_state.get( 'openaq_country_id', '' ),
				key='openaq_country_id',
				disabled=(openaq_mode != 'locations'),
				placeholder='Optional numeric country ID'
			)
			
			openaq_coordinates = st.text_input(
				'Coordinates',
				value=st.session_state.get( 'openaq_coordinates', '' ),
				key='openaq_coordinates',
				disabled=(openaq_mode != 'locations'),
				placeholder='latitude,longitude'
			)
			
			openaq_radius = st.number_input(
				'Radius (meters)',
				min_value=1,
				max_value=500000,
				value=int( st.session_state.get( 'openaq_radius', 25000 ) ),
				step=1,
				key='openaq_radius',
				disabled=(openaq_mode != 'locations')
			)
			
			filter_c1, filter_c2 = st.columns( 2 )
			
			with filter_c1:
				openaq_providers_id = st.text_input(
					'Providers ID',
					value=st.session_state.get( 'openaq_providers_id', '' ),
					key='openaq_providers_id',
					disabled=(openaq_mode != 'locations'),
					placeholder='Optional provider ID'
				)
			
			with filter_c2:
				openaq_parameters_id = st.text_input(
					'Parameters ID',
					value=st.session_state.get( 'openaq_parameters_id', '' ),
					key='openaq_parameters_id',
					disabled=(openaq_mode != 'locations'),
					placeholder='Optional parameter ID'
				)
			
			page_c1, page_c2 = st.columns( 2 )
			
			with page_c1:
				openaq_limit = st.number_input(
					'Limit',
					min_value=1,
					max_value=500,
					value=int( st.session_state.get( 'openaq_limit', 25 ) ),
					step=1,
					key='openaq_limit',
					disabled=(openaq_mode != 'locations')
				)
			
			with page_c2:
				openaq_page = st.number_input(
					'Page',
					min_value=1,
					max_value=10000,
					value=int( st.session_state.get( 'openaq_page', 1 ) ),
					step=1,
					key='openaq_page',
					disabled=(openaq_mode != 'locations')
				)
			
			openaq_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'openaq_timeout', 20 ) ),
				step=1,
				key='openaq_timeout'
			)
			
			st.caption(
				'OpenAQ v3 requires an API key. Locations mode supports discovery. '
				'Latest mode retrieves the latest measurements for one location.'
			)
			
			btn_c1, btn_c2 = st.columns( 2 )
			
			with btn_c1:
				openaq_submit = st.button(
					'Submit',
					key='openaq_submit'
				)
			
			with btn_c2:
				st.button(
					'Clear',
					key='openaq_clear',
					on_click=_clear_openaq_state
				)
		
		with col_right:
			if openaq_submit:
				try:
					f = OpenAQ( )
					result = f.fetch(
						mode=str( openaq_mode ),
						location_id=None if not str( openaq_location_id ).strip( ) else int( openaq_location_id ),
						country_id=None if not str( openaq_country_id ).strip( ) else int( openaq_country_id ),
						coordinates=str( openaq_coordinates ).strip( ),
						radius=int( openaq_radius ),
						providers_id=str( openaq_providers_id ).strip( ),
						parameters_id=str( openaq_parameters_id ).strip( ),
						limit=int( openaq_limit ),
						page=int( openaq_page ),
						time=int( openaq_timeout )
					)
					
					st.session_state[ 'openaq_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'OpenAQ request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'openaq_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'location_id' in params:
						st.markdown(
							f"**Location ID:** {params.get( 'location_id', '' )}"
						)
					if 'country_id' in params:
						st.markdown(
							f"**Country ID:** {params.get( 'country_id', '' )}"
						)
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						first_name = str( summary.get( 'first_name', '' ) or '' )
						if first_name:
							st.markdown( f"**First Result:** {first_name}" )
						else:
							st.markdown( '**First Result:** N/A' )
					
					with sum_c3:
						first_parameter = str(
							summary.get( 'first_parameter', '' ) or ''
						)
						if first_parameter:
							st.markdown( f"**First Parameter:** {first_parameter}" )
						else:
							st.markdown( '**First Parameter:** N/A' )
					
					first_country = str( summary.get( 'first_country', '' ) or '' )
					if first_country:
						st.markdown( f"**Country:** {first_country}" )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### OpenAQ Results' )
					df_openaq = pd.DataFrame( rows )
					
					if not df_openaq.empty:
						st.dataframe(
							df_openaq,
							use_container_width=True,
							hide_index=True
						)
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str(
								item.get( 'Name', '' ) or
								item.get( 'Location Id', '' ) or
								item.get( 'Parameter', '' ) or
								f'Record {idx}'
							)
							
							with st.expander(
									f'Record {idx}: {label}',
									expanded=False
							):
								st.json( item )
					else:
						st.info( 'No displayable OpenAQ rows were found.' )
				else:
					st.info( 'No OpenAQ records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- NASA FIRMS
	with st.expander( label='NASA FIRMS', expanded=False ):
		if 'firms_results' not in st.session_state:
			st.session_state[ 'firms_results' ] = { }
		
		if 'firms_clear_request' not in st.session_state:
			st.session_state[ 'firms_clear_request' ] = False
		
		if st.session_state.get( 'firms_clear_request', False ):
			st.session_state[ 'firms_mode' ] = 'area'
			st.session_state[ 'firms_source' ] = 'VIIRS_SNPP_NRT'
			st.session_state[ 'firms_area_coordinates' ] = 'world'
			st.session_state[ 'firms_day_range' ] = 1
			st.session_state[ 'firms_date' ] = ''
			st.session_state[ 'firms_sensor' ] = 'ALL'
			st.session_state[ 'firms_timeout' ] = 20
			st.session_state[ 'firms_results' ] = { }
			st.session_state[ 'firms_clear_request' ] = False
		
		def _clear_firms_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the NASA FIRMS expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'firms_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			firms_mode = st.selectbox(
				'Mode',
				options=[ 'area', 'data-availability' ],
				index=[ 'area', 'data-availability' ].index(
					st.session_state.get( 'firms_mode', 'area' )
				),
				key='firms_mode'
			)
			
			firms_source = st.selectbox(
				'Source',
				options=[
						'LANDSAT_NRT',
						'MODIS_NRT',
						'MODIS_SP',
						'VIIRS_NOAA20_NRT',
						'VIIRS_NOAA20_SP',
						'VIIRS_NOAA21_NRT',
						'VIIRS_SNPP_NRT',
						'VIIRS_SNPP_SP'
				],
				index=[
						'LANDSAT_NRT',
						'MODIS_NRT',
						'MODIS_SP',
						'VIIRS_NOAA20_NRT',
						'VIIRS_NOAA20_SP',
						'VIIRS_NOAA21_NRT',
						'VIIRS_SNPP_NRT',
						'VIIRS_SNPP_SP'
				].index(
					st.session_state.get( 'firms_source', 'VIIRS_SNPP_NRT' )
				),
				key='firms_source',
				disabled=(firms_mode != 'area')
			)
			
			firms_area_coordinates = st.text_input(
				'Area Coordinates',
				value=st.session_state.get( 'firms_area_coordinates', 'world' ),
				key='firms_area_coordinates',
				disabled=(firms_mode != 'area'),
				placeholder='west,south,east,north or world'
			)
			
			firms_day_range = st.number_input(
				'Day Range',
				min_value=1,
				max_value=5,
				value=int( st.session_state.get( 'firms_day_range', 1 ) ),
				step=1,
				key='firms_day_range',
				disabled=(firms_mode != 'area')
			)
			
			firms_date = st.text_input(
				'Date',
				value=st.session_state.get( 'firms_date', '' ),
				key='firms_date',
				disabled=(firms_mode != 'area'),
				placeholder='Optional YYYY-MM-DD'
			)
			
			firms_sensor = st.selectbox(
				'Sensor',
				options=[
						'ALL',
						'LANDSAT_NRT',
						'MODIS_NRT',
						'MODIS_SP',
						'VIIRS_NOAA20_NRT',
						'VIIRS_NOAA20_SP',
						'VIIRS_NOAA21_NRT',
						'VIIRS_SNPP_NRT',
						'VIIRS_SNPP_SP'
				],
				index=[
						'ALL',
						'LANDSAT_NRT',
						'MODIS_NRT',
						'MODIS_SP',
						'VIIRS_NOAA20_NRT',
						'VIIRS_NOAA20_SP',
						'VIIRS_NOAA21_NRT',
						'VIIRS_SNPP_NRT',
						'VIIRS_SNPP_SP'
				].index(
					st.session_state.get( 'firms_sensor', 'ALL' )
				),
				key='firms_sensor',
				disabled=(firms_mode != 'data-availability')
			)
			
			firms_timeout = st.number_input(
				'Timeout (seconds)',
				min_value=5,
				max_value=120,
				value=int( st.session_state.get( 'firms_timeout', 20 ) ),
				step=1,
				key='firms_timeout'
			)
			
			st.caption(
				'FIRMS requires a MAP_KEY. Area mode retrieves fire detections for a '
				'bounding box or world. Data-availability mode reports available dates.'
			)
			
			btn_c1, btn_c2 = st.columns( 2 )
			
			with btn_c1:
				firms_submit = st.button(
					'Submit',
					key='firms_submit'
				)
			
			with btn_c2:
				st.button(
					'Clear',
					key='firms_clear',
					on_click=_clear_firms_state
				)
		
		with col_right:
			if firms_submit:
				try:
					f = Firms( )
					result = f.fetch(
						mode=str( firms_mode ),
						source=str( firms_source ).strip( ),
						area_coordinates=str( firms_area_coordinates ).strip( ),
						day_range=int( firms_day_range ),
						date=str( firms_date ).strip( ),
						sensor=str( firms_sensor ).strip( ),
						time=int( firms_timeout )
					)
					
					st.session_state[ 'firms_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'NASA FIRMS request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'firms_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					params = result.get( 'params', { } ) or { }
					if 'source' in params:
						st.markdown( f"**Source:** {params.get( 'source', '' )}" )
					if 'sensor' in params:
						st.markdown( f"**Sensor:** {params.get( 'sensor', '' )}" )
				
				summary = result.get( 'summary', { } ) or { }
				if summary:
					st.markdown( '#### Result Summary' )
					
					sum_c1, sum_c2, sum_c3 = st.columns( 3 )
					
					with sum_c1:
						st.metric( 'Count', int( summary.get( 'count', 0 ) or 0 ) )
					
					with sum_c2:
						first_date = str( summary.get( 'first_date', '' ) or '' )
						if first_date:
							st.markdown( f"**First Date:** {first_date}" )
						else:
							st.markdown( '**First Date:** N/A' )
					
					with sum_c3:
						first_lat = str( summary.get( 'first_lat', '' ) or '' )
						first_lon = str( summary.get( 'first_lon', '' ) or '' )
						if first_lat and first_lon:
							st.markdown( f"**First Point:** {first_lat}, {first_lon}" )
						else:
							st.markdown( '**First Point:** N/A' )
				
				params = result.get( 'params', { } ) or { }
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				rows = result.get( 'rows', [ ] ) or [ ]
				if rows:
					st.markdown( '#### FIRMS Results' )
					df_firms = pd.DataFrame( rows )
					
					if not df_firms.empty:
						st.dataframe(
							df_firms,
							use_container_width=True,
							hide_index=True
						)
						
						top_rows = rows[ : min( 10, len( rows ) ) ]
						for idx, item in enumerate( top_rows, start=1 ):
							label = str(
								item.get( 'Acq Date', '' ) or
								item.get( 'Date', '' ) or
								f'Record {idx}'
							)
							
							with st.expander(
									f'Record {idx}: {label}',
									expanded=False
							):
								st.json( item )
					else:
						st.info( 'No displayable FIRMS rows were found.' )
				else:
					st.info( 'No FIRMS records were returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.text( str( result.get( 'raw', '' ) ) )
					
# ==============================================================================
# ASTRONOMICAL MODE
# ==============================================================================
elif mode == 'Astronomical':
	st.subheader( f'🌌 Physics & Astronomical Data' )
	st.divider( )
	
	# -------- Satellite Center
	with st.expander( label='Satellite Center', expanded=False ):
		if 'satellitecenter_results' not in st.session_state:
			st.session_state[ 'satellitecenter_results' ] = { }
		
		if 'satellitecenter_clear_request' not in st.session_state:
			st.session_state[ 'satellitecenter_clear_request' ] = False
		
		if st.session_state.get( 'satellitecenter_clear_request', False ):
			st.session_state[ 'satellitecenter_mode' ] = 'observatories'
			st.session_state[ 'satellitecenter_query' ] = ''
			st.session_state[ 'satellitecenter_start_time' ] = ''
			st.session_state[ 'satellitecenter_end_time' ] = ''
			st.session_state[ 'satellitecenter_coordinate_systems' ] = 'gse'
			st.session_state[ 'satellitecenter_resolution_factor' ] = 1
			st.session_state[ 'satellitecenter_timeout' ] = 20
			st.session_state[ 'satellitecenter_results' ] = { }
			st.session_state[ 'satellitecenter_clear_request' ] = False
		
		def _clear_satellitecenter_state( ) -> None:
			st.session_state[ 'satellitecenter_clear_request' ] = True
		
		def _safe_dataframe( rows: Any ) -> pd.DataFrame:
			try:
				if isinstance( rows, pd.DataFrame ):
					return rows
				
				if isinstance( rows, list ):
					if rows and all( isinstance( x, dict ) for x in rows ):
						return pd.DataFrame( rows )
					return pd.DataFrame( { 'Value': [ str( x ) for x in rows ] } )
				
				if isinstance( rows, dict ):
					return pd.json_normalize( rows )
				
				return pd.DataFrame( { 'Value': [ str( rows ) ] } )
			except Exception:
				return pd.DataFrame( )
		
		def _render_satellite_table( title: str, rows: Any ) -> None:
			df_local = _safe_dataframe( rows )
			if not df_local.empty:
				st.markdown( title )
				st.dataframe( df_local, use_container_width=True, hide_index=True )
			else:
				st.info( 'No displayable rows were found.' )
		
		col_left, col_right = st.columns( 2, border=True )
		with col_left:
			satellite_mode = st.selectbox(
				'Mode',
				options=[ 'observatories', 'ground_stations', 'locations' ],
				index=[ 'observatories', 'ground_stations', 'locations' ].index(
					st.session_state.get( 'satellitecenter_mode', 'observatories' )
				),
				key='satellitecenter_mode'
			)
			
			satellite_query = st.text_area(
				'Observatory Query',
				height=90,
				key='satellitecenter_query',
				placeholder=(
						'Examples:\n'
						'iss\n'
						'mms1,mms2\n'
						'themisb\n'
						'\n'
						'Used for locations mode only. Leave blank for observatories '
						'and ground stations.'
				),
				disabled=(satellite_mode != 'locations')
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				satellite_start_time = st.text_input(
					'Start Time (UTC)',
					value=st.session_state.get( 'satellitecenter_start_time', '' ),
					key='satellitecenter_start_time',
					placeholder='2026-03-15T00:00:00Z',
					disabled=(satellite_mode != 'locations')
				)
			
			with c2:
				satellite_end_time = st.text_input(
					'End Time (UTC)',
					value=st.session_state.get( 'satellitecenter_end_time', '' ),
					key='satellitecenter_end_time',
					placeholder='2026-03-15T02:00:00Z',
					disabled=(satellite_mode != 'locations')
				)
			
			c3, c4, c5 = st.columns( 3 )
			
			with c3:
				satellite_coordinate_systems = st.text_input(
					'Coordinate Systems',
					value=st.session_state.get( 'satellitecenter_coordinate_systems', 'gse' ),
					key='satellitecenter_coordinate_systems',
					placeholder='gse or geo,gsm',
					disabled=(satellite_mode != 'locations')
				)
			
			with c4:
				satellite_resolution_factor = st.number_input(
					'Resolution Factor',
					min_value=1,
					max_value=1000,
					value=int( st.session_state.get( 'satellitecenter_resolution_factor', 1 ) ),
					step=1,
					key='satellitecenter_resolution_factor',
					disabled=(satellite_mode != 'locations')
				)
			
			with c5:
				satellite_timeout = st.number_input(
					'Timeout',
					min_value=1,
					max_value=120,
					value=int( st.session_state.get( 'satellitecenter_timeout', 20 ) ),
					step=1,
					key='satellitecenter_timeout'
				)
			
			st.caption(
				'No API key is required for SSCWeb. For locations mode, use UTC ISO '
				'8601 timestamps and observatory IDs returned by the observatories '
				'service.'
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				satellite_submit = st.button(
					'Submit',
					key='satellitecenter_submit',
					use_container_width=True
				)
			
			with b2:
				st.button(
					'Clear',
					key='satellitecenter_clear',
					on_click=_clear_satellitecenter_state,
					use_container_width=True
				)
		
		with col_right:
			st.markdown( 'Results' )
			
			if satellite_submit:
				try:
					f = SatelliteCenter( )
					result = f.fetch(
						mode=satellite_mode,
						query=satellite_query,
						start_time=satellite_start_time,
						end_time=satellite_end_time,
						coordinate_systems=satellite_coordinate_systems,
						resolution_factor=int( satellite_resolution_factor ),
						time=int( satellite_timeout )
					)
					
					st.session_state[ 'satellitecenter_results' ] = {
							'request': {
									'mode': satellite_mode,
									'query': satellite_query,
									'start_time': satellite_start_time,
									'end_time': satellite_end_time,
									'coordinate_systems': satellite_coordinate_systems,
									'resolution_factor': int( satellite_resolution_factor ),
									'timeout': int( satellite_timeout ),
							},
							'data': result or { },
					}
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Satellite Center request failed.' )
					st.exception( exc )
			
			result_wrapper = st.session_state.get( 'satellitecenter_results', { } )
			
			if not result_wrapper:
				st.text( 'No results.' )
			else:
				if (
						isinstance( result_wrapper, dict )
						and 'request' in result_wrapper
						and 'data' in result_wrapper
				):
					request_meta = result_wrapper.get( 'request', { } )
					result = result_wrapper.get( 'data', { } )
				else:
					request_meta = {
							'mode': satellite_mode,
							'query': satellite_query,
							'start_time': satellite_start_time,
							'end_time': satellite_end_time,
							'coordinate_systems': satellite_coordinate_systems,
							'resolution_factor': int( satellite_resolution_factor ),
							'timeout': int( satellite_timeout ),
					}
					result = result_wrapper if isinstance( result_wrapper, dict ) else { }
				
				render_mode = str( request_meta.get( 'mode', 'observatories' ) )
				
				st.markdown( '#### Request Metadata' )
				st.json( request_meta )
				
				if render_mode == 'observatories':
					items = result.get( 'Observatory', [ ] ) if isinstance( result, dict ) else [ ]
					
					if items:
						summary_rows: List[ Dict[ str, Any ] ] = [ ]
						
						for item in items:
							if isinstance( item, dict ):
								location_value = ''
								geo_value = item.get( 'GeoLocation', { } )
								
								if isinstance( geo_value, dict ):
									lat_value = geo_value.get( 'Latitude', '' )
									lon_value = geo_value.get( 'Longitude', '' )
									if str( lat_value ).strip( ) or str( lon_value ).strip( ):
										location_value = f'{lat_value}, {lon_value}'
								
								summary_rows.append(
									{
											'Id': item.get( 'Id', '' ),
											'Name': item.get( 'Name', '' ),
											'Resolution': item.get( 'Resolution', '' ),
											'StartTime': item.get( 'StartTime', '' ),
											'EndTime': item.get( 'EndTime', '' ),
											'GeoLocation': location_value,
									}
								)
						
						_render_satellite_table(
							f'#### Observatories ({len( summary_rows )})',
							summary_rows )
					else:
						st.info( 'No observatories were returned.' )
				
				elif render_mode == 'ground_stations':
					items = [ ]
					
					if isinstance( result, dict ):
						items = result.get( 'GroundStation', [ ] )
						if not items:
							items = result.get( 'GroundStations', [ ] )
					
					if items:
						summary_rows: List[ Dict[ str, Any ] ] = [ ]
						
						for item in items:
							if isinstance( item, dict ):
								coords = item.get( 'Location', { } )
								location_value = ''
								
								if isinstance( coords, dict ):
									lat_value = coords.get( 'Latitude', '' )
									lon_value = coords.get( 'Longitude', '' )
									if str( lat_value ).strip( ) or str( lon_value ).strip( ):
										location_value = f'{lat_value}, {lon_value}'
								
								summary_rows.append(
									{
											'Id': item.get( 'Id', '' ),
											'Name': item.get( 'Name', '' ),
											'Code': item.get( 'Code', '' ),
											'Location': location_value,
									}
								)
						
						_render_satellite_table(
							f'#### Ground Stations ({len( summary_rows )})',
							summary_rows )
					else:
						st.markdown( '#### Ground Stations' )
						st.json( result )
				
				elif render_mode == 'locations':
					st.markdown( '#### Locations' )
					
					if isinstance( result, dict ):
						if 'Data' in result and isinstance( result.get( 'Data' ), list ):
							_render_satellite_table( '##### Position Samples', result.get( 'Data', [ ] ) )
						elif 'Coordinates' in result and isinstance( result.get( 'Coordinates' ), list ):
							_render_satellite_table(
								'##### Coordinates',
								result.get( 'Coordinates', [ ] ) )
						else:
							flat_rows: List[ Dict[ str, Any ] ] = [ ]
							for key, value in result.items( ):
								if isinstance( value, (str, int, float, bool) ) or value is None:
									flat_rows.append( { 'Field': key, 'Value': value } )
							
							if flat_rows:
								_render_satellite_table( '##### Summary', flat_rows )
							else:
								st.json( result )
					else:
						st.write( result )
				
				else:
					st.json( result )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- Astro Catalog
	with st.expander( label='Astro Catalog', expanded=False ):
		if 'astrocatalog_results' not in st.session_state:
			st.session_state[ 'astrocatalog_results' ] = { }
		
		if 'astrocatalog_clear_request' not in st.session_state:
			st.session_state[ 'astrocatalog_clear_request' ] = False
		
		if st.session_state.get( 'astrocatalog_clear_request', False ):
			st.session_state[ 'astrocatalog_mode' ] = 'object_query'
			st.session_state[ 'astrocatalog_query' ] = ''
			st.session_state[ 'astrocatalog_quantity' ] = ''
			st.session_state[ 'astrocatalog_attributes' ] = ''
			st.session_state[ 'astrocatalog_arguments' ] = ''
			st.session_state[ 'astrocatalog_ra' ] = ''
			st.session_state[ 'astrocatalog_dec' ] = ''
			st.session_state[ 'astrocatalog_radius' ] = 2
			st.session_state[ 'astrocatalog_format' ] = 'json'
			st.session_state[ 'astrocatalog_timeout' ] = 20
			st.session_state[ 'astrocatalog_results' ] = { }
			st.session_state[ 'astrocatalog_clear_request' ] = False
		
		def _clear_astrocatalog_state( ) -> None:
			st.session_state[ 'astrocatalog_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			astro_mode = st.selectbox(
				'Mode',
				options=[ 'object_query', 'cone_search' ],
				index=[ 'object_query', 'cone_search' ].index(
					st.session_state.get( 'astrocatalog_mode', 'object_query' )
				),
				key='astrocatalog_mode'
			)
			
			astro_query = st.text_area(
				'Object Query',
				height=80,
				key='astrocatalog_query',
				placeholder=(
						'Examples:\n'
						'SN1987A\n'
						'AT2024abc\n'
						'GW170817\n'
						'\n'
						'Used for object_query mode.'
				),
				disabled=(astro_mode != 'object_query')
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				astro_quantity = st.text_input(
					'Quantity',
					value=st.session_state.get( 'astrocatalog_quantity', '' ),
					key='astrocatalog_quantity',
					placeholder='Example: photometry',
					disabled=(astro_mode != 'object_query')
				)
			
			with c2:
				astro_attributes = st.text_input(
					'Attributes',
					value=st.session_state.get( 'astrocatalog_attributes', '' ),
					key='astrocatalog_attributes',
					placeholder='Example: time,magnitude,band',
					disabled=(astro_mode != 'object_query')
				)
			
			astro_arguments = st.text_area(
				'Arguments',
				height=80,
				key='astrocatalog_arguments',
				placeholder=(
						'Optional query arguments.\n'
						'Examples:\n'
						'time=2450000\n'
						'band=V'
				),
				disabled=(astro_mode != 'object_query')
			)
			
			c3, c4, c5 = st.columns( 3 )
			
			with c3:
				astro_ra = st.text_input(
					'RA',
					value=st.session_state.get( 'astrocatalog_ra', '' ),
					key='astrocatalog_ra',
					placeholder='13:09:48.09',
					disabled=(astro_mode != 'cone_search')
				)
			
			with c4:
				astro_dec = st.text_input(
					'Dec',
					value=st.session_state.get( 'astrocatalog_dec', '' ),
					key='astrocatalog_dec',
					placeholder='+27:57:34.8',
					disabled=(astro_mode != 'cone_search')
				)
			
			with c5:
				astro_radius = st.number_input(
					'Radius (arcsec)',
					min_value=1,
					max_value=3600,
					value=int( st.session_state.get( 'astrocatalog_radius', 2 ) ),
					step=1,
					key='astrocatalog_radius',
					disabled=(astro_mode != 'cone_search')
				)
			
			c6, c7 = st.columns( 2 )
			
			with c6:
				astro_format = st.selectbox(
					'Format',
					options=[ 'json', 'csv', 'tsv' ],
					index=[ 'json', 'csv', 'tsv' ].index(
						st.session_state.get( 'astrocatalog_format', 'json' )
					),
					key='astrocatalog_format'
				)
			
			with c7:
				astro_timeout = st.number_input(
					'Timeout',
					min_value=1,
					max_value=120,
					value=int( st.session_state.get( 'astrocatalog_timeout', 20 ) ),
					step=1,
					key='astrocatalog_timeout'
				)
			
			st.caption(
				'No API key is required for Open Astronomy Catalog. '
				'Use object_query for named events and cone_search for coordinate searches.'
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				astro_submit = st.button(
					'Submit',
					key='astrocatalog_submit',
					use_container_width=True
				)
			with b2:
				st.button(
					'Clear',
					key='astrocatalog_clear',
					on_click=_clear_astrocatalog_state,
					use_container_width=True
				)
		
		with col_right:
			st.markdown( 'Results' )
			
			if astro_submit:
				try:
					f = AstroCatalog( )
					result = f.fetch(
						mode=astro_mode,
						query=astro_query,
						quantity=astro_quantity,
						attributes=astro_attributes,
						arguments=astro_arguments,
						ra=astro_ra,
						dec=astro_dec,
						radius=int( astro_radius ),
						data_format=astro_format,
						time=int( astro_timeout )
					)
					
					st.session_state[ 'astrocatalog_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Astronomy Catalog request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'astrocatalog_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				st.markdown( '#### Request Metadata' )
				st.json(
					{
							'mode': astro_mode,
							'query': astro_query,
							'quantity': astro_quantity,
							'attributes': astro_attributes,
							'arguments': astro_arguments,
							'ra': astro_ra,
							'dec': astro_dec,
							'radius': int( astro_radius ),
							'format': astro_format,
					}
				)
				
				parsed_result = result
				
				if isinstance( result, str ):
					text_value = result.strip( )
					
					if astro_format == 'json':
						try:
							parsed_result = json.loads( text_value )
						except Exception:
							parsed_result = result
					else:
						parsed_result = result
				
				if isinstance( parsed_result, list ):
					st.markdown( f'#### Result Rows ({len( parsed_result )})' )
					
					df_catalog = pd.DataFrame( parsed_result )
					if not df_catalog.empty:
						st.dataframe(
							df_catalog,
							use_container_width=True,
							hide_index=True
						)
					else:
						st.text_area(
							'Results',
							value=str( parsed_result ),
							height=320
						)
				
				elif isinstance( parsed_result, dict ):
					candidate_rows: List[ Dict[ str, Any ] ] = [ ]
					
					for key in [ 'results', 'items', 'data', 'objects' ]:
						value = parsed_result.get( key, None )
						if isinstance( value, list ) and value:
							candidate_rows = [
									item for item in value
									if isinstance( item, dict )
							]
							break
					
					if candidate_rows:
						st.markdown( f'#### Result Rows ({len( candidate_rows )})' )
						df_catalog = pd.DataFrame( candidate_rows )
						
						if not df_catalog.empty:
							st.dataframe(
								df_catalog,
								use_container_width=True,
								hide_index=True
							)
						else:
							st.json( parsed_result )
					
					else:
						title_value = (
								parsed_result.get( 'name' )
								or parsed_result.get( 'alias' )
								or parsed_result.get( 'event' )
								or parsed_result.get( 'id' )
								or 'Catalog Result'
						)
						
						st.markdown( f'### {title_value}' )
						
						top_fields: Dict[ str, Any ] = { }
						for key in [
								'name',
								'alias',
								'ra',
								'dec',
								'redshift',
								'type',
								'claimedtype',
								'schema'
						]:
							if key in parsed_result:
								top_fields[ key ] = parsed_result.get( key )
						
						if top_fields:
							st.json( top_fields )
						
						for key in [ 'summary', 'description', 'comments' ]:
							if key in parsed_result and str( parsed_result.get( key ) ).strip( ):
								st.markdown( f'#### {key.title( )}' )
								st.write( str( parsed_result.get( key ) ) )
						
						if not top_fields:
							st.json( parsed_result )
				
				elif isinstance( parsed_result, str ):
					st.markdown( '#### Result Text' )
					st.text_area(
						'Results',
						value=parsed_result,
						height=320
					)
				
				else:
					st.text_area(
						'Results',
						value=str( parsed_result ),
						height=320
					)
				
				with st.expander( 'Raw Result', expanded=False ):
					if isinstance( result, (dict, list) ):
						st.json( result )
					else:
						st.text_area(
							'Raw',
							value=str( result ),
							height=240
						)
	
	# -------- Astro Query
	with st.expander( label='Astro Query', expanded=False ):
		if 'astroquery_results' not in st.session_state:
			st.session_state[ 'astroquery_results' ] = { }
		
		if 'astroquery_clear_request' not in st.session_state:
			st.session_state[ 'astroquery_clear_request' ] = False
		
		if st.session_state.get( 'astroquery_clear_request', False ):
			st.session_state[ 'astroquery_mode' ] = 'object_search'
			st.session_state[ 'astroquery_query' ] = ''
			st.session_state[ 'astroquery_ra' ] = ''
			st.session_state[ 'astroquery_dec' ] = ''
			st.session_state[ 'astroquery_radius' ] = 0.5
			st.session_state[ 'astroquery_radius_unit' ] = 'deg'
			st.session_state[ 'astroquery_row_limit' ] = 100
			st.session_state[ 'astroquery_results' ] = { }
			st.session_state[ 'astroquery_clear_request' ] = False
		
		def _clear_astroquery_state( ) -> None:
			st.session_state[ 'astroquery_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			astroquery_mode = st.selectbox(
				'Mode',
				options=[ 'object_search', 'object_ids', 'region_search' ],
				index=[ 'object_search', 'object_ids', 'region_search' ].index(
					st.session_state.get( 'astroquery_mode', 'object_search' )
				),
				key='astroquery_mode'
			)
			
			astroquery_query = st.text_area(
				'Object Query',
				height=80,
				key='astroquery_query',
				placeholder=(
						'Examples:\n'
						'M81\n'
						'Sirius\n'
						'NGC 1300\n'
						'\n'
						'Used for object_search and object_ids.'
				),
				disabled=(astroquery_mode == 'region_search')
			)
			
			c1, c2, c3 = st.columns( 3 )
			
			with c1:
				astroquery_ra = st.text_input(
					'RA',
					value=st.session_state.get( 'astroquery_ra', '' ),
					key='astroquery_ra',
					placeholder='13:09:48.09',
					disabled=(astroquery_mode != 'region_search'),
					help='Right Ascension of the search center, e.g. 13:09:48.09.'
				)
			
			with c2:
				astroquery_dec = st.text_input(
					'Dec',
					value=st.session_state.get( 'astroquery_dec', '' ),
					key='astroquery_dec',
					placeholder='-23:22:53.3',
					disabled=(astroquery_mode != 'region_search'),
					help='Declination of the search center, e.g. -23:22:53.3.'
				)
			
			with c3:
				astroquery_radius = st.number_input(
					'Radius',
					min_value=0.001,
					max_value=60.0,
					value=float( st.session_state.get( 'astroquery_radius', 0.5 ) ),
					step=0.1,
					key='astroquery_radius',
					disabled=(astroquery_mode != 'region_search'),
					help='Cone-search radius around the RA/Dec sky position.'
				)
			
			c4, c5 = st.columns( 2 )
			
			with c4:
				astroquery_radius_unit = st.selectbox(
					'Radius Unit',
					options=[ 'deg', 'arcmin', 'arcsec' ],
					index=[ 'deg', 'arcmin', 'arcsec' ].index(
						st.session_state.get( 'astroquery_radius_unit', 'deg' )
					),
					key='astroquery_radius_unit',
					disabled=(astroquery_mode != 'region_search')
				)
			
			with c5:
				astroquery_row_limit = st.number_input(
					'Row Limit',
					min_value=1,
					max_value=10000,
					value=int( st.session_state.get( 'astroquery_row_limit', 100 ) ),
					step=1,
					key='astroquery_row_limit'
				)
			
			st.caption(
				'No API key is required for basic astroquery SIMBAD queries. '
				'Use object_search for a named object, object_ids for alternate names, '
				'and region_search for a cone search around RA/Dec.'
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				astroquery_submit = st.button(
					'Submit',
					key='astroquery_submit',
					use_container_width=True
				)
			with b2:
				st.button(
					'Clear',
					key='astroquery_clear',
					on_click=_clear_astroquery_state,
					use_container_width=True
				)
		
		with col_right:
			st.markdown( 'Results' )
			
			if astroquery_submit:
				try:
					f = AstroQuery( )
					result = f.fetch(
						mode=astroquery_mode,
						query=astroquery_query,
						ra=astroquery_ra,
						dec=astroquery_dec,
						radius=float( astroquery_radius ),
						radius_unit=astroquery_radius_unit,
						row_limit=int( astroquery_row_limit )
					)
					
					st.session_state[ 'astroquery_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Astro Query request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'astroquery_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_col1, meta_col2 = st.columns( 2 )
				
				with meta_col1:
					if isinstance( result, dict ) and 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if isinstance( result, dict ) and 'query' in result:
						st.markdown( f"**Query:** {result.get( 'query', '' )}" )
					if isinstance( result, dict ) and 'ra' in result:
						st.markdown( f"**RA:** {result.get( 'ra', '' )}" )
				
				with meta_col2:
					if isinstance( result, dict ) and 'row_limit' in result:
						st.markdown( f"**Row Limit:** {result.get( 'row_limit', '' )}" )
					if isinstance( result, dict ) and 'dec' in result:
						st.markdown( f"**Dec:** {result.get( 'dec', '' )}" )
					if isinstance( result, dict ) and 'radius' in result:
						st.markdown(
							f"**Radius:** {result.get( 'radius', '' )} "
							f"{result.get( 'radius_unit', '' )}"
						)
				
				rows = result.get( 'rows', [ ] ) if isinstance( result, dict ) else [ ]
				columns = result.get( 'columns', [ ] ) if isinstance( result, dict ) else [ ]
				
				if columns:
					st.markdown( '#### Columns' )
					st.write( ', '.join( [ str( c ) for c in columns ] ) )
				
				if not rows:
					st.info( 'No rows returned.' )
				else:
					df_rows = pd.DataFrame( rows )
					
					if not df_rows.empty and columns:
						ordered_columns = [ c for c in columns if c in df_rows.columns ]
						if ordered_columns:
							df_rows = df_rows[ ordered_columns ]
					
					st.markdown( f'#### Result Rows ({len( df_rows )})' )
					
					if not df_rows.empty:
						st.dataframe(
							df_rows,
							use_container_width=True,
							hide_index=True
						)
					else:
						st.info( 'No displayable rows were returned.' )
					
					with st.expander( 'Row Details', expanded=False ):
						for idx, row in enumerate( rows, start=1 ):
							label = row.get( 'MAIN_ID', f'Row {idx}' )
							with st.expander( f'Row {idx}: {label}', expanded=False ):
								st.json( row )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- Star Map
	with st.expander( label='Star Map', expanded=False ):
		if 'starmap_results' not in st.session_state:
			st.session_state[ 'starmap_results' ] = { }
		
		if 'starmap_clear_request' not in st.session_state:
			st.session_state[ 'starmap_clear_request' ] = False
		
		if st.session_state.get( 'starmap_clear_request', False ):
			st.session_state[ 'starmap_mode' ] = 'object_link'
			st.session_state[ 'starmap_query' ] = ''
			st.session_state[ 'starmap_ra' ] = 15.2976
			st.session_state[ 'starmap_dec' ] = -17.5892
			st.session_state[ 'starmap_zoom' ] = 5
			st.session_state[ 'starmap_image_source' ] = 'DSS2'
			st.session_state[ 'starmap_box_color' ] = 'yellow'
			st.session_state[ 'starmap_show_box' ] = True
			st.session_state[ 'starmap_show_grid' ] = True
			st.session_state[ 'starmap_show_lines' ] = True
			st.session_state[ 'starmap_show_boundaries' ] = True
			st.session_state[ 'starmap_show_const_names' ] = False
			st.session_state[ 'starmap_timeout' ] = 20
			st.session_state[ 'starmap_results' ] = { }
			st.session_state[ 'starmap_clear_request' ] = False
		
		def _clear_starmap_state( ) -> None:
			st.session_state[ 'starmap_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			starmap_mode = st.selectbox(
				'Mode',
				options=[ 'object_link', 'coordinate_link', 'snapshot' ],
				index=[ 'object_link', 'coordinate_link', 'snapshot' ].index(
					st.session_state.get( 'starmap_mode', 'object_link' ) ),
				key='starmap_mode'
			)
			
			starmap_query = st.text_area(
				'Object Query',
				height=80,
				key='starmap_query',
				placeholder='Examples: Polaris, M31, NGC 1300, Used for object_link mode only.',
				disabled=(starmap_mode != 'object_link')
			)
			
			c1, c2, c3 = st.columns( 3 )
			
			with c1:
				starmap_ra = st.number_input(
					'RA (hours)',
					min_value=0.0,
					max_value=24.0,
					value=float( st.session_state.get( 'starmap_ra', 15.2976 ) ),
					step=0.0001,
					format='%.4f',
					key='starmap_ra',
					disabled=(starmap_mode == 'object_link'),
					help='Right Ascension of the sky center in hours. Example: 15.2976'
				)
			
			with c2:
				starmap_dec = st.number_input(
					'Dec (degrees)',
					min_value=-90.0,
					max_value=90.0,
					value=float( st.session_state.get( 'starmap_dec', -17.5892 ) ),
					step=0.0001,
					format='%.4f',
					key='starmap_dec',
					disabled=(starmap_mode == 'object_link'),
					help='Declination of the sky center in degrees. Example: -17.5892'
				)
			
			with c3:
				starmap_zoom = st.number_input(
					'Zoom',
					min_value=1,
					max_value=18,
					value=int( st.session_state.get( 'starmap_zoom', 5 ) ),
					step=1,
					key='starmap_zoom',
					help='Smaller values show a wider field; larger values zoom in.'
				)
			
			c4, c5 = st.columns( 2 )
			
			with c4:
				starmap_image_source = st.selectbox(
					'Image Source',
					options=[ 'DSS2', 'SDSS', 'SDSS-III', 'GALEX', 'IRAS', 'RASS', 'H-Alpha' ],
					index=[ 'DSS2', 'SDSS', 'SDSS-III', 'GALEX', 'IRAS', 'RASS', 'H-Alpha' ].index(
						st.session_state.get( 'starmap_image_source', 'DSS2' ) ),
					key='starmap_image_source',
					disabled=(starmap_mode != 'snapshot'),
					help='Sky survey source used for snapshot generation.'
				)
			
			with c5:
				starmap_box_color = st.text_input(
					'Box Color',
					value=st.session_state.get( 'starmap_box_color', 'yellow' ),
					key='starmap_box_color',
					placeholder='yellow',
					disabled=(starmap_mode == 'snapshot')
				)
			
			c6, c7 = st.columns( 2 )
			
			with c6:
				starmap_show_box = st.checkbox(
					'Show Box',
					value=st.session_state.get( 'starmap_show_box', True ),
					key='starmap_show_box',
					disabled=(starmap_mode == 'snapshot')
				)
			
			with c7:
				starmap_show_grid = st.checkbox(
					'Show Grid',
					value=st.session_state.get( 'starmap_show_grid', True ),
					key='starmap_show_grid',
					disabled=(starmap_mode == 'object_link')
				)
			
			c8, c9 = st.columns( 2 )
			
			with c8:
				starmap_show_lines = st.checkbox(
					'Show Constellation Lines',
					value=st.session_state.get( 'starmap_show_lines', True ),
					key='starmap_show_lines',
					disabled=False
				)
			
			with c9:
				starmap_show_boundaries = st.checkbox(
					'Show Constellation Boundaries',
					value=st.session_state.get( 'starmap_show_boundaries', True ),
					key='starmap_show_boundaries',
					disabled=False
				)
			
			starmap_show_const_names = st.checkbox(
				'Show Constellation Names',
				value=st.session_state.get( 'starmap_show_const_names', False ),
				key='starmap_show_const_names',
				disabled=(starmap_mode != 'snapshot')
			)
			
			starmap_timeout = st.number_input(
				'Timeout',
				min_value=1,
				max_value=120,
				value=int( st.session_state.get( 'starmap_timeout', 20 ) ),
				step=1,
				key='starmap_timeout'
			)
			
			st.caption(
				'No API key is required. '
				'Use object_link for a named object, coordinate_link for RA/Dec-centered interactive maps, '
				'and snapshot for a static sky image page.'
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				starmap_submit = st.button( 'Submit', key='starmap_submit' )
			with b2:
				st.button( 'Clear', key='starmap_clear', on_click=_clear_starmap_state )
		
		with col_right:
			st.markdown( 'Results' )
			
			if starmap_submit:
				try:
					f = StarMap( )
					result = f.fetch(
						mode=starmap_mode,
						query=starmap_query,
						ra=float( starmap_ra ),
						dec=float( starmap_dec ),
						zoom=int( starmap_zoom ),
						image_source=starmap_image_source,
						box_color=starmap_box_color,
						show_box=bool( starmap_show_box ),
						show_grid=bool( starmap_show_grid ),
						show_lines=bool( starmap_show_lines ),
						show_boundaries=bool( starmap_show_boundaries ),
						show_const_names=bool( starmap_show_const_names ),
						time=int( starmap_timeout ) )
					
					st.session_state[ 'starmap_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Star Map request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'starmap_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_col1, meta_col2 = st.columns( 2 )
				
				with meta_col1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'object' in result:
						st.markdown( f"**Object:** {result.get( 'object', '' )}" )
					if 'ra' in result:
						st.markdown( f"**RA:** {result.get( 'ra', '' )}" )
				
				with meta_col2:
					if 'zoom' in result:
						st.markdown( f"**Zoom:** {result.get( 'zoom', '' )}" )
					if 'dec' in result:
						st.markdown( f"**Dec:** {result.get( 'dec', '' )}" )
					if 'image_source' in result:
						st.markdown( f"**Image Source:** {result.get( 'image_source', '' )}" )
				
				if result.get( 'interactive_url', '' ):
					st.markdown( f"**Interactive URL:** {result.get( 'interactive_url', '' )}" )
				
				if result.get( 'snapshot_page_url', '' ):
					st.markdown( f"**Snapshot Page URL:** {result.get( 'snapshot_page_url', '' )}" )
				
				preferred_image_url = result.get( 'preferred_image_url', '' )
				if preferred_image_url:
					st.markdown( '#### Preferred Image' )
					st.image( preferred_image_url, use_container_width=True )
				
				image_links = result.get( 'image_links', { } ) or { }
				if image_links:
					st.markdown( '#### Available Image Links' )
					st.json( image_links )
				
				if result.get( 'params', { } ):
					st.markdown( '#### Request Parameters' )
					st.json( result.get( 'params', { } ) )
				
				if result.get( 'html_preview', '' ):
					st.markdown( '#### HTML Preview' )
					st.text_area(
						'',
						value=result.get( 'html_preview', '' ),
						height=220,
						key='starmap_html_preview'
					)
	
	# -------- SIMBAD
	with st.expander( label='SIMBAD', expanded=False ):
		if 'simbad_results' not in st.session_state:
			st.session_state[ 'simbad_results' ] = { }
		
		if 'simbad_clear_request' not in st.session_state:
			st.session_state[ 'simbad_clear_request' ] = False
		
		if st.session_state.get( 'simbad_clear_request', False ):
			st.session_state[ 'simbad_mode' ] = 'object_search'
			st.session_state[ 'simbad_query' ] = 'Polaris'
			st.session_state[ 'simbad_ra' ] = '02:31:49.09'
			st.session_state[ 'simbad_dec' ] = '+89:15:50.8'
			st.session_state[ 'simbad_radius' ] = 0.5
			st.session_state[ 'simbad_radius_unit' ] = 'deg'
			st.session_state[ 'simbad_row_limit' ] = 100
			st.session_state[ 'simbad_results' ] = { }
			st.session_state[ 'simbad_clear_request' ] = False
		
		def _clear_simbad_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the SIMBAD expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'simbad_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			simbad_mode = st.selectbox(
				'Mode',
				options=[ 'object_search', 'object_ids', 'region_search' ],
				index=[ 'object_search', 'object_ids', 'region_search' ].index(
					st.session_state.get( 'simbad_mode', 'object_search' )
				),
				key='simbad_mode',
				help='Choose named-object lookup, alternate identifiers, or cone search.'
			)
			
			simbad_query = st.text_area(
				'Object Name',
				height=80,
				key='simbad_query',
				placeholder=(
						'Examples:\n'
						'Polaris\n'
						'M 31\n'
						'NGC 1300\n'
						'\n'
						'Used for object_search and object_ids.'
				),
				disabled=(simbad_mode == 'region_search')
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				simbad_ra = st.text_input(
					'Right Ascension',
					value=st.session_state.get( 'simbad_ra', '02:31:49.09' ),
					key='simbad_ra',
					placeholder='13:09:48.09',
					disabled=(simbad_mode != 'region_search'),
					help='Hourangle format recommended for this wrapper.'
				)
			
			with c2:
				simbad_dec = st.text_input(
					'Declination',
					value=st.session_state.get( 'simbad_dec', '+89:15:50.8' ),
					key='simbad_dec',
					placeholder='-23:22:53.3',
					disabled=(simbad_mode != 'region_search')
				)
			
			c3, c4, c5 = st.columns( 3 )
			
			with c3:
				simbad_radius = st.number_input(
					'Radius',
					min_value=0.001,
					max_value=60.0,
					value=float( st.session_state.get( 'simbad_radius', 0.5 ) ),
					step=0.1,
					key='simbad_radius',
					disabled=(simbad_mode != 'region_search'),
					help='Cone-search radius around the RA/Dec position.'
				)
			
			with c4:
				simbad_radius_unit = st.selectbox(
					'Radius Unit',
					options=[ 'deg', 'arcmin', 'arcsec' ],
					index=[ 'deg', 'arcmin', 'arcsec' ].index(
						st.session_state.get( 'simbad_radius_unit', 'deg' )
					),
					key='simbad_radius_unit',
					disabled=(simbad_mode != 'region_search')
				)
			
			with c5:
				simbad_row_limit = st.number_input(
					'Row Limit',
					min_value=1,
					max_value=10000,
					value=int( st.session_state.get( 'simbad_row_limit', 100 ) ),
					step=1,
					key='simbad_row_limit'
				)
			
			st.caption(
				'SIMBAD named-object queries do not require an API key. '
				'Use object_search for a record lookup, object_ids for aliases, '
				'and region_search for a cone search around sky coordinates.'
			)
			
			b1, b2 = st.columns( 2 )
			
			with b1:
				simbad_submit = st.button(
					'Submit',
					key='simbad_submit'
				)
			
			with b2:
				st.button(
					'Clear',
					key='simbad_clear',
					on_click=_clear_simbad_state
				)
		
		with col_right:
			st.markdown( 'Results' )
			
			if simbad_submit:
				try:
					f = AstroQuery( )
					result = f.fetch(
						mode=simbad_mode,
						query=simbad_query,
						ra=simbad_ra,
						dec=simbad_dec,
						radius=float( simbad_radius ),
						radius_unit=simbad_radius_unit,
						row_limit=int( simbad_row_limit )
					)
					
					st.session_state[ 'simbad_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'SIMBAD request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'simbad_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if isinstance( result, dict ) and 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if isinstance( result, dict ) and 'query' in result:
						st.markdown( f"**Query:** {result.get( 'query', '' )}" )
					if isinstance( result, dict ) and 'ra' in result:
						st.markdown( f"**RA:** {result.get( 'ra', '' )}" )
				
				with meta_c2:
					if isinstance( result, dict ) and 'row_limit' in result:
						st.markdown( f"**Row Limit:** {result.get( 'row_limit', '' )}" )
					if isinstance( result, dict ) and 'dec' in result:
						st.markdown( f"**Dec:** {result.get( 'dec', '' )}" )
					if isinstance( result, dict ) and 'radius' in result:
						st.markdown(
							f"**Radius:** {result.get( 'radius', '' )} "
							f"{result.get( 'radius_unit', '' )}"
						)
				
				columns = result.get( 'columns', [ ] ) if isinstance( result, dict ) else [ ]
				rows = result.get( 'rows', [ ] ) if isinstance( result, dict ) else [ ]
				
				if columns:
					st.markdown( '#### Columns' )
					st.write( columns )
				
				if rows:
					st.markdown( '#### Rows' )
					df_simbad = pd.DataFrame( rows )
					st.dataframe( df_simbad, use_container_width=True, hide_index=True )
				else:
					st.text( 'No rows returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- Space Weather
	with st.expander( label='Space Weather', expanded=False ):
		if 'spaceweather_results' not in st.session_state:
			st.session_state[ 'spaceweather_results' ] = { }
		
		if 'spaceweather_clear_request' not in st.session_state:
			st.session_state[ 'spaceweather_clear_request' ] = False
		
		if st.session_state.get( 'spaceweather_clear_request', False ):
			st.session_state[ 'spaceweather_mode' ] = 'cme'
			st.session_state[ 'spaceweather_start_date' ] = '2026-03-01'
			st.session_state[ 'spaceweather_end_date' ] = '2026-03-15'
			st.session_state[ 'spaceweather_location' ] = 'ALL'
			st.session_state[ 'spaceweather_catalog' ] = 'ALL'
			st.session_state[ 'spaceweather_notification_type' ] = 'all'
			st.session_state[ 'spaceweather_most_accurate_only' ] = True
			st.session_state[ 'spaceweather_complete_entry_only' ] = True
			st.session_state[ 'spaceweather_speed' ] = 0
			st.session_state[ 'spaceweather_half_angle' ] = 0
			st.session_state[ 'spaceweather_keyword' ] = ''
			st.session_state[ 'spaceweather_timeout' ] = 20
			st.session_state[ 'spaceweather_results' ] = { }
			st.session_state[ 'spaceweather_clear_request' ] = False
		
		def _clear_spaceweather_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the Space Weather expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'spaceweather_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			spaceweather_mode = st.selectbox(
				'Mode',
				options=[
						'cme', 'cme_analysis', 'gst', 'ips', 'flr',
						'sep', 'mpc', 'rbe', 'hss', 'wsa_enlil', 'notifications'
				],
				index=[
						'cme', 'cme_analysis', 'gst', 'ips', 'flr',
						'sep', 'mpc', 'rbe', 'hss', 'wsa_enlil', 'notifications'
				].index(
					st.session_state.get( 'spaceweather_mode', 'cme' )
				),
				key='spaceweather_mode',
				help='Choose the documented DONKI endpoint to query.'
			)
			
			d1, d2 = st.columns( 2 )
			
			with d1:
				spaceweather_start_date = st.text_input(
					'Start Date',
					value=st.session_state.get( 'spaceweather_start_date', '2026-03-01' ),
					key='spaceweather_start_date',
					placeholder='2026-03-01'
				)
			
			with d2:
				spaceweather_end_date = st.text_input(
					'End Date',
					value=st.session_state.get( 'spaceweather_end_date', '2026-03-15' ),
					key='spaceweather_end_date',
					placeholder='2026-03-15'
				)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				spaceweather_location = st.text_input(
					'Location',
					value=st.session_state.get( 'spaceweather_location', 'ALL' ),
					key='spaceweather_location',
					placeholder='ALL or Earth',
					disabled=(spaceweather_mode != 'ips')
				)
			
			with c2:
				spaceweather_catalog = st.text_input(
					'Catalog',
					value=st.session_state.get( 'spaceweather_catalog', 'ALL' ),
					key='spaceweather_catalog',
					placeholder='ALL or SWRC_CATALOG',
					disabled=(spaceweather_mode not in [ 'ips', 'cme_analysis' ])
				)
			
			c3, c4 = st.columns( 2 )
			
			with c3:
				spaceweather_notification_type = st.text_input(
					'Notification Type',
					value=st.session_state.get( 'spaceweather_notification_type', 'all' ),
					key='spaceweather_notification_type',
					placeholder='all or FLR',
					disabled=(spaceweather_mode != 'notifications')
				)
			
			with c4:
				spaceweather_keyword = st.text_input(
					'Keyword',
					value=st.session_state.get( 'spaceweather_keyword', '' ),
					key='spaceweather_keyword',
					placeholder='swpc_annex',
					disabled=(spaceweather_mode != 'cme_analysis')
				)
			
			c5, c6 = st.columns( 2 )
			
			with c5:
				spaceweather_speed = st.number_input(
					'Speed',
					min_value=0,
					max_value=5000,
					value=int( st.session_state.get( 'spaceweather_speed', 0 ) ),
					step=10,
					key='spaceweather_speed',
					disabled=(spaceweather_mode != 'cme_analysis')
				)
			
			with c6:
				spaceweather_half_angle = st.number_input(
					'Half Angle',
					min_value=0,
					max_value=180,
					value=int( st.session_state.get( 'spaceweather_half_angle', 0 ) ),
					step=1,
					key='spaceweather_half_angle',
					disabled=(spaceweather_mode != 'cme_analysis')
				)
			
			c7, c8, c9 = st.columns( 3 )
			
			with c7:
				spaceweather_most_accurate_only = st.checkbox(
					'Most Accurate Only',
					value=bool( st.session_state.get( 'spaceweather_most_accurate_only', True ) ),
					key='spaceweather_most_accurate_only',
					disabled=(spaceweather_mode != 'cme_analysis')
				)
			
			with c8:
				spaceweather_complete_entry_only = st.checkbox(
					'Complete Entry Only',
					value=bool( st.session_state.get( 'spaceweather_complete_entry_only', True ) ),
					key='spaceweather_complete_entry_only',
					disabled=(spaceweather_mode != 'cme_analysis')
				)
			
			with c9:
				spaceweather_timeout = st.number_input(
					'Timeout',
					min_value=1,
					max_value=120,
					value=int( st.session_state.get( 'spaceweather_timeout', 20 ) ),
					step=1,
					key='spaceweather_timeout'
				)
			
			st.caption(
				'Examples: cme for coronal mass ejections, gst for geomagnetic storms, '
				'ips with Location=Earth, notifications with Type=all, or '
				'cme_analysis with Catalog=ALL and Speed=500.'
			)
			
			b1, b2 = st.columns( 2 )
			
			with b1:
				spaceweather_submit = st.button(
					'Submit',
					key='spaceweather_submit'
				)
			
			with b2:
				st.button(
					'Clear',
					key='spaceweather_clear',
					on_click=_clear_spaceweather_state
				)
		
		with col_right:
			if spaceweather_submit:
				try:
					f = SpaceWeather( )
					result = f.fetch(
						mode=spaceweather_mode,
						start_date=str( spaceweather_start_date ),
						end_date=str( spaceweather_end_date ),
						location=str( spaceweather_location or 'ALL' ),
						catalog=str( spaceweather_catalog or 'ALL' ),
						notification_type=str( spaceweather_notification_type or 'all' ),
						most_accurate_only=bool( spaceweather_most_accurate_only ),
						complete_entry_only=bool( spaceweather_complete_entry_only ),
						speed=int( spaceweather_speed ),
						half_angle=int( spaceweather_half_angle ),
						keyword=str( spaceweather_keyword or '' ),
						time=int( spaceweather_timeout )
					)
					
					st.session_state[ 'spaceweather_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Space Weather request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'satellitecenter_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				st.markdown( '#### Result Summary' )
				
				if satellite_mode == 'observatories':
					items = result.get( 'Observatory', [ ] ) if isinstance( result, dict ) else [ ]
					
					if items:
						df_sat = pd.DataFrame( items )
						if not df_sat.empty:
							st.caption( f'Observatories returned: {len( df_sat )}' )
							st.dataframe( df_sat, use_container_width=True, hide_index=True )
						else:
							st.info( 'No displayable observatory rows were found.' )
						
						for idx, item in enumerate( items, start=1 ):
							label = item.get( 'Id', f'Observatory {idx}' )
							with st.expander( f'Observatory {idx}: {label}', expanded=False ):
								st.json( item )
					else:
						st.info( 'No observatories returned.' )
				
				elif satellite_mode == 'ground_stations':
					items = result.get( 'GroundStation', [ ] ) if isinstance( result, dict ) else [ ]
					
					if items:
						df_sat = pd.DataFrame( items )
						if not df_sat.empty:
							st.caption( f'Ground stations returned: {len( df_sat )}' )
							st.dataframe( df_sat, use_container_width=True, hide_index=True )
						else:
							st.info( 'No displayable ground-station rows were found.' )
						
						for idx, item in enumerate( items, start=1 ):
							label = item.get( 'Id', f'Ground Station {idx}' )
							with st.expander( f'Ground Station {idx}: {label}', expanded=False ):
								st.json( item )
					else:
						st.info( 'No ground stations returned.' )
				
				else:
					data_items = result.get( 'Data', [ ] ) if isinstance( result, dict ) else [ ]
					
					if data_items:
						summary_rows: List[ Dict[ str, Any ] ] = [ ]
						
						for item in data_items:
							if isinstance( item, dict ):
								summary_rows.append(
									{
											'Id': item.get( 'Id', '' ),
											'CoordinateSystem': item.get( 'CoordinateSystem', '' ),
											'Start': item.get( 'StartTime', '' ),
											'End': item.get( 'EndTime', '' )
									}
								)
						
						df_sat = pd.DataFrame( summary_rows )
						if not df_sat.empty:
							st.caption( f'Location sets returned: {len( df_sat )}' )
							st.dataframe( df_sat, use_container_width=True, hide_index=True )
						else:
							st.info( 'No displayable location-set summary rows were found.' )
						
						for idx, item in enumerate( data_items, start=1 ):
							label = item.get( 'Id', f'Trajectory {idx}' )
							with st.expander( f'Location Set {idx}: {label}', expanded=False ):
								st.json( item )
					else:
						st.info( 'No location data returned.' )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- Star Chart
	with st.expander( label='Star Chart', expanded=False ):
		if 'starchart_results' not in st.session_state:
			st.session_state[ 'starchart_results' ] = { }
		
		if 'starchart_clear_request' not in st.session_state:
			st.session_state[ 'starchart_clear_request' ] = False
		
		if st.session_state.get( 'starchart_clear_request', False ):
			st.session_state[ 'starchart_mode' ] = 'object_chart'
			st.session_state[ 'starchart_query' ] = 'Polaris'
			st.session_state[ 'starchart_ra' ] = 2.5302
			st.session_state[ 'starchart_dec' ] = 89.2642
			st.session_state[ 'starchart_zoom' ] = 5
			st.session_state[ 'starchart_image_source' ] = 'DSS2'
			st.session_state[ 'starchart_box_color' ] = 'yellow'
			st.session_state[ 'starchart_show_box' ] = True
			st.session_state[ 'starchart_show_grid' ] = True
			st.session_state[ 'starchart_show_lines' ] = True
			st.session_state[ 'starchart_show_boundaries' ] = True
			st.session_state[ 'starchart_show_const_names' ] = False
			st.session_state[ 'starchart_width' ] = 900
			st.session_state[ 'starchart_height' ] = 450
			st.session_state[ 'starchart_magnitude' ] = 7.5
			st.session_state[ 'starchart_timeout' ] = 20
			st.session_state[ 'starchart_results' ] = { }
			st.session_state[ 'starchart_clear_request' ] = False
		
		def _clear_starchart_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the Star Chart expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'starchart_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			starchart_mode = st.selectbox(
				'Mode',
				options=[ 'object_search', 'object_chart', 'coordinate_chart', 'static_chart' ],
				index=[ 'object_search', 'object_chart', 'coordinate_chart', 'static_chart' ].index(
					st.session_state.get( 'starchart_mode', 'object_chart' )
				),
				key='starchart_mode'
			)
			
			starchart_query = st.text_area(
				'Object Query',
				height=80,
				key='starchart_query',
				placeholder=(
						'Examples:\n'
						'Polaris\n'
						'M31\n'
						'NGC 1300\n'
						'\n'
						'Used for object_search and object_chart.'
				),
				disabled=(starchart_mode not in [ 'object_search', 'object_chart' ])
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				starchart_ra = st.number_input(
					'RA',
					value=float( st.session_state.get( 'starchart_ra', 2.5302 ) ),
					format='%.4f',
					key='starchart_ra',
					disabled=(starchart_mode not in [ 'coordinate_chart', 'static_chart' ])
				)
			
			with c2:
				starchart_dec = st.number_input(
					'Dec',
					value=float( st.session_state.get( 'starchart_dec', 89.2642 ) ),
					format='%.4f',
					key='starchart_dec',
					disabled=(starchart_mode not in [ 'coordinate_chart', 'static_chart' ])
				)
			
			c3, c4, c5 = st.columns( 3 )
			
			with c3:
				starchart_zoom = st.number_input(
					'Zoom',
					min_value=0,
					max_value=18,
					value=int( st.session_state.get( 'starchart_zoom', 5 ) ),
					step=1,
					key='starchart_zoom'
				)
			
			with c4:
				starchart_width = st.number_input(
					'Width',
					min_value=100,
					max_value=2400,
					value=int( st.session_state.get( 'starchart_width', 900 ) ),
					step=50,
					key='starchart_width',
					disabled=(starchart_mode != 'static_chart')
				)
			
			with c5:
				starchart_height = st.number_input(
					'Height',
					min_value=100,
					max_value=2400,
					value=int( st.session_state.get( 'starchart_height', 450 ) ),
					step=50,
					key='starchart_height',
					disabled=(starchart_mode != 'static_chart')
				)
			
			c6, c7, c8 = st.columns( 3 )
			
			with c6:
				starchart_image_source = st.selectbox(
					'Image Source',
					options=[ 'DSS2', 'SDSS' ],
					index=[ 'DSS2', 'SDSS' ].index(
						st.session_state.get( 'starchart_image_source', 'DSS2' )
					),
					key='starchart_image_source'
				)
			
			with c7:
				starchart_box_color = st.text_input(
					'Box Color',
					value=st.session_state.get( 'starchart_box_color', 'yellow' ),
					key='starchart_box_color',
					disabled=(starchart_mode not in [ 'object_chart', 'coordinate_chart' ])
				)
			
			with c8:
				starchart_magnitude = st.number_input(
					'Magnitude',
					min_value=0.0,
					max_value=20.0,
					value=float( st.session_state.get( 'starchart_magnitude', 7.5 ) ),
					step=0.1,
					key='starchart_magnitude',
					disabled=(starchart_mode != 'static_chart')
				)
			
			c9, c10 = st.columns( 2 )
			
			with c9:
				starchart_show_box = st.checkbox(
					'Show Box',
					value=bool( st.session_state.get( 'starchart_show_box', True ) ),
					key='starchart_show_box',
					disabled=(starchart_mode not in [ 'object_chart', 'coordinate_chart' ])
				)
			
			with c10:
				starchart_show_grid = st.checkbox(
					'Show Grid',
					value=bool( st.session_state.get( 'starchart_show_grid', True ) ),
					key='starchart_show_grid',
					disabled=(starchart_mode not in [ 'coordinate_chart', 'static_chart' ])
				)
			
			c11, c12 = st.columns( 2 )
			
			with c11:
				starchart_show_lines = st.checkbox(
					'Show Lines',
					value=bool( st.session_state.get( 'starchart_show_lines', True ) ),
					key='starchart_show_lines',
					disabled=(starchart_mode not in [ 'coordinate_chart', 'static_chart' ])
				)
			
			with c12:
				starchart_show_boundaries = st.checkbox(
					'Show Boundaries',
					value=bool( st.session_state.get( 'starchart_show_boundaries', True ) ),
					key='starchart_show_boundaries',
					disabled=(starchart_mode not in [ 'coordinate_chart', 'static_chart' ])
				)
			
			starchart_show_const_names = st.checkbox(
				'Show Constellation Names',
				value=bool( st.session_state.get( 'starchart_show_const_names', False ) ),
				key='starchart_show_const_names',
				disabled=(starchart_mode != 'static_chart')
			)
			
			starchart_timeout = st.number_input(
				'Timeout',
				min_value=1,
				max_value=120,
				value=int( st.session_state.get( 'starchart_timeout', 20 ) ),
				step=1,
				key='starchart_timeout'
			)
			
			st.caption(
				'Examples: object_search with Polaris, object_chart with M31, '
				'coordinate_chart with RA=2.5302 and Dec=89.2642, or '
				'static_chart with DSS2 and 900x450 output.'
			)
			
			b1, b2 = st.columns( 2 )
			
			with b1:
				starchart_submit = st.button(
					'Submit',
					key='starchart_submit'
				)
			
			with b2:
				st.button(
					'Clear',
					key='starchart_clear',
					on_click=_clear_starchart_state
				)
		
		with col_right:
			if starchart_submit:
				try:
					f = StarChart( )
					result = f.fetch(
						mode=starchart_mode,
						query=str( starchart_query or '' ),
						ra=float( starchart_ra ),
						dec=float( starchart_dec ),
						zoom=int( starchart_zoom ),
						image_source=str( starchart_image_source ),
						box_color=str( starchart_box_color or 'yellow' ),
						show_box=bool( starchart_show_box ),
						show_grid=bool( starchart_show_grid ),
						show_lines=bool( starchart_show_lines ),
						show_boundaries=bool( starchart_show_boundaries ),
						show_const_names=bool( starchart_show_const_names ),
						width=int( starchart_width ),
						height=int( starchart_height ),
						magnitude=float( starchart_magnitude ),
						time=int( starchart_timeout )
					)
					
					st.session_state[ 'starchart_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Star Chart request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'starchart_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				mode_value = result.get( 'mode', '' ) if isinstance( result, dict ) else ''
				params = result.get( 'params', { } ) if isinstance( result, dict ) else { }
				search_payload = result.get( 'search', { } ) if isinstance( result, dict ) else { }
				data = result.get( 'data', { } ) if isinstance( result, dict ) else { }
				chart_url = result.get( 'chart_url', '' ) if isinstance( result, dict ) else ''
				image_url = result.get( 'image_url', '' ) if isinstance( result, dict ) else ''
				
				st.markdown( '#### Chart Summary' )
				
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if mode_value:
						st.markdown( f'**Mode:** {mode_value}' )
					if result.get( 'url', '' ):
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
					if params:
						if 'query' in params and params.get( 'query', '' ):
							st.markdown( f"**Query:** {params.get( 'query', '' )}" )
						if 'ra' in params and 'dec' in params:
							st.markdown(
								f"**Coordinates:** RA={params.get( 'ra', '' )}, "
								f"Dec={params.get( 'dec', '' )}"
							)
				
				with meta_c2:
					if chart_url:
						st.markdown( f'**Chart Link:** {chart_url}' )
					if image_url:
						st.markdown( f'**Image URL:** {image_url}' )
					if 'zoom' in params:
						st.markdown( f"**Zoom:** {params.get( 'zoom', '' )}" )
				
				if image_url:
					st.markdown( '#### Preview' )
					st.image( image_url, use_container_width=True )
				
				if search_payload:
					st.markdown( '#### Object Search' )
					
					if isinstance( search_payload, list ):
						items = [ item for item in search_payload if isinstance( item, dict ) ]
					elif isinstance( search_payload, dict ):
						items = [ search_payload ]
					else:
						items = [ ]
					
					if items:
						for idx, item in enumerate( items, start=1 ):
							title_value = (
									item.get( 'name' )
									or item.get( 'title' )
									or item.get( 'object' )
									or f'Result {idx}'
							)
							
							with st.expander( f'Result {idx}: {title_value}', expanded=False ):
								summary_parts: List[ str ] = [ ]
								
								for key in [ 'ra', 'dec', 'type', 'constellation', 'magnitude' ]:
									if key in item and str( item.get( key ) ).strip( ):
										summary_parts.append( f'{key}={item.get( key )}' )
								
								if summary_parts:
									st.caption( ' | '.join( summary_parts ) )
								
								st.json( item )
					else:
						st.json( search_payload )
				
				if data:
					st.markdown( '#### Chart Data' )
					
					if isinstance( data, list ):
						df_chart = pd.DataFrame( data )
						if not df_chart.empty:
							st.dataframe( df_chart, use_container_width=True, hide_index=True )
						else:
							st.json( data )
					elif isinstance( data, dict ):
						key_fields = { }
						for key in [ 'name', 'title', 'ra', 'dec', 'type', 'constellation',
						             'magnitude' ]:
							if key in data:
								key_fields[ key ] = data.get( key )
						
						if key_fields:
							st.json( key_fields )
						else:
							st.json( data )
					else:
						st.write( data )
				
				if params:
					with st.expander( 'Request Parameters', expanded=False ):
						st.json( params )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )
	
	# -------- Nearby Objects
	with st.expander( label='Near Earth Objects', expanded=False ):
		if 'nearbyobjects_results' not in st.session_state:
			st.session_state[ 'nearbyobjects_results' ] = { }
		
		if 'nearbyobjects_clear_request' not in st.session_state:
			st.session_state[ 'nearbyobjects_clear_request' ] = False
		
		if st.session_state.get( 'nearbyobjects_clear_request', False ):
			st.session_state[ 'nearbyobjects_mode' ] = 'close_approaches'
			st.session_state[ 'nearbyobjects_start_date' ] = '2026-03-01'
			st.session_state[ 'nearbyobjects_end_date' ] = '2026-03-31'
			st.session_state[ 'nearbyobjects_query' ] = 'Apophis'
			st.session_state[ 'nearbyobjects_query_type' ] = 'sstr'
			st.session_state[ 'nearbyobjects_dist_max' ] = '10LD'
			st.session_state[ 'nearbyobjects_body' ] = 'Earth'
			st.session_state[ 'nearbyobjects_sort' ] = 'date'
			st.session_state[ 'nearbyobjects_limit' ] = 20
			st.session_state[ 'nearbyobjects_dv' ] = 6.0
			st.session_state[ 'nearbyobjects_dur' ] = 360
			st.session_state[ 'nearbyobjects_stay' ] = 8
			st.session_state[ 'nearbyobjects_launch' ] = '2020-2045'
			st.session_state[ 'nearbyobjects_h' ] = 26.0
			st.session_state[ 'nearbyobjects_occ' ] = 7
			st.session_state[ 'nearbyobjects_include_physical' ] = True
			st.session_state[ 'nearbyobjects_include_close_approaches' ] = True
			st.session_state[ 'nearbyobjects_ca_body' ] = 'Earth'
			st.session_state[ 'nearbyobjects_include_discovery' ] = True
			st.session_state[ 'nearbyobjects_timeout' ] = 20
			st.session_state[ 'nearbyobjects_results' ] = { }
			st.session_state[ 'nearbyobjects_clear_request' ] = False
		
		def _clear_nearbyobjects_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the Near Earth Objects expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'nearbyobjects_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			nearby_mode = st.selectbox(
				'Mode',
				options=[
						'close_approaches',
						'object_lookup',
						'nhats_summary',
						'nhats_object',
						'fireballs'
				],
				index=[
						'close_approaches',
						'object_lookup',
						'nhats_summary',
						'nhats_object',
						'fireballs'
				].index(
					st.session_state.get( 'nearbyobjects_mode', 'close_approaches' )
				),
				key='nearbyobjects_mode',
				help='Choose close approaches, single-object lookup, NHATS screening, or fireball data.'
			)
			
			d1, d2 = st.columns( 2 )
			
			with d1:
				nearby_start_date = st.text_input(
					'Start Date',
					value=st.session_state.get( 'nearbyobjects_start_date', '2026-03-01' ),
					key='nearbyobjects_start_date',
					placeholder='2026-03-01',
					disabled=(nearby_mode not in [ 'close_approaches', 'fireballs' ])
				)
			
			with d2:
				nearby_end_date = st.text_input(
					'End Date',
					value=st.session_state.get( 'nearbyobjects_end_date', '2026-03-31' ),
					key='nearbyobjects_end_date',
					placeholder='2026-03-31',
					disabled=(nearby_mode != 'close_approaches')
				)
			
			nearby_query = st.text_area(
				'Object Query / Designation',
				height=80,
				key='nearbyobjects_query',
				placeholder=(
						'Examples:\n'
						'Apophis\n'
						'Eros\n'
						'99942\n'
						'2000 SG344'
				),
				disabled=(nearby_mode not in [ 'object_lookup', 'nhats_object' ])
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				nearby_query_type = st.selectbox(
					'Query Type',
					options=[ 'sstr', 'spk', 'des' ],
					index=[ 'sstr', 'spk', 'des' ].index(
						st.session_state.get( 'nearbyobjects_query_type', 'sstr' )
					),
					key='nearbyobjects_query_type',
					disabled=(nearby_mode != 'object_lookup')
				)
			
			with c2:
				nearby_dist_max = st.text_input(
					'Distance Max',
					value=st.session_state.get( 'nearbyobjects_dist_max', '10LD' ),
					key='nearbyobjects_dist_max',
					placeholder='10LD or 0.05AU',
					disabled=(nearby_mode != 'close_approaches')
				)
			
			c3, c4, c5 = st.columns( 3 )
			
			with c3:
				nearby_body = st.text_input(
					'Body',
					value=st.session_state.get( 'nearbyobjects_body', 'Earth' ),
					key='nearbyobjects_body',
					placeholder='Earth',
					disabled=(nearby_mode != 'close_approaches')
				)
			
			with c4:
				nearby_sort = st.text_input(
					'Sort',
					value=st.session_state.get( 'nearbyobjects_sort', 'date' ),
					key='nearbyobjects_sort',
					placeholder='date or dist',
					disabled=(nearby_mode != 'close_approaches')
				)
			
			with c5:
				nearby_limit = st.number_input(
					'Limit',
					min_value=1,
					max_value=500,
					value=int( st.session_state.get( 'nearbyobjects_limit', 20 ) ),
					step=1,
					key='nearbyobjects_limit'
				)
			
			st.markdown( '#### NHATS Filters' )
			
			n1, n2, n3 = st.columns( 3 )
			
			with n1:
				nearby_dv = st.number_input(
					'ΔV',
					min_value=0.0,
					max_value=20.0,
					value=float( st.session_state.get( 'nearbyobjects_dv', 6.0 ) ),
					step=0.1,
					key='nearbyobjects_dv',
					disabled=(nearby_mode not in [ 'nhats_summary', 'nhats_object' ])
				)
			
			with n2:
				nearby_dur = st.number_input(
					'Duration',
					min_value=1,
					max_value=3000,
					value=int( st.session_state.get( 'nearbyobjects_dur', 360 ) ),
					step=1,
					key='nearbyobjects_dur',
					disabled=(nearby_mode not in [ 'nhats_summary', 'nhats_object' ])
				)
			
			with n3:
				nearby_stay = st.number_input(
					'Stay',
					min_value=0,
					max_value=365,
					value=int( st.session_state.get( 'nearbyobjects_stay', 8 ) ),
					step=1,
					key='nearbyobjects_stay',
					disabled=(nearby_mode not in [ 'nhats_summary', 'nhats_object' ])
				)
			
			n4, n5, n6 = st.columns( 3 )
			
			with n4:
				nearby_launch = st.text_input(
					'Launch Window',
					value=st.session_state.get( 'nearbyobjects_launch', '2020-2045' ),
					key='nearbyobjects_launch',
					placeholder='2020-2045',
					disabled=(nearby_mode not in [ 'nhats_summary', 'nhats_object' ])
				)
			
			with n5:
				nearby_h = st.number_input(
					'H Max',
					min_value=0.0,
					max_value=40.0,
					value=float( st.session_state.get( 'nearbyobjects_h', 26.0 ) ),
					step=0.1,
					key='nearbyobjects_h',
					disabled=(nearby_mode != 'nhats_summary')
				)
			
			with n6:
				nearby_occ = st.number_input(
					'OCC Max',
					min_value=0,
					max_value=20,
					value=int( st.session_state.get( 'nearbyobjects_occ', 7 ) ),
					step=1,
					key='nearbyobjects_occ',
					disabled=(nearby_mode != 'nhats_summary')
				)
			
			st.markdown( '#### SBDB Options' )
			
			s1, s2, s3 = st.columns( 3 )
			
			with s1:
				nearby_include_physical = st.checkbox(
					'Physical Params',
					value=bool( st.session_state.get( 'nearbyobjects_include_physical', True ) ),
					key='nearbyobjects_include_physical',
					disabled=(nearby_mode != 'object_lookup')
				)
			
			with s2:
				nearby_include_close_approaches = st.checkbox(
					'CA Data',
					value=bool( st.session_state.get( 'nearbyobjects_include_close_approaches', True ) ),
					key='nearbyobjects_include_close_approaches',
					disabled=(nearby_mode != 'object_lookup')
				)
			
			with s3:
				nearby_include_discovery = st.checkbox(
					'Discovery',
					value=bool( st.session_state.get( 'nearbyobjects_include_discovery', True ) ),
					key='nearbyobjects_include_discovery',
					disabled=(nearby_mode != 'object_lookup')
				)
			
			nearby_ca_body = st.text_input(
				'CA Body',
				value=st.session_state.get( 'nearbyobjects_ca_body', 'Earth' ),
				key='nearbyobjects_ca_body',
				placeholder='Earth',
				disabled=(nearby_mode != 'object_lookup')
			)
			
			nearby_timeout = st.number_input(
				'Timeout',
				min_value=1,
				max_value=120,
				value=int( st.session_state.get( 'nearbyobjects_timeout', 20 ) ),
				step=1,
				key='nearbyobjects_timeout'
			)
			
			st.caption(
				'Examples: close_approaches with Distance Max=10LD and Body=Earth; '
				'object_lookup with Query=Apophis and Query Type=sstr; '
				'nhats_object with Query=99942; fireballs with Start Date=2014-01-01.'
			)
			
			b1, b2 = st.columns( 2 )
			
			with b1:
				nearby_submit = st.button(
					'Submit',
					key='nearbyobjects_submit'
				)
			
			with b2:
				st.button(
					'Clear',
					key='nearbyobjects_clear',
					on_click=_clear_nearbyobjects_state
				)
		
		with col_right:
			if nearby_submit:
				try:
					f = NearbyObjects( )
					result = f.fetch(
						mode=nearby_mode,
						start_date=str( nearby_start_date ),
						end_date=str( nearby_end_date ),
						query=str( nearby_query or '' ).strip( ),
						query_type=str( nearby_query_type ),
						dist_max=str( nearby_dist_max or '10LD' ),
						body=str( nearby_body or 'Earth' ),
						sort=str( nearby_sort or 'date' ),
						limit=int( nearby_limit ),
						dv=float( nearby_dv ),
						dur=int( nearby_dur ),
						stay=int( nearby_stay ),
						launch=str( nearby_launch or '2020-2045' ),
						h=float( nearby_h ),
						occ=int( nearby_occ ),
						include_physical=bool( nearby_include_physical ),
						include_close_approaches=bool( nearby_include_close_approaches ),
						ca_body=str( nearby_ca_body or 'Earth' ),
						include_discovery=bool( nearby_include_discovery ),
						time=int( nearby_timeout )
					)
					
					st.session_state[ 'nearbyobjects_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Near Earth Objects request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'nearbyobjects_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				meta_c1, meta_c2 = st.columns( 2 )
				
				with meta_c1:
					if 'mode' in result:
						st.markdown( f"**Mode:** {result.get( 'mode', '' )}" )
					if 'url' in result:
						st.markdown( f"**URL:** {result.get( 'url', '' )}" )
				
				with meta_c2:
					if 'count' in result:
						st.markdown( f"**Count:** {result.get( 'count', '' )}" )
				
				if result.get( 'params', { } ):
					st.markdown( '#### Request Parameters' )
					st.json( result.get( 'params', { } ) )
				
				if result.get( 'fields', [ ] ) and result.get( 'data', [ ] ):
					st.markdown( '#### Results' )
					df_nearby = pd.DataFrame(
						result.get( 'data', [ ] ),
						columns=result.get( 'fields', [ ] )
					)
					st.dataframe( df_nearby, use_container_width=True, hide_index=True )
				
				elif 'data' in result:
					st.markdown( '#### Results' )
					data = result.get( 'data', { } )
					if isinstance( data, list ):
						if data:
							df_nearby = pd.DataFrame( data )
							st.dataframe( df_nearby, use_container_width=True, hide_index=True )
						else:
							st.text( 'No rows returned.' )
					else:
						st.json( data )
				
				if result.get( 'signature', { } ):
					with st.expander( 'Signature', expanded=False ):
						st.json( result.get( 'signature', { } ) )
				
				with st.expander( 'Raw Result', expanded=False ):
					st.json( result )

# ==============================================================================
# POPULATION MODE
# ==============================================================================
elif mode == 'Population':
	st.subheader( f'⚕️ Population & Public Health Data' )
	st.divider( )
	
	# -------- U.S. Census Bureau
	with st.expander( label='U.S. Census Bureau', expanded=False ):
		if 'census_results' not in st.session_state:
			st.session_state[ 'census_results' ] = { }
		
		if 'census_clear_request' not in st.session_state:
			st.session_state[ 'census_clear_request' ] = False
		
		if st.session_state.get( 'census_clear_request', False ):
			st.session_state[ 'census_mode' ] = 'variables'
			st.session_state[ 'census_year' ] = '2022'
			st.session_state[ 'census_dataset' ] = 'acs/acs5'
			st.session_state[ 'census_fields' ] = 'NAME,B01001_001E'
			st.session_state[ 'census_for' ] = 'state:*'
			st.session_state[ 'census_in' ] = ''
			st.session_state[ 'census_predicates' ] = ''
			st.session_state[ 'census_timeout' ] = 20
			st.session_state[ 'census_results' ] = { }
			st.session_state[ 'census_clear_request' ] = False
		
		def _clear_census_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the Census expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'census_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			census_mode = st.selectbox(
				'Mode',
				options=[ 'variables', 'data' ],
				index=[ 'variables', 'data' ].index(
					st.session_state.get( 'census_mode', 'variables' )
				),
				key='census_mode',
				help=(
						'variables = dataset variable metadata; '
						'data = tabular Census query using get/for/in.'
				)
			)
			
			census_year = st.text_input(
				'Year',
				value=st.session_state.get( 'census_year', '2022' ),
				key='census_year',
				placeholder='2022'
			)
			
			census_dataset = st.text_input(
				'Dataset',
				value=st.session_state.get( 'census_dataset', 'acs/acs5' ),
				key='census_dataset',
				placeholder='acs/acs5'
			)
			
			census_fields = st.text_area(
				'Fields (get)',
				value=st.session_state.get( 'census_fields', 'NAME,B01001_001E' ),
				height=90,
				key='census_fields',
				placeholder='NAME,B01001_001E',
				disabled=(census_mode != 'data')
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				census_for = st.text_input(
					'For',
					value=st.session_state.get( 'census_for', 'state:*' ),
					key='census_for',
					placeholder='state:*',
					disabled=(census_mode != 'data')
				)
			
			with c2:
				census_in = st.text_input(
					'In',
					value=st.session_state.get( 'census_in', '' ),
					key='census_in',
					placeholder='state:24',
					disabled=(census_mode != 'data')
				)
			
			census_predicates = st.text_area(
				'Predicates',
				value=st.session_state.get( 'census_predicates', '' ),
				height=90,
				key='census_predicates',
				placeholder='SEX=1\nAGE=15',
				disabled=(census_mode != 'data'),
				help='Optional newline-delimited key=value filters.'
			)
			
			census_timeout = st.number_input(
				'Timeout',
				min_value=1,
				max_value=120,
				value=int( st.session_state.get( 'census_timeout', 20 ) ),
				step=1,
				key='census_timeout'
			)
			
			st.caption(
				'Examples: dataset = acs/acs5, fields = NAME,B01001_001E, '
				'for = state:*'
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				census_submit = st.button(
					'Submit',
					key='census_submit',
					use_container_width=True
				)
			
			with b2:
				st.button(
					'Clear',
					key='census_clear',
					on_click=_clear_census_state,
					use_container_width=True
				)
		
		with col_right:
			result = st.session_state.get( 'census_results', { } )
			
			if census_submit:
				try:
					f = CensusData( )
					result = f.fetch(
						mode=census_mode,
						year=str( census_year ),
						dataset=str( census_dataset ),
						fields=str( census_fields ),
						geography_for=str( census_for ),
						geography_in=str( census_in ),
						predicates=str( census_predicates ),
						time=int( census_timeout )
					)
					
					st.session_state[ 'census_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Census request failed.' )
					st.exception( exc )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_result_metadata( result )
				
				if result.get( 'mode', '' ) == 'variables':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					variables = payload.get( 'variables', { } ) if isinstance( payload, dict ) else { }
					
					rows: List[ Dict[ str, Any ] ] = [ ]
					if isinstance( variables, dict ):
						for name, meta in variables.items( ):
							if isinstance( meta, dict ):
								rows.append(
									{
											'Name': name,
											'Label': meta.get( 'label', '' ),
											'Concept': meta.get( 'concept', '' ),
											'PredicateType': meta.get( 'predicateType', '' ),
											'Group': meta.get( 'group', '' ),
											'Limit': meta.get( 'limit', '' ),
									}
								)
					
					_render_summary_kv(
						'#### Summary',
						{
								'Year': census_year,
								'Dataset': census_dataset,
								'VariableCount': len( rows ),
						}
					)
					_render_rows_table( '#### Variables', rows )
				
				elif result.get( 'mode', '' ) == 'data':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					rows = payload.get( 'rows', [ ] ) if isinstance( payload, dict ) else [ ]
					
					_render_summary_kv(
						'#### Summary',
						{
								'Year': census_year,
								'Dataset': census_dataset,
								'Fields': census_fields,
								'For': census_for,
								'In': census_in,
								'RowCount': len( rows ) if isinstance( rows, list ) else 0,
						}
					)
					_render_rows_table( '#### Data Rows', rows if isinstance( rows, list ) else [ ] )
				
				_render_fallback_raw( result )
	
	# -------- CDC SOCRATA
	with st.expander( label='CDC Socrata', expanded=False ):
		if 'socrata_results' not in st.session_state:
			st.session_state[ 'socrata_results' ] = { }
		
		if 'socrata_clear_request' not in st.session_state:
			st.session_state[ 'socrata_clear_request' ] = False
		
		if st.session_state.get( 'socrata_clear_request', False ):
			st.session_state[ 'socrata_mode' ] = 'rows'
			st.session_state[ 'socrata_domain' ] = 'data.cdc.gov'
			st.session_state[ 'socrata_dataset_id' ] = 'q8xq-ygsk'
			st.session_state[ 'socrata_select' ] = ''
			st.session_state[ 'socrata_where' ] = ''
			st.session_state[ 'socrata_order' ] = ''
			st.session_state[ 'socrata_group' ] = ''
			st.session_state[ 'socrata_limit' ] = 25
			st.session_state[ 'socrata_offset' ] = 0
			st.session_state[ 'socrata_timeout' ] = 20
			st.session_state[ 'socrata_results' ] = { }
			st.session_state[ 'socrata_clear_request' ] = False
		
		def _clear_socrata_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the Socrata expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'socrata_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			socrata_mode = st.selectbox(
				'Mode',
				options=[ 'rows', 'metadata' ],
				index=[ 'rows', 'metadata' ].index(
					st.session_state.get( 'socrata_mode', 'rows' )
				),
				key='socrata_mode',
				help='rows = query dataset rows; metadata = inspect dataset metadata.'
			)
			
			socrata_domain = st.text_input(
				'Domain',
				value=st.session_state.get( 'socrata_domain', 'data.cdc.gov' ),
				key='socrata_domain',
				placeholder='data.cdc.gov'
			)
			
			socrata_dataset_id = st.text_input(
				'Dataset ID',
				value=st.session_state.get( 'socrata_dataset_id', 'q8xq-ygsk' ),
				key='socrata_dataset_id',
				placeholder='q8xq-ygsk'
			)
			
			socrata_select = st.text_area(
				'Select',
				value=st.session_state.get( 'socrata_select', '' ),
				height=80,
				key='socrata_select',
				placeholder='locationname,datavaluetype,datavalue',
				disabled=(socrata_mode != 'rows')
			)
			
			socrata_where = st.text_area(
				'Where',
				value=st.session_state.get( 'socrata_where', '' ),
				height=100,
				key='socrata_where',
				placeholder="year = '2020'",
				disabled=(socrata_mode != 'rows')
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				socrata_order = st.text_input(
					'Order',
					value=st.session_state.get( 'socrata_order', '' ),
					key='socrata_order',
					placeholder='locationname ASC',
					disabled=(socrata_mode != 'rows')
				)
			
			with c2:
				socrata_group = st.text_input(
					'Group',
					value=st.session_state.get( 'socrata_group', '' ),
					key='socrata_group',
					placeholder='locationname',
					disabled=(socrata_mode != 'rows')
				)
			
			c3, c4, c5 = st.columns( 3 )
			
			with c3:
				socrata_limit = st.number_input(
					'Limit',
					min_value=1,
					max_value=50000,
					value=int( st.session_state.get( 'socrata_limit', 25 ) ),
					step=1,
					key='socrata_limit',
					disabled=(socrata_mode != 'rows')
				)
			
			with c4:
				socrata_offset = st.number_input(
					'Offset',
					min_value=0,
					max_value=1000000,
					value=int( st.session_state.get( 'socrata_offset', 0 ) ),
					step=1,
					key='socrata_offset',
					disabled=(socrata_mode != 'rows')
				)
			
			with c5:
				socrata_timeout = st.number_input(
					'Timeout',
					min_value=1,
					max_value=120,
					value=int( st.session_state.get( 'socrata_timeout', 20 ) ),
					step=1,
					key='socrata_timeout'
				)
			
			st.caption(
				'Example dataset: q8xq-ygsk on data.cdc.gov. '
				'Use SoQL clauses for select, where, order, and group.'
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				socrata_submit = st.button(
					'Submit',
					key='socrata_submit',
					use_container_width=True
				)
			
			with b2:
				st.button(
					'Clear',
					key='socrata_clear',
					on_click=_clear_socrata_state,
					use_container_width=True
				)
		
		with col_right:
			result = st.session_state.get( 'socrata_results', { } )
			
			if socrata_submit:
				try:
					f = Socrata( )
					result = f.fetch(
						mode=socrata_mode,
						domain=str( socrata_domain ),
						dataset_id=str( socrata_dataset_id ),
						select=str( socrata_select ),
						where=str( socrata_where ),
						order=str( socrata_order ),
						group=str( socrata_group ),
						limit=int( socrata_limit ),
						offset=int( socrata_offset ),
						time=int( socrata_timeout )
					)
					
					st.session_state[ 'socrata_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Socrata request failed.' )
					st.exception( exc )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_result_metadata( result )
				
				if result.get( 'mode', '' ) == 'metadata':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					_render_summary_kv(
						'#### Summary',
						{
								'Name': payload.get( 'name', '' ) if isinstance( payload, dict ) else '',
								'Description': payload.get( 'description', '' ) if isinstance( payload, dict ) else '',
								'RowsUpdatedAt': payload.get( 'rowsUpdatedAt', '' ) if isinstance( payload, dict ) else '',
								'ViewType': payload.get( 'viewType', '' ) if isinstance( payload, dict ) else '',
								'Columns': len( payload.get( 'columns', [ ] ) ) if isinstance( payload, dict ) else 0,
						}
					)
					
					rows: List[ Dict[ str, Any ] ] = [ ]
					columns_payload = payload.get( 'columns', [ ] ) if isinstance( payload, dict ) else [ ]
					for item in columns_payload:
						if isinstance( item, dict ):
							rows.append(
								{
										'Name': item.get( 'name', '' ),
										'FieldName': item.get( 'fieldName', '' ),
										'DataType': item.get( 'dataTypeName', '' ),
										'Description': item.get( 'description', '' ),
								}
							)
					
					_render_rows_table( '#### Columns', rows )
				
				elif result.get( 'mode', '' ) == 'rows':
					rows = result.get( 'data', [ ] ) if isinstance( result, dict ) else [ ]
					
					_render_summary_kv(
						'#### Summary',
						{
								'Domain': socrata_domain,
								'DatasetId': socrata_dataset_id,
								'Limit': int( socrata_limit ),
								'Offset': int( socrata_offset ),
								'RowCount': len( rows ) if isinstance( rows, list ) else 0,
						}
					)
					_render_rows_table( '#### Rows', rows if isinstance( rows, list ) else [ ] )
				
				_render_fallback_raw( result )
	
	# -------- US Health Data
	with st.expander( label='U.S. Health', expanded=False ):
		if 'healthdata_results' not in st.session_state:
			st.session_state[ 'healthdata_results' ] = { }
		
		if 'healthdata_clear_request' not in st.session_state:
			st.session_state[ 'healthdata_clear_request' ] = False
		
		if st.session_state.get( 'healthdata_clear_request', False ):
			st.session_state[ 'healthdata_mode' ] = 'rows'
			st.session_state[ 'healthdata_domain' ] = 'healthdata.gov'
			st.session_state[ 'healthdata_dataset_id' ] = ''
			st.session_state[ 'healthdata_select' ] = ''
			st.session_state[ 'healthdata_where' ] = ''
			st.session_state[ 'healthdata_order' ] = ''
			st.session_state[ 'healthdata_group' ] = ''
			st.session_state[ 'healthdata_limit' ] = 25
			st.session_state[ 'healthdata_offset' ] = 0
			st.session_state[ 'healthdata_timeout' ] = 20
			st.session_state[ 'healthdata_results' ] = { }
			st.session_state[ 'healthdata_clear_request' ] = False
		
		def _clear_healthdata_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the HealthData expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'healthdata_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			healthdata_mode = st.selectbox(
				'Mode',
				options=[ 'rows', 'metadata' ],
				index=[ 'rows', 'metadata' ].index(
					st.session_state.get( 'healthdata_mode', 'rows' )
				),
				key='healthdata_mode',
				help='rows = query dataset rows; metadata = inspect dataset metadata.'
			)
			
			healthdata_domain = st.text_input(
				'Domain',
				value=st.session_state.get( 'healthdata_domain', 'healthdata.gov' ),
				key='healthdata_domain',
				placeholder='healthdata.gov'
			)
			
			healthdata_dataset_id = st.text_input(
				'Dataset ID',
				value=st.session_state.get( 'healthdata_dataset_id', '' ),
				key='healthdata_dataset_id',
				placeholder='dataset id'
			)
			
			healthdata_select = st.text_area(
				'Select',
				value=st.session_state.get( 'healthdata_select', '' ),
				height=80,
				key='healthdata_select',
				placeholder='column1,column2',
				disabled=(healthdata_mode != 'rows')
			)
			
			healthdata_where = st.text_area(
				'Where',
				value=st.session_state.get( 'healthdata_where', '' ),
				height=100,
				key='healthdata_where',
				placeholder="year = '2024'",
				disabled=(healthdata_mode != 'rows')
			)
			
			c1, c2 = st.columns( 2 )
			
			with c1:
				healthdata_order = st.text_input(
					'Order',
					value=st.session_state.get( 'healthdata_order', '' ),
					key='healthdata_order',
					placeholder='column1 ASC',
					disabled=(healthdata_mode != 'rows')
				)
			
			with c2:
				healthdata_group = st.text_input(
					'Group',
					value=st.session_state.get( 'healthdata_group', '' ),
					key='healthdata_group',
					placeholder='column1',
					disabled=(healthdata_mode != 'rows')
				)
			
			c3, c4, c5 = st.columns( 3 )
			
			with c3:
				healthdata_limit = st.number_input(
					'Limit',
					min_value=1,
					max_value=50000,
					value=int( st.session_state.get( 'healthdata_limit', 25 ) ),
					step=1,
					key='healthdata_limit',
					disabled=(healthdata_mode != 'rows')
				)
			
			with c4:
				healthdata_offset = st.number_input(
					'Offset',
					min_value=0,
					max_value=1000000,
					value=int( st.session_state.get( 'healthdata_offset', 0 ) ),
					step=1,
					key='healthdata_offset',
					disabled=(healthdata_mode != 'rows')
				)
			
			with c5:
				healthdata_timeout = st.number_input(
					'Timeout',
					min_value=1,
					max_value=120,
					value=int( st.session_state.get( 'healthdata_timeout', 20 ) ),
					step=1,
					key='healthdata_timeout'
				)
			
			st.caption(
				'HealthData.gov exposes developer tools and open API access. '
				'Use SoQL-style clauses for select, where, order, and group.'
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				healthdata_submit = st.button(
					'Submit',
					key='healthdata_submit',
					use_container_width=True
				)
			
			with b2:
				st.button(
					'Clear',
					key='healthdata_clear',
					on_click=_clear_healthdata_state,
					use_container_width=True
				)
		
		with col_right:
			result = st.session_state.get( 'healthdata_results', { } )
			
			if healthdata_submit:
				try:
					f = HealthData( )
					result = f.fetch(
						mode=healthdata_mode,
						domain=str( healthdata_domain ),
						dataset_id=str( healthdata_dataset_id ),
						select=str( healthdata_select ),
						where=str( healthdata_where ),
						order=str( healthdata_order ),
						group=str( healthdata_group ),
						limit=int( healthdata_limit ),
						offset=int( healthdata_offset ),
						time=int( healthdata_timeout )
					)
					
					st.session_state[ 'healthdata_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'HealthData request failed.' )
					st.exception( exc )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_result_metadata( result )
				
				if result.get( 'mode', '' ) == 'metadata':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					_render_summary_kv(
						'#### Summary',
						{
								'Name': payload.get( 'name', '' ) if isinstance( payload, dict ) else '',
								'Description': payload.get( 'description', '' ) if isinstance( payload, dict ) else '',
								'RowsUpdatedAt': payload.get( 'rowsUpdatedAt', '' ) if isinstance( payload, dict ) else '',
								'ViewType': payload.get( 'viewType', '' ) if isinstance( payload, dict ) else '',
								'Columns': len( payload.get( 'columns', [ ] ) ) if isinstance( payload, dict ) else 0,
						}
					)
					
					rows: List[ Dict[ str, Any ] ] = [ ]
					columns_payload = payload.get( 'columns', [ ] ) if isinstance( payload, dict ) else [ ]
					for item in columns_payload:
						if isinstance( item, dict ):
							rows.append(
								{
										'Name': item.get( 'name', '' ),
										'FieldName': item.get( 'fieldName', '' ),
										'DataType': item.get( 'dataTypeName', '' ),
										'Description': item.get( 'description', '' ),
								}
							)
					
					_render_rows_table( '#### Columns', rows )
				
				elif result.get( 'mode', '' ) == 'rows':
					rows = result.get( 'data', [ ] ) if isinstance( result, dict ) else [ ]
					
					_render_summary_kv(
						'#### Summary',
						{
								'Domain': healthdata_domain,
								'DatasetId': healthdata_dataset_id,
								'Limit': int( healthdata_limit ),
								'Offset': int( healthdata_offset ),
								'RowCount': len( rows ) if isinstance( rows, list ) else 0,
						}
					)
					_render_rows_table( '#### Rows', rows if isinstance( rows, list ) else [ ] )
				
				_render_fallback_raw( result )
	
	# -------- WHO Global Health
	with st.expander( label='WHO Global', expanded=False ):
		if 'who_results' not in st.session_state:
			st.session_state[ 'who_results' ] = { }
		
		if 'who_clear_request' not in st.session_state:
			st.session_state[ 'who_clear_request' ] = False
		
		if st.session_state.get( 'who_clear_request', False ):
			st.session_state[ 'who_mode' ] = 'indicator_registry'
			st.session_state[ 'who_query_path' ] = ''
			st.session_state[ 'who_format' ] = 'json'
			st.session_state[ 'who_timeout' ] = 20
			st.session_state[ 'who_results' ] = { }
			st.session_state[ 'who_clear_request' ] = False
		
		def _clear_who_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the WHO Global Health expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'who_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			who_mode = st.selectbox(
				'Mode',
				options=[ 'indicator_registry', 'athena' ],
				index=[ 'indicator_registry', 'athena' ].index(
					st.session_state.get( 'who_mode', 'indicator_registry' )
				),
				key='who_mode',
				help=(
						'indicator_registry = WHO metadata landing content; '
						'athena = configurable WHO GHO query path.'
				)
			)
			
			who_query_path = st.text_area(
				'Query Path',
				value=st.session_state.get( 'who_query_path', '' ),
				height=100,
				key='who_query_path',
				placeholder='Indicator',
				disabled=(who_mode != 'athena'),
				help='Path appended after the WHO GHO API base endpoint.'
			)
			
			who_format = st.selectbox(
				'Format',
				options=[ 'json', 'xml' ],
				index=[ 'json', 'xml' ].index(
					st.session_state.get( 'who_format', 'json' )
				),
				key='who_format',
				disabled=(who_mode != 'athena')
			)
			
			who_timeout = st.number_input(
				'Timeout',
				min_value=1,
				max_value=120,
				value=int( st.session_state.get( 'who_timeout', 20 ) ),
				step=1,
				key='who_timeout'
			)
			
			st.caption(
				'WHO documents both GHO OData API and Athena API endpoints. '
				'Use athena mode for direct query-path requests.'
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				who_submit = st.button(
					'Submit',
					key='who_submit',
					use_container_width=True
				)
			
			with b2:
				st.button(
					'Clear',
					key='who_clear',
					on_click=_clear_who_state,
					use_container_width=True
				)
		
		with col_right:
			result = st.session_state.get( 'who_results', { } )
			
			if who_submit:
				try:
					f = GlobalHealthData( )
					result = f.fetch(
						mode=who_mode,
						query_path=str( who_query_path ),
						fmt=str( who_format ),
						time=int( who_timeout )
					)
					
					st.session_state[ 'who_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'WHO Global Health request failed.' )
					st.exception( exc )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_result_metadata( result )
				
				if result.get( 'mode', '' ) == 'indicator_registry':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					_render_summary_kv(
						'#### Summary',
						{
								'Mode': result.get( 'mode', '' ),
								'HasHtml': isinstance( payload, dict ) and bool( payload.get( 'html', '' ) ),
						}
					)
					
					if isinstance( payload, dict ) and payload.get( 'html', '' ):
						_render_html_preview(
							'#### Indicator Registry Preview',
							str( payload.get( 'html', '' ) ) )
					else:
						st.json( payload )
				
				elif result.get( 'mode', '' ) == 'athena':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					if isinstance( payload, dict ) and isinstance( payload.get( 'value', [ ] ), list ):
						rows = payload.get( 'value', [ ] )
						_render_summary_kv(
							'#### Summary',
							{
									'QueryPath': who_query_path,
									'Format': who_format,
									'ResultCount': len( rows ),
							}
						)
						_render_rows_table( '#### Athena Results', rows )
					elif isinstance( payload, dict ) and payload.get( 'text', '' ):
						_render_summary_kv(
							'#### Summary',
							{
									'QueryPath': who_query_path,
									'Format': who_format,
									'HasText': True,
							}
						)
						st.markdown( '#### Response' )
						st.code( str( payload.get( 'text', '' ) )[ :8000 ] )
					else:
						st.json( payload )
				
				_render_fallback_raw( result )
	
	# -------- United Nations Data
	with st.expander( label='United Nations', expanded=False ):
		if 'un_results' not in st.session_state:
			st.session_state[ 'un_results' ] = { }
		
		if 'un_clear_request' not in st.session_state:
			st.session_state[ 'un_clear_request' ] = False
		
		if st.session_state.get( 'un_clear_request', False ):
			st.session_state[ 'un_mode' ] = 'datasets'
			st.session_state[ 'un_query_path' ] = ''
			st.session_state[ 'un_timeout' ] = 20
			st.session_state[ 'un_results' ] = { }
			st.session_state[ 'un_clear_request' ] = False
		
		def _clear_un_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the United Nations expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'un_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			un_mode = st.selectbox(
				'Mode',
				options=[ 'datasets', 'sdmx_query' ],
				index=[ 'datasets', 'sdmx_query' ].index(
					st.session_state.get( 'un_mode', 'datasets' )
				),
				key='un_mode',
				help=(
						'datasets = UNdata dataset catalog landing content; '
						'sdmx_query = direct REST SDMX query path.'
				)
			)
			
			un_query_path = st.text_area(
				'Query Path',
				value=st.session_state.get( 'un_query_path', '' ),
				height=120,
				key='un_query_path',
				placeholder='data/DF_SDG_GLH/..SI_POV_DAY1...........?',
				disabled=(un_mode != 'sdmx_query'),
				help='Path appended after https://data.un.org/WS/rest/'
			)
			
			un_timeout = st.number_input(
				'Timeout',
				min_value=1,
				max_value=120,
				value=int( st.session_state.get( 'un_timeout', 20 ) ),
				step=1,
				key='un_timeout'
			)
			
			st.caption(
				'UNdata documents SDMX REST access and a public dataset catalog. '
				'Use sdmx_query mode for direct REST query paths.'
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				un_submit = st.button(
					'Submit',
					key='un_submit',
					use_container_width=True
				)
			
			with b2:
				st.button(
					'Clear',
					key='un_clear',
					on_click=_clear_un_state,
					use_container_width=True
				)
		
		with col_right:
			result = st.session_state.get( 'un_results', { } )
			
			if un_submit:
				try:
					f = UnitedNations( )
					result = f.fetch(
						mode=un_mode,
						query_path=str( un_query_path ),
						time=int( un_timeout )
					)
					
					st.session_state[ 'un_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'United Nations request failed.' )
					st.exception( exc )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_result_metadata( result )
				
				if result.get( 'mode', '' ) == 'datasets':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					_render_summary_kv(
						'#### Summary',
						{
								'Mode': result.get( 'mode', '' ),
								'HasHtml': isinstance( payload, dict ) and bool( payload.get( 'html', '' ) ),
						}
					)
					
					if isinstance( payload, dict ) and payload.get( 'html', '' ):
						_render_html_preview(
							'#### Dataset Catalog Preview',
							str( payload.get( 'html', '' ) ) )
					else:
						st.json( payload )
				
				elif result.get( 'mode', '' ) == 'sdmx_query':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					_render_summary_kv(
						'#### Summary',
						{
								'Mode': result.get( 'mode', '' ),
								'QueryPath': un_query_path,
								'TextPayload': isinstance( payload, dict ) and bool( payload.get( 'text', '' ) ),
						}
					)
					
					if isinstance( payload, dict ) and payload.get( 'text', '' ):
						st.markdown( '#### Query Response' )
						st.code( str( payload.get( 'text', '' ) )[ :8000 ] )
					else:
						st.json( payload )
				
				_render_fallback_raw( result )
	
	# -------- World Population
	with st.expander( label='World Population', expanded=False ):
		if 'worldpop_results' not in st.session_state:
			st.session_state[ 'worldpop_results' ] = { }
		
		if 'worldpop_clear_request' not in st.session_state:
			st.session_state[ 'worldpop_clear_request' ] = False
		
		if st.session_state.get( 'worldpop_clear_request', False ):
			st.session_state[ 'worldpop_mode' ] = 'catalog'
			st.session_state[ 'worldpop_query' ] = ''
			st.session_state[ 'worldpop_asset_path' ] = ''
			st.session_state[ 'worldpop_page' ] = 1
			st.session_state[ 'worldpop_page_size' ] = 25
			st.session_state[ 'worldpop_timeout' ] = 20
			st.session_state[ 'worldpop_results' ] = { }
			st.session_state[ 'worldpop_clear_request' ] = False
		
		def _clear_worldpop_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the World Population expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'worldpop_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			worldpop_mode = st.selectbox(
				'Mode',
				options=[ 'catalog', 'search', 'raster_metadata' ],
				index=[ 'catalog', 'search', 'raster_metadata' ].index(
					st.session_state.get( 'worldpop_mode', 'catalog' )
				),
				key='worldpop_mode',
				help=(
						'catalog = API landing content; '
						'search = catalog-style search; '
						'raster_metadata = direct asset or metadata path.'
				)
			)
			
			worldpop_query = st.text_area(
				'Query',
				value=st.session_state.get( 'worldpop_query', '' ),
				height=90,
				key='worldpop_query',
				placeholder='population Ghana 2020',
				disabled=(worldpop_mode != 'search')
			)
			
			worldpop_asset_path = st.text_area(
				'Asset Path',
				value=st.session_state.get( 'worldpop_asset_path', '' ),
				height=100,
				key='worldpop_asset_path',
				placeholder='data/pop/wpgp?iso3=GHA',
				disabled=(worldpop_mode != 'raster_metadata')
			)
			
			c1, c2, c3 = st.columns( 3 )
			
			with c1:
				worldpop_page = st.number_input(
					'Page',
					min_value=1,
					max_value=100000,
					value=int( st.session_state.get( 'worldpop_page', 1 ) ),
					step=1,
					key='worldpop_page',
					disabled=(worldpop_mode != 'search')
				)
			
			with c2:
				worldpop_page_size = st.number_input(
					'Page Size',
					min_value=1,
					max_value=500,
					value=int( st.session_state.get( 'worldpop_page_size', 25 ) ),
					step=1,
					key='worldpop_page_size',
					disabled=(worldpop_mode != 'search')
				)
			
			with c3:
				worldpop_timeout = st.number_input(
					'Timeout',
					min_value=1,
					max_value=120,
					value=int( st.session_state.get( 'worldpop_timeout', 20 ) ),
					step=1,
					key='worldpop_timeout'
				)
			
			st.caption(
				'WorldPop publishes API-based access and STAC-oriented discovery. '
				'Use raster_metadata mode for direct API paths.'
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				worldpop_submit = st.button(
					'Submit',
					key='worldpop_submit',
					use_container_width=True
				)
			
			with b2:
				st.button(
					'Clear',
					key='worldpop_clear',
					on_click=_clear_worldpop_state,
					use_container_width=True
				)
		
		with col_right:
			result = st.session_state.get( 'worldpop_results', { } )
			
			if worldpop_submit:
				try:
					f = WorldPopulation( )
					result = f.fetch(
						mode=worldpop_mode,
						query=str( worldpop_query ),
						asset_path=str( worldpop_asset_path ),
						page=int( worldpop_page ),
						page_size=int( worldpop_page_size ),
						time=int( worldpop_timeout )
					)
					
					st.session_state[ 'worldpop_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'World Population request failed.' )
					st.exception( exc )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_result_metadata( result )
				
				if result.get( 'mode', '' ) == 'catalog':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					_render_summary_kv(
						'#### Summary',
						{
								'Mode': result.get( 'mode', '' ),
								'HasHtml': isinstance( payload, dict ) and bool( payload.get( 'html', '' ) ),
						}
					)
					
					if isinstance( payload, dict ) and payload.get( 'html', '' ):
						_render_html_preview(
							'#### Catalog Preview',
							str( payload.get( 'html', '' ) ) )
					else:
						st.json( payload )
				
				elif result.get( 'mode', '' ) == 'search':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					if isinstance( payload, dict ) and isinstance( payload.get( 'results', [ ] ), list ):
						rows = payload.get( 'results', [ ] )
						_render_summary_kv(
							'#### Summary',
							{
									'Query': worldpop_query,
									'Page': worldpop_page,
									'PageSize': worldpop_page_size,
									'ResultCount': len( rows ),
							}
						)
						_render_rows_table( '#### Search Results', rows )
					else:
						st.json( payload )
				
				elif result.get( 'mode', '' ) == 'raster_metadata':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					_render_summary_kv(
						'#### Summary',
						{
								'AssetPath': worldpop_asset_path,
								'HasText': isinstance( payload, dict ) and bool( payload.get( 'text', '' ) ),
						}
					)
					
					if isinstance( payload, dict ) and payload.get( 'text', '' ):
						st.markdown( '#### Metadata Response' )
						st.code( str( payload.get( 'text', '' ) )[ :8000 ] )
					else:
						st.json( payload )
				
				_render_fallback_raw( result )
	
	# -------- CDC WONDER
	with st.expander( label='CDC Wonder', expanded=False ):
		if 'wonder_results' not in st.session_state:
			st.session_state[ 'wonder_results' ] = { }
		
		if 'wonder_clear_request' not in st.session_state:
			st.session_state[ 'wonder_clear_request' ] = False
		
		if st.session_state.get( 'wonder_clear_request', False ):
			st.session_state[ 'wonder_mode' ] = 'metadata_template'
			st.session_state[ 'wonder_dataset_id' ] = 'D76'
			st.session_state[ 'wonder_request_xml' ] = ''
			st.session_state[ 'wonder_timeout' ] = 20
			st.session_state[ 'wonder_results' ] = { }
			st.session_state[ 'wonder_clear_request' ] = False
		
		def _clear_wonder_state( ) -> None:
			'''
				Purpose:
				--------
				Flag the CDC WONDER expander state for reset on the next rerun.

				Parameters:
				-----------
				None

				Returns:
				--------
				None
			'''
			st.session_state[ 'wonder_clear_request' ] = True
		
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			wonder_mode = st.selectbox(
				'Mode',
				options=[ 'metadata_template', 'query_xml' ],
				index=[ 'metadata_template', 'query_xml' ].index(
					st.session_state.get( 'wonder_mode', 'metadata_template' )
				),
				key='wonder_mode',
				help=(
						'metadata_template = build a starter XML request; '
						'query_xml = submit a raw XML request to CDC WONDER.'
				)
			)
			
			wonder_dataset_id = st.text_input(
				'Dataset ID',
				value=st.session_state.get( 'wonder_dataset_id', 'D76' ),
				key='wonder_dataset_id',
				placeholder='D76'
			)
			
			wonder_request_xml = st.text_area(
				'Request XML',
				value=st.session_state.get( 'wonder_request_xml', '' ),
				height=240,
				key='wonder_request_xml',
				placeholder='<request>...</request>',
				disabled=(wonder_mode != 'query_xml')
			)
			
			wonder_timeout = st.number_input(
				'Timeout',
				min_value=1,
				max_value=120,
				value=int( st.session_state.get( 'wonder_timeout', 20 ) ),
				step=1,
				key='wonder_timeout'
			)
			
			st.caption(
				'CDC WONDER requires POST requests with request_xml and '
				'acceptance of data-use restrictions.'
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				wonder_submit = st.button(
					'Submit',
					key='wonder_submit',
					use_container_width=True
				)
			
			with b2:
				st.button(
					'Clear',
					key='wonder_clear',
					on_click=_clear_wonder_state,
					use_container_width=True
				)
		
		with col_right:
			result = st.session_state.get( 'wonder_results', { } )
			
			if wonder_submit:
				try:
					f = Wonder( )
					result = f.fetch(
						mode=wonder_mode,
						dataset_id=str( wonder_dataset_id ),
						request_xml=str( wonder_request_xml ),
						time=int( wonder_timeout )
					)
					
					st.session_state[ 'wonder_results' ] = result or { }
					
					if (
							wonder_mode == 'metadata_template'
							and isinstance( result, dict )
							and isinstance( result.get( 'data', { } ), dict )
					):
						template_xml = result.get( 'data', { } ).get( 'request_xml', '' )
						st.session_state[ 'wonder_request_xml' ] = template_xml
					
					st.rerun( )
				
				except Exception as exc:
					st.error( 'CDC WONDER request failed.' )
					st.exception( exc )
			
			if not result:
				st.text( 'No results.' )
			else:
				_render_result_metadata( result )
				
				if result.get( 'mode', '' ) == 'metadata_template':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					if isinstance( payload, dict ):
						_render_summary_kv(
							'#### Template Summary',
							{
									'DatasetId': payload.get( 'dataset_id', '' ),
									'Notes': payload.get( 'notes', '' ),
							}
						)
						
						template_xml = str( payload.get( 'request_xml', '' ) )
						if template_xml:
							_render_xml_preview( '#### Starter XML', template_xml )
						else:
							st.info( 'No starter XML returned.' )
				
				elif result.get( 'mode', '' ) == 'query_xml':
					payload = result.get( 'data', { } ) if isinstance( result, dict ) else { }
					
					if isinstance( payload, dict ):
						xml_text = str( payload.get( 'xml', '' ) )
						
						_render_summary_kv(
							'#### Response Summary',
							{
									'DatasetId': wonder_dataset_id,
									'Characters': len( xml_text ),
									'HasXml': bool( xml_text.strip( ) ),
							}
						)
						
						_render_xml_preview( '#### XML Response', xml_text )
					else:
						st.info( 'No XML response returned.' )
				
				_render_fallback_raw( result )

	# -------- PubMed Search Loader
	with st.expander( label='Pub Med', icon='🧬', expanded=False ):
		query = st.text_input( 'PubMed Query', key='pubmed_query' )
		max_docs = st.number_input(
			'Max Documents',
			min_value=1,
			max_value=100,
			value=5,
			step=1,
			key='pubmed_max_docs'
		)
		
		col_load, col_clear, col_save = st.columns( 3 )
		load_pubmed = col_load.button( 'Load', key='pubmed_load' )
		clear_pubmed = col_clear.button( 'Clear', key='pubmed_clear' )
		
		can_save = (
				st.session_state.get( 'active_loader' ) == 'PubMedSearchLoader'
				and isinstance( st.session_state.get( 'raw_text' ), str )
				and st.session_state.get( 'raw_text' ).strip( )
		)
		
		if can_save:
			col_save.download_button(
				'Save',
				data=st.session_state.get( 'raw_text' ),
				file_name='pubmed_loader_output.txt',
				mime='text/plain',
				key='pubmed_save'
			)
		else:
			col_save.button( 'Save', key='pubmed_save_disabled', disabled=True )
		
		if clear_pubmed:
			remaining = _clear_loader_documents( 'PubMedSearchLoader' )
			st.info( f'PubMed Loader state cleared. Remaining documents: {remaining}.' )
		
		if load_pubmed and query.strip( ):
			try:
				loader = PubMedSearchLoader( query=query.strip( ), load_max_docs=int( max_docs ) )
				documents = loader.load( ) or [ ]
				count = _promote_loader_documents( documents, 'PubMedSearchLoader' )
				st.success( f'Loaded {count} PubMed document(s).' )
			except Exception as e:
				st.error( str( e ) )
		
		# ------- Open City Data Loader
		with st.expander( label='Open City Data Loader', icon='🏙️', expanded=False ):
			city_id = st.text_input( 'City ID', value='data.sfgov.org', key='open_city_id' )
			dataset_id = st.text_input( 'Dataset ID', key='open_city_dataset_id' )
			limit = st.number_input(
				'Limit',
				min_value=1,
				max_value=5000,
				value=100,
				step=10,
				key='open_city_limit'
			)
			
			col_load, col_clear, col_save = st.columns( 3 )
			load_open_city = col_load.button( 'Load', key='open_city_load' )
			clear_open_city = col_clear.button( 'Clear', key='open_city_clear' )
			
			can_save = (
					st.session_state.get( 'active_loader' ) == 'OpenCityLoader'
					and isinstance( st.session_state.get( 'raw_text' ), str )
					and st.session_state.get( 'raw_text' ).strip( )
			)
			
			if can_save:
				col_save.download_button(
					'Save',
					data=st.session_state.get( 'raw_text' ),
					file_name='open_city_loader_output.txt',
					mime='text/plain',
					key='open_city_save'
				)
			else:
				col_save.button( 'Save', key='open_city_save_disabled', disabled=True )
			
			if clear_open_city:
				clear_if_active( 'OpenCityLoader' )
				st.session_state.raw_text = _rebuild_raw_text_from_documents( )
				st.session_state[ '_loader_status' ] = 'Open City Data Loader state cleared.'
				st.rerun( )
			
			if load_open_city and city_id and dataset_id:
				loader = OpenCityLoader( )
				documents = loader.load(
					city_id=city_id,
					dataset_id=dataset_id,
					limit=int( limit )
				) or [ ]
				st.session_state.documents = documents
				st.session_state.raw_documents = list( documents )
				st.session_state.raw_text = '\n\n'.join(
					d.page_content for d in documents
					if hasattr( d, 'page_content' ) and isinstance( d.page_content, str )
					and d.page_content.strip( )
				)
				st.session_state.processed_text = None
				st.session_state.tokens = None
				st.session_state.vocabulary = None
				st.session_state.token_counts = None
				st.session_state.active_loader = 'OpenCityLoader'
				st.session_state[ '_loader_status' ] = (
						f'Loaded {len( documents )} Open City document(s).'
				)
				st.rerun( )
	
	# -------- Open City Loader
	with st.expander( label='Open City', icon='🏙️', expanded=False ):
		city_id = st.text_input( 'City ID', value='data.sfgov.org', key='open_city_id' )
		dataset_id = st.text_input( 'Dataset ID', key='open_city_dataset_id' )
		limit = st.number_input(
			'Limit',
			min_value=1,
			max_value=5000,
			value=100,
			step=10,
			key='open_city_limit'
		)
		
		col_load, col_clear, col_save = st.columns( 3 )
		load_open_city = col_load.button( 'Load', key='open_city_load' )
		clear_open_city = col_clear.button( 'Clear', key='open_city_clear' )
		
		can_save = (
				st.session_state.get( 'active_loader' ) == 'OpenCityLoader'
				and isinstance( st.session_state.get( 'raw_text' ), str )
				and st.session_state.get( 'raw_text' ).strip( )
		)
		
		if can_save:
			col_save.download_button(
				'Save',
				data=st.session_state.get( 'raw_text' ),
				file_name='open_city_loader_output.txt',
				mime='text/plain',
				key='open_city_save'
			)
		else:
			col_save.button( 'Save', key='open_city_save_disabled', disabled=True )
		
		if clear_open_city:
			remaining = _clear_loader_documents( 'OpenCityLoader' )
			st.info( f'Open City Data Loader state cleared. Remaining documents: {remaining}.' )
		
		if load_open_city and city_id.strip( ) and dataset_id.strip( ):
			try:
				loader = OpenCityLoader(
					city_id=city_id.strip( ),
					dataset_id=dataset_id.strip( ),
					limit=int( limit )
				)
				
				documents = loader.load( ) or [ ]
				count = _promote_loader_documents( documents, 'OpenCityLoader' )
				st.success( f'Loaded {count} Open City document(s).' )
			except Exception as e:
				st.error( str( e ) )
			
# ==============================================================================
# TEXT GENERATION MODE
# ==============================================================================
elif mode == 'Generation':
	st.subheader( f'🧠  Generative AI' )
	st.divider( )
	
	# -------- ChatGPT
	with st.expander( label='ChatGPT', expanded=True ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			chat_prompt = st.text_area( 'Prompt', value='', height=120, key='chat_prompt' )
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			p_row4 = st.columns( 2 )
			p_row5 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				_chat_models = cfg.GPT_MODELS if hasattr( cfg, 'GPT_MODELS' ) and cfg.GPT_MODELS else \
					[ 'gpt-5.4', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-4.1' ]
				chat_model = _model_selector(
					key_prefix='chat',
					label='Model',
					options=_chat_models,
					default_model=(
							'gpt-5-mini' if 'gpt-5-mini' in _chat_models else _chat_models[ 0 ]),
				)
			
			with p_row1[ 1 ]:
				chat_temperature = st.slider(
					'Temperature',
					min_value=0.0,
					max_value=2.0,
					value=0.7,
					step=0.05,
					key='chat_temperature',
				)
			
			with p_row2[ 0 ]:
				chat_max_tokens = st.number_input(
					'Max Tokens',
					min_value=1,
					max_value=32768,
					value=2048,
					step=1,
					key='chat_max_tokens',
				)
			
			with p_row2[ 1 ]:
				chat_top_p = st.slider(
					'Top-P',
					min_value=0.0,
					max_value=1.0,
					value=1.0,
					step=0.01,
					key='chat_top_p',
				)
			
			with p_row3[ 0 ]:
				chat_seed = st.number_input(
					'Seed',
					min_value=0,
					max_value=2_147_483_647,
					value=0,
					step=1,
					key='chat_seed',
				)
			
			with p_row3[ 1 ]:
				chat_json_mode = st.checkbox(
					'JSON Mode',
					value=False,
					key='chat_json_mode',
				)
			
			with p_row4[ 0 ]:
				chat_reasoning = st.checkbox(
					'Reasoning',
					value=False,
					key='chat_reasoning',
				)
			
			with p_row4[ 1 ]:
				chat_web_search = st.checkbox(
					'Web Search',
					value=False,
					key='chat_web_search',
				)
			
			with p_row5[ 0 ]:
				chat_store = st.checkbox(
					'Store',
					value=True,
					key='chat_store',
				)
			
			with p_row5[ 1 ]:
				chat_stream = st.checkbox(
					'Stream',
					value=False,
					key='chat_stream',
				)
			
			_chat_supports_reasoning = (
					str( chat_model ).strip( ).lower( ).startswith( 'gpt-5' )
					or str( chat_model ).strip( ).lower( ).startswith( 'o' )
			)
			
			if _chat_supports_reasoning and chat_reasoning:
				chat_reasoning_effort = st.selectbox(
					'Reasoning Effort',
					options=[ 'minimal', 'low', 'medium', 'high' ],
					index=1,
					key='chat_reasoning_effort',
				)
			else:
				chat_reasoning_effort = None
			
			chat_system = st.text_area(
				'System',
				value='',
				height=120,
				key='chat_system',
			)
			
			if chat_web_search:
				chat_domains = st.text_area(
					'Preferred Search Domains (one per line or comma-separated)',
					value='',
					height=90,
					key='chat_domains',
					help='Examples: openai.com, platform.openai.com, arxiv.org',
				)
			else:
				chat_domains = ''
			
			btn_row = st.columns( 2 )
			with btn_row[ 0 ]:
				chat_submit = st.button( 'Submit', key='chat_submit' )
			with btn_row[ 1 ]:
				chat_clear = st.button( 'Clear', key='chat_clear' )
		
		with col_right:
			chat_output = st.empty( )
		
		# -----------------------------
		# Clear Button
		# -----------------------------
		if chat_clear:
			st.session_state.update(
				{
						'chat_prompt': '',
						'chat_system': '',
						'chat_domains': '',
						'chat_json_mode': False,
						'chat_reasoning': False,
						'chat_web_search': False,
						'chat_store': True,
						'chat_stream': False,
						'chat_seed': 0,
				}
			)
			st.rerun( )
		
		# -----------------------------
		# Submit Button
		# -----------------------------
		if chat_submit:
			try:
				if not str( chat_prompt ).strip( ):
					raise ValueError( 'Prompt cannot be empty.' )
				
				chat_domains_list = [ ]
				if chat_domains and str( chat_domains ).strip( ):
					_domain_entries = re.split( r'[\n,;]+', str( chat_domains ) )
					for _entry in _domain_entries:
						_value = str( _entry ).strip( ).lower( )
						if not _value:
							continue
						
						if not _value.startswith( 'http://' ) and not _value.startswith( 'https://' ):
							_value = f'https://{_value}'
						
						_parsed = urlparse( _value )
						_domain = (_parsed.netloc or _parsed.path or '').strip( ).lower( )
						_domain = re.sub( r':\d+$', '', _domain )
						_domain = _domain.lstrip( '.' )
						
						if _domain.startswith( 'www.' ):
							_domain = _domain[ 4: ]
						
						if _domain and _domain not in chat_domains_list:
							chat_domains_list.append( _domain )
				
				fetcher = Chat( )
				params = \
					{
							'model': chat_model,
							'temperature': float( chat_temperature ),
							'max_tokens': int( chat_max_tokens ),
							'top_p': float( chat_top_p ),
							'seed': int( chat_seed ) if int( chat_seed ) > 0 else None,
							'system': chat_system if str( chat_system ).strip( ) else None,
							'response_format': ('json' if chat_json_mode else None),
							'reasoning_effort': (
									chat_reasoning_effort
									if _chat_supports_reasoning and chat_reasoning and chat_reasoning_effort
									else None
							),
							'web_search': bool( chat_web_search ),
							'search_domains': chat_domains_list if chat_domains_list else None,
							'store': bool( chat_store ),
							'stream': bool( chat_stream ),
							'parallel_tool_calls': True,
							'tool_choice': 'auto',
					}
				
				params = { k: v for k, v in params.items( ) if v is not None }
				result = _invoke_provider( fetcher, chat_prompt, params )
				_render_output( chat_output, result )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- Groq
	with st.expander( label='Grok', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			groq_prompt = st.text_area(
				'Prompt',
				value='',
				height=120,
				key='groq_prompt_chat',
			)
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			p_row4 = st.columns( 2 )
			p_row5 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				_grok_models = cfg.GROK_MODELS if hasattr( cfg, 'GROK_MODELS' ) and cfg.GROK_MODELS else \
					[ 'grok-4-1-fast-reasoning',
					  'grok-4-fast-reasoning',
					  'grok-4',
					  'grok-code-fast-1',
					  'grok-3-mini' ]
				groq_model = _model_selector(
					key_prefix='groq',
					label='Model',
					options=_grok_models,
					default_model=(
							'grok-4-fast-reasoning' if 'grok-4-fast-reasoning' in _grok_models else
							_grok_models[ 0 ]),
				)
			
			with p_row1[ 1 ]:
				groq_temperature = st.slider(
					'Temperature',
					min_value=0.0,
					max_value=2.0,
					value=0.7,
					step=0.05,
					key='groq_temperature_chat',
				)
			
			with p_row2[ 0 ]:
				groq_max_tokens = st.number_input(
					'Max Tokens',
					min_value=1,
					max_value=32768,
					value=2048,
					step=1,
					key='groq_max_tokens_chat',
				)
			
			with p_row2[ 1 ]:
				groq_top_p = st.slider(
					'Top-P',
					min_value=0.0,
					max_value=1.0,
					value=1.0,
					step=0.01,
					key='groq_top_p_chat',
				)
			
			with p_row3[ 0 ]:
				groq_seed = st.number_input(
					'Seed',
					min_value=0,
					max_value=2_147_483_647,
					value=0,
					step=1,
					key='groq_seed_chat',
				)
			
			with p_row3[ 1 ]:
				groq_json_mode = st.checkbox(
					'JSON Mode',
					value=False,
					key='groq_json_mode_chat',
				)
			
			with p_row4[ 0 ]:
				groq_reasoning = st.checkbox(
					'Reasoning',
					value=False,
					key='groq_reasoning_chat',
					help='Use for models that support explicit reasoning controls. Grok 4 models reason natively.',
				)
			
			with p_row4[ 1 ]:
				groq_web_search = st.checkbox(
					'Web Search',
					value=False,
					key='groq_web_search_chat',
				)
			
			with p_row5[ 0 ]:
				groq_store = st.checkbox(
					'Store',
					value=True,
					key='groq_store_chat',
				)
			
			with p_row5[ 1 ]:
				groq_stream = st.checkbox(
					'Stream',
					value=False,
					key='groq_stream_chat',
				)
			
			_groq_supports_reasoning_effort = 'grok-3-mini' in str( groq_model ).strip( ).lower( )
			_groq_is_reasoning_model = (
					'grok-4' in str( groq_model ).strip( ).lower( )
					or 'reasoning' in str( groq_model ).strip( ).lower( )
			)
			
			if _groq_supports_reasoning_effort and groq_reasoning:
				groq_reasoning_effort = st.selectbox(
					'Reasoning Effort',
					options=[ 'low', 'high' ],
					index=0,
					key='groq_reasoning_effort_chat',
				)
			else:
				groq_reasoning_effort = None
			
			groq_system = st.text_area(
				'System',
				value='',
				height=120,
				key='groq_system_chat',
			)
			
			if not _groq_is_reasoning_model:
				groq_stop = st.text_area(
					'Stop Sequences (one per line)',
					value='',
					height=90,
					key='groq_stop_chat',
				)
			else:
				groq_stop = ''
				st.caption( 'Stop sequences are omitted for Grok reasoning models.' )
			
			if groq_web_search:
				groq_domains = st.text_area(
					'Allowed Search Domains (one per line or comma-separated)',
					value='',
					height=90,
					key='groq_domains_chat',
					help='Examples: x.ai, docs.x.ai, arxiv.org',
				)
			else:
				groq_domains = ''
			
			btn_row = st.columns( 2 )
			with btn_row[ 0 ]:
				groq_submit = st.button( 'Submit', key='groq_submit_chat' )
			with btn_row[ 1 ]:
				groq_clear = st.button( 'Clear', key='groq_clear_chat' )
		
		with col_right:
			groq_output = st.empty( )
		
		if groq_clear:
			st.session_state.update(
				{
						'groq_prompt_chat': '',
						'groq_system_chat': '',
						'groq_stop_chat': '',
						'groq_domains_chat': '',
						'groq_json_mode_chat': False,
						'groq_reasoning_chat': False,
						'groq_web_search_chat': False,
						'groq_store_chat': True,
						'groq_stream_chat': False,
						'groq_seed_chat': 0,
				}
			)
			st.rerun( )
		
		if groq_submit:
			try:
				if not str( groq_prompt ).strip( ):
					raise ValueError( 'Prompt cannot be empty.' )
				
				groq_domains_list = [ ]
				if groq_domains and str( groq_domains ).strip( ):
					_domain_entries = re.split( r'[\n,;]+', str( groq_domains ) )
					for _entry in _domain_entries:
						_value = str( _entry ).strip( ).lower( )
						if not _value:
							continue
						
						if not _value.startswith( 'http://' ) and not _value.startswith( 'https://' ):
							_value = f'https://{_value}'
						
						_parsed = urlparse( _value )
						_domain = (_parsed.netloc or _parsed.path or '').strip( ).lower( )
						_domain = re.sub( r':\d+$', '', _domain )
						_domain = _domain.lstrip( '.' )
						
						if _domain.startswith( 'www.' ):
							_domain = _domain[ 4: ]
						
						if _domain and _domain not in groq_domains_list:
							groq_domains_list.append( _domain )
				
				stop_lines = [ s.strip( ) for s in (groq_stop or '').splitlines( ) if s.strip( ) ]
				
				fetcher = Grok( )
				params = \
					{
							'model': groq_model,
							'temperature': float( groq_temperature ),
							'max_tokens': int( groq_max_tokens ),
							'top_p': float( groq_top_p ),
							'seed': int( groq_seed ) if int( groq_seed ) > 0 else None,
							'system': groq_system if str( groq_system ).strip( ) else None,
							'response_format': ('json' if groq_json_mode else None),
							'reasoning_effort': (
									groq_reasoning_effort
									if _groq_supports_reasoning_effort and groq_reasoning and groq_reasoning_effort
									else None
							),
							'web_search': bool( groq_web_search ),
							'search_domains': groq_domains_list if groq_domains_list else None,
							'stop': stop_lines if stop_lines and not _groq_is_reasoning_model else None,
							'stream': bool( groq_stream ),
							'store': bool( groq_store ),
							'parallel_tool_calls': True,
							'tool_choice': 'auto',
					}
				
				params = { k: v for k, v in params.items( ) if v is not None }
				result = _invoke_provider( fetcher, groq_prompt, params )
				_render_output( groq_output, result )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- CLAUDE
	with st.expander( label='Claude', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			claude_prompt = st.text_area(
				'Prompt',
				value='',
				height=120,
				key='claude_prompt_chat',
			)
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			p_row4 = st.columns( 2 )
			p_row5 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				_claude_models = (
						cfg.CLAUDE_MODELS
						if hasattr( cfg, 'CLAUDE_MODELS' ) and cfg.CLAUDE_MODELS
						else [
								'claude-opus-4-6',
								'claude-sonnet-4-6',
								'claude-haiku-4-5',
								'claude-3-5-haiku-latest',
						]
				)
				claude_model = _model_selector(
					key_prefix='claude',
					label='Model',
					options=_claude_models,
					default_model=('claude-sonnet-4-6' if 'claude-sonnet-4-6' in _claude_models else
					               _claude_models[ 0 ]),
				)
			
			with p_row1[ 1 ]:
				claude_temperature = st.slider(
					'Temperature',
					min_value=0.0,
					max_value=1.0,
					value=0.7,
					step=0.05,
					key='claude_temperature_chat',
				)
			
			with p_row2[ 0 ]:
				claude_max_tokens = st.number_input(
					'Max Tokens',
					min_value=1,
					max_value=65536,
					value=2048,
					step=1,
					key='claude_max_tokens_chat',
				)
			
			with p_row2[ 1 ]:
				claude_top_p = st.slider(
					'Top-P',
					min_value=0.0,
					max_value=1.0,
					value=1.0,
					step=0.01,
					key='claude_top_p_chat',
				)
			
			with p_row3[ 0 ]:
				claude_top_k = st.number_input(
					'Top-k',
					min_value=0,
					max_value=500,
					value=0,
					step=1,
					key='claude_top_k_chat',
				)
			
			with p_row3[ 1 ]:
				claude_thinking = st.checkbox(
					'Reasoning',
					value=False,
					key='claude_thinking_chat',
					help='Anthropic exposes this as extended thinking with a token budget.',
				)
			
			with p_row4[ 0 ]:
				claude_web_search = st.checkbox(
					'Web Search',
					value=False,
					key='claude_web_search_chat',
				)
			
			with p_row4[ 1 ]:
				claude_stop = st.text_area(
					'Stop Sequences (one per line)',
					value='',
					height=80,
					key='claude_stop_chat',
				)
			
			with p_row5[ 0 ]:
				if claude_thinking:
					claude_thinking_budget = st.number_input(
						'Thinking Budget',
						min_value=1024,
						max_value=64000,
						value=1024,
						step=1024,
						key='claude_thinking_budget_chat',
					)
				else:
					claude_thinking_budget = None
			
			with p_row5[ 1 ]:
				claude_system = st.text_area(
					'System',
					value='',
					height=100,
					key='claude_system_chat',
				)
			
			if claude_web_search:
				claude_domains = st.text_area(
					'Allowed Search Domains (one per line or comma-separated)',
					value='',
					height=90,
					key='claude_domains_chat',
					help='Examples: docs.anthropic.com, arxiv.org, github.com',
				)
				
				claude_blocked_domains = st.text_area(
					'Blocked Search Domains (one per line or comma-separated)',
					value='',
					height=90,
					key='claude_blocked_domains_chat',
					help='Optional denylist for domains you do not want Claude to use.',
				)
			else:
				claude_domains = ''
				claude_blocked_domains = ''
			
			if claude_thinking:
				st.caption(
					'When Reasoning is enabled, temperature and top-k are omitted to match Anthropic compatibility rules.'
				)
			
			btn_row = st.columns( 2 )
			with btn_row[ 0 ]:
				claude_submit = st.button( 'Submit', key='claude_submit_chat' )
			with btn_row[ 1 ]:
				claude_clear = st.button( 'Clear', key='claude_clear_chat' )
		
		with col_right:
			claude_output = st.empty( )
		
		if claude_clear:
			st.session_state.update(
				{
						'claude_prompt_chat': '',
						'claude_stop_chat': '',
						'claude_system_chat': '',
						'claude_domains_chat': '',
						'claude_blocked_domains_chat': '',
						'claude_thinking_chat': False,
						'claude_web_search_chat': False,
				}
			)
			st.rerun( )
		
		if claude_submit:
			try:
				if not str( claude_prompt ).strip( ):
					raise ValueError( 'Prompt cannot be empty.' )
				
				claude_domains_list = [ ]
				if claude_domains and str( claude_domains ).strip( ):
					_domain_entries = re.split( r'[\n,;]+', str( claude_domains ) )
					for _entry in _domain_entries:
						_value = str( _entry ).strip( ).lower( )
						if not _value:
							continue
						
						if not _value.startswith( 'http://' ) and not _value.startswith( 'https://' ):
							_value = f'https://{_value}'
						
						_parsed = urlparse( _value )
						_domain = (_parsed.netloc or _parsed.path or '').strip( ).lower( )
						_domain = re.sub( r':\d+$', '', _domain )
						_domain = _domain.lstrip( '.' )
						
						if _domain.startswith( 'www.' ):
							_domain = _domain[ 4: ]
						
						if _domain and _domain not in claude_domains_list:
							claude_domains_list.append( _domain )
				
				claude_blocked_domains_list = [ ]
				if claude_blocked_domains and str( claude_blocked_domains ).strip( ):
					_block_entries = re.split( r'[\n,;]+', str( claude_blocked_domains ) )
					for _entry in _block_entries:
						_value = str( _entry ).strip( ).lower( )
						if not _value:
							continue
						
						if not _value.startswith( 'http://' ) and not _value.startswith( 'https://' ):
							_value = f'https://{_value}'
						
						_parsed = urlparse( _value )
						_domain = (_parsed.netloc or _parsed.path or '').strip( ).lower( )
						_domain = re.sub( r':\d+$', '', _domain )
						_domain = _domain.lstrip( '.' )
						
						if _domain.startswith( 'www.' ):
							_domain = _domain[ 4: ]
						
						if _domain and _domain not in claude_blocked_domains_list:
							claude_blocked_domains_list.append( _domain )
				
				stop_lines = [ s.strip( ) for s in (claude_stop or '').splitlines( ) if s.strip( ) ]
				
				fetcher = Claude( )
				params = \
					{
							'model': claude_model,
							'temperature': float( claude_temperature ),
							'max_tokens': int( claude_max_tokens ),
							'top_p': float( claude_top_p ),
							'top_k': int( claude_top_k ) if int( claude_top_k ) > 0 else None,
							'stop_sequences': stop_lines if stop_lines else None,
							'system': claude_system if claude_system.strip( ) else None,
							'thinking': bool( claude_thinking ),
							'thinking_budget': int( claude_thinking_budget ) if claude_thinking and claude_thinking_budget else None,
							'web_search': bool( claude_web_search ),
							'search_domains': claude_domains_list if claude_domains_list else None,
							'blocked_domains': claude_blocked_domains_list if claude_blocked_domains_list else None,
					}
				
				if claude_thinking:
					params.pop( 'temperature', None )
					params.pop( 'top_k', None )
				
				params = { k: v for k, v in params.items( ) if v is not None }
				result = _invoke_provider( fetcher, claude_prompt, params )
				_render_output( claude_output, result )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- GEMINI
	with st.expander( label='Gemini', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			gemini_prompt = st.text_area(
				'Prompt',
				value='',
				height=160,
				key='gemini_prompt_chat',
			)
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			p_row4 = st.columns( 2 )
			p_row5 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				_gemini_models = (
						cfg.GEMINI_MODELS
						if hasattr( cfg, 'GEMINI_MODELS' ) and cfg.GEMINI_MODELS
						else [ 'gemini-2.5-flash', 'gemini-2.5-flash-lite' ]
				)
				gemini_model = _model_selector(
					key_prefix='gemini',
					label='Model',
					options=_gemini_models,
					default_model=('gemini-2.5-flash' if 'gemini-2.5-flash' in _gemini_models else
					               _gemini_models[ 0 ]),
				)
			
			with p_row1[ 1 ]:
				gemini_temperature = st.slider(
					'Temperature',
					min_value=0.0,
					max_value=2.0,
					value=0.7,
					step=0.05,
					key='gemini_temperature_chat',
				)
			
			with p_row2[ 0 ]:
				gemini_max_tokens = st.number_input(
					'Max Tokens',
					min_value=1,
					max_value=32768,
					value=2048,
					step=1,
					key='gemini_max_tokens_chat',
				)
			
			with p_row2[ 1 ]:
				gemini_top_p = st.slider(
					'Top-p',
					min_value=0.0,
					max_value=1.0,
					value=1.0,
					step=0.01,
					key='gemini_top_p_chat',
				)
			
			with p_row3[ 0 ]:
				gemini_top_k = st.number_input(
					'Top-k',
					min_value=0,
					max_value=500,
					value=0,
					step=1,
					key='gemini_top_k_chat',
				)
			
			with p_row3[ 1 ]:
				gemini_candidate_count = st.number_input(
					'Candidates',
					min_value=1,
					max_value=8,
					value=1,
					step=1,
					key='gemini_candidate_count_chat',
				)
			
			with p_row4[ 0 ]:
				gemini_seed = st.number_input(
					'Seed',
					min_value=0,
					max_value=2_147_483_647,
					value=0,
					step=1,
					key='gemini_seed_chat',
				)
			
			with p_row4[ 1 ]:
				gemini_json_mode = st.checkbox(
					'JSON Mode',
					value=False,
					key='gemini_json_mode_chat',
				)
			
			with p_row5[ 0 ]:
				gemini_grounding = st.checkbox(
					'Grounding',
					value=False,
					key='gemini_grounding_chat',
					help='Enable Google Search grounding for supported Gemini models.',
				)
			
			with p_row5[ 1 ]:
				gemini_reasoning = st.checkbox(
					'Reasoning',
					value=False,
					key='gemini_reasoning_chat',
					help='Uses Gemini thinking configuration where supported.',
				)
			
			_gemini_supports_reasoning = str( gemini_model ).strip( ).lower( ).startswith( 'gemini-3' )
			
			if _gemini_supports_reasoning and gemini_reasoning:
				r_row = st.columns( 2 )
				
				with r_row[ 0 ]:
					gemini_thinking_level = st.selectbox(
						'Thinking Level',
						options=[ 'minimal', 'low', 'medium', 'high' ],
						index=1,
						key='gemini_thinking_level_chat',
					)
				
				with r_row[ 1 ]:
					gemini_include_thoughts = st.checkbox(
						'Include Thoughts',
						value=False,
						key='gemini_include_thoughts_chat',
					)
			else:
				gemini_thinking_level = None
				gemini_include_thoughts = False
			
			gemini_stop = st.text_area(
				'Stop Sequences (one per line)',
				value='',
				height=80,
				key='gemini_stop_chat',
			)
			
			gemini_system = st.text_area(
				'System',
				value='',
				height=110,
				key='gemini_system_chat',
			)
			
			if gemini_grounding:
				gemini_domains = st.text_area(
					'Preferred Search Domains (one per line or comma-separated)',
					value='',
					height=90,
					key='gemini_domains_chat',
					help='Used as preferred source guidance for grounded Gemini responses.',
				)
			else:
				gemini_domains = ''
			
			if gemini_reasoning and not _gemini_supports_reasoning:
				st.caption(
					'Reasoning controls are only sent for Gemini 3 model names. Older Gemini models run without thinking_config.'
				)
			
			btn_row = st.columns( 2 )
			with btn_row[ 0 ]:
				gemini_submit = st.button( 'Submit', key='gemini_submit_chat' )
			with btn_row[ 1 ]:
				gemini_clear = st.button( 'Clear', key='gemini_clear_chat' )
		
		with col_right:
			gemini_output = st.empty( )
		
		if gemini_clear:
			st.session_state.update(
				{
						'gemini_prompt_chat': '',
						'gemini_system_chat': '',
						'gemini_domains_chat': '',
						'gemini_stop_chat': '',
						'gemini_json_mode_chat': False,
						'gemini_grounding_chat': False,
						'gemini_reasoning_chat': False,
						'gemini_include_thoughts_chat': False,
						'gemini_seed_chat': 0,
				}
			)
			st.rerun( )
		
		if gemini_submit:
			try:
				if not str( gemini_prompt ).strip( ):
					raise ValueError( 'Prompt cannot be empty.' )
				
				gemini_domains_list = [ ]
				if gemini_domains and str( gemini_domains ).strip( ):
					_domain_entries = re.split( r'[\n,;]+', str( gemini_domains ) )
					for _entry in _domain_entries:
						_value = str( _entry ).strip( ).lower( )
						if not _value:
							continue
						
						if not _value.startswith( 'http://' ) and not _value.startswith( 'https://' ):
							_value = f'https://{_value}'
						
						_parsed = urlparse( _value )
						_domain = (_parsed.netloc or _parsed.path or '').strip( ).lower( )
						_domain = re.sub( r':\d+$', '', _domain )
						_domain = _domain.lstrip( '.' )
						
						if _domain.startswith( 'www.' ):
							_domain = _domain[ 4: ]
						
						if _domain and _domain not in gemini_domains_list:
							gemini_domains_list.append( _domain )
				
				stop_lines = [ s.strip( ) for s in (gemini_stop or '').splitlines( ) if s.strip( ) ]
				
				fetcher = Gemini( )
				params = \
					{
							'model': gemini_model,
							'temperature': float( gemini_temperature ),
							'max_tokens': int( gemini_max_tokens ),
							'top_p': float( gemini_top_p ),
							'top_k': int( gemini_top_k ) if int( gemini_top_k ) > 0 else None,
							'candidate_count': int( gemini_candidate_count ),
							'seed': int( gemini_seed ) if int( gemini_seed ) > 0 else None,
							'system': gemini_system if gemini_system.strip( ) else None,
							'response_format': ('json' if gemini_json_mode else None),
							'stop_sequences': stop_lines if stop_lines else None,
							'grounding': bool( gemini_grounding ),
							'search_domains': gemini_domains_list if gemini_domains_list else None,
							'reasoning': bool( gemini_reasoning and _gemini_supports_reasoning ),
							'thinking_level': gemini_thinking_level if _gemini_supports_reasoning and gemini_reasoning else None,
							'include_thoughts': bool( gemini_include_thoughts ) if _gemini_supports_reasoning and gemini_reasoning else False,
					}
				
				params = { k: v for k, v in params.items( ) if v is not None }
				result = _invoke_provider( fetcher, gemini_prompt, params )
				_render_output( gemini_output, result )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- Mistral
	with st.expander( label='Mistral', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			mistral_prompt = st.text_area(
				'Prompt',
				value='',
				height=40,
				key='mistral_prompt_chat'
			)
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				mistral_model = _model_selector(
					key_prefix='mistral',
					label='Model',
					options=[
							'mistral-large-latest',
							'mistral-medium-latest',
							'mistral-small-latest',
							'open-mistral-7b',
							'Custom...',
					],
					default_model='mistral-large-latest',
				)
			
			with p_row1[ 1 ]:
				mistral_temperature = st.slider(
					'Temperature',
					min_value=0.0,
					max_value=2.0,
					value=0.7,
					step=0.05,
					key='mistral_temperature_chat',
				)
			
			with p_row2[ 0 ]:
				mistral_max_tokens = st.number_input(
					'Max Tokens',
					min_value=1,
					max_value=32768,
					value=1024,
					step=1,
					key='mistral_max_tokens_chat',
				)
			
			with p_row2[ 1 ]:
				mistral_top_p = st.slider(
					'Top-p',
					min_value=0.0,
					max_value=1.0,
					value=1.0,
					step=0.01,
					key='mistral_top_p_chat',
				)
			
			with p_row3[ 0 ]:
				mistral_seed = st.number_input(
					'Seed',
					min_value=0,
					max_value=2_147_483_647,
					value=0,
					step=1,
					key='mistral_seed_chat',
				)
			
			with p_row3[ 1 ]:
				mistral_safe_mode = st.checkbox(
					'Safe Mode',
					value=False,
					key='mistral_safe_mode_chat'
				)
			
			mistral_system = st.text_area(
				'System',
				value='',
				height=100,
				key='mistral_system_chat',
			)
			
			btn_row = st.columns( 2 )
			with btn_row[ 0 ]:
				mistral_submit = st.button( 'Submit', key='mistral_submit_chat' )
			with btn_row[ 1 ]:
				mistral_clear = st.button( 'Clear', key='mistral_clear_chat' )
		
		with col_right:
			mistral_output = st.empty( )
		
		if mistral_clear:
			st.session_state.update( {
					'mistral_prompt_chat': "",
					'mistral_system_chat': "",
			} )
			st.rerun( )
		
		if mistral_submit:
			try:
				fetcher = Mistral( )
				params = {
						'model': mistral_model,
						'temperature': float( mistral_temperature ),
						'max_tokens': int( mistral_max_tokens ),
						'top_p': float( mistral_top_p ),
						'seed': int( mistral_seed ) if int( mistral_seed ) > 0 else None,
						'safe_mode': bool( mistral_safe_mode ),
						'system': mistral_system if mistral_system.strip( ) else None,
				}
				
				params = { k: v for k, v in params.items( ) if v is not None }
				
				result = _invoke_provider( fetcher, mistral_prompt, params )
				_render_output( mistral_output, result )
			
			except Exception as exc:
				st.error( str( exc ) )

# ==============================================================================
# DATA MANAGEMENT MODE
# ==============================================================================
elif mode == 'Management':
	st.subheader( f'🏛️ Data Management', help=cfg.DATA_MANAGEMENT )
	st.divider( )
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		tabs = st.tabs( [ '📥 Import', '🗂 Browse', '💉 CRUD', '📊 Explore', '🔎 Filter',
		                  '🧮 Aggregate', '📈 Visualize', '⚙ Admin', '🧠 SQL' ] )
		
		tables = list_tables( )
		if not tables:
			st.info( 'No tables available.' )
		else:
			table = st.selectbox( 'Table', tables )
			df_full = read_table( table )
		
		# ------------------------------------------------------------------------------
		# UPLOAD TAB
		# ------------------------------------------------------------------------------
		with tabs[ 0 ]:
			uploaded_file = st.file_uploader( 'Upload Excel File', type=[ 'xlsx' ] )
			overwrite = st.checkbox( 'Overwrite existing tables', value=True )
			if uploaded_file:
				try:
					sheets = pd.read_excel( uploaded_file, sheet_name=None )
					with create_connection( ) as conn:
						conn.execute( 'BEGIN' )
						for sheet_name, df in sheets.items( ):
							table_name = create_identifier( sheet_name )
							if overwrite:
								conn.execute( f'DROP TABLE IF EXISTS "{table_name}"' )
							
							# --- Create Table ---
							columns = [ ]
							df.columns = [ create_identifier( c ) for c in df.columns ]
							for col in df.columns:
								sql_type = get_sqlite_type( df[ col ].dtype )
								columns.append( f'"{col}" {sql_type}' )
							
							create_stmt = (
									f'CREATE TABLE "{table_name}" '
									f'({", ".join( columns )});'
							)
							
							conn.execute( create_stmt )
							
							# --- Insert Data ---
							placeholders = ", ".join( [ "?" ] * len( df.columns ) )
							insert_stmt = (
									f'INSERT INTO "{table_name}" '
									f'VALUES ({placeholders});'
							)
							
							conn.executemany( insert_stmt,
								df.where( pd.notnull( df ), None ).values.tolist( ) )
						
						conn.commit( )
					
					st.success( 'Import completed successfully (transaction committed).' )
					st.rerun( )
				
				except Exception as e:
					try:
						conn.rollback( )
					except:
						pass
					st.error( f'Import failed — transaction rolled back.\n\n{e}' )
		
		# ------------------------------------------------------------------------------
		# BROWSE TAB
		# ------------------------------------------------------------------------------
		with tabs[ 1 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='table_name' )
				df = read_table( table )
				render_table( df )
			else:
				st.info( 'No tables available.' )
		
		# ------------------------------------------------------------------------------
		# CRUD
		# ------------------------------------------------------------------------------
		with tabs[ 2 ]:
			tables = list_tables( )
			if not tables:
				st.info( 'No tables available.' )
			else:
				table = st.selectbox( 'Select Table', tables, key='crud_table' )
				df = read_table( table )
				schema = create_schema( table )
				
				# Build type map
				type_map = { col[ 1 ]: col[ 2 ].upper( ) for col in schema if col[ 1 ] != 'rowid' }
				
				# ------------------------------------------------------------------
				# INSERT
				# ------------------------------------------------------------------
				st.subheader( 'Insert Row' )
				insert_data = { }
				for column, col_type in type_map.items( ):
					if 'INT' in col_type:
						insert_data[ column ] = st.number_input( column, step=1, key=f'ins_{column}' )
					
					elif 'REAL' in col_type:
						insert_data[ column ] = st.number_input( column, format='%.6f', key=f'ins_{column}' )
					
					elif 'BOOL' in col_type:
						insert_data[ column ] = 1 if st.checkbox( column, key=f'ins_{column}' ) else 0
					
					else:
						insert_data[ column ] = st.text_input( column, key=f'ins_{column}' )
				
				if st.button( 'Insert Row' ):
					cols = list( insert_data.keys( ) )
					placeholders = ', '.join( [ '?' ] * len( cols ) )
					stmt = f'INSERT INTO "{table}" ({", ".join( cols )}) VALUES ({placeholders});'
					
					with create_connection( ) as conn:
						conn.execute( stmt, list( insert_data.values( ) ) )
						conn.commit( )
					
					st.success( 'Row inserted.' )
					st.rerun( )
				
				# ------------------------------------------------------------------
				# UPDATE
				# ------------------------------------------------------------------
				st.subheader( 'Update Row' )
				rowid = st.number_input( 'Row ID', min_value=1, step=1 )
				update_data = { }
				for column, col_type in type_map.items( ):
					if 'INT' in col_type:
						val = st.number_input( column, step=1, key=f'upd_{column}' )
						update_data[ column ] = val
					
					elif 'REAL' in col_type:
						val = st.number_input( column, format='%.6f', key=f'upd_{column}' )
						update_data[ column ] = val
					
					elif 'BOOL' in col_type:
						val = 1 if st.checkbox( column, key=f'upd_{column}' ) else 0
						update_data[ column ] = val
					
					else:
						val = st.text_input( column, key=f"upd_{column}" )
						update_data[ column ] = val
				
				if st.button( 'Update Row' ):
					set_clause = ', '.join( [ f'{c}=?' for c in update_data ] )
					stmt = f'UPDATE {table} SET {set_clause} WHERE rowid=?;'
					
					with create_connection( ) as conn:
						conn.execute( stmt, list( update_data.values( ) ) + [ rowid ] )
						conn.commit( )
					
					st.success( 'Row updated.' )
					st.rerun( )
				
				# ------------------------------------------------------------------
				# DELETE
				# ------------------------------------------------------------------
				st.subheader( 'Delete Row' )
				delete_id = st.number_input( 'Row ID to Delete', min_value=1, step=1 )
				if st.button( 'Delete Row' ):
					with create_connection( ) as conn:
						conn.execute( f'DELETE FROM {table} WHERE rowid=?;', (delete_id,) )
						conn.commit( )
					
					st.success( 'Row deleted.' )
					st.rerun( )
		
		# ------------------------------------------------------------------------------
		# EXPLORE
		# ------------------------------------------------------------------------------
		with tabs[ 3 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='explore_table' )
				page_size = st.slider( 'Rows per page', 10, 500, 50 )
				page = st.number_input( 'Page', min_value=1, step=1 )
				offset = (page - 1) * page_size
				df_page = read_table( table, page_size, offset )
				render_table( df_page )
		
		# ------------------------------------------------------------------------------
		# FILTER
		# ------------------------------------------------------------------------------
		with tabs[ 4 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='filter_table' )
				df = read_table( table )
				column = st.selectbox( 'Column', df.columns )
				value = st.text_input( 'Contains' )
				if value:
					df = df[ df[ column ].astype( str ).str.contains( value ) ]
				
				render_table( df )
		
		# ------------------------------------------------------------------------------
		# AGGREGATE
		# ------------------------------------------------------------------------------
		with tabs[ 5 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='agg_table' )
				df = read_table( table )
				numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
				if numeric_cols:
					col = st.selectbox( 'Column', numeric_cols )
					agg = st.selectbox( 'Function', [ 'SUM', 'AVG', 'COUNT' ] )
					if agg == 'SUM':
						st.metric( 'Result', df[ col ].sum( ) )
					elif agg == 'AVG':
						st.metric( 'Result', df[ col ].mean( ) )
					elif agg == 'COUNT':
						st.metric( 'Result', df[ col ].count( ) )
		
		# ------------------------------------------------------------------------------
		# VISUALIZE
		# ------------------------------------------------------------------------------
		with tabs[ 6 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='viz_table' )
				df = read_table( table )
				create_visualization( df )
		
		# ------------------------------------------------------------------------------
		# ADMIN
		# ------------------------------------------------------------------------------
		with tabs[ 7 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='admin_table' )
			
			st.divider( )
			
			st.subheader( 'Data Profiling' )
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='profile_table' )
				if st.button( 'Generate Profile' ):
					profile_df = create_profile_table( table )
					render_table( profile_df )
			
			st.subheader( 'Drop Table' )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table to Drop', tables, key='admin_drop_table' )
				
				# Initialize confirmation state
				if 'dm_confirm_drop' not in st.session_state:
					st.session_state.dm_confirm_drop = False
				
				# Step 1: Initial Drop click
				if st.button( 'Drop Table', key='admin_drop_button' ):
					st.session_state.dm_confirm_drop = True
				
				# Step 2: Confirmation UI
				if st.session_state.dm_confirm_drop:
					st.warning( f'You are about to permanently delete table {table}. '
					            'This action cannot be undone.' )
					
					col1, col2 = st.columns( 2 )
					
					if col1.button( 'Confirm Drop', key='admin_confirm_drop' ):
						try:
							drop_table( table )
							st.success( f'Table {table} dropped successfully.' )
						except Exception as e:
							st.error( f'Drop failed: {e}' )
						
						st.session_state.dm_confirm_drop = False
						st.rerun( )
					
					if col2.button( 'Cancel', key='admin_cancel_drop' ):
						st.session_state.dm_confirm_drop = False
						st.rerun( )
				
				df = read_table( table )
				col = st.selectbox( 'Create Index On', df.columns )
				
				if st.button( 'Create Index' ):
					create_index( table, col )
					st.success( 'Index created.' )
			
			st.divider( )
			
			st.subheader( 'Create Custom Table' )
			new_table_name = st.text_input( 'Table Name' )
			column_count = st.number_input( 'Number of Columns', min_value=1, max_value=20, value=1 )
			columns = [ ]
			for i in range( column_count ):
				st.markdown( f'### Column {i + 1}' )
				col_name = st.text_input( 'Column Name', key=f'col_name_{i}' )
				col_type = st.selectbox( 'Column Type', [ 'INTEGER', 'REAL', 'TEXT' ],
					key=f'col_type_{i}' )
				
				not_null = st.checkbox( 'NOT NULL', key=f'not_null_{i}' )
				primary_key = st.checkbox( 'PRIMARY KEY', key=f'pk_{i}' )
				auto_inc = st.checkbox( 'AUTOINCREMENT (INTEGER only)', key=f'ai_{i}' )
				
				columns.append( {
						'name': col_name,
						'type': col_type,
						'not_null': not_null,
						'primary_key': primary_key,
						'auto_increment': auto_inc } )
			
			if st.button( 'Create Table' ):
				try:
					create_custom_table( new_table_name, columns )
					st.success( 'Table created successfully.' )
					st.rerun( )
				
				except Exception as e:
					st.error( f'Error: {e}' )
			
			st.divider( )
			st.subheader( 'Schema Viewer' )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='schema_view_table' )
				
				# Column schema
				schema = create_schema( table )
				schema_df = DataFrame(
					schema,
					columns=[ 'cid', 'name', 'type', 'notnull', 'default', 'pk' ] )
				
				st.markdown( "### Columns" )
				st.data_editor(
					make_display_safe( schema_df ),
					hide_index=True,
					use_container_width=True,
					disabled=True )
				
				# Row count
				with create_connection( ) as conn:
					count = conn.execute(
						f'SELECT COUNT(*) FROM "{table}"'
					).fetchone( )[ 0 ]
				
				st.metric( "Row Count", f"{count:,}" )
				
				# Indexes
				indexes = get_indexes( table )
				if indexes:
					idx_df = DataFrame(
						indexes,
						columns=[ 'seq', 'name', 'unique', 'origin', 'partial' ]
					)
					st.markdown( "### Indexes" )
					st.data_editor(
						make_display_safe( idx_df ),
						hide_index=True,
						use_container_width=True,
						disabled=True )
				else:
					st.info( "No indexes defined." )
			
			st.divider( )
			st.subheader( "ALTER TABLE Operations" )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='alter_table_select' )
				operation = st.selectbox( 'Operation',
					[ 'Add Column', 'Rename Column', 'Rename Table', 'Drop Column' ] )
				
				if operation == 'Add Column':
					new_col = st.text_input( 'Column Name' )
					col_type = st.selectbox( 'Column Type', [ 'INTEGER', 'REAL', 'TEXT' ] )
					
					if st.button( 'Add Column' ):
						add_column( table, new_col, col_type )
						st.success( 'Column added.' )
						st.rerun( )
				
				elif operation == 'Rename Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					old_col = st.selectbox( 'Column to Rename', col_names )
					new_col = st.text_input( 'New Column Name' )
					
					if st.button( 'Rename Column' ):
						rename_column( table, old_col, new_col )
						st.success( 'Column renamed.' )
						st.rerun( )
				
				elif operation == 'Rename Table':
					new_name = st.text_input( 'New Table Name' )
					
					if st.button( 'Rename Table' ):
						rename_table( table, new_name )
						st.success( 'Table renamed.' )
						st.rerun( )
				
				elif operation == 'Drop Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					drop_col = st.selectbox( 'Column to Drop', col_names )
					
					if st.button( 'Drop Column' ):
						drop_column( table, drop_col )
						st.success( 'Column dropped.' )
						st.rerun( )
		
		# ------------------------------------------------------------------------------
		# SQL
		# ------------------------------------------------------------------------------
		with tabs[ 8 ]:
			st.subheader( 'SQL Console' )
			query = st.text_area( 'Enter SQL Query' )
			if st.button( 'Run Query' ):
				if not is_safe_query( query ):
					st.error( 'Query blocked: Only read-only SELECT statements are allowed.' )
				else:
					try:
						start_time = time.perf_counter( )
						with create_connection( ) as conn:
							result = pd.read_sql_query( query, conn )
						
						end_time = time.perf_counter( )
						elapsed = end_time - start_time
						
						# ----------------------------------------------------------
						# Display Results
						# ----------------------------------------------------------
						st.dataframe( result, use_container_width=True )
						row_count = len( result )
						
						# ----------------------------------------------------------
						# Execution Metrics
						# ----------------------------------------------------------
						col1, col2 = st.columns( 2 )
						col1.metric( 'Rows Returned', f'{row_count:,}' )
						col2.metric( 'Execution Time (seconds)', f'{elapsed:.6f}' )
						
						# Optional slow query warning
						if elapsed > 2.0:
							st.warning( 'Slow query detected (> 2 seconds). Consider indexing.' )
						
						# ----------------------------------------------------------
						# Download
						# ----------------------------------------------------------
						if not result.empty:
							csv = result.to_csv( index=False ).encode( 'utf-8' )
							st.download_button( 'Download CSV', csv,
								'query_results.csv', 'text/csv' )
					
					except Exception as e:
						st.error( f'Execution failed: {e}' )
