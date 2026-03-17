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
import inspect
from astroquery.simbad import Simbad
import base64

from bs4 import BeautifulSoup

import config as cfg
from collections import deque
import datetime as dt
import html as html_lib
import json
import numpy as np
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from langchain_core.documents import Document
from loaders import (PdfLoader, WordLoader, ExcelLoader, MarkdownLoader,
                     HtmlLoader, TextLoader, CsvLoader, OutlookLoader,
                     WebLoader, ArXivLoader, WikiLoader, YouTubeLoader,
                     RecursiveCharacterTextSplitter, PowerPointLoader)

from generators import Chat, Claude, Grok, Mistral, Gemini
from fetchers import (
	Wikipedia, TheNews, SatelliteCenter, WebFetcher,
	GoogleWeather, Grokipedia, OpenWeather, NavalObservatory,
	GoogleSearch, GoogleDrive, GoogleMaps, NearbyObjects, OpenScience,
	EarthObservatory, SpaceWeather, AstroCatalog, AstroQuery, StarMap,
	GovData, Congress, InternetArchive, StarChart, HistoricalWeather, GoogleGeocoding )

import plotly.graph_objects as px
import pandas as pd
from pandas import DataFrame
import streamlit as st
import scrapers
import sqlite3
from sqlite3 import Connection
from urllib.parse import urljoin, urlparse

# =====================================================================
# SESSION STATE
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

if 'loader_documents' not in st.session_state:
	st.session_state[ 'loader_documents' ] = [ ]

if 'loader_path' not in st.session_state:
	st.session_state[ 'loader_path' ] = ''
	
if 'loader_text' not in st.session_state:
	st.session_state[ 'loader_text' ] = ''
	
if 'loader_files' not in st.session_state:
	st.session_state[ 'loader_files' ] = ''

# ------------- SCRAPPER VARIABLES --------------

if 'target_url' not in st.session_state:
	st.session_state[ 'target_url' ] = ''
	
	
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
	st.text( 'Configuration' )
	st.divider( )
	
	# ---------------------------
	# API Keys
	# ---------------------------
	with st.expander( 'API Keys', expanded=False ):
		for attr in dir( cfg ):
			if attr.endswith( '_API_KEY' ) or attr.endswith( '_TOKEN' ):
				current = getattr( cfg, attr, "" ) or ""
				val = st.text_input( attr, value=current, type='password' )
				if val:
					os.environ[ attr ] = val
	
	st.divider( )
	st.text( 'Mode' )
	mode = st.sidebar.radio( label='Mode', options=modes, label_visibility='collapsed' )
	if mode:
		st.session_state[ 'mode' ] = mode
	else:
		st.session_state[ 'mode' ] = 'Loaders'
		
# =============================================================================
# DOCUMENT LOADING MODE
# =============================================================================
if mode == modes[ 0 ]:
	st.subheader( f'📤  {modes[ 0 ]}' )
	st.divider( )
	
	if 'loader_clear_request' not in st.session_state:
		st.session_state[ 'loader_clear_request' ] = False
	
	if 'loader_uploader_nonce' not in st.session_state:
		st.session_state[ 'loader_uploader_nonce' ] = 0
	
	if 'loader_selected_type' not in st.session_state:
		st.session_state[ 'loader_selected_type' ] = 'Auto'
	
	if 'remote_loader_results' not in st.session_state:
		st.session_state[ 'remote_loader_results' ] = { }
	
	if st.session_state.get( 'loader_clear_request', False ):
		st.session_state[ 'loader_results' ] = { }
		st.session_state[ 'loader_documents' ] = [ ]
		st.session_state[ 'loader_path' ] = ''
		st.session_state[ 'loader_selected_type' ] = 'Auto'
		st.session_state[ 'remote_loader_results' ] = { }
		st.session_state[ 'loader_clear_request' ] = False
	
	def _clear_loader_state( ) -> None:
		st.session_state[ 'loader_clear_request' ] = True
		st.session_state[ 'loader_uploader_nonce' ] += 1
	
	def _get_local_loader_from_selection(
			file_path: str,
			selected_type: str ) -> tuple[ Any | None, str ]:
		path_obj = Path( file_path )
		suffix = path_obj.suffix.lower( ).lstrip( '.' )
		effective_type = selected_type
		
		if selected_type == 'Auto':
			if suffix == 'pdf':
				effective_type = 'PDF'
			elif suffix == 'docx':
				effective_type = 'Word'
			elif suffix in ('xlsx', 'xls'):
				effective_type = 'Excel'
			elif suffix == 'md':
				effective_type = 'Markdown'
			elif suffix == 'pptx':
				effective_type = 'PowerPoint'
			elif suffix in ('html', 'htm'):
				effective_type = 'HTML'
			elif suffix in ('txt', 'text', 'log'):
				effective_type = 'Text'
			elif suffix == 'csv':
				effective_type = 'CSV'
			elif suffix == 'msg':
				effective_type = 'Outlook'
			else:
				effective_type = 'Unsupported'
		
		if effective_type == 'PDF':
			return PdfLoader( ), effective_type
		elif effective_type == 'Word':
			return WordLoader( ), effective_type
		elif effective_type == 'Excel':
			return ExcelLoader( ), effective_type
		elif effective_type == 'Markdown':
			return MarkdownLoader( ), effective_type
		elif effective_type == 'PowerPoint':
			return PowerPointLoader( ), effective_type
		elif effective_type == 'HTML':
			return HtmlLoader( ), effective_type
		elif effective_type == 'Text':
			return TextLoader( ), effective_type
		elif effective_type == 'CSV':
			return CsvLoader( ), effective_type
		elif effective_type == 'Outlook':
			return OutlookLoader( ), effective_type
		else:
			return None, effective_type
	
	def _load_local_from_path(
			file_path: str,
			selected_type: str ) -> tuple[ list[ Document ] | None, dict[ str, Any ] ]:
		path_obj = Path( file_path )
		suffix = path_obj.suffix.lower( ).lstrip( '.' )
		out: dict[ str, Any ] = {
				'file': str( path_obj ),
				'type': suffix,
				'selected_loader': selected_type,
		}
		docs: list[ Document ] | None = None
		
		ld, effective_type = _get_local_loader_from_selection(
			file_path=str( path_obj ),
			selected_type=selected_type )
		
		out[ 'resolved_loader' ] = effective_type
		
		if ld is None:
			out[ 'skipped' ] = 'No compatible local file loader is available for this file type.'
			return docs, out
		
		if effective_type == 'CSV':
			docs = ld.load(
				path=str( path_obj ),
				columns=None,
				csv_args=None )
		else:
			docs = ld.load( str( path_obj ) )
		
		return docs, out
	
	top_left, top_right = st.columns( [ 1.35, 1.0 ], border=False )
	
	with top_left:
		with st.expander( 'Local Loaders', expanded=True ):
			col_local, col_actions = st.columns( [ 1.15, 0.85 ], border=True )
			
			with col_local:
				selected_loader_type = st.selectbox(
					label='Loader Type',
					options=[
							'Auto',
							'PDF',
							'Word',
							'Excel',
							'Markdown',
							'PowerPoint',
							'HTML',
							'Text',
							'CSV',
							'Outlook',
					],
					key='loader_selected_type' )
				
				uploaded_files = st.file_uploader(
					'Choose file(s)',
					type=[
							'pdf',
							'docx',
							'xlsx',
							'xls',
							'pptx',
							'md',
							'html',
							'htm',
							'txt',
							'text',
							'log',
							'csv',
							'msg',
					],
					accept_multiple_files=True,
					key=f'loader_uploaded_files_{st.session_state[ "loader_uploader_nonce" ]}' )
				
				st.caption(
					'Auto detects the local loader from the file extension. '
					'Use a specific loader only when all selected files share the same format.'
				)
			
			with col_actions:
				loader_text = st.text_area(
					'Enter one local file path per line',
					height=120,
					key='loader_path' )
				
				la1, la2 = st.columns( 2 )
				
				with la1:
					do_load = st.button(
						'Load Local Files',
						key='loader_load_btn',
						width='stretch' )
				
				with la2:
					st.button(
						'Clear',
						key='loader_clear_btn',
						on_click=_clear_loader_state,
						width='stretch' )
	
	with top_right:
		with st.expander( 'Remote Loaders', expanded=True ):
			remote_type = st.selectbox(
				label='Remote Loader',
				options=[ 'Web', 'ArXiv', 'Wikipedia', 'YouTube' ],
				key='remote_loader_type' )
			
			if remote_type == 'Web':
				web_urls_text = st.text_area(
					'Enter one URL per line',
					height=120,
					key='remote_web_urls' )
				
				if st.button(
						'Load Web Pages',
						key='remote_web_load_btn',
						width='stretch' ):
					try:
						urls = [
								u.strip( )
								for u in (web_urls_text or '').splitlines( )
								if u.strip( )
						]
						
						loader = WebLoader( )
						docs = loader.load( urls=urls )
						
						st.session_state[ 'remote_loader_results' ] = {
								'type': 'Web',
								'input': urls,
								'documents_loaded': len( docs ) if isinstance( docs, list ) else 0,
								'documents': docs or [ ],
						}
						st.rerun( )
					except Exception as exc:
						st.error( str( exc ) )
			
			elif remote_type == 'ArXiv':
				arxiv_query = st.text_input(
					'Enter an ArXiv query',
					key='remote_arxiv_query' )
				
				if st.button(
						'Load ArXiv Results',
						key='remote_arxiv_load_btn',
						width='stretch' ):
					try:
						loader = ArXivLoader( )
						docs = loader.load( question=str( arxiv_query ) )
						
						st.session_state[ 'remote_loader_results' ] = {
								'type': 'ArXiv',
								'input': arxiv_query,
								'documents_loaded': len( docs ) if isinstance( docs, list ) else 0,
								'documents': docs or [ ],
						}
						st.rerun( )
					except Exception as exc:
						st.error( str( exc ) )
			
			elif remote_type == 'Wikipedia':
				wiki_query = st.text_input(
					'Enter a Wikipedia query',
					key='remote_wiki_query' )
				
				if st.button(
						'Load Wikipedia Results',
						key='remote_wiki_load_btn',
						width='stretch' ):
					try:
						loader = WikiLoader( )
						docs = loader.load( question=str( wiki_query ) )
						
						st.session_state[ 'remote_loader_results' ] = {
								'type': 'Wikipedia',
								'input': wiki_query,
								'documents_loaded': len( docs ) if isinstance( docs, list ) else 0,
								'documents': docs or [ ],
						}
						st.rerun( )
					except Exception as exc:
						st.error( str( exc ) )
			
			elif remote_type == 'YouTube':
				youtube_url = st.text_input(
					'Enter a YouTube URL',
					key='remote_youtube_url' )
				
				if st.button(
						'Load YouTube Transcript',
						key='remote_youtube_load_btn',
						width='stretch' ):
					try:
						loader = YouTubeLoader( )
						docs = loader.load( youtube_url=str( youtube_url ) )
						
						st.session_state[ 'remote_loader_results' ] = {
								'type': 'YouTube',
								'input': youtube_url,
								'documents_loaded': len( docs ) if isinstance( docs, list ) else 0,
								'documents': docs or [ ],
						}
						st.rerun( )
					except Exception as exc:
						st.error( str( exc ) )
	
	if do_load:
		results: Dict[ str, Any ] = { }
		documents: list[ Document ] = [ ]
		file_outputs: list[ dict[ str, Any ] ] = [ ]
		
		if uploaded_files:
			for f in uploaded_files:
				name = getattr( f, 'name', 'uploaded' )
				out: dict[ str, Any ] = { 'file': name }
				
				try:
					tmp_dir = cfg.BASE_DIR / 'stores' / 'tmp_uploads'
					tmp_dir.mkdir( parents=True, exist_ok=True )
					tmp_path = tmp_dir / name
					
					with open( tmp_path, 'wb' ) as fp:
						fp.write( f.getbuffer( ) )
					
					docs, out = _load_local_from_path(
						file_path=str( tmp_path ),
						selected_type=selected_loader_type )
					
					if isinstance( docs, list ):
						documents.extend( docs )
						out[ 'documents_loaded' ] = len( docs )
				except Exception as exc:
					out[ 'error' ] = str( exc )
				
				file_outputs.append( out )
		
		paths = [ p.strip( ) for p in (loader_text or '').splitlines( ) if p.strip( ) ]
		if paths:
			for file_path in paths:
				out: dict[ str, Any ] = { 'file': file_path }
				try:
					docs, out = _load_local_from_path(
						file_path=file_path,
						selected_type=selected_loader_type )
					
					if isinstance( docs, list ):
						documents.extend( docs )
						out[ 'documents_loaded' ] = len( docs )
				except Exception as exc:
					out[ 'error' ] = str( exc )
				
				file_outputs.append( out )
		
		results[ 'selected_loader' ] = selected_loader_type
		results[ 'files' ] = file_outputs
		st.session_state[ 'loader_results' ] = results
		st.session_state[ 'loader_documents' ] = documents
		st.rerun( )
	
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	
	if st.session_state.get( 'loader_documents' ):
		st.markdown( '##### Loaded Documents' )
		for idx, doc in enumerate( st.session_state[ 'loader_documents' ], start=1 ):
			with st.expander( f'Local Document {idx}', expanded=False ):
				st.text_area( '', value=(doc.page_content or ''), height=260 )
				if getattr( doc, 'metadata', None ):
					st.json( doc.metadata )
	
	if st.session_state.get( 'loader_results' ):
		st.markdown( '##### Local Load Results' )
		st.json( st.session_state[ 'loader_results' ] )
	
	remote_results = st.session_state.get( 'remote_loader_results', { } )
	if remote_results:
		st.markdown( '##### Remote Load Results' )
		st.json(
			{
					'type': remote_results.get( 'type', '' ),
					'input': remote_results.get( 'input', '' ),
					'documents_loaded': remote_results.get( 'documents_loaded', 0 ),
			}
		)
		
		for idx, doc in enumerate( remote_results.get( 'documents', [ ] ), start=1 ):
			with st.expander( f'Remote Document {idx}', expanded=False ):
				st.text_area( '', value=(doc.page_content or ''), height=260 )
				if getattr( doc, 'metadata', None ):
					st.json( doc.metadata )

# =============================================================================
# SCRAPING MODE
# ==============================================================================
elif mode == modes[ 1 ]:
	st.subheader( f'🕷️ { modes[ 1 ] }' )
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
elif mode == modes[ 2 ]:
	st.subheader( f'🏛️  {modes[ 2 ]}' )
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
	with st.expander( label='Congress', expanded=False ):
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

# ==============================================================================
# SATELLITE MODE
# ==============================================================================
elif mode == modes[ 3 ]:
	st.subheader( f'🚀  {modes[ 3 ]}' )
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
					value=st.session_state.get(
						'satellitecenter_coordinate_systems',
						'gse'
					),
					key='satellitecenter_coordinate_systems',
					placeholder='gse or geo,gsm',
					disabled=(satellite_mode != 'locations')
				)
			
			with c4:
				satellite_resolution_factor = st.number_input(
					'Resolution Factor',
					min_value=1,
					max_value=1000,
					value=int(
						st.session_state.get( 'satellitecenter_resolution_factor', 1 )
					),
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
					
					st.session_state[ 'satellitecenter_results' ] = result or { }
					st.rerun( )
				
				except Exception as exc:
					st.error( 'Satellite Center request failed.' )
					st.exception( exc )
			
			result = st.session_state.get( 'satellitecenter_results', { } )
			
			if not result:
				st.text( 'No results.' )
			else:
				st.markdown( '#### Request Metadata' )
				st.json(
					{
							'mode': satellite_mode,
							'query': satellite_query,
							'start_time': satellite_start_time,
							'end_time': satellite_end_time,
							'coordinate_systems': satellite_coordinate_systems,
							'resolution_factor': int( satellite_resolution_factor ),
					}
				)
				
				if satellite_mode == 'observatories':
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
						
						st.markdown( f'#### Observatories ({len( summary_rows )})' )
						df_satellite = pd.DataFrame( summary_rows )
						
						if not df_satellite.empty:
							st.dataframe(
								df_satellite,
								use_container_width=True,
								hide_index=True
							)
						else:
							st.info( 'No displayable observatory rows were found.' )
						
						with st.expander( 'Observatory Details', expanded=False ):
							for idx, item in enumerate( items, start=1 ):
								label = item.get( 'Id', f'Observatory {idx}' )
								with st.expander(
										f'Observatory {idx}: {label}',
										expanded=False
								):
									st.json( item )
					else:
						st.info( 'No observatories returned.' )
				
				elif satellite_mode == 'ground_stations':
					items = result.get( 'GroundStation', [ ] ) if isinstance( result, dict ) else [ ]
					
					if items:
						summary_rows: List[ Dict[ str, Any ] ] = [ ]
						
						for item in items:
							if isinstance( item, dict ):
								location_value = ''
								geo_value = item.get( 'Location', { } )
								
								if isinstance( geo_value, dict ):
									lat_value = geo_value.get( 'Latitude', '' )
									lon_value = geo_value.get( 'Longitude', '' )
									if str( lat_value ).strip( ) or str( lon_value ).strip( ):
										location_value = f'{lat_value}, {lon_value}'
								
								summary_rows.append(
									{
											'Id': item.get( 'Id', '' ),
											'Name': item.get( 'Name', '' ),
											'Provider': item.get( 'Provider', '' ),
											'Type': item.get( 'Type', '' ),
											'Location': location_value,
									}
								)
						
						st.markdown( f'#### Ground Stations ({len( summary_rows )})' )
						df_satellite = pd.DataFrame( summary_rows )
						
						if not df_satellite.empty:
							st.dataframe(
								df_satellite,
								use_container_width=True,
								hide_index=True
							)
						else:
							st.info( 'No displayable ground-station rows were found.' )
						
						with st.expander( 'Ground Station Details', expanded=False ):
							for idx, item in enumerate( items, start=1 ):
								label = item.get( 'Id', f'Ground Station {idx}' )
								with st.expander(
										f'Ground Station {idx}: {label}',
										expanded=False
								):
									st.json( item )
					else:
						st.info( 'No ground stations returned.' )
				
				else:
					data_items = result.get( 'Data', [ ] ) if isinstance( result, dict ) else [ ]
					
					if data_items:
						summary_rows: List[ Dict[ str, Any ] ] = [ ]
						
						for item in data_items:
							if isinstance( item, dict ):
								coordinates_value = item.get( 'Coordinates', [ ] )
								point_count = 0
								
								if isinstance( coordinates_value, list ):
									point_count = len( coordinates_value )
								
								summary_rows.append(
									{
											'Id': item.get( 'Id', '' ),
											'CoordinateSystem': item.get(
												'CoordinateSystem',
												''
											),
											'StartTime': item.get( 'StartTime', '' ),
											'EndTime': item.get( 'EndTime', '' ),
											'PointCount': point_count,
									}
								)
						
						st.markdown( f'#### Location Sets ({len( summary_rows )})' )
						df_satellite = pd.DataFrame( summary_rows )
						
						if not df_satellite.empty:
							st.dataframe(
								df_satellite,
								use_container_width=True,
								hide_index=True
							)
						else:
							st.info( 'No displayable location rows were found.' )
						
						with st.expander( 'Location Set Details', expanded=False ):
							for idx, item in enumerate( data_items, start=1 ):
								label = item.get( 'Id', f'Trajectory {idx}' )
								with st.expander(
										f'Location Set {idx}: {label}',
										expanded=False
								):
									st.json( item )
					else:
						st.info( 'No location data returned.' )
				
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
	
	# -------- Earth Observatory
	with st.expander( label='Earth Observatory', expanded=False ):
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
# TEXT GENERATION MODE
# ==============================================================================

elif mode == modes[ 4 ]:
	st.subheader( f'🧠  {modes[ 4 ]}' )
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
elif mode == modes[ 5 ]:
	st.subheader( f'🏛️ {modes[ 5 ]}', help=cfg.DATA_MANAGEMENT )
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
