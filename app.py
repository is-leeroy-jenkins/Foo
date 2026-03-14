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
import config as cfg
import json
import numpy as np
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


from langchain_core.documents import Document
from loaders import ( PdfLoader, WordLoader, ExcelLoader, MarkdownLoader,
                     HtmlLoader, YouTubeLoader, RecursiveCharacterTextSplitter,
                      PowerPointLoader )
from fetchers import (
	Wikipedia, TheNews, SatelliteCenter, WebFetcher,
	GoogleWeather, Grokipedia, OpenWeather, NavalObservatory,
	GoogleSearch, GoogleDrive, GoogleMaps, NearbyObjects, OpenScience,
	EarthObservatory, SpaceWeather, AstroCatalog, AstroQuery, StarMap,
	GovData, Congress, InternetArchive, Chat, Claude,
	Groq, Mistral, Gemini, StarChart
)

import pandas as pd
from pandas import DataFrame
import streamlit as st
import scrapers
import sqlite3
from sqlite3 import Connection




# ======================================================================================
# SESSION STATE
# ======================================================================================

if 'mode' not in st.session_state or st.session_state[ 'mode' ] is None:
	st.session_state[ 'mode' ] = 'Document Loading'
	
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
	
	
# ======================================================================================
# UTILITITES
# ======================================================================================

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
                PromptsId
                INTEGER
                NOT
                NULL
                PRIMARY
                KEY
                AUTOINCREMENT,
                Caption
                TEXT,
                Name
                TEXT
            (
                80
            ),
                Text TEXT,
                Version TEXT
            (
                80
            ),
                ID TEXT
            (
                80
            )
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

def apply_filters( df: DataFrame ) -> DataFrame:
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
		
		fig = go.Figure( data=[ go.Histogram( x=values ) ] )
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
		
		fig = go.Figure( data=[ go.Bar( x=x_values, y=y_values ) ] )
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
		
		fig = go.Figure( data=[ go.Scatter( x=x_values, y=y_values, mode='lines' ) ] )
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
		
		fig = go.Figure( data=[ go.Scatter( x=x_values, y=y_values, mode='markers' ) ] )
		fig.update_layout( xaxis_title=x, yaxis_title=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Box':
		if not numeric_cols:
			st.info( 'No numeric columns available.' )
			return
		
		col = st.selectbox( 'Column', numeric_cols, key='viz_box_col' )
		values = pd.to_numeric( df_plot[ col ], errors='coerce' ).dropna( ).tolist( )
		
		fig = go.Figure( data=[ go.Box( y=values, name=col ) ] )
		fig.update_layout( yaxis_title=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Pie':
		if not categorical_cols:
			st.info( 'No categorical columns available.' )
			return
		
		col = st.selectbox( 'Category Column', categorical_cols )
		counts = df_plot[ col ].astype( str ).value_counts( )
		
		fig = go.Figure(
			data=[ go.Pie( labels=counts.index.tolist( ), values=counts.values.tolist( ) ) ] )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Correlation':
		if len( numeric_cols ) < 2:
			st.info( 'At least two numeric columns are required.' )
			return
		
		corr_df = DataFrame( )
		for col in numeric_cols:
			corr_df[ col ] = pd.to_numeric( df_plot[ col ], errors='coerce' )
		
		corr = corr_df.corr( )
		
		fig = go.Figure(
			data=[ go.Heatmap(
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


# ======================================================================================
# APP SET-UP
# ======================================================================================

style_subheaders( )
st.logo( cfg.LOGO, size='large' )
st.set_page_config( page_title=cfg.APP_TITLE, layout='wide', page_icon=cfg.FAVICON )
col_left, col_center, col_right = st.columns( [ 1, 2, 1 ], vertical_alignment='top' )

# ======================================================================================
# SIDEBAR
# ======================================================================================
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
	mode = st.sidebar.radio( label='Mode', options=cfg.MODES, label_visibility='collapsed' )
	if mode:
		st.session_state[ 'mode' ] = mode
	else:
		st.session_state[ 'mode' ] = 'Loaders'
		
# ======================================================================================
# DOCUMENT LOADING MODE
# ======================================================================================
if mode == 'Document Loading':
	st.subheader( '📤   Document Loading' )
	st.divider( )
	loader_path = st.session_state.get( 'loader_path', '' )
	loader_results = st.session_state.get( 'loader_results', { } )
	loader_documents = st.session_state.get( 'loader_documents', [ ] )
	loader_text = st.session_state.get( 'loader_text', '' )
	loader_files = st.session_state.get( 'loader_files', [ ] )
	loader = None
	
	col_browse, col_text = st.columns( [ 0.7, 0.3 ], border=True )
	with col_browse:
		uploaded_files = st.file_uploader( 'Choose file(s)',
			type=[ 'pdf', 'docx', 'xlsx', 'xls', 'pptx', 'txt', 'md' ], accept_multiple_files=True,
			key='loader_uploaded_files', )
		
		# -----------------------------------------------
		# Checkbox row (unchanged: its own row)
		# -----------------------------------------------
		col1, col2, col3, col4, col5, col6  = st.columns( 6 )
		
		with col1:
			do_pdf = st.checkbox( label='PDF', key='pdf_cb' )
		with col2:
			do_word = st.checkbox( label='Word', key='word_cb' )
		with col3:
			do_excel = st.checkbox( label='Excel', key='excel_cb' )
		with col4:
			do_markdown = st.checkbox( label='Markdown', key='markdown_cb' )
		with col5:
			do_powerpoint = st.checkbox( label='Powerpoint', key='powerpoint_cb' )
		with col6:
			do_text = st.checkbox( label='Text', key='text_cb' )


	with col_text:
		loader_text = st.text_area( 'Enter one URL or file path', height=20,
			key='loader_path' )

	b1, b2 = st.columns( 2 )
	with b1:
		do_load = st.button( 'Load', key='loader_load_btn' )

	with b2:
		do_clear = st.button( 'Clear', key='loader_clear_btn' )

	if do_clear:
		st.session_state.update(
		{
			'loader_results': { },
			'loader_documents': [ ],
			'loader_path': '',
			'loader_uploaded_files': None,
		} )
		st.rerun( )

	# ---------------------------
	# Execute loads
	# ---------------------------
	if do_load:
		results: Dict[ str, Any ] = { }
		documents: list[ Document ] = [ ]

		# Always use the correct object name.
		extractor = scrapers.WebExtractor( )

	    # -----------------------------------------------
		# 1) URL scraping (from right text area)
	    # -----------------------------------------------
		urls = [ u.strip( ) for u in ( loader_text or "" ).splitlines( ) if u.strip( ) ]
		if urls:
			url_outputs: list[ dict[ str, Any ] ] = [ ]

			for url in urls:
				output: dict[ str, Any ] = { "url": url }

				try:
					if do_pdf:
						output[ 'pdf' ] = extractor.scrape_hyperlinks( url )
					if do_word:
						output[ 'word' ] = extractor.scrape_links( url )
					if do_excel:
						output[ 'excel' ] = extractor.scrape_tables( url )

					if do_markdown and hasattr( extractor, 'scrape_' ):
						output[ 'markdown' ] = extractor.scrape_( url )
					if do_powerpoint and hasattr( extractor, 'scrape_' ):
						output[ 'powerpoint' ] = extractor.scrape_( url )
					if do_youtube and hasattr( extractor, 'scrape_' ):
						output[ 'youtube' ] = extractor.scrape_( url )

				except Exception as exc:
					output[ 'error' ] = str( exc )

				url_outputs.append( output )

			results[ 'urls' ] = url_outputs

	    # -----------------------------------------------
		# 2) Local file loading (from left uploader)
	    # -----------------------------------------------
		if uploaded_files:
			file_outputs: list[ dict[ str, Any ] ] = [ ]

			for f in uploaded_files:
				name = getattr( f, 'name', 'uploaded' )
				suffix = Path( name ).suffix.lower( ).lstrip( '.' )

				out: dict[ str, Any ] = { 'file': name, 'type': suffix }

				try:
					tmp_dir = ( cf.BASE_DIR / 'stores' / 'tmp_uploads' )
					tmp_dir.mkdir( parents=True, exist_ok=True )
					tmp_path = tmp_dir / name

					with open( tmp_path, 'wb' ) as fp:
						fp.write( f.getbuffer( ) )
						
					if suffix == 'pdf' and do_pdf:
						ld = PdfLoader( )
						docs = ld.load( str( tmp_path ) )
						if isinstance( docs, list ):
							documents.extend( docs )

					elif suffix == 'docx' and do_word:
						ld = WordLoader( )
						docs = ld.load( str( tmp_path ) )
						if isinstance( docs, list ):
							documents.extend( docs )

					elif suffix in ( 'xlsx', 'xls' ) and do_excel:
						ld = ExcelLoader( )
						docs = ld.load( str( tmp_path ) )
						if isinstance( docs, list ):
							documents.extend( docs )

					else:
						out[ 'skipped' ] = 'Checkbox for this file type is not selected.'

				except Exception as exc:
					out[ 'error' ] = str( exc )

				file_outputs.append( out )

			results[ 'files' ] = file_outputs

		st.session_state.update( { 'loader_results': results, 'loader_documents': documents, } )
		st.rerun( )

	# -----------------------------------------------
	# Render results (below checkboxes)
	# -----------------------------------------------
	st.markdown( "----" )

	# 1) Documents (from local loaders)
	if st.session_state[ 'loader_documents' ]:
		st.markdown( '### Loaded Documents' )
		for idx, doc in enumerate( st.session_state[ 'loader_documents' ], start=1 ):
			with st.expander( f'Document {idx}', expanded=False ):
				st.text_area( '', value=( doc.page_content or "" ), height=260 )
				if getattr( doc, 'metadata', None ):
					st.json( doc.metadata )

	# 2) Raw results (from URL scraping and file processing metadata)
	if st.session_state[ 'loader_results' ]:
		st.markdown( '### Load Results' )
		st.json( st.session_state[ 'loader_results' ] )
		
# ======================================================================================
# SCRAPING MODE
# ======================================================================================
elif mode == 'Web Scrapping':
	st.subheader( '🕷️ Web Scrapping' )
	st.divider( )
	col_left, col_right = st.columns([1, 2], border=True)
	with col_left:
		target_url = st.text_input( 'Target URL', placeholder='https://example.com',
			key='webfetcher_url' )

		st.markdown( '#### Extraction Options' )

		fetcher = WebFetcher( )
		raw_names = [ name for name in fetcher.__dir__()
			if name.startswith('scrape') ]

		VALID_SCRAPERS: dict[str, str] = {
			'scrape_images': 'Images',
			'scrape_hyperlinks': 'Hyperlinks',
			'scrape_blockquotes': 'Blockquotes',
			'scrape_sections': 'Sections',
			'scrape_divisions': 'Divisions',
			'scrape_tables': 'Tables',
			'scrape_lists': 'Lists',
			'scrape_paragraphs': 'Paragraphs',
		}

		available_methods: dict[ str, callable ] = { }

		for name in raw_names:
			if name in VALID_SCRAPERS and hasattr( fetcher, name ):
				available_methods[name] = getattr( fetcher, name )

		selected_methods: list[ str ] = [ ]

		for method_name, label in VALID_SCRAPERS.items():
			if method_name in available_methods:
				if st.checkbox(label, key=f'wf_{method_name}'):
					selected_methods.append( method_name )

		run_scraper = st.button( 'Run Scraper', key='webfetcher_run' )

	with col_right:
		output = st.empty( )

		if run_scraper:
			try:
				if not target_url:
					raise ValueError('A target URL is required.')

				if not selected_methods:
					raise ValueError('At least one scraper must be selected.')

				results: dict[str, list[str]] = { }

				for method_name in selected_methods:
					method = available_methods[method_name]
					data = method(target_url)

					if data is None:
						results[method_name] = []
					elif isinstance(data, list):
						results[method_name] = data
					else:
						results[method_name] = [str(data)]

				with output.container():
					for method_name, items in results.items():
						st.markdown( f'#### {VALID_SCRAPERS[method_name]}' )

						if not items:
							st.info('No results returned.')
							continue

						for idx, item in enumerate(items, start=1):
							st.write(f"{idx}. {item}")

			except Exception as exc:
				st.error( str( exc ) )

# ======================================================================================
# FETCHING MODE
# ======================================================================================
elif mode == 'Data Collection':
	st.subheader( '🏛️  Data Archives & Collections' )
	st.divider( )
	st.session_state.setdefault( "arxiv_input", "" )
	st.session_state.setdefault( "arxiv_results", [ ] )
	
	# -------- ArXiv
	with st.expander( label='ArXiv', expanded=True ):
		col1, col2 = st.columns( 2, border=True )
		
		with col1:
			arxiv_input = st.text_area( 'Query', height=40, key='arxiv_input', )
			
			b1, b2 = st.columns( 2 )
			
			with b1:
				if st.button( 'Submit', key='arxiv_submit' ):
					try:
						queries = [ q.strip( ) for q in arxiv_input.splitlines( ) if q.strip( ) ]
						if not queries:
							st.warning( 'No input provided.' )
						else:
							from fetchers import ArXiv
							
							f = ArXiv( )
							results = [ ]
							
							for q in queries:
								docs = f.fetch( q )
								if isinstance( docs, list ):
									results.append( docs )
								elif isinstance( docs, list ):
									results.extend( docs )
							
							st.session_state.update( { 'arxiv_results': results } )
							st.rerun( )
					
					except Exception as exc:
						st.error( 'ArXiv request failed.' )
						st.exception( exc )
			
			with b2:
				if st.button( 'Clear', key='arxiv_clear' ):
					st.session_state.update( { 'arxiv_input': '', 'arxiv_results': [ ] } )
					st.rerun( )
		
		with col2:
			st.markdown( 'Results' )
			
			if not st.session_state[ 'arxiv_results' ]:
				st.text( 'No results.' )
			else:
				for idx, doc in enumerate( st.session_state[ 'arxiv_results' ], start=1 ):
					with st.expander( f'Document {idx}', expanded=False ):
						if isinstance( doc, Document ):
							st.text_area( 'Content', value=doc.page_content or '', height=300 )
							if doc.metadata:
								st.json( doc.metadata )
						else:
							st.write( doc )
		
	# -------- Google Drive
	with st.expander( label='Google Drive', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			gd_query = st.text_area( 'Google Drive Query', value='', height=40,
				key='googledrive_query' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				gd_submit = st.button( 'Submit', key='googledrive_submit' )
			with b2:
				gd_clear = st.button( 'Clear', key='googledrive_clear' )
		
		with col_right:
			gd_output = st.empty( )
		
		if gd_clear:
			st.session_state.update( { 'googledrive_query': "" } )
			st.rerun( )
		
		if gd_submit:
			try:
				f = GoogleDrive( )
				docs = f.fetch( gd_query )
				
				if docs:
					with gd_output.container( ):
						for idx, doc in enumerate( docs, start=1 ):
							st.markdown( f"**Document {idx}**" )
							st.text_area( '', value=doc.page_content, height=200 )
				else:
					gd_output.info( 'No documents returned.' )
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- Wikipedia
	with st.expander( label='Wikipedia', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			wiki_query = st.text_area( 'Wikipedia Query', value='', height=40,
				key='wikipedia_query' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				wiki_submit = st.button( 'Submit', key='wikipedia_submit' )
			with b2:
				wiki_clear = st.button( 'Clear', key='wikipedia_clear' )
		
		with col_right:
			wiki_output = st.empty( )
		
		if wiki_clear:
			st.session_state.update( { 'wikipedia_query': '' } )
			st.rerun( )
		
		if wiki_submit:
			try:
				f = Wikipedia( )
				docs = f.fetch( wiki_query )
				
				if docs:
					with wiki_output.container( ):
						for idx, doc in enumerate( docs, start=1 ):
							st.markdown( f"**Document {idx}**" )
							st.text_area( '', value=doc.page_content, height=200 )
				else:
					wiki_output.info( 'No documents returned.' )
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- The News API
	with st.expander( label='The News API', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			news_query = st.text_area( 'News Query', value='',
				height=40, key='thenews_query' )
			
			news_api_key = st.text_input( 'API Key', value='', type='password',
				key='thenews_api_key' )
			
			news_timeout = st.number_input( 'Timeout (seconds)', min_value=1, max_value=60,
				value=10, step=1, key='thenews_timeout' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				news_submit = st.button( 'Submit', key='thenews_submit' )
			with b2:
				news_clear = st.button( 'Clear', key='thenews_clear' )
		
		with col_right:
			news_output = st.empty( )
		
		if news_clear:
			st.session_state.update( { 'thenews_query': '', 'thenews_api_key': '',
					'thenews_timeout': 10 } )
			st.rerun( )
		
		if news_submit:
			try:
				f = TheNews( )
				if news_api_key:
					f.api_key = news_api_key
				
				result = f.fetch(
					query=news_query,
					time=int( news_timeout )
				)
				
				if result and getattr( result, 'text', None ):
					news_output.text_area( 'Result', value=result.text, height=300 )
				else:
					news_output.info( 'No results returned.' )
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- Google Search
	with st.expander( label="Google Search", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			google_query = st.text_area( "Query", value='', height=40,
				key='googlesearch_query' )
			
			google_num_results = st.number_input( 'Number of Results', min_value=1,
				max_value=50, value=10, step=1, key='googlesearch_num_results' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				google_submit = st.button( 'Submit', key='googlesearch_submit' )
			with b2:
				google_clear = st.button( 'Clear', key='googlesearch_clear' )
		
		with col_right:
			google_output = st.empty( )
		
		if google_clear:
			st.session_state.update( { 'googlesearch_query': '', 'googlesearch_num_results': 10 } )
			st.rerun( )
		
		if google_submit:
			try:
				f = GoogleSearch( )
				result = f.fetch( keywords=google_query, results=int( google_num_results ) )
				
				txt = render_google_results( result )
				google_output.text_area( 'Results', value=txt, height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- Naval Observatory
	with st.expander( label='US Naval Observatory', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			naval_query = st.text_area( 'Query', value='', height=40,
				key='navalobservatory_query' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				naval_submit = st.button( 'Submit', key='navalobservatory_submit' )
			with b2:
				naval_clear = st.button( 'Clear', key='navalobservatory_clear' )
		
		with col_right:
			naval_output = st.empty( )
		
		if naval_clear:
			st.session_state.update( { 'navalobservatory_query': '', } )
			st.rerun( )
		
		if naval_submit:
			try:
				f = NavalObservatory( )
				result = f.fetch( naval_query )
				
				if not result:
					naval_output.info( 'No results returned.' )
				else:
					naval_output.text_area( 'Results', value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- Open Science
	with st.expander( label='Open Science', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			openscience_query = st.text_area( 'Query', value='', height=40,
				key='openscience_query' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				openscience_submit = st.button( 'Submit', key='openscience_submit' )
			with b2:
				openscience_clear = st.button( 'Clear', key='openscience_clear' )
		
		with col_right:
			openscience_output = st.empty( )
		
		if openscience_clear:
			st.session_state.update( { 'openscience_query': '', } )
			st.rerun( )
		
		if openscience_submit:
			try:
				f = OpenScience( )
				result = f.fetch( openscience_query )
				
				if not result:
					openscience_output.info( 'No results returned.' )
				else:
					openscience_output.text_area( 'Results', value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- Gov Data
	with st.expander( label='Gov Info', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			govdata_query = st.text_area( 'Query', value='', height=40, key='govdata_query' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				govdata_submit = st.button( 'Submit', key='govdata_submit' )
			with b2:
				govdata_clear = st.button( 'Clear', key='govdata_clear' )
		
		with col_right:
			govdata_output = st.empty( )
		
		if govdata_clear:
			st.session_state.update( { 'govdata_query': '', } )
			st.rerun( )
		
		if govdata_submit:
			try:
				f = GovData( )
				result = f.fetch( govdata_query )
				
				if not result:
					govdata_output.info( 'No results returned.' )
				else:
					govdata_output.text_area( 'Results', value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- Congress
	with st.expander( label='Congress', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			congress_query = st.text_area( 'Query', value='', height=40, key='congress_query' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				congress_submit = st.button( 'Submit', key='congress_submit' )
			with b2:
				congress_clear = st.button( 'Clear', key='congress_clear' )
		
		with col_right:
			congress_output = st.empty( )
		
		if congress_clear:
			st.session_state.update( { 'congress_query': '', } )
			st.rerun( )
		
		if congress_submit:
			try:
				f = Congress( )
				result = f.fetch( congress_query )
				
				if not result:
					congress_output.info( 'No results returned.' )
				else:
					congress_output.text_area( 'Results', value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- Internet Archive
	with st.expander( label='Internet Archive', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			ia_query = st.text_area(
				'Query',
				value='',
				height=40,
				key='internetarchive_query'
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				ia_submit = st.button( 'Submit', key='internetarchive_submit' )
			with b2:
				ia_clear = st.button( 'Clear', key='internetarchive_clear' )
		
		with col_right:
			ia_output = st.empty( )
		
		if ia_clear:
			st.session_state.update( { 'internetarchive_query': '', } )
			st.rerun( )
		
		if ia_submit:
			try:
				f = InternetArchive( )
				result = f.fetch( ia_query )
				
				if not result:
					ia_output.info( 'No results returned.' )
				else:
					ia_output.text_area( 'Results', value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )

# ======================================================================================
# TEXT GENERATION MODE
# ======================================================================================
elif mode == 'Generative AI':
	st.subheader( '🧠  Generative AI' )
	st.divider( )
	# -------- Chat GPT
	with st.expander( label='ChatGPT', expanded=True ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			chat_prompt = st.text_area( 'Prompt', value='', height=40, key='chat_prompt' )
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				chat_model = _model_selector( key_prefix='chat', label='Model',
					options=[ 'gpt-4o-mini', 'gpt-4.1-mini', 'gpt-4.1', 'o3-mini',  ],
					default_model='gpt-4o-mini', )
			
			with p_row1[ 1 ]:
				chat_temperature = st.slider( 'Temperature', min_value=0.0, max_value=2.0,
					value=0.7, step=0.05, key='chat_temperature', )
			
			with p_row2[ 0 ]:
				chat_max_tokens = st.number_input( 'Max Tokens', min_value=1, max_value=32768,
					value=1024, step=1, key='chat_max_tokens', )
			
			with p_row2[ 1 ]:
				chat_top_p = st.slider( 'Top-P', min_value=0.0, max_value=1.0,
					value=1.0, step=0.01, key='chat_top_p', )
			
			with p_row3[ 0 ]:
				chat_seed = st.number_input( 'Seed', min_value=0, max_value=2_147_483_647,
					value=0, step=1, key='chat_seed', )
			
			with p_row3[ 1 ]:
				chat_json_mode = st.checkbox( 'JSON Mode', value=False, key='chat_json_mode' )
			
			chat_system = st.text_area( 'System', value='', height=100, key='chat_system' )
			
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
			st.session_state.update( { "chat_prompt": '', "chat_system": "" } )
			st.rerun( )
		
		# -----------------------------
		# Submit Button
		# -----------------------------
		if chat_submit:
			try:
				fetcher = Chat( )
				params = \
				{ 
						'model': chat_model, 
				        'temperature': float( chat_temperature ),
				        'max_tokens': int( chat_max_tokens ), 
				        'top_p': float( chat_top_p ), 
				        'seed': int( chat_seed ) if int( chat_seed ) > 0 else None, 
				        'system': chat_system if chat_system.strip( ) else None, 
				        'response_format': ('json' if chat_json_mode else None),
				}
				
				params = { k: v for k, v in params.items( ) if v is not None }
				result = _invoke_provider( fetcher, chat_prompt, params )
				_render_output( chat_output, result )
			
			except Exception as exc:
				st.error( str( exc ) )
		
	# -------- GROQ
	with st.expander( label='Groq', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			groq_prompt = st.text_area( 'Prompt', value='', height=40,
				key='groq_prompt_chat' )
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				groq_model = _model_selector( key_prefix='groq', label='Model',
					options=[ 'llama3-70b-8192', 'llama3-8b-8192', 'mixtral-8x7b-32768', ],
					default_model='llama3-70b-8192', )
			
			with p_row1[ 1 ]:
				groq_temperature = st.slider( 'Temperature', min_value=0.0, max_value=2.0,
					value=0.7, step=0.05, key='groq_temperature_chat', )
			
			with p_row2[ 0 ]:
				groq_max_tokens = st.number_input( 'Max Tokens', min_value=1, max_value=32768,
					value=1024, step=1, key='groq_max_tokens_chat', )
			
			with p_row2[ 1 ]:
				groq_top_p = st.slider( 'Top-P', min_value=0.0,
					max_value=1.0, value=1.0, step=0.01, key='groq_top_p_chat', )
			
			with p_row3[ 0 ]:
				groq_stop = st.text_area( 'Stop Sequences (one per line)', value='', height=80,
					key='groq_stop_chat', )
			
			with p_row3[ 1 ]:
				groq_stream = st.checkbox( 'Stream', value=False, key='groq_stream_chat' )
			
			btn_row = st.columns( 2 )
			with btn_row[ 0 ]:
				groq_submit = st.button( 'Submit', key='groq_submit_chat' )
			with btn_row[ 1 ]:
				groq_clear = st.button( 'Clear', key='groq_clear_chat' )
		
		with col_right:
			groq_output = st.empty( )
		
		
		if groq_clear:
			st.session_state.update( { 'groq_prompt_chat': '', 'groq_stop_chat': '' } )
			st.rerun( )
		
		if groq_submit:
			try:
				fetcher = Groq( )
				stop_lines = [ s.strip( ) for s in (groq_stop or "").splitlines( ) if
				               s.strip( ) ]
				
				params = \
				{
						'model': groq_model,
						'temperature': float( groq_temperature ),
						'max_tokens': int( groq_max_tokens ),
						'top_p': float( groq_top_p ),
						'stop': stop_lines if stop_lines else None,
						'stream': bool( groq_stream ),
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
			claude_prompt = st.text_area( 'Prompt', value='', height=40, key='claude_prompt_chat' )
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				claude_model = _model_selector( key_prefix='claude', label='Model',
					options=[
							'claude-3-5-sonnet-latest',
							'claude-3-5-haiku-latest',
							'claude-3-opus-latest',
							'Custom...',
					],
					default_model='claude-3-5-sonnet-latest',
				)
			
			with p_row1[ 1 ]:
				claude_temperature = st.slider( 'Temperature', min_value=0.0, max_value=1.0,
					value=0.7, step=0.05, key='claude_temperature_chat', )
			
			with p_row2[ 0 ]:
				claude_max_tokens = st.number_input( 'Max Tokens', min_value=1, max_value=8192,
					value=1024, step=1, key='claude_max_tokens_chat', )
			
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
				claude_stop = st.text_area(
					'Stop Sequences (one per line)',
					value='',
					height=80,
					key='claude_stop_chat',
				)
			
			claude_system = st.text_area(
				'System',
				value='',
				height=100,
				key='claude_system_chat',
			)
			
			btn_row = st.columns( 2 )
			with btn_row[ 0 ]:
				claude_submit = st.button( 'Submit', key='claude_submit_chat' )
			with btn_row[ 1 ]:
				claude_clear = st.button( 'Clear', key='claude_clear_chat' )
		
		with col_right:
			claude_output = st.empty( )
			
		if claude_clear:
			st.session_state.update( {
					'claude_prompt_chat': '',
					'claude_stop_chat': '',
					'claude_system_chat': ''
			} )
			st.rerun( )
		
		if claude_submit:
			try:
				fetcher = Claude( )
				stop_lines = [ s.strip( ) for s in (claude_stop or "").splitlines( ) if
				               s.strip( ) ]
				
				params = {
						'model': claude_model,
						'temperature': float( claude_temperature ),
						'max_tokens': int( claude_max_tokens ),
						'top_p': float( claude_top_p ),
						'top_k': int( claude_top_k ) if int( claude_top_k ) > 0 else None,
						'stop_sequences': stop_lines if stop_lines else None,
						'system': claude_system if claude_system.strip( ) else None,
				}
				
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
				height=180,
				key='gemini_prompt_chat'
			)
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				gemini_model = _model_selector(
					key_prefix='gemini',
					label='Model',
					options=[
							'gemini-1.5-pro',
							'gemini-1.5-flash',
							'gemini-2.0-flash',
							'Custom...',
					],
					default_model='gemini-1.5-pro',
				)
			
			with p_row1[ 1 ]:
				gemini_temperature = st.slider(
					"Temperature",
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
					value=1024,
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
			
			gemini_system = st.text_area(
				'System',
				value='',
				height=100,
				key='gemini_system_chat',
			)
			
			btn_row = st.columns( 2 )
			with btn_row[ 0 ]:
				gemini_submit = st.button( 'Submit', key='gemini_submit_chat' )
			with btn_row[ 1 ]:
				gemini_clear = st.button( 'Clear', key='gemini_clear_chat' )
		
		with col_right:
			gemini_output = st.empty( )
			
		if gemini_clear:
			st.session_state.update( { 'gemini_prompt_chat': '', 'gemini_system_chat': '', } )
			st.rerun( )
		
		if gemini_submit:
			try:
				fetcher = Gemini( )
				params = \
				{
					'model': gemini_model,
					'temperature': float( gemini_temperature ),
					'max_tokens': int( gemini_max_tokens ),
					'top_p': float( gemini_top_p ),
					'top_k': int( gemini_top_k ) if int( gemini_top_k ) > 0 else None,
					'candidate_count': int( gemini_candidate_count ),
					'system': gemini_system if gemini_system.strip( ) else None,
				}
				
				params = { k: v for k, v in params.items( ) if v is not None }
				
				result = _invoke_provider( fetcher, gemini_prompt, params )
				_render_output( gemini_output, result )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- MISTRAL
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

# ======================================================================================
# SATELLITE MODE
# ======================================================================================
elif mode == 'Satellite Data':
	st.subheader( '🚀  Satellite Data' )
	st.divider( )
	# -------- Google Maps
	with st.expander( label='Google Maps', expanded=True ):
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
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			gw_location = st.text_area( 'Location', value='', height=40,
				key='googleweather_location' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				gw_submit = st.button( 'Submit', key='googleweather_submit' )
			with b2:
				gw_clear = st.button( 'Clear', key='googleweather_clear' )
		
		with col_right:
			gw_output = st.empty( )
			
			if gw_clear:
				st.session_state.update( { 'googleweather_location': '', } )
				st.rerun( )
			
			if gw_submit:
				try:
					f = GoogleWeather( )
					result = f.fetch_current( address=gw_location )
					if not result:
						gw_output.info( 'No results returned.' )
					else:
						gw_output.text_area( 'Results', value=result.text, height=300 )
				
				except Exception as exc:
					st.error( str( exc ) )
					
	# -------- Satellite Center
	with st.expander( label='Satellite Center', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			satellite_query = st.text_area( 'Query', value='', height=40,
				key='satellitecenter_query' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				satellite_submit = st.button( 'Submit', key='satellitecenter_submit' )
			with b2:
				satellite_clear = st.button( 'Clear', key='satellitecenter_clear' )
		
		with col_right:
			satellite_output = st.empty( )
		
		if satellite_clear:
			st.session_state.update( { 'satellitecenter_query': '', } )
			st.rerun( )
		
		if satellite_submit:
			try:
				f = SatelliteCenter( )
				result = f.fetch( satellite_query )
				
				if not result:
					satellite_output.info( 'No results returned.' )
				else:
					satellite_output.text_area( 'Results', value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
		
	# -------- Astro Catalog
	with st.expander( label='Astronomy Catalog', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			astro_query = st.text_area( 'Query', value='',
				height=40, key='astrocatalog_query' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				astro_submit = st.button( 'Submit', key='astrocatalog_submit' )
			with b2:
				astro_clear = st.button( 'Clear', key='astrocatalog_clear' )
		
		with col_right:
			astro_output = st.empty( )
		
		if astro_clear:
			st.session_state.update( { 'astrocatalog_query': '', } )
			st.rerun( )
		
		if astro_submit:
			try:
				f = AstroCatalog( )
				result = f.fetch( astro_query )
				
				if not result:
					astro_output.info( "No results returned." )
				else:
					astro_output.text_area( "Results", value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
				
	# -------- Astro Query
	with st.expander( label='Astro Query', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			astroquery_query = st.text_area( 'Query', value='',
				height=40, key='astroquery_query' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				astroquery_submit = st.button( 'Submit', key='astroquery_submit' )
			with b2:
				astroquery_clear = st.button( 'Clear', key='astroquery_clear' )
		
		with col_right:
			astroquery_output = st.empty( )
		
		if astroquery_clear:
			st.session_state.update( { 'astroquery_query': "", } )
			st.rerun( )
		
		if astroquery_submit:
			try:
				f = AstroQuery( )
				result = f.fetch( astroquery_query )
				
				if not result:
					astroquery_output.info( 'No results returned.' )
				else:
					astroquery_output.text_area( 'Results', value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
		
	# -------- Star Map
	with st.expander( label='Star Map', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			starmap_query = st.text_area(
				'Query',
				value='',
				height=40,
				key='starmap_query'
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				starmap_submit = st.button( 'Submit', key='starmap_submit' )
			with b2:
				starmap_clear = st.button( 'Clear', key='starmap_clear' )
		
		with col_right:
			starmap_output = st.empty( )
		
		if starmap_clear:
			st.session_state.update( { 'starmap_query': '', } )
			st.rerun( )
		
		if starmap_submit:
			try:
				f = StarMap( )
				result = f.fetch( starmap_query )
				
				if not result:
					starmap_output.info( 'No results returned.' )
				else:
					starmap_output.text_area( 'Results', value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
				
	# -------- Open Weather
	with st.expander( label='Open Weather', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			openweather_query = st.text_area( 'Location', value='',
				height=40, key='openweather_query' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				openweather_submit = st.button( "Submit", key="openweather_submit" )
			with b2:
				openweather_clear = st.button( "Clear", key="openweather_clear" )
		
		with col_right:
			openweather_output = st.empty( )
		
		if openweather_clear:
			st.session_state.update( {
					"openweather_query": "",
			} )
			st.rerun( )
		
		if openweather_submit:
			try:
				f = OpenWeather( )
				result = f.fetch( openweather_query )
				
				if not result:
					openweather_output.info( "No results returned." )
				else:
					openweather_output.text_area( "Results", value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- Open Meteo
	with st.expander( label='Open Meteorology', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			latitude = st.number_input( 'Latitude', value=0.0,
				format='%.6f', key='openmeteo_latitude' )
			
			longitude = st.number_input( 'Longitude', value=0.0,
				format='%.6f', key='openmeteo_longitude' )
			
			days = st.number_input( 'Forecast Days', min_value=1, max_value=14,
				value=7, step=1, key='openmeteo_days' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				om_submit = st.button( 'Submit', key='openmeteo_submit' )
			with b2:
				om_clear = st.button( 'Clear', key='openmeteo_clear' )
		
		with col_right:
			om_output = st.empty( )
		
		if om_clear:
			st.session_state.update( { 'openmeteo_latitude': 0.0, 'openmeteo_longitude': 0.0,
					'openmeteo_days': 7 } )
			st.rerun( )
		
		if om_submit:
			try:
				f = OpenWeather( )
				result = f.fetch(
					latitude=float( latitude ),
					longitude=float( longitude ),
					days=int( days )
				)
				
				if result:
					om_output.text_area( 'Forecast Data', value=str( result ), height=300 )
				else:
					om_output.info( 'No data returned.' )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- Simbad Fetcher
	with st.expander( label='Simbad', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			simbad_query = st.text_area( 'Astronomical Object Query', value='', height=120,
				key='simbad_query' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				simbad_submit = st.button( 'Submit', key='simbad_submit' )
			with b2:
				simbad_clear = st.button( 'Clear', key='simbad_clear' )
		
		with col_right:
			simbad_output = st.empty( )
		
		if simbad_clear:
			st.session_state.update( { 'simbad_query': '' } )
			st.rerun( )
		
		if simbad_submit:
			try:
				f = Simbad( )
				result = f.fetch( simbad_query )
				
				if result:
					simbad_output.text_area( 'Result', value=str( result ), height=300 )
				else:
					simbad_output.info( 'No results returned.' )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- Earth Observatory
	with st.expander( label='Earth Observatory', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			earth_query = st.text_area( 'Query', value='',
				height=40, key='earthobservatory_query' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				earth_submit = st.button( 'Submit', key='earthobservatory_submit' )
			with b2:
				earth_clear = st.button( 'Clear', key='earthobservatory_clear' )
		
		with col_right:
			earth_output = st.empty( )
		
		if earth_clear:
			st.session_state.update( { 'earthobservatory_query': '', } )
			st.rerun( )
		
		if earth_submit:
			try:
				f = EarthObservatory( )
				result = f.fetch( earth_query )
				
				if not result:
					earth_output.info( 'No results returned.' )
				else:
					earth_output.text_area( 'Results', value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- Space Weather
	with st.expander( label='Space Weather', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			spaceweather_query = st.text_area( 'Query', value='',
				height=40, key='spaceweather_query' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				spaceweather_submit = st.button( 'Submit', key='spaceweather_submit' )
			with b2:
				spaceweather_clear = st.button( 'Clear', key='spaceweather_clear' )
		
		with col_right:
			spaceweather_output = st.empty( )
		
		if spaceweather_clear:
			st.session_state.update( { 'spaceweather_query': '', } )
			st.rerun( )
		
		if spaceweather_submit:
			try:
				f = SpaceWeather( )
				result = f.fetch( spaceweather_query )
				
				if not result:
					spaceweather_output.info( 'No results returned.' )
				else:
					spaceweather_output.text_area( 'Results', value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------- Star Chart
	with st.expander( label='Star Chart', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			starchart_query = st.text_area( 'Query', value='',
				height=40, key='starchart_query' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				starchart_submit = st.button( 'Submit', key='starchart_submit' )
			with b2:
				starchart_clear = st.button( 'Clear', key='starchart_clear' )
		
		with col_right:
			starchart_output = st.empty( )
		
		if starchart_clear:
			st.session_state.update( { 'starchart_query': '', } )
			st.rerun( )
		
		if starchart_submit:
			try:
				f = StarChart( )
				result = f.fetch( starchart_query )
				
				if not result:
					starchart_output.info( 'No results returned.' )
				else:
					starchart_output.text_area( 'Results', value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
		
	# -------- Nearby Objects
	with st.expander( label='Near Earth Objects', expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			nearby_query = st.text_area( 'Query', value='',
				height=40, key='nearbyobjects_query' )
			
			b1, b2 = st.columns( 2 )
			with b1:
				nearby_submit = st.button( 'Submit', key='nearbyobjects_submit' )
			with b2:
				nearby_clear = st.button( 'Clear', key='nearbyobjects_clear' )
		
		with col_right:
			nearby_output = st.empty( )
		
		if nearby_clear:
			st.session_state.update( { 'nearbyobjects_query': "", } )
			st.rerun( )
		
		if nearby_submit:
			try:
				f = NearbyObjects( )
				result = f.fetch( nearby_query )
				
				if not result:
					nearby_output.info( 'No results returned.' )
				else:
					nearby_output.text_area( 'Results', value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )

# ==============================================================================
# DATA MANAGEMENT MODE
# ==============================================================================
elif mode == 'Data Management':
	st.subheader( "🏛️ Data Management", help=cfg.DATA_MANAGEMENT )
	st.divider( )
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		tabs = st.tabs( [ "📥 Import", "🗂 Browse", "💉 CRUD", "📊 Explore", "🔎 Filter",
		                  "🧮 Aggregate", "📈 Visualize", "⚙ Admin", "🧠 SQL" ] )
		
		tables = list_tables( )
		if not tables:
			st.info( "No tables available." )
		else:
			table = st.selectbox( "Table", tables )
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
		# CRUD (Schema-Aware)
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
						insert_data[
							column ] = st.number_input( column, step=1, key=f'ins_{column}' )
					
					elif 'REAL' in col_type:
						insert_data[
							column ] = st.number_input( column, format='%.6f', key=f'ins_{column}' )
					
					elif 'BOOL' in col_type:
						insert_data[
							column ] = 1 if st.checkbox( column, key=f'ins_{column}' ) else 0
					
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
