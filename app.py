from __future__ import annotations

import inspect
import json
import os
import sqlite3
from langchain_core.documents import Document
from pathlib import Path
from typing import Any, Dict

import streamlit as st

import config
import scrapers
import fetchers
from fetchers import (Wikipedia, TheNews, SatelliteCenter, Simbad,
                      GoogleWeather, Grokipedia, OpenWeather, NavalObservatory,
                      GoogleSearch, GoogleDrive, GoogleMaps, NearbyObjects, OpenScience,
                      EarthObservatory, SpaceWeather, AstroCatalog, AstroQuery, StarMap,
                      GovData, Congress, InternetArchive, OpenAI, Claude, GrokipediaClient,
                      Groq, Mistral, Gemini, StarChart)

# ======================================================================================
# Paths / Database
# ======================================================================================

BASE_DIR = Path( __file__ ).resolve( ).parent
DB_PATH = BASE_DIR / "stores" / "sqlite" / "datamodels" / "Data.db"

DB_PATH.parent.mkdir( parents=True, exist_ok=True )
if not DB_PATH.exists( ):
	conn = sqlite3.connect( DB_PATH )
	conn.execute( "PRAGMA journal_mode=WAL;" )
	conn.commit( )
	conn.close( )

# ======================================================================================
# Chat Helpers
# ======================================================================================


def _filter_kwargs_for_callable( fn: Any, kwargs: dict[ str, Any ] ) -> dict[ str, Any ]:
	try:
		sig = inspect.signature( fn )
		accepted = set( sig.parameters.keys( ) )
		return { k: v for k, v in kwargs.items( ) if k in accepted }
	except Exception:
		return kwargs

def _invoke_provider( fetcher: Any, prompt: str, params: dict[ str, Any ] ) -> Any:
	# Prefer fetch(); fallback to chat()/invoke() if that is what the provider exposes.
	if hasattr( fetcher, "fetch" ) and callable( getattr( fetcher, "fetch" ) ):
		fn = getattr( fetcher, "fetch" )
		safe = _filter_kwargs_for_callable( fn, params )
		try:
			return fn( prompt, **safe )
		except TypeError:
			# Some implementations may expect `query=` instead of positional prompt.
			safe2 = _filter_kwargs_for_callable( fn, { **safe,
			                                           "query": prompt } )
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
	
	raise RuntimeError( f"Provider '{type( fetcher ).__name__}' does not expose "
	                    f"fetch/chat/invoke." )

def _render_output( container: Any, result: Any ) -> None:
	if result is None:
		container.info( "No response returned." )
		return
	
	# Documents (LangChain)
	if isinstance( result, list ) and result and isinstance( result[ 0 ], Document ):
		with container.container( ):
			for idx, doc in enumerate( result, start=1 ):
				with st.expander( f"Document {idx}", expanded=False ):
					st.text_area( label="", value=(doc.page_content or ""), height=300 )
					if doc.metadata:
						st.json( doc.metadata )
		return
	
	# Any other object -> string
	container.text_area( label="Response", value=str( result ), height=320 )

def _model_selector(
		key_prefix: str,
		label: str,
		options: list[ str ],
		default_model: str
) -> str:
	# Always provide a selectbox; allow custom model entry if not in list.
	base_options = options[ : ]
	if "Custom..." not in base_options:
		base_options.append( "Custom..." )
	
	idx_default = 0
	if default_model in base_options:
		idx_default = base_options.index( default_model )
	
	selected = st.selectbox(
		label=label,
		options=base_options,
		index=idx_default,
		key=f"{key_prefix}_model_select",
	)
	
	if selected == "Custom...":
		return st.text_input(
			label="Custom Model",
			value=default_model,
			key=f"{key_prefix}_model_custom",
		)
	
	return selected

# ======================================================================================
# Google Helper
# ======================================================================================

def render_google_results(response) -> str:
    try:
        data = response.json()
    except Exception:
        return "Failed to decode response."

    items = data.get("items", [])
    if not items:
        return "No results returned."

    lines = []

    for idx, item in enumerate(items, start=1):
        title = item.get("title", "Untitled")
        snippet = item.get("snippet", "")
        link = item.get("link", "")

        lines.append(f"{idx}. {title}")
        if snippet:
            lines.append(snippet)
        if link:
            lines.append(link)
        lines.append("")

    return "\n".join(lines)
	    
# ======================================================================================
# Introspection helpers
# ======================================================================================

IGNORED_CLASSES = { "WebCrawler",
                    "GlobalImagery" }
IGNORED_METHODS = { "create_schema",
                    "__init__" }

def is_public_method( name: str ) -> bool:
	return not name.startswith( "_" ) and name not in IGNORED_METHODS

def get_fetcher_classes( ) -> Dict[ str, type ]:
	classes = { }
	for name, cls in inspect.getmembers( fetchers, inspect.isclass ):
		if cls.__module__ != fetchers.__name__:
			continue
		if name in IGNORED_CLASSES:
			continue
		classes[ name ] = cls
	return classes

# ======================================================================================
# Streamlit setup
# ======================================================================================

st.set_page_config( page_title="Foo", layout="wide", page_icon=config.FAVICON )
st.title( "" )

# ======================================================================================
# Sidebar — Global configuration ONLY
# ======================================================================================

with st.sidebar:
	st.header( "Configuration" )
	
	with st.expander( "API Keys", expanded=False ):
		for attr in dir( config ):
			if attr.endswith( "_API_KEY" ) or attr.endswith( "_TOKEN" ):
				current = getattr( config, attr, "" ) or ""
				value = st.text_input( attr, value=current, type="password" )
				if value:
					os.environ[ attr ] = value

# ======================================================================================
# Tabs
# ======================================================================================

tab_fetchers, tab_scrapers, tab_chat, tab_maps, tab_data = st.tabs(
	[ "Fetchers",
	  "Scrapers",
	  "Chat",
	  "Maps",
	  "Data" ]
)

# ======================================================================================
# SCRAPERS TAB — WebExtractor
# ======================================================================================

with tab_scrapers:
	st.subheader( "" )
	
	extractor = scrapers.WebExtractor( )
	
	urls_raw = st.text_area( "URLs", height=150 )
	
	col1, col2, col3 = st.columns( 3 )
	with col1:
		do_text = st.checkbox( "Text", value=True )
	with col2:
		do_links = st.checkbox( "Links", value=False )
	with col3:
		do_tables = st.checkbox( "Tables", value=False )
	
	if st.button( "Run" ):
		urls = [ u.strip( ) for u in urls_raw.splitlines( ) if u.strip( ) ]
		
		if not urls:
			st.warning( "No URLs provided." )
		else:
			for url in urls:
				st.markdown( f"### {url}" )
				output: Dict[ str, Any ] = { }
				
				try:
					if do_text:
						output[ "text" ] = extractor.scrape( url )
					if do_links:
						output[ "links" ] = extractor.scrape_links( url )
					if do_tables:
						output[ "tables" ] = extractor.scrape_tables( url )
					
					st.json( output )
				
				except Exception as exc:
					st.error( "Error" )
					st.exception( exc )

# ======================================================================================
# FETCHERS TAB — ArXiv (reconstructed, minimal, safe)
# ======================================================================================

with tab_fetchers:
	st.subheader( "" )
	
	# -----------------------------
	# Session state
	# -----------------------------
	st.session_state.setdefault( "arxiv_input", "" )
	st.session_state.setdefault( "arxiv_results", [ ] )
	
	with st.expander( "ArXiv", expanded=True ):
		col1, col2 = st.columns( 2, border=True )
		
		# -----------------------------
		# Input column
		# -----------------------------
		with col1:
			arxiv_input = st.text_area(
				"Query",
				height=200,
				key="arxiv_input",
			)
			
			btn_col1, btn_col2 = st.columns( 2 )
			
			with btn_col1:
				if st.button( "Submit", key="arxiv_submit" ):
					try:
						queries = [
								q.strip( )
								for q in arxiv_input.splitlines( )
								if q.strip( )
						]
						
						if not queries:
							st.warning( "No input provided." )
						else:
							from fetchers import ArXiv
							
							fetcher = ArXiv( )
							results = [ ]
							
							for q in queries:
								docs = fetcher.fetch( q )
								
								if isinstance( docs, Document ):
									results.append( docs )
								elif isinstance( docs, list ):
									results.extend( docs )
							
							st.session_state[ "arxiv_results" ] = results
					
					except Exception as exc:
						st.error( "ArXiv request failed." )
						st.exception( exc )
			
			with btn_col2:
				if st.button( "Clear", key="arxiv_clear" ):
					st.session_state[ "arxiv_input" ] = ""
					st.session_state[ "arxiv_results" ] = [ ]
		
		# -----------------------------
		# Output column
		# -----------------------------
		with col2:
			st.markdown( "Results" )
			
			if not st.session_state[ "arxiv_results" ]:
				st.text( "No results." )
			else:
				for idx, doc in enumerate( st.session_state[ "arxiv_results" ], start=1 ):
					with st.expander( f"Document {idx}", expanded=False ):
						if isinstance( doc, Document ):
							st.text_area(
								label="Content",
								value=doc.page_content or "",
								height=300,
							)
							
							if doc.metadata:
								st.json( doc.metadata )
						else:
							st.write( doc )
	
	# -------------------------------
	# GoogleDrive Fetcher
	# -------------------------------
	with st.expander( "GoogleDrive", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			gd_query = st.text_area(
				label="Google Drive Query",
				value="",
				height=150,
				key="googledrive_query"
			)
			
			btn_col1, btn_col2 = st.columns( 2 )
			
			with btn_col1:
				gd_submit = st.button( "Submit", key="googledrive_submit" )
			
			with btn_col2:
				gd_clear = st.button( "Clear", key="googledrive_clear" )
		
		with col_right:
			gd_output = st.empty( )
		
		if gd_clear:
			st.session_state[ "googledrive_query" ] = ""
			gd_output.empty( )
		
		if gd_submit:
			try:
				fetcher = GoogleDrive( )
				documents = fetcher.fetch( gd_query )
				
				if documents:
					with gd_output.container( ):
						for idx, doc in enumerate( documents, start=1 ):
							st.markdown( f"**Document {idx}**" )
							st.text_area(
								label="",
								value=doc.page_content,
								height=200
							)
				else:
					gd_output.info( "No documents returned." )
			
			except Exception as exc:
				error = Error( exc )
				error.module = "app"
				error.cause = "GoogleDrive"
				error.method = "fetch"
				ErrorDialog( error ).show( )
	
	# -------------------------------
	# Wikipedia Fetcher
	# -------------------------------
	with st.expander( "Wikipedia", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			wiki_query = st.text_area(
				label="Wikipedia Query",
				value="",
				height=150,
				key="wikipedia_query"
			)
			
			btn_col1, btn_col2 = st.columns( 2 )
			
			with btn_col1:
				wiki_submit = st.button( "Submit", key="wikipedia_submit" )
			
			with btn_col2:
				wiki_clear = st.button( "Clear", key="wikipedia_clear" )
		
		with col_right:
			wiki_output = st.empty( )
		
		if wiki_clear:
			st.session_state[ "wikipedia_query" ] = ""
			wiki_output.empty( )
		
		if wiki_submit:
			try:
				fetcher = Wikipedia( )
				documents = fetcher.fetch( wiki_query )
				
				if documents:
					with wiki_output.container( ):
						for idx, doc in enumerate( documents, start=1 ):
							st.markdown( f"**Document {idx}**" )
							st.text_area(
								label="",
								value=doc.page_content,
								height=200
							)
				else:
					wiki_output.info( "No documents returned." )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------------------------------
	# TheNews Fetcher
	# -------------------------------
	with st.expander( "TheNews", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			news_query = st.text_area(
				label="News Query",
				value="",
				height=120,
				key="thenews_query"
			)
			
			news_api_key = st.text_input(
				label="API Key",
				value="",
				type="password",
				key="thenews_api_key"
			)
			
			news_timeout = st.number_input(
				label="Timeout (seconds)",
				min_value=1,
				max_value=60,
				value=10,
				step=1,
				key="thenews_timeout"
			)
			
			btn_col1, btn_col2 = st.columns( 2 )
			
			with btn_col1:
				news_submit = st.button( "Submit", key="thenews_submit" )
			
			with btn_col2:
				news_clear = st.button( "Clear", key="thenews_clear" )
		
		with col_right:
			news_output = st.empty( )
		
		if news_clear:
			st.session_state[ "thenews_query" ] = ""
			st.session_state[ "thenews_api_key" ] = ""
			news_output.empty( )
		
		if news_submit:
			try:
				fetcher = TheNews( )
				
				if news_api_key:
					fetcher.api_key = news_api_key
				
				result = fetcher.fetch(
					query=news_query,
					time=int( news_timeout )
				)
				
				if result and getattr( result, "text", None ):
					news_output.text_area(
						label="Result",
						value=result.text,
						height=300
					)
				else:
					news_output.info( "No results returned." )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------------------------------
	# OpenMeteo Fetcher
	# -------------------------------
	with st.expander( "OpenMeteo", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			latitude = st.number_input(
				label="Latitude",
				value=0.0,
				format="%.6f",
				key="openmeteo_latitude"
			)
			
			longitude = st.number_input(
				label="Longitude",
				value=0.0,
				format="%.6f",
				key="openmeteo_longitude"
			)
			
			days = st.number_input(
				label="Forecast Days",
				min_value=1,
				max_value=14,
				value=7,
				step=1,
				key="openmeteo_days"
			)
			
			btn_col1, btn_col2 = st.columns( 2 )
			
			with btn_col1:
				om_submit = st.button( "Submit", key="openmeteo_submit" )
			
			with btn_col2:
				om_clear = st.button( "Clear", key="openmeteo_clear" )
		
		with col_right:
			om_output = st.empty( )
		
		if om_clear:
			st.session_state[ "openmeteo_latitude" ] = 0.0
			st.session_state[ "openmeteo_longitude" ] = 0.0
			st.session_state[ "openmeteo_days" ] = 7
			om_output.empty( )
		
		if om_submit:
			try:
				fetcher = OpenWeather( )
				
				result = fetcher.fetch(
					latitude=float( latitude ),
					longitude=float( longitude ),
					days=int( days )
				)
				
				if result:
					om_output.text_area(
						label="Forecast Data",
						value=str( result ),
						height=300
					)
				else:
					om_output.info( "No data returned." )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------------------------------
	# Simbad Fetcher
	# -------------------------------
	with st.expander( "Simbad", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			simbad_query = st.text_area(
				label="Astronomical Object Query",
				value="",
				height=120,
				key="simbad_query"
			)
			
			btn_col1, btn_col2 = st.columns( 2 )
			
			with btn_col1:
				simbad_submit = st.button( "Submit", key="simbad_submit" )
			
			with btn_col2:
				simbad_clear = st.button( "Clear", key="simbad_clear" )
		
		with col_right:
			simbad_output = st.empty( )
		
		if simbad_clear:
			st.session_state[ "simbad_query" ] = ""
			simbad_output.empty( )
		
		if simbad_submit:
			try:
				fetcher = SimbadFetcher( )  # class name as defined in fetchers.py
				result = fetcher.fetch( simbad_query )
				
				if result:
					simbad_output.text_area(
						label="Result",
						value=str( result ),
						height=300
					)
				else:
					simbad_output.info( "No results returned." )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------------------------------
	# GoogleSearch Fetcher
	# -------------------------------
	with st.expander( "GoogleSearch", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			google_query = st.text_area(
				label="Query",
				value="",
				height=150,
				key="googlesearch_query"
			)
			
			google_num_results = st.number_input(
				label="Number of Results",
				min_value=1,
				max_value=50,
				value=10,
				step=1,
				key="googlesearch_num_results"
			)
			
			btn_col1, btn_col2 = st.columns( 2 )
			
			with btn_col1:
				google_submit = st.button( "Submit", key="googlesearch_submit" )
			
			with btn_col2:
				google_clear = st.button( "Clear", key="googlesearch_clear" )
		
		with col_right:
			google_output = st.empty()
		
		if google_clear:
			st.session_state[ "googlesearch_query" ] = ""
			st.session_state[ "googlesearch_num_results" ] = 10
			google_output.empty( )
		
		if google_submit:
			try:
				# API key is expected via sidebar / environment
				fetcher = GoogleSearch( )
				
				result = fetcher.fetch(
					keywords=google_query,
					results=int( google_num_results )
				)

				results_text = render_google_results( result )
				st.session_state[ "google_results_text" ] = results_text
				if not result:
					google_output.info( "No results returned." )
				else:
					google_output.text_area(
						label="Results",
						value=render_google_results( result ),
						height=300
					)
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------------------------------
	# GoogleMaps Fetcher
	# -------------------------------
	with st.expander( "GoogleMaps", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			gm_query = st.text_area(
				label="Query",
				value="",
				height=150,
				key="googlemaps_query"
			)
			
			gm_radius = st.number_input(
				label="Radius (meters)",
				min_value=1,
				max_value=50000,
				value=5000,
				step=100,
				key="googlemaps_radius"
			)
			
			btn_col1, btn_col2 = st.columns( 2 )
			
			with btn_col1:
				gm_submit = st.button( "Submit", key="googlemaps_submit" )
			
			with btn_col2:
				gm_clear = st.button( "Clear", key="googlemaps_clear" )
		
		with col_right:
			gm_output = st.empty( )
		
		if gm_clear:
			st.session_state[ "googlemaps_query" ] = ""
			st.session_state[ "googlemaps_radius" ] = 5000
			gm_output.empty( )
		
		if gm_submit:
			try:
				# API key expected via sidebar / environment
				fetcher = GoogleMaps( )
				
				result = fetcher.fetch(
					query=gm_query,
					radius=int( gm_radius )
				)
				
				if not result:
					gm_output.info( "No results returned." )
				else:
					gm_output.text_area(
						label="Results",
						value=str( result ),
						height=300
					)
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------------------------------
	# GoogleWeather Fetcher
	# -------------------------------
	with st.expander( "GoogleWeather", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			gw_location = st.text_area(
				label="Location",
				value="",
				height=120,
				key="googleweather_location"
			)
			
			btn_col1, btn_col2 = st.columns( 2 )
			
			with btn_col1:
				gw_submit = st.button( "Submit", key="googleweather_submit" )
			
			with btn_col2:
				gw_clear = st.button( "Clear", key="googleweather_clear" )
		
		with col_right:
			gw_output = st.empty( )
		
		if gw_clear:
			st.session_state[ "googleweather_location" ] = ""
			gw_output.empty( )
		
		if gw_submit:
			try:
				# API key expected via sidebar / environment
				fetcher = GoogleWeather( )
				
				result = fetcher.fetch(
					location=gw_location
				)
				
				if not result:
					gw_output.info( "No results returned." )
				else:
					gw_output.text_area(
						label="Results",
						value=str( result ),
						height=300
					)
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------------------------------
	# NavalObservatory Fetcher
	# -------------------------------
	with st.expander( "NavalObservatory", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			naval_query = st.text_area(
				label="Query",
				value="",
				height=150,
				key="navalobservatory_query"
			)
			
			btn_col1, btn_col2 = st.columns( 2 )
			
			with btn_col1:
				naval_submit = st.button( "Submit", key="navalobservatory_submit" )
			
			with btn_col2:
				naval_clear = st.button( "Clear", key="navalobservatory_clear" )
		
		with col_right:
			naval_output = st.empty( )
		
		if naval_clear:
			st.session_state[ "navalobservatory_query" ] = ""
			naval_output.empty( )
		
		if naval_submit:
			try:
				# API key (if required) is expected via sidebar / environment
				fetcher = NavalObservatory( )
				
				result = fetcher.fetch( naval_query )
				
				if not result:
					naval_output.info( "No results returned." )
				else:
					naval_output.text_area(
						label="Results",
						value=str( result ),
						height=300
					)
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------------------------------
	# SatelliteCenter Fetcher
	# -------------------------------
	with st.expander( "SatelliteCenter", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			satellite_query = st.text_area(
				label="Query",
				value="",
				height=150,
				key="satellitecenter_query"
			)
			
			btn_col1, btn_col2 = st.columns( 2 )
			
			with btn_col1:
				satellite_submit = st.button( "Submit", key="satellitecenter_submit" )
			
			with btn_col2:
				satellite_clear = st.button( "Clear", key="satellitecenter_clear" )
		
		with col_right:
			satellite_output = st.empty( )
		
		if satellite_clear:
			st.session_state[ "satellitecenter_query" ] = ""
			satellite_output.empty( )
		
		if satellite_submit:
			try:
				# API key (if required) is expected via sidebar / environment
				fetcher = SatelliteCenter( )
				
				result = fetcher.fetch( satellite_query )
				
				if not result:
					satellite_output.info( "No results returned." )
				else:
					satellite_output.text_area(
						label="Results",
						value=str( result ),
						height=300
					)
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------------------------------
	# NearbyObjects Fetcher
	# -------------------------------
	with st.expander( "NearbyObjects", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
	
	with col_left:
		nearby_query = st.text_area(
			label="Query",
			value="",
			height=150,
			key="nearbyobjects_query"
		)
		
		btn_col1, btn_col2 = st.columns( 2 )
		
		with btn_col1:
			nearby_submit = st.button( "Submit", key="nearbyobjects_submit" )
		
		with btn_col2:
			nearby_clear = st.button( "Clear", key="nearbyobjects_clear" )
	
	with col_right:
		nearby_output = st.empty( )
	
	if nearby_clear:
		st.session_state[ "nearbyobjects_query" ] = ""
		nearby_output.empty( )
	
	if nearby_submit:
		try:
			# API key (if required) is expected via sidebar / environment
			fetcher = NearbyObjects( )
			
			result = fetcher.fetch( nearby_query )
			
			if not result:
				nearby_output.info( "No results returned." )
			else:
				nearby_output.text_area(
					label="Results",
					value=str( result ),
					height=300
				)
		
		except Exception as exc:
			st.error( str( exc ) )
	
	# -------------------------------
	# OpenScience Fetcher
	# -------------------------------
	with st.expander( "OpenScience", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
	
	with col_left:
		openscience_query = st.text_area(
			label="Query",
			value="",
			height=150,
			key="openscience_query"
		)
		
		btn_col1, btn_col2 = st.columns( 2 )
		
		with btn_col1:
			openscience_submit = st.button( "Submit", key="openscience_submit" )
		
		with btn_col2:
			openscience_clear = st.button( "Clear", key="openscience_clear" )
	
	with col_right:
		openscience_output = st.empty( )
	
	if openscience_clear:
		st.session_state[ "openscience_query" ] = ""
		openscience_output.empty( )
	
	if openscience_submit:
		try:
			# API key (if required) is expected via sidebar / environment
			fetcher = OpenScience( )
			
			result = fetcher.fetch( openscience_query )
			
			if not result:
				openscience_output.info( "No results returned." )
			else:
				openscience_output.text_area(
					label="Results",
					value=str( result ),
					height=300
				)
		
		except Exception as exc:
			st.error( str( exc ) )
	
	# -------------------------------
	# EarthObservatory Fetcher
	# -------------------------------
	with st.expander( "EarthObservatory", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
	
	with col_left:
		earth_query = st.text_area(
			label="Query",
			value="",
			height=150,
			key="earthobservatory_query"
		)
		
		btn_col1, btn_col2 = st.columns( 2 )
		
		with btn_col1:
			earth_submit = st.button( "Submit", key="earthobservatory_submit" )
		
		with btn_col2:
			earth_clear = st.button( "Clear", key="earthobservatory_clear" )
	
	with col_right:
		earth_output = st.empty( )
	
	if earth_clear:
		st.session_state[ "earthobservatory_query" ] = ""
		earth_output.empty( )
	
	if earth_submit:
		try:
			# API key (if required) is expected via sidebar / environment
			fetcher = EarthObservatory( )
			
			result = fetcher.fetch( earth_query )
			
			if not result:
				earth_output.info( "No results returned." )
			else:
				earth_output.text_area(
					label="Results",
					value=str( result ),
					height=300
				)
		
		except Exception as exc:
			st.error( str( exc ) )
	
	# -------------------------------
	# SpaceWeather Fetcher
	# -------------------------------
	with st.expander( "SpaceWeather", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
	
	with col_left:
		spaceweather_query = st.text_area(
			label="Query",
			value="",
			height=150,
			key="spaceweather_query"
		)
		
		btn_col1, btn_col2 = st.columns( 2 )
		
		with btn_col1:
			spaceweather_submit = st.button( "Submit", key="spaceweather_submit" )
		
		with btn_col2:
			spaceweather_clear = st.button( "Clear", key="spaceweather_clear" )
	
	with col_right:
		spaceweather_output = st.empty( )
	
	if spaceweather_clear:
		st.session_state[ "spaceweather_query" ] = ""
		spaceweather_output.empty( )
	
	if spaceweather_submit:
		try:
			# API key (if required) is expected via sidebar / environment
			fetcher = SpaceWeather( )
			
			result = fetcher.fetch( spaceweather_query )
			
			if not result:
				spaceweather_output.info( "No results returned." )
			else:
				spaceweather_output.text_area(
					label="Results",
					value=str( result ),
					height=300
				)
		
		except Exception as exc:
			st.error( str( exc ) )
	
	# -------------------------------
	# AstroCatalog Fetcher
	# -------------------------------
	with st.expander( "AstroCatalog", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
	
	with col_left:
		astro_query = st.text_area(
			label="Query",
			value="",
			height=150,
			key="astrocatalog_query"
		)
		
		btn_col1, btn_col2 = st.columns( 2 )
		
		with btn_col1:
			astro_submit = st.button( "Submit", key="astrocatalog_submit" )
		
		with btn_col2:
			astro_clear = st.button( "Clear", key="astrocatalog_clear" )
	
	with col_right:
		astro_output = st.empty( )
	
	if astro_clear:
		st.session_state[ "astrocatalog_query" ] = ""
		astro_output.empty( )
	
	if astro_submit:
		try:
			# API key (if required) is expected via sidebar / environment
			fetcher = AstroCatalog( )
			
			result = fetcher.fetch( astro_query )
			
			if not result:
				astro_output.info( "No results returned." )
			else:
				astro_output.text_area(
					label="Results",
					value=str( result ),
					height=300
				)
		
		except Exception as exc:
			st.error( str( exc ) )
	
	# -------------------------------
	# AstroQuery Fetcher
	# -------------------------------
	with st.expander( "AstroQuery", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
	
	with col_left:
		astroquery_query = st.text_area(
			label="Query",
			value="",
			height=150,
			key="astroquery_query"
		)
		
		btn_col1, btn_col2 = st.columns( 2 )
		
		with btn_col1:
			astroquery_submit = st.button( "Submit", key="astroquery_submit" )
		
		with btn_col2:
			astroquery_clear = st.button( "Clear", key="astroquery_clear" )
	
	with col_right:
		astroquery_output = st.empty( )
	
	if astroquery_clear:
		st.session_state[ "astroquery_query" ] = ""
		astroquery_output.empty( )
	
	if astroquery_submit:
		try:
			# API key (if required) is expected via sidebar / environment
			fetcher = AstroQuery( )
			
			result = fetcher.fetch( astroquery_query )
			
			if not result:
				astroquery_output.info( "No results returned." )
			else:
				astroquery_output.text_area(
					label="Results",
					value=str( result ),
					height=300
				)
		
		except Exception as exc:
			st.error( str( exc ) )
	
	# -------------------------------
	# StarMap Fetcher
	# -------------------------------
	with st.expander( "StarMap", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			starmap_query = st.text_area(
				label="Query",
				value="",
				height=150,
				key="starmap_query"
			)
			
			btn_col1, btn_col2 = st.columns( 2 )
			
			with btn_col1:
				starmap_submit = st.button( "Submit", key="starmap_submit" )
			
			with btn_col2:
				starmap_clear = st.button( "Clear", key="starmap_clear" )
		
		with col_right:
			starmap_output = st.empty( )
		
		if starmap_clear:
			st.session_state[ "starmap_query" ] = ""
			starmap_output.empty( )
		
		if starmap_submit:
			try:
				# API key (if required) is expected via sidebar / environment
				fetcher = StarMap( )
				
				result = fetcher.fetch( starmap_query )
				
				if not result:
					starmap_output.info( "No results returned." )
				else:
					starmap_output.text_area(
						label="Results",
						value=str( result ),
						height=300
					)
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------------------------------
	# GovData Fetcher
	# -------------------------------
	with st.expander( "GovData", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			govdata_query = st.text_area(
				label="Query",
				value="",
				height=150,
				key="govdata_query"
			)
			
			btn_col1, btn_col2 = st.columns( 2 )
			
			with btn_col1:
				govdata_submit = st.button( "Submit", key="govdata_submit" )
			
			with btn_col2:
				govdata_clear = st.button( "Clear", key="govdata_clear" )
		
		with col_right:
			govdata_output = st.empty( )
		
		if govdata_clear:
			st.session_state[ "govdata_query" ] = ""
			govdata_output.empty( )
		
		if govdata_submit:
			try:
				# API key (if required) is expected via sidebar / environment
				fetcher = GovData( )
				
				result = fetcher.search_criteria( govdata_query )
				
				if not result:
					govdata_output.info( "No results returned." )
				else:
					govdata_output.text_area(
						label="Results",
						value=str( result ),
						height=300
					)
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------------------------------
	# StarChart Fetcher
	# -------------------------------
	with st.expander( "StarChart", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			starchart_query = st.text_area(
				label="Query",
				value="",
				height=150,
				key="starchart_query"
			)
			
			btn_col1, btn_col2 = st.columns( 2 )
			
			with btn_col1:
				starchart_submit = st.button( "Submit", key="starchart_submit" )
			
			with btn_col2:
				starchart_clear = st.button( "Clear", key="starchart_clear" )
		
		with col_right:
			starchart_output = st.empty( )
		
		if starchart_clear:
			st.session_state[ "starchart_query" ] = ""
			starchart_output.empty( )
		
		if starchart_submit:
			try:
				# API key (if required) is expected via sidebar / environment
				fetcher = StarChart( )
				
				result = fetcher.fetch( starchart_query )
				
				if not result:
					starchart_output.info( "No results returned." )
				else:
					starchart_output.text_area(
						label="Results",
						value=str( result ),
						height=300
					)
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------------------------------
	# Congress Fetcher
	# -------------------------------
	with st.expander( "Congress", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			congress_query = st.text_area(
				label="Query",
				value="",
				height=150,
				key="congress_query"
			)
			
			btn_col1, btn_col2 = st.columns( 2 )
			
			with btn_col1:
				congress_submit = st.button( "Submit", key="congress_submit" )
			
			with btn_col2:
				congress_clear = st.button( "Clear", key="congress_clear" )
		
		with col_right:
			congress_output = st.empty( )
		
		if congress_clear:
			st.session_state[ "congress_query" ] = ""
			congress_output.empty( )
		
		if congress_submit:
			try:
				# API key (if required) is expected via sidebar / environment
				fetcher = Congress( )
				
				result = fetcher.fetch( congress_query )
				
				if not result:
					congress_output.info( "No results returned." )
				else:
					congress_output.text_area(
						label="Results",
						value=str( result ),
						height=300
					)
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------------------------------
	# InternetArchive Fetcher
	# -------------------------------
	with st.expander( "InternetArchive", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			ia_query = st.text_area(
				label="Query",
				value="",
				height=150,
				key="internetarchive_query"
			)
			
			btn_col1, btn_col2 = st.columns( 2 )
			
			with btn_col1:
				ia_submit = st.button( "Submit", key="internetarchive_submit" )
			
			with btn_col2:
				ia_clear = st.button( "Clear", key="internetarchive_clear" )
		
		with col_right:
			ia_output = st.empty( )
		
		if ia_clear:
			st.session_state[ "internetarchive_query" ] = ""
			ia_output.empty( )
		
		if ia_submit:
			try:
				# API key (if required) is expected via sidebar / environment
				fetcher = InternetArchive( )
				
				result = fetcher.fetch( ia_query )
				
				if not result:
					ia_output.info( "No results returned." )
				else:
					ia_output.text_area(
						label="Results",
						value=str( result ),
						height=300
					)
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# -------------------------------
	# OpenWeather Fetcher
	# -------------------------------
	with st.expander( "OpenWeather", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			openweather_query = st.text_area(
				label="Location",
				value="",
				height=150,
				key="openweather_query"
			)
			
			btn_col1, btn_col2 = st.columns( 2 )
			
			with btn_col1:
				openweather_submit = st.button( "Submit", key="openweather_submit" )
			
			with btn_col2:
				openweather_clear = st.button( "Clear", key="openweather_clear" )
		
		with col_right:
			openweather_output = st.empty( )
		
		if openweather_clear:
			st.session_state[ "openweather_query" ] = ""
			openweather_output.empty( )
		
		if openweather_submit:
			try:
				# API key (if required) is expected via sidebar / environment
				fetcher = OpenWeather( )
				
				result = fetcher.fetch( openweather_query )
				
				if not result:
					openweather_output.info( "No results returned." )
				else:
					openweather_output.text_area(
						label="Results",
						value=str( result ),
						height=300
					)
			
			except Exception as exc:
				st.error( str( exc ) )

# ======================================================================================
# DATA TAB — SQLite inspection
# ======================================================================================

with tab_data:
	st.subheader( "Data" )
	
	conn = sqlite3.connect( f"file:{DB_PATH.as_posix( )}?mode=ro", uri=True )
	cur = conn.cursor( )
	cur.execute( "SELECT name FROM sqlite_master WHERE type='table';" )
	tables = [ r[ 0 ] for r in cur.fetchall( ) ]
	conn.close( )
	
	st.write( tables )

# --------------------------------------------------------------------------------------
# Chat Tab
# --------------------------------------------------------------------------------------
with tab_chat:
	with st.expander( "Chat", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			chat_prompt = st.text_area( "Prompt", value="", height=180, key="chat_prompt" )
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				chat_model = _model_selector(
					key_prefix="chat",
					label="Model",
					options=[ "gpt-4o-mini",
					          "gpt-4.1-mini",
					          "gpt-4.1",
					          "o3-mini",
					          "Custom..." ],
					default_model="gpt-4o-mini",
				)
			
			with p_row1[ 1 ]:
				chat_temperature = st.slider(
					"Temperature",
					min_value=0.0,
					max_value=2.0,
					value=0.7,
					step=0.05,
					key="chat_temperature",
				)
			
			with p_row2[ 0 ]:
				chat_max_tokens = st.number_input(
					"Max Tokens",
					min_value=1,
					max_value=32768,
					value=1024,
					step=1,
					key="chat_max_tokens",
				)
			
			with p_row2[ 1 ]:
				chat_top_p = st.slider(
					"Top-p",
					min_value=0.0,
					max_value=1.0,
					value=1.0,
					step=0.01,
					key="chat_top_p",
				)
			
			with p_row3[ 0 ]:
				chat_seed = st.number_input(
					"Seed",
					min_value=0,
					max_value=2_147_483_647,
					value=0,
					step=1,
					key="chat_seed",
				)
			
			with p_row3[ 1 ]:
				chat_json_mode = st.checkbox( "JSON Mode", value=False, key="chat_json_mode" )
			
			chat_system = st.text_area(
				"System",
				value="",
				height=100,
				key="chat_system",
			)
			
			btn_row = st.columns( 2 )
			with btn_row[ 0 ]:
				chat_submit = st.button( "Submit", key="chat_submit" )
			with btn_row[ 1 ]:
				chat_clear = st.button( "Clear", key="chat_clear" )
		
		with col_right:
			chat_output = st.empty( )
		
		if chat_clear:
			st.session_state[ "chat_prompt" ] = ""
			st.session_state[ "chat_system" ] = ""
			chat_output.empty( )
		
		if chat_submit:
			try:
				fetcher = Chat( )
				params = {
						"model": chat_model,
						"temperature": float( chat_temperature ),
						"max_tokens": int( chat_max_tokens ),
						"top_p": float( chat_top_p ),
						"seed": int( chat_seed ) if int( chat_seed ) > 0 else None,
						"system": chat_system if chat_system.strip( ) else None,
						"response_format": ("json" if chat_json_mode else None),
				}
				# Remove None to avoid surprising providers
				params = { k: v for k, v in params.items( ) if v is not None }
				
				result = _invoke_provider( fetcher, chat_prompt, params )
				_render_output( chat_output, result )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# --------------------------------------------------------------------------------------
	# Groq
	# --------------------------------------------------------------------------------------
	with st.expander( "Groq", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			groq_prompt = st.text_area( "Prompt", value="", height=180, key="groq_prompt_chat" )
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				groq_model = _model_selector(
					key_prefix="groq",
					label="Model",
					options=[
							"llama3-70b-8192",
							"llama3-8b-8192",
							"mixtral-8x7b-32768",
							"Custom...",
					],
					default_model="llama3-70b-8192",
				)
			
			with p_row1[ 1 ]:
				groq_temperature = st.slider(
					"Temperature",
					min_value=0.0,
					max_value=2.0,
					value=0.7,
					step=0.05,
					key="groq_temperature_chat",
				)
			
			with p_row2[ 0 ]:
				groq_max_tokens = st.number_input(
					"Max Tokens",
					min_value=1,
					max_value=32768,
					value=1024,
					step=1,
					key="groq_max_tokens_chat",
				)
			
			with p_row2[ 1 ]:
				groq_top_p = st.slider(
					"Top-p",
					min_value=0.0,
					max_value=1.0,
					value=1.0,
					step=0.01,
					key="groq_top_p_chat",
				)
			
			with p_row3[ 0 ]:
				groq_stop = st.text_area(
					"Stop Sequences (one per line)",
					value="",
					height=80,
					key="groq_stop_chat",
				)
			
			with p_row3[ 1 ]:
				groq_stream = st.checkbox( "Stream", value=False, key="groq_stream_chat" )
			
			btn_row = st.columns( 2 )
			with btn_row[ 0 ]:
				groq_submit = st.button( "Submit", key="groq_submit_chat" )
			with btn_row[ 1 ]:
				groq_clear = st.button( "Clear", key="groq_clear_chat" )
		
		with col_right:
			groq_output = st.empty( )
		
		if groq_clear:
			st.session_state[ "groq_prompt_chat" ] = ""
			st.session_state[ "groq_stop_chat" ] = ""
			groq_output.empty( )
		
		if groq_submit:
			try:
				fetcher = Groq( )
				stop_lines = [ s.strip( ) for s in (groq_stop or "").splitlines( ) if s.strip( ) ]
				params = {
						"model": groq_model,
						"temperature": float( groq_temperature ),
						"max_tokens": int( groq_max_tokens ),
						"top_p": float( groq_top_p ),
						"stop": stop_lines if stop_lines else None,
						"stream": bool( groq_stream ),
				}
				params = { k: v for k, v in params.items( ) if v is not None }
				
				result = _invoke_provider( fetcher, groq_prompt, params )
				_render_output( groq_output, result )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# --------------------------------------------------------------------------------------
	# Claude (Anthropic)
	# --------------------------------------------------------------------------------------
	with st.expander( "Claude", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			claude_prompt = st.text_area( "Prompt", value="", height=180,
				key="claude_prompt_chat" )
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				claude_model = _model_selector(
					key_prefix="claude",
					label="Model",
					options=[
							"claude-3-5-sonnet-latest",
							"claude-3-5-haiku-latest",
							"claude-3-opus-latest",
							"Custom...",
					],
					default_model="claude-3-5-sonnet-latest",
				)
			
			with p_row1[ 1 ]:
				claude_temperature = st.slider(
					"Temperature",
					min_value=0.0,
					max_value=1.0,
					value=0.7,
					step=0.05,
					key="claude_temperature_chat",
				)
			
			with p_row2[ 0 ]:
				claude_max_tokens = st.number_input(
					"Max Tokens",
					min_value=1,
					max_value=8192,
					value=1024,
					step=1,
					key="claude_max_tokens_chat",
				)
			
			with p_row2[ 1 ]:
				claude_top_p = st.slider(
					"Top-p",
					min_value=0.0,
					max_value=1.0,
					value=1.0,
					step=0.01,
					key="claude_top_p_chat",
				)
			
			with p_row3[ 0 ]:
				claude_top_k = st.number_input(
					"Top-k",
					min_value=0,
					max_value=500,
					value=0,
					step=1,
					key="claude_top_k_chat",
				)
			
			with p_row3[ 1 ]:
				claude_stop = st.text_area(
					"Stop Sequences (one per line)",
					value="",
					height=80,
					key="claude_stop_chat",
				)
			
			claude_system = st.text_area(
				"System",
				value="",
				height=100,
				key="claude_system_chat",
			)
			
			btn_row = st.columns( 2 )
			with btn_row[ 0 ]:
				claude_submit = st.button( "Submit", key="claude_submit_chat" )
			with btn_row[ 1 ]:
				claude_clear = st.button( "Clear", key="claude_clear_chat" )
		
		with col_right:
			claude_output = st.empty( )
		
		if claude_clear:
			st.session_state[ "claude_prompt_chat" ] = ""
			st.session_state[ "claude_stop_chat" ] = ""
			st.session_state[ "claude_system_chat" ] = ""
			claude_output.empty( )
		
		if claude_submit:
			try:
				fetcher = Claude( )
				stop_lines = [ s.strip( ) for s in (claude_stop or "").splitlines( ) if s.strip(
				) ]
				params = {
						"model": claude_model,
						"temperature": float( claude_temperature ),
						"max_tokens": int( claude_max_tokens ),
						"top_p": float( claude_top_p ),
						"top_k": int( claude_top_k ) if int( claude_top_k ) > 0 else None,
						"stop_sequences": stop_lines if stop_lines else None,
						"system": claude_system if claude_system.strip( ) else None,
				}
				params = { k: v for k, v in params.items( ) if v is not None }
				
				result = _invoke_provider( fetcher, claude_prompt, params )
				_render_output( claude_output, result )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# --------------------------------------------------------------------------------------
	# Gemini
	# --------------------------------------------------------------------------------------
	with st.expander( "Gemini", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			gemini_prompt = st.text_area( "Prompt", value="", height=180,
				key="gemini_prompt_chat" )
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				gemini_model = _model_selector(
					key_prefix="gemini",
					label="Model",
					options=[
							"gemini-1.5-pro",
							"gemini-1.5-flash",
							"gemini-2.0-flash",
							"Custom...",
					],
					default_model="gemini-1.5-pro",
				)
			
			with p_row1[ 1 ]:
				gemini_temperature = st.slider(
					"Temperature",
					min_value=0.0,
					max_value=2.0,
					value=0.7,
					step=0.05,
					key="gemini_temperature_chat",
				)
			
			with p_row2[ 0 ]:
				gemini_max_tokens = st.number_input(
					"Max Tokens",
					min_value=1,
					max_value=32768,
					value=1024,
					step=1,
					key="gemini_max_tokens_chat",
				)
			
			with p_row2[ 1 ]:
				gemini_top_p = st.slider(
					"Top-p",
					min_value=0.0,
					max_value=1.0,
					value=1.0,
					step=0.01,
					key="gemini_top_p_chat",
				)
			
			with p_row3[ 0 ]:
				gemini_top_k = st.number_input(
					"Top-k",
					min_value=0,
					max_value=500,
					value=0,
					step=1,
					key="gemini_top_k_chat",
				)
			
			with p_row3[ 1 ]:
				gemini_candidate_count = st.number_input(
					"Candidates",
					min_value=1,
					max_value=8,
					value=1,
					step=1,
					key="gemini_candidate_count_chat",
				)
			
			gemini_system = st.text_area(
				"System",
				value="",
				height=100,
				key="gemini_system_chat",
			)
			
			btn_row = st.columns( 2 )
			with btn_row[ 0 ]:
				gemini_submit = st.button( "Submit", key="gemini_submit_chat" )
			with btn_row[ 1 ]:
				gemini_clear = st.button( "Clear", key="gemini_clear_chat" )
		
		with col_right:
			gemini_output = st.empty( )
		
		if gemini_clear:
			st.session_state[ "gemini_prompt_chat" ] = ""
			st.session_state[ "gemini_system_chat" ] = ""
			gemini_output.empty( )
		
		if gemini_submit:
			try:
				fetcher = Gemini( )
				params = {
						"model": gemini_model,
						"temperature": float( gemini_temperature ),
						"max_tokens": int( gemini_max_tokens ),
						"top_p": float( gemini_top_p ),
						"top_k": int( gemini_top_k ) if int( gemini_top_k ) > 0 else None,
						"candidate_count": int( gemini_candidate_count ),
						"system": gemini_system if gemini_system.strip( ) else None,
				}
				params = { k: v for k, v in params.items( ) if v is not None }
				
				result = _invoke_provider( fetcher, gemini_prompt, params )
				_render_output( gemini_output, result )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# --------------------------------------------------------------------------------------
	# Mistral
	# --------------------------------------------------------------------------------------
	with st.expander( "Mistral", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			mistral_prompt = st.text_area( "Prompt", value="", height=180,
				key="mistral_prompt_chat" )
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				mistral_model = _model_selector(
					key_prefix="mistral",
					label="Model",
					options=[
							"mistral-large-latest",
							"mistral-medium-latest",
							"mistral-small-latest",
							"open-mistral-7b",
							"Custom...",
					],
					default_model="mistral-large-latest",
				)
			
			with p_row1[ 1 ]:
				mistral_temperature = st.slider(
					"Temperature",
					min_value=0.0,
					max_value=2.0,
					value=0.7,
					step=0.05,
					key="mistral_temperature_chat",
				)
			
			with p_row2[ 0 ]:
				mistral_max_tokens = st.number_input(
					"Max Tokens",
					min_value=1,
					max_value=32768,
					value=1024,
					step=1,
					key="mistral_max_tokens_chat",
				)
			
			with p_row2[ 1 ]:
				mistral_top_p = st.slider(
					"Top-p",
					min_value=0.0,
					max_value=1.0,
					value=1.0,
					step=0.01,
					key="mistral_top_p_chat",
				)
			
			with p_row3[ 0 ]:
				mistral_seed = st.number_input(
					"Seed",
					min_value=0,
					max_value=2_147_483_647,
					value=0,
					step=1,
					key="mistral_seed_chat",
				)
			
			with p_row3[ 1 ]:
				mistral_safe_mode = st.checkbox( "Safe Mode", value=False,
					key="mistral_safe_mode_chat" )
			
			mistral_system = st.text_area(
				"System",
				value="",
				height=100,
				key="mistral_system_chat",
			)
			
			btn_row = st.columns( 2 )
			with btn_row[ 0 ]:
				mistral_submit = st.button( "Submit", key="mistral_submit_chat" )
			with btn_row[ 1 ]:
				mistral_clear = st.button( "Clear", key="mistral_clear_chat" )
		
		with col_right:
			mistral_output = st.empty( )
		
		if mistral_clear:
			st.session_state[ "mistral_prompt_chat" ] = ""
			st.session_state[ "mistral_system_chat" ] = ""
			mistral_output.empty( )
		
		if mistral_submit:
			try:
				fetcher = Mistral( )
				params = {
						"model": mistral_model,
						"temperature": float( mistral_temperature ),
						"max_tokens": int( mistral_max_tokens ),
						"top_p": float( mistral_top_p ),
						"seed": int( mistral_seed ) if int( mistral_seed ) > 0 else None,
						"safe_mode": bool( mistral_safe_mode ),
						"system": mistral_system if mistral_system.strip( ) else None,
				}
				params = { k: v for k, v in params.items( ) if v is not None }
				
				result = _invoke_provider( fetcher, mistral_prompt, params )
				_render_output( mistral_output, result )
			
			except Exception as exc:
				st.error( str( exc ) )
