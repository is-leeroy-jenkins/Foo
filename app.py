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
from loaders import PdfLoader, WordLoader, ExcelLoader
import fetchers
from fetchers import (
	Wikipedia, TheNews, SatelliteCenter, Simbad, WebFetcher,
	GoogleWeather, Grokipedia, OpenWeather, NavalObservatory,
	GoogleSearch, GoogleDrive, GoogleMaps, NearbyObjects, OpenScience,
	EarthObservatory, SpaceWeather, AstroCatalog, AstroQuery, StarMap,
	GovData, Congress, InternetArchive, OpenAI, Claude, GrokipediaClient,
	Groq, Mistral, Gemini, StarChart
)

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
					st.text_area( "", value=(doc.page_content or ""), height=300 )
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

# ======================================================================================
# GOOGLE SEARCH FORMATTER
# ======================================================================================

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

# ======================================================================================
# Streamlit Setup
# ======================================================================================

st.set_page_config( page_title='Foo', layout='wide', page_icon=config.FAVICON )

col_left, col_center, col_right = st.columns( [ 1, 2, 1 ], vertical_alignment='top' )

with col_left:
    st.image( 'resources/images/foo_logo.png', width=80 )

# ======================================================================================
# Sidebar — Global API keys only
# ======================================================================================

with st.sidebar:
	st.header( "Configuration" )
	
	with st.expander( "API Keys", expanded=False ):
		for attr in dir( config ):
			if attr.endswith( "_API_KEY" ) or attr.endswith( "_TOKEN" ):
				current = getattr( config, attr, "" ) or ""
				val = st.text_input( attr, value=current, type="password" )
				if val:
					os.environ[ attr ] = val

# ======================================================================================
# Tabs
# ======================================================================================

tab_loaders, tab_scrapers, tab_fetchers, tab_chat, tab_maps, tab_lockers, tab_data = st.tabs(
	[ "Loaders",
	  "Scrapers",
	  "Fetchers",
	  "Yappers",
	  "Mappers",
	  "Lockers",
	  "Data" ] )

# ======================================================================================
# SCRAPERS TAB — unchanged except for state fixes later
# ======================================================================================

with tab_loaders:
	loader = scrapers.WebExtractor( )
	urls_raw = st.text_area( "URLs", height=40 )
	
	col1, col2, col3, col4, col5, col6 = st.columns( 6 )
	with col1:
		do_pdf = st.checkbox( label='PDF', key='pdf_cb' )
	with col2:
		do_word = st.checkbox( label="Word", key='word_cb'  )
	with col3:
		do_excel = st.checkbox( label="Excel", key='excel_cb' )
	with col4:
		do_markdown = st.checkbox( label="Markdown", key='markdown_cb'  )
	with col5:
		do_powerpoint = st.checkbox( label="Powerpoint", key='powerpoint_cb'  )
	with col6:
		do_youtube = st.checkbox( label="Youtube", key='youtube_cb'  )
	
	if st.button( "Load" ):
		urls = [ u.strip( ) for u in urls_raw.splitlines( ) if u.strip( ) ]
		if not urls:
			st.warning( "No URLs provided." )
		else:
			for url in urls:
				st.markdown( f"### {url}" )
				output = { }
				
				try:
					if do_pdf:
						output[ "pdf" ] = extractor.scrape_links( url )
					if do_word:
						output[ "word" ] = extractor.scrape_links( url )
					if do_excel:
						output[ "excel" ] = extractor.scrape_tables( url )
					if do_markdown:
						output[ "markdown" ] = extractor.scrape_( url )
					if do_powerpoint:
						output[ "powerpoint" ] = extractor.scrape_( url )
					if do_youtube:
						output[ "youtube" ] = extractor.scrape_( url )
					
					st.json( output )
				
				except Exception as exc:
					st.error( "Error" )
					st.exception( exc )
					
# ======================================================================================
# Scrapers
# ======================================================================================
with tab_scrapers:
	col_left, col_right = st.columns([1, 2], border=True)
	with col_left:
		target_url = st.text_input(
			"Target URL",
			placeholder="https://example.com",
			key="webfetcher_url"
		)

		st.markdown("#### Extraction Options")

		fetcher = WebFetcher()

		# ------------------------------------------------------------------
		# Discover scrape* methods from WebFetcher.__dir__()
		# ------------------------------------------------------------------
		raw_names = [
			name for name in fetcher.__dir__()
			if name.startswith("scrape")
		]

		# Deduplicate + filter out invalid / misspelled names
		VALID_SCRAPERS: dict[str, str] = {
			"scrape_images": "Images",
			"scrape_hyperlinks": "Hyperlinks",
			"scrape_blockquotes": "Blockquotes",
			"scrape_sections": "Sections",
			"scrape_divisions": "Divisions",
			"scrape_tables": "Tables",
			"scrape_lists": "Lists",
			"scrape_paragraphs": "Paragraphs",
		}

		available_methods: dict[str, callable] = {}

		for name in raw_names:
			if name in VALID_SCRAPERS and hasattr(fetcher, name):
				available_methods[name] = getattr(fetcher, name)

		selected_methods: list[str] = []

		for method_name, label in VALID_SCRAPERS.items():
			if method_name in available_methods:
				if st.checkbox(label, key=f"wf_{method_name}"):
					selected_methods.append(method_name)

		run_scraper = st.button("Run Scraper", key="webfetcher_run")

	with col_right:
		output = st.empty()

		if run_scraper:
			try:
				if not target_url:
					raise ValueError("A target URL is required.")

				if not selected_methods:
					raise ValueError("At least one scraper must be selected.")

				results: dict[str, list[str]] = {}

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
						st.markdown(f"#### {VALID_SCRAPERS[method_name]}")

						if not items:
							st.info("No results returned.")
							continue

						for idx, item in enumerate(items, start=1):
							st.write(f"{idx}. {item}")

			except Exception as exc:
				st.error( str( exc ) )



# ======================================================================================
# FETCHERS TAB — ArXiv
# ======================================================================================
with tab_fetchers:
	st.markdown( "#### Fetcher" )
	
	st.session_state.setdefault( "arxiv_input", "" )
	st.session_state.setdefault( "arxiv_results", [ ] )
	
	with st.expander( "ArXiv", expanded=True ):
		col1, col2 = st.columns( 2, border=True )
		
		with col1:
			arxiv_input = st.text_area(
				"Query",
				height=40,
				key="arxiv_input",
			)
			
			b1, b2 = st.columns( 2 )
			
			with b1:
				if st.button( "Submit", key="arxiv_submit" ):
					try:
						queries = [ q.strip( ) for q in arxiv_input.splitlines( ) if q.strip( ) ]
						if not queries:
							st.warning( "No input provided." )
						else:
							from fetchers import ArXiv
							
							f = ArXiv( )
							results = [ ]
							
							for q in queries:
								docs = f.fetch( q )
								if isinstance( docs, Document ):
									results.append( docs )
								elif isinstance( docs, list ):
									results.extend( docs )
							
							st.session_state.update( {
									"arxiv_results": results
							} )
							st.rerun( )
					
					except Exception as exc:
						st.error( "ArXiv request failed." )
						st.exception( exc )
			
			with b2:
				if st.button( "Clear", key="arxiv_clear" ):
					st.session_state.update( {
							"arxiv_input": "",
							"arxiv_results": [ ]
					} )
					st.rerun( )
		
		with col2:
			st.markdown( "Results" )
			
			if not st.session_state[ "arxiv_results" ]:
				st.text( "No results." )
			else:
				for idx, doc in enumerate( st.session_state[ "arxiv_results" ], start=1 ):
					with st.expander( f"Document {idx}", expanded=False ):
						if isinstance( doc, Document ):
							st.text_area( "Content", value=doc.page_content or "", height=300 )
							if doc.metadata:
								st.json( doc.metadata )
						else:
							st.write( doc )
		
	# ======================================================================================
	# GoogleDrive Fetcher
	# ======================================================================================
	with st.expander( "Google Drive", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			gd_query = st.text_area(
				"Google Drive Query",
				value="",
				height=40,
				key="googledrive_query"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				gd_submit = st.button( "Submit", key="googledrive_submit" )
			with b2:
				gd_clear = st.button( "Clear", key="googledrive_clear" )
		
		with col_right:
			gd_output = st.empty( )
		
		if gd_clear:
			st.session_state.update( {
					"googledrive_query": ""
			} )
			st.rerun( )
		
		if gd_submit:
			try:
				f = GoogleDrive( )
				docs = f.fetch( gd_query )
				
				if docs:
					with gd_output.container( ):
						for idx, doc in enumerate( docs, start=1 ):
							st.markdown( f"**Document {idx}**" )
							st.text_area( "", value=doc.page_content, height=200 )
				else:
					gd_output.info( "No documents returned." )
			except Exception as exc:
				st.error( str( exc ) )
	
	# ======================================================================================
	# Wikipedia Fetcher
	# ======================================================================================
	with st.expander( "Wikipedia", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			wiki_query = st.text_area(
				"Wikipedia Query",
				value="",
				height=40,
				key="wikipedia_query"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				wiki_submit = st.button( "Submit", key="wikipedia_submit" )
			with b2:
				wiki_clear = st.button( "Clear", key="wikipedia_clear" )
		
		with col_right:
			wiki_output = st.empty( )
		
		if wiki_clear:
			st.session_state.update( {
					"wikipedia_query": ""
			} )
			st.rerun( )
		
		if wiki_submit:
			try:
				f = Wikipedia( )
				docs = f.fetch( wiki_query )
				
				if docs:
					with wiki_output.container( ):
						for idx, doc in enumerate( docs, start=1 ):
							st.markdown( f"**Document {idx}**" )
							st.text_area( "", value=doc.page_content, height=200 )
				else:
					wiki_output.info( "No documents returned." )
			except Exception as exc:
				st.error( str( exc ) )
	
	# ======================================================================================
	# TheNews
	# ======================================================================================
	with st.expander( "The News API", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			news_query = st.text_area(
				"News Query",
				value="",
				height=40,
				key="thenews_query"
			)
			
			news_api_key = st.text_input(
				"API Key",
				value="",
				type="password",
				key="thenews_api_key"
			)
			
			news_timeout = st.number_input(
				"Timeout (seconds)",
				min_value=1,
				max_value=60,
				value=10,
				step=1,
				key="thenews_timeout"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				news_submit = st.button( "Submit", key="thenews_submit" )
			with b2:
				news_clear = st.button( "Clear", key="thenews_clear" )
		
		with col_right:
			news_output = st.empty( )
		
		if news_clear:
			st.session_state.update( {
					"thenews_query": "",
					"thenews_api_key": "",
					"thenews_timeout": 10
			} )
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
				
				if result and getattr( result, "text", None ):
					news_output.text_area( "Result", value=result.text, height=300 )
				else:
					news_output.info( "No results returned." )
			except Exception as exc:
				st.error( str( exc ) )
	
	# ======================================================================================
	# GoogleSearch
	# ======================================================================================
	with st.expander( "Google Search", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			google_query = st.text_area(
				"Query",
				value="",
				height=40,
				key="googlesearch_query"
			)
			
			google_num_results = st.number_input(
				"Number of Results",
				min_value=1,
				max_value=50,
				value=10,
				step=1,
				key="googlesearch_num_results"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				google_submit = st.button( "Submit", key="googlesearch_submit" )
			with b2:
				google_clear = st.button( "Clear", key="googlesearch_clear" )
		
		with col_right:
			google_output = st.empty( )
		
		if google_clear:
			st.session_state.update( {
					"googlesearch_query": "",
					"googlesearch_num_results": 10
			} )
			st.rerun( )
		
		if google_submit:
			try:
				f = GoogleSearch( )
				result = f.fetch(
					keywords=google_query,
					results=int( google_num_results )
				)
				
				txt = render_google_results( result )
				google_output.text_area( "Results", value=txt, height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	
	
	# ======================================================================================
	# NavalObservatory
	# ======================================================================================
	with st.expander( "US Naval Observatory", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			naval_query = st.text_area(
				"Query",
				value="",
				height=40,
				key="navalobservatory_query"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				naval_submit = st.button( "Submit", key="navalobservatory_submit" )
			with b2:
				naval_clear = st.button( "Clear", key="navalobservatory_clear" )
		
		with col_right:
			naval_output = st.empty( )
		
		if naval_clear:
			st.session_state.update( {
					"navalobservatory_query": "",
			} )
			st.rerun( )
		
		if naval_submit:
			try:
				f = NavalObservatory( )
				result = f.fetch( naval_query )
				
				if not result:
					naval_output.info( "No results returned." )
				else:
					naval_output.text_area( "Results", value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# ======================================================================================
	# OpenScience
	# ======================================================================================
	with st.expander( "Open Science", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			openscience_query = st.text_area(
				"Query",
				value="",
				height=40,
				key="openscience_query"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				openscience_submit = st.button( "Submit", key="openscience_submit" )
			with b2:
				openscience_clear = st.button( "Clear", key="openscience_clear" )
		
		with col_right:
			openscience_output = st.empty( )
		
		if openscience_clear:
			st.session_state.update( {
					"openscience_query": "",
			} )
			st.rerun( )
		
		if openscience_submit:
			try:
				f = OpenScience( )
				result = f.fetch( openscience_query )
				
				if not result:
					openscience_output.info( "No results returned." )
				else:
					openscience_output.text_area( "Results", value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# ======================================================================================
	# GovData
	# ======================================================================================
	with st.expander( "Gov Info", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			govdata_query = st.text_area(
				"Query",
				value="",
				height=40,
				key="govdata_query"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				govdata_submit = st.button( "Submit", key="govdata_submit" )
			with b2:
				govdata_clear = st.button( "Clear", key="govdata_clear" )
		
		with col_right:
			govdata_output = st.empty( )
		
		if govdata_clear:
			st.session_state.update( {
					"govdata_query": "",
			} )
			st.rerun( )
		
		if govdata_submit:
			try:
				f = GovData( )
				result = f.fetch( govdata_query )
				
				if not result:
					govdata_output.info( "No results returned." )
				else:
					govdata_output.text_area( "Results", value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# ======================================================================================
	# Congress
	# ======================================================================================
	with st.expander( "Congress", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			congress_query = st.text_area(
				"Query",
				value="",
				height=40,
				key="congress_query"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				congress_submit = st.button( "Submit", key="congress_submit" )
			with b2:
				congress_clear = st.button( "Clear", key="congress_clear" )
		
		with col_right:
			congress_output = st.empty( )
		
		if congress_clear:
			st.session_state.update( {
					"congress_query": "",
			} )
			st.rerun( )
		
		if congress_submit:
			try:
				f = Congress( )
				result = f.fetch( congress_query )
				
				if not result:
					congress_output.info( "No results returned." )
				else:
					congress_output.text_area( "Results", value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# ======================================================================================
	# InternetArchive
	# ======================================================================================
	with st.expander( "Internet Archive", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			ia_query = st.text_area(
				"Query",
				value="",
				height=40,
				key="internetarchive_query"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				ia_submit = st.button( "Submit", key="internetarchive_submit" )
			with b2:
				ia_clear = st.button( "Clear", key="internetarchive_clear" )
		
		with col_right:
			ia_output = st.empty( )
		
		if ia_clear:
			st.session_state.update( {
					"internetarchive_query": "",
			} )
			st.rerun( )
		
		if ia_submit:
			try:
				f = InternetArchive( )
				result = f.fetch( ia_query )
				
				if not result:
					ia_output.info( "No results returned." )
				else:
					ia_output.text_area( "Results", value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )

# ======================================================================================
# DATA TAB
# ======================================================================================

with tab_data:
	st.subheader( "" )
	
	conn = sqlite3.connect( f"file:{DB_PATH.as_posix( )}?mode=ro", uri=True )
	cur = conn.cursor( )
	cur.execute( "SELECT name FROM sqlite_master WHERE type='table';" )
	tables = [ row[ 0 ] for row in cur.fetchall( ) ]
	conn.close( )
	
	st.write( tables )

# ======================================================================================
# CHAT TAB — all clear() handlers replaced with update()+rerun()
# ======================================================================================

with tab_chat:
	# -----------------------------
	# Chat (OpenAI)
	# -----------------------------
	with st.expander( "Chat", expanded=True ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			chat_prompt = st.text_area(
				"Prompt",
				value="",
				height=40,
				key="chat_prompt"
			)
			
			p_row1 = st.columns( 2 )
			p_row2 = st.columns( 2 )
			p_row3 = st.columns( 2 )
			
			with p_row1[ 0 ]:
				chat_model = _model_selector(
					key_prefix="chat",
					label="Model",
					options=[
							"gpt-4o-mini",
							"gpt-4.1-mini",
							"gpt-4.1",
							"o3-mini",
							"Custom..."
					],
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
				chat_json_mode = st.checkbox(
					"JSON Mode",
					value=False,
					key="chat_json_mode"
				)
			
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
		
		# -----------------------------
		# FIXED clear()
		# -----------------------------
		if chat_clear:
			st.session_state.update( {
					"chat_prompt": "",
					"chat_system": ""
			} )
			st.rerun( )
		
		# -----------------------------
		# Submit
		# -----------------------------
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
				
				params = { k: v for k, v in params.items( ) if v is not None }
				
				result = _invoke_provider( fetcher, chat_prompt, params )
				_render_output( chat_output, result )
			
			except Exception as exc:
				st.error( str( exc ) )
		
	# ======================================================================================
	# GROQ
	# ======================================================================================
	with st.expander( "Groq", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			groq_prompt = st.text_area(
				"Prompt",
				value="",
				height=40,
				key="groq_prompt_chat"
			)
			
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
				groq_stream = st.checkbox(
					"Stream",
					value=False,
					key="groq_stream_chat"
				)
			
			btn_row = st.columns( 2 )
			with btn_row[ 0 ]:
				groq_submit = st.button( "Submit", key="groq_submit_chat" )
			with btn_row[ 1 ]:
				groq_clear = st.button( "Clear", key="groq_clear_chat" )
		
		with col_right:
			groq_output = st.empty( )
		
		
		if groq_clear:
			st.session_state.update( {
					"groq_prompt_chat": "",
					"groq_stop_chat": ""
			} )
			st.rerun( )
		
		if groq_submit:
			try:
				fetcher = Groq( )
				stop_lines = [ s.strip( ) for s in (groq_stop or "").splitlines( ) if
				               s.strip( ) ]
				
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
	
	# ======================================================================================
	# CLAUDE
	# ======================================================================================
	with st.expander( "Claude", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			claude_prompt = st.text_area(
				"Prompt",
				value="",
				height=40,
				key="claude_prompt_chat"
			)
			
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
			st.session_state.update( {
					"claude_prompt_chat": "",
					"claude_stop_chat": "",
					"claude_system_chat": ""
			} )
			st.rerun( )
		
		if claude_submit:
			try:
				fetcher = Claude( )
				stop_lines = [ s.strip( ) for s in (claude_stop or "").splitlines( ) if
				               s.strip( ) ]
				
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
	
	# ======================================================================================
	# GEMINI
	# ======================================================================================
	with st.expander( "Gemini", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			gemini_prompt = st.text_area(
				"Prompt",
				value="",
				height=180,
				key="gemini_prompt_chat"
			)
			
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
			st.session_state.update( {
					"gemini_prompt_chat": "",
					"gemini_system_chat": "",
			} )
			st.rerun( )
		
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
	
	# ======================================================================================
	# MISTRAL
	# ======================================================================================
	with st.expander( "Mistral", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			mistral_prompt = st.text_area(
				"Prompt",
				value="",
				height=40,
				key="mistral_prompt_chat"
			)
			
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
				mistral_safe_mode = st.checkbox(
					"Safe Mode",
					value=False,
					key="mistral_safe_mode_chat"
				)
			
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
			st.session_state.update( {
					"mistral_prompt_chat": "",
					"mistral_system_chat": "",
			} )
			st.rerun( )
		
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

# ======================================================================================
# MAPS TAB
# ======================================================================================
with tab_maps:
	
	with st.expander( "Google Maps", expanded=True ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			gm_query = st.text_area(
				"Address",
				value="",
				height=40,
				key="googlemaps_query"
			)
			
			gm_radius = st.number_input(
				"Radius (meters)",
				min_value=1,
				max_value=50000,
				value=5000,
				step=100,
				key="googlemaps_radius"
			)
			
			m1, m2 = st.columns( 2 )
			with m1:
				gm_submit = st.button( "Submit", key="googlemaps_submit" )
			with m2:
				gm_clear = st.button( "Clear", key="googlemaps_clear" )
		
		with col_right:
			gm_output = st.empty( )
			
			if gm_clear:
				st.session_state.update( {
						"googlemaps_query": "",
						"googlemaps_radius": 5000
				} )
				st.rerun( )
			
			if gm_submit:
				try:
					gm = GoogleMaps( )
					loc = gm.geocode_location( gm_query )
					coords = f'{loc[ 0 ]}, {loc[ 1 ]}'
					gm_output.text_area( "Coords", value=coords, height=300 )
				except Exception as exc:
					st.error( exc )
	
	# ======================================================================================
	# GoogleWeather
	# ======================================================================================
	with st.expander( "Google Weather", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			gw_location = st.text_area(
				"Location",
				value="",
				height=40,
				key="googleweather_location"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				gw_submit = st.button( "Submit", key="googleweather_submit" )
			with b2:
				gw_clear = st.button( "Clear", key="googleweather_clear" )
		
		with col_right:
			gw_output = st.empty( )
			
			if gw_clear:
				st.session_state.update( {
						"googleweather_location": "",
				} )
				st.rerun( )
			
			if gw_submit:
				try:
					f = GoogleWeather( )
					result = f.fetch_current( address=gw_location )
					if not result:
						gw_output.info( "No results returned." )
					else:
						gw_output.text_area( "Results", value=result.text, height=300 )
				
				except Exception as exc:
					st.error( str( exc ) )
					
	# ======================================================================================
	# SatelliteCenter
	# ======================================================================================
	with st.expander( "Satellite Center", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			satellite_query = st.text_area(
				"Query",
				value="",
				height=40,
				key="satellitecenter_query"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				satellite_submit = st.button( "Submit", key="satellitecenter_submit" )
			with b2:
				satellite_clear = st.button( "Clear", key="satellitecenter_clear" )
		
		with col_right:
			satellite_output = st.empty( )
		
		if satellite_clear:
			st.session_state.update( {
					"satellitecenter_query": "",
			} )
			st.rerun( )
		
		if satellite_submit:
			try:
				f = SatelliteCenter( )
				result = f.fetch( satellite_query )
				
				if not result:
					satellite_output.info( "No results returned." )
				else:
					satellite_output.text_area( "Results", value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
		
	# ======================================================================================
	# AstroCatalog
	# ======================================================================================
	with st.expander( "Astronomy Catalog", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			astro_query = st.text_area(
				"Query",
				value="",
				height=40,
				key="astrocatalog_query"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				astro_submit = st.button( "Submit", key="astrocatalog_submit" )
			with b2:
				astro_clear = st.button( "Clear", key="astrocatalog_clear" )
		
		with col_right:
			astro_output = st.empty( )
		
		if astro_clear:
			st.session_state.update( {
					"astrocatalog_query": "",
			} )
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
				
	# ======================================================================================
	# AstroQuery
	# ======================================================================================
	with st.expander( "Astro Query", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			astroquery_query = st.text_area(
				"Query",
				value="",
				height=40,
				key="astroquery_query"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				astroquery_submit = st.button( "Submit", key="astroquery_submit" )
			with b2:
				astroquery_clear = st.button( "Clear", key="astroquery_clear" )
		
		with col_right:
			astroquery_output = st.empty( )
		
		if astroquery_clear:
			st.session_state.update( {
					"astroquery_query": "",
			} )
			st.rerun( )
		
		if astroquery_submit:
			try:
				f = AstroQuery( )
				result = f.fetch( astroquery_query )
				
				if not result:
					astroquery_output.info( "No results returned." )
				else:
					astroquery_output.text_area( "Results", value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
		
	# ======================================================================================
	# StarMap
	# ======================================================================================
	with st.expander( "Star Map", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			starmap_query = st.text_area(
				"Query",
				value="",
				height=40,
				key="starmap_query"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				starmap_submit = st.button( "Submit", key="starmap_submit" )
			with b2:
				starmap_clear = st.button( "Clear", key="starmap_clear" )
		
		with col_right:
			starmap_output = st.empty( )
		
		if starmap_clear:
			st.session_state.update( {
					"starmap_query": "",
			} )
			st.rerun( )
		
		if starmap_submit:
			try:
				f = StarMap( )
				result = f.fetch( starmap_query )
				
				if not result:
					starmap_output.info( "No results returned." )
				else:
					starmap_output.text_area( "Results", value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
				
	# ======================================================================================
	# OpenWeather
	# ======================================================================================
	with st.expander( "Open Weather", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			openweather_query = st.text_area(
				"Location",
				value="",
				height=40,
				key="openweather_query"
			)
			
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
	
	# ======================================================================================
	# Open Meteo / OpenWeather
	# ======================================================================================
	with st.expander( "Open Meteorology", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			latitude = st.number_input(
				"Latitude",
				value=0.0,
				format="%.6f",
				key="openmeteo_latitude"
			)
			
			longitude = st.number_input(
				"Longitude",
				value=0.0,
				format="%.6f",
				key="openmeteo_longitude"
			)
			
			days = st.number_input(
				"Forecast Days",
				min_value=1,
				max_value=14,
				value=7,
				step=1,
				key="openmeteo_days"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				om_submit = st.button( "Submit", key="openmeteo_submit" )
			with b2:
				om_clear = st.button( "Clear", key="openmeteo_clear" )
		
		with col_right:
			om_output = st.empty( )
		
		if om_clear:
			st.session_state.update( {
					"openmeteo_latitude": 0.0,
					"openmeteo_longitude": 0.0,
					"openmeteo_days": 7
			} )
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
					om_output.text_area( "Forecast Data", value=str( result ), height=300 )
				else:
					om_output.info( "No data returned." )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# ======================================================================================
	# Simbad Fetcher
	# ======================================================================================
	with st.expander( "Simbad", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			simbad_query = st.text_area(
				"Astronomical Object Query",
				value="",
				height=120,
				key="simbad_query"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				simbad_submit = st.button( "Submit", key="simbad_submit" )
			with b2:
				simbad_clear = st.button( "Clear", key="simbad_clear" )
		
		with col_right:
			simbad_output = st.empty( )
		
		if simbad_clear:
			st.session_state.update( {
					"simbad_query": ""
			} )
			st.rerun( )
		
		if simbad_submit:
			try:
				f = Simbad( )
				result = f.fetch( simbad_query )
				
				if result:
					simbad_output.text_area( "Result", value=str( result ), height=300 )
				else:
					simbad_output.info( "No results returned." )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# ======================================================================================
	# EarthObservatory
	# ======================================================================================
	with st.expander( "Earth Observatory", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			earth_query = st.text_area(
				"Query",
				value="",
				height=40,
				key="earthobservatory_query"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				earth_submit = st.button( "Submit", key="earthobservatory_submit" )
			with b2:
				earth_clear = st.button( "Clear", key="earthobservatory_clear" )
		
		with col_right:
			earth_output = st.empty( )
		
		if earth_clear:
			st.session_state.update( {
					"earthobservatory_query": "",
			} )
			st.rerun( )
		
		if earth_submit:
			try:
				f = EarthObservatory( )
				result = f.fetch( earth_query )
				
				if not result:
					earth_output.info( "No results returned." )
				else:
					earth_output.text_area( "Results", value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# ======================================================================================
	# SpaceWeather
	# ======================================================================================
	with st.expander( "Space Weather", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			spaceweather_query = st.text_area(
				"Query",
				value="",
				height=40,
				key="spaceweather_query"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				spaceweather_submit = st.button( "Submit", key="spaceweather_submit" )
			with b2:
				spaceweather_clear = st.button( "Clear", key="spaceweather_clear" )
		
		with col_right:
			spaceweather_output = st.empty( )
		
		if spaceweather_clear:
			st.session_state.update( {
					"spaceweather_query": "",
			} )
			st.rerun( )
		
		if spaceweather_submit:
			try:
				f = SpaceWeather( )
				result = f.fetch( spaceweather_query )
				
				if not result:
					spaceweather_output.info( "No results returned." )
				else:
					spaceweather_output.text_area( "Results", value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
	
	# ======================================================================================
	# StarChart
	# ======================================================================================
	with st.expander( "Star Chart", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			starchart_query = st.text_area(
				"Query",
				value="",
				height=40,
				key="starchart_query"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				starchart_submit = st.button( "Submit", key="starchart_submit" )
			with b2:
				starchart_clear = st.button( "Clear", key="starchart_clear" )
		
		with col_right:
			starchart_output = st.empty( )
		
		if starchart_clear:
			st.session_state.update( {
					"starchart_query": "",
			} )
			st.rerun( )
		
		if starchart_submit:
			try:
				f = StarChart( )
				result = f.fetch( starchart_query )
				
				if not result:
					starchart_output.info( "No results returned." )
				else:
					starchart_output.text_area( "Results", value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
		
	# ======================================================================================
	# NearbyObjects
	# ======================================================================================
	with st.expander( "Near Earth Objects", expanded=False ):
		col_left, col_right = st.columns( 2, border=True )
		
		with col_left:
			nearby_query = st.text_area(
				"Query",
				value="",
				height=40,
				key="nearbyobjects_query"
			)
			
			b1, b2 = st.columns( 2 )
			with b1:
				nearby_submit = st.button( "Submit", key="nearbyobjects_submit" )
			with b2:
				nearby_clear = st.button( "Clear", key="nearbyobjects_clear" )
		
		with col_right:
			nearby_output = st.empty( )
		
		if nearby_clear:
			st.session_state.update( {
					"nearbyobjects_query": "",
			} )
			st.rerun( )
		
		if nearby_submit:
			try:
				f = NearbyObjects( )
				result = f.fetch( nearby_query )
				
				if not result:
					nearby_output.info( "No results returned." )
				else:
					nearby_output.text_area( "Results", value=str( result ), height=300 )
			
			except Exception as exc:
				st.error( str( exc ) )
				
# ======================================================================================
# END OF FILE
# ======================================================================================
