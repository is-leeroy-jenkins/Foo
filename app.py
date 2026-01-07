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
                      GoogleSearch, GoogleDrive, GoogleMaps, )

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

st.set_page_config( page_title="Foo", layout="wide" )
st.title( "Foo" )

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

tab_scrapers, tab_fetchers, tab_data, tab_chat = st.tabs(
	[ "Scrapers",
	  "Fetchers",
	  "Data",
	  "Chat" ]
)

# ======================================================================================
# SCRAPERS TAB — WebExtractor
# ======================================================================================

with tab_scrapers:
	st.subheader( "Scrapers" )
	
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
	st.subheader( "Fetchers" )
	
	# -----------------------------
	# Session state
	# -----------------------------
	st.session_state.setdefault( "arxiv_input", "" )
	st.session_state.setdefault( "arxiv_results", [ ] )
	
	with st.expander( "ArXiv", expanded=True ):
		col1, col2 = st.columns( 2 )
		
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
		col_left, col_right = st.columns( 2 )
		
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
		col_left, col_right = st.columns( 2 )
		
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
		col_left, col_right = st.columns( 2 )
		
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
				error = Error( exc )
				error.module = "app"
				error.cause = "TheNews"
				error.method = "fetch"
				ErrorDialog( error ).show( )
	
	# -------------------------------
	# OpenMeteo Fetcher
	# -------------------------------
	with st.expander( "OpenMeteo", expanded=False ):
		col_left, col_right = st.columns( 2 )
		
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
				error = Error( exc )
				error.module = "app"
				error.cause = "OpenMeteo"
				error.method = "fetch"
				ErrorDialog( error ).show( )
	
	# -------------------------------
	# Simbad Fetcher
	# -------------------------------
	with st.expander( "Simbad", expanded=False ):
		col_left, col_right = st.columns( 2 )
		
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
				error = Error( exc )
				error.module = "app"
				error.cause = "Simbad"
				error.method = "fetch"
				ErrorDialog( error ).show( )
	
	# -------------------------------
	# GoogleSearch Fetcher
	# -------------------------------
	with st.expander( "GoogleSearch", expanded=False ):
		col_left, col_right = st.columns( 2 )
		
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
			google_output = st.empty( )
		
		if google_clear:
			st.session_state[ "googlesearch_query" ] = ""
			st.session_state[ "googlesearch_num_results" ] = 10
			google_output.empty( )
		
		if google_submit:
			try:
				# API key is expected via sidebar / environment
				fetcher = GoogleSearch( )
				
				result = fetcher.fetch(
					query=google_query,
					num_results=int( google_num_results )
				)
				
				if not result:
					google_output.info( "No results returned." )
				else:
					google_output.text_area(
						label="Results",
						value=str( result ),
						height=300
					)
			
			except Exception as exc:
				error = Error( exc )
				error.module = "app"
				error.cause = "GoogleSearch"
				error.method = "fetch"
				ErrorDialog( error ).show( )
	
	# -------------------------------
	# GoogleMaps Fetcher
	# -------------------------------
	with st.expander( "GoogleMaps", expanded=False ):
		col_left, col_right = st.columns( 2 )
		
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
				error = Error( exc )
				error.module = "app"
				error.cause = "GoogleMaps"
				error.method = "fetch"
				ErrorDialog( error ).show( )
	
	# -------------------------------
	# GoogleWeather Fetcher
	# -------------------------------
	with st.expander( "GoogleWeather", expanded=False ):
		col_left, col_right = st.columns( 2 )
		
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
				error = Error( exc )
				error.module = "app"
				error.cause = "GoogleWeather"
				error.method = "fetch"
				ErrorDialog( error ).show( )
	
	# -------------------------------
	# NavalObservatory Fetcher
	# -------------------------------
	with st.expander( "NavalObservatory", expanded=False ):
		col_left, col_right = st.columns( 2 )
		
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
				error = Error( exc )
				error.module = "app"
				error.cause = "NavalObservatory"
				error.method = "fetch"
				ErrorDialog( error ).show( )
	
	# -------------------------------
	# SatelliteCenter Fetcher
	# -------------------------------
	with st.expander( "SatelliteCenter", expanded=False ):
		col_left, col_right = st.columns( 2 )
		
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
				error = Error( exc )
				error.module = "app"
				error.cause = "SatelliteCenter"
				error.method = "fetch"
				ErrorDialog( error ).show( )


# -------------------------------
# NearbyObjects Fetcher
# -------------------------------

with st.expander("NearbyObjects", expanded=False):
    col_left, col_right = st.columns(2)

    with col_left:
        nearby_query = st.text_area(
            label="Query",
            value="",
            height=150,
            key="nearbyobjects_query"
        )

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            nearby_submit = st.button("Submit", key="nearbyobjects_submit")

        with btn_col2:
            nearby_clear = st.button("Clear", key="nearbyobjects_clear")

    with col_right:
        nearby_output = st.empty()

    if nearby_clear:
        st.session_state["nearbyobjects_query"] = ""
        nearby_output.empty()

    if nearby_submit:
        try:
            # API key (if required) is expected via sidebar / environment
            fetcher = NearbyObjects()

            result = fetcher.fetch(nearby_query)

            if not result:
                nearby_output.info("No results returned.")
            else:
                nearby_output.text_area(
                    label="Results",
                    value=str(result),
                    height=300
                )

        except Exception as exc:
            error = Error(exc)
            error.module = "app"
            error.cause = "NearbyObjects"
            error.method = "fetch"
            ErrorDialog(error).show()

# -------------------------------
# OpenScience Fetcher
# -------------------------------

with st.expander("OpenScience", expanded=False):
    col_left, col_right = st.columns(2)

    with col_left:
        openscience_query = st.text_area(
            label="Query",
            value="",
            height=150,
            key="openscience_query"
        )

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            openscience_submit = st.button("Submit", key="openscience_submit")

        with btn_col2:
            openscience_clear = st.button("Clear", key="openscience_clear")

    with col_right:
        openscience_output = st.empty()

    if openscience_clear:
        st.session_state["openscience_query"] = ""
        openscience_output.empty()

    if openscience_submit:
        try:
            # API key (if required) is expected via sidebar / environment
            fetcher = OpenScience()

            result = fetcher.fetch(openscience_query)

            if not result:
                openscience_output.info("No results returned.")
            else:
                openscience_output.text_area(
                    label="Results",
                    value=str(result),
                    height=300
                )

        except Exception as exc:
            error = Error(exc)
            error.module = "app"
            error.cause = "OpenScience"
            error.method = "fetch"
            ErrorDialog(error).show()

# -------------------------------
# EarthObservatory Fetcher
# -------------------------------

with st.expander("EarthObservatory", expanded=False):
    col_left, col_right = st.columns(2)

    with col_left:
        earth_query = st.text_area(
            label="Query",
            value="",
            height=150,
            key="earthobservatory_query"
        )

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            earth_submit = st.button("Submit", key="earthobservatory_submit")

        with btn_col2:
            earth_clear = st.button("Clear", key="earthobservatory_clear")

    with col_right:
        earth_output = st.empty()

    if earth_clear:
        st.session_state["earthobservatory_query"] = ""
        earth_output.empty()

    if earth_submit:
        try:
            # API key (if required) is expected via sidebar / environment
            fetcher = EarthObservatory()

            result = fetcher.fetch(earth_query)

            if not result:
                earth_output.info("No results returned.")
            else:
                earth_output.text_area(
                    label="Results",
                    value=str(result),
                    height=300
                )

        except Exception as exc:
            error = Error(exc)
            error.module = "app"
            error.cause = "EarthObservatory"
            error.method = "fetch"
            ErrorDialog(error).show()

# -------------------------------
# SpaceWeather Fetcher
# -------------------------------

with st.expander("SpaceWeather", expanded=False):
    col_left, col_right = st.columns(2)

    with col_left:
        spaceweather_query = st.text_area(
            label="Query",
            value="",
            height=150,
            key="spaceweather_query"
        )

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            spaceweather_submit = st.button("Submit", key="spaceweather_submit")

        with btn_col2:
            spaceweather_clear = st.button("Clear", key="spaceweather_clear")

    with col_right:
        spaceweather_output = st.empty()

    if spaceweather_clear:
        st.session_state["spaceweather_query"] = ""
        spaceweather_output.empty()

    if spaceweather_submit:
        try:
            # API key (if required) is expected via sidebar / environment
            fetcher = SpaceWeather()

            result = fetcher.fetch(spaceweather_query)

            if not result:
                spaceweather_output.info("No results returned.")
            else:
                spaceweather_output.text_area(
                    label="Results",
                    value=str(result),
                    height=300
                )

        except Exception as exc:
            error = Error(exc)
            error.module = "app"
            error.cause = "SpaceWeather"
            error.method = "fetch"
            ErrorDialog(error).show()
# -------------------------------
# AstroCatalog Fetcher
# -------------------------------

with st.expander("AstroCatalog", expanded=False):
    col_left, col_right = st.columns(2)

    with col_left:
        astro_query = st.text_area(
            label="Query",
            value="",
            height=150,
            key="astrocatalog_query"
        )

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            astro_submit = st.button("Submit", key="astrocatalog_submit")

        with btn_col2:
            astro_clear = st.button("Clear", key="astrocatalog_clear")

    with col_right:
        astro_output = st.empty()

    if astro_clear:
        st.session_state["astrocatalog_query"] = ""
        astro_output.empty()

    if astro_submit:
        try:
            # API key (if required) is expected via sidebar / environment
            fetcher = AstroCatalog()

            result = fetcher.fetch(astro_query)

            if not result:
                astro_output.info("No results returned.")
            else:
                astro_output.text_area(
                    label="Results",
                    value=str(result),
                    height=300
                )

        except Exception as exc:
            error = Error(exc)
            error.module = "app"
            error.cause = "AstroCatalog"
            error.method = "fetch"
            ErrorDialog(error).show()

# -------------------------------
# AstroQuery Fetcher
# -------------------------------

with st.expander("AstroQuery", expanded=False):
    col_left, col_right = st.columns(2)

    with col_left:
        astroquery_query = st.text_area(
            label="Query",
            value="",
            height=150,
            key="astroquery_query"
        )

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            astroquery_submit = st.button("Submit", key="astroquery_submit")

        with btn_col2:
            astroquery_clear = st.button("Clear", key="astroquery_clear")

    with col_right:
        astroquery_output = st.empty()

    if astroquery_clear:
        st.session_state["astroquery_query"] = ""
        astroquery_output.empty()

    if astroquery_submit:
        try:
            # API key (if required) is expected via sidebar / environment
            fetcher = AstroQuery()

            result = fetcher.fetch(astroquery_query)

            if not result:
                astroquery_output.info("No results returned.")
            else:
                astroquery_output.text_area(
                    label="Results",
                    value=str(result),
                    height=300
                )

        except Exception as exc:
            error = Error(exc)
            error.module = "app"
            error.cause = "AstroQuery"
            error.method = "fetch"
            ErrorDialog(error).show()


# -------------------------------
# StarMap Fetcher
# -------------------------------

with st.expander("StarMap", expanded=False):
    col_left, col_right = st.columns(2)

    with col_left:
        starmap_query = st.text_area(
            label="Query",
            value="",
            height=150,
            key="starmap_query"
        )

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            starmap_submit = st.button("Submit", key="starmap_submit")

        with btn_col2:
            starmap_clear = st.button("Clear", key="starmap_clear")

    with col_right:
        starmap_output = st.empty()

    if starmap_clear:
        st.session_state["starmap_query"] = ""
        starmap_output.empty()

    if starmap_submit:
        try:
            # API key (if required) is expected via sidebar / environment
            fetcher = StarMap()

            result = fetcher.fetch(starmap_query)

            if not result:
                starmap_output.info("No results returned.")
            else:
                starmap_output.text_area(
                    label="Results",
                    value=str(result),
                    height=300
                )

        except Exception as exc:
            error = Error(exc)
            error.module = "app"
            error.cause = "StarMap"
            error.method = "fetch"
            ErrorDialog(error).show()



# -------------------------------
# GovData Fetcher
# -------------------------------

with st.expander("GovData", expanded=False):
    col_left, col_right = st.columns(2)

    with col_left:
        govdata_query = st.text_area(
            label="Query",
            value="",
            height=150,
            key="govdata_query"
        )

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            govdata_submit = st.button("Submit", key="govdata_submit")

        with btn_col2:
            govdata_clear = st.button("Clear", key="govdata_clear")

    with col_right:
        govdata_output = st.empty()

    if govdata_clear:
        st.session_state["govdata_query"] = ""
        govdata_output.empty()

    if govdata_submit:
        try:
            # API key (if required) is expected via sidebar / environment
            fetcher = GovData()

            result = fetcher.fetch(govdata_query)

            if not result:
                govdata_output.info("No results returned.")
            else:
                govdata_output.text_area(
                    label="Results",
                    value=str(result),
                    height=300
                )

        except Exception as exc:
            error = Error(exc)
            error.module = "app"
            error.cause = "GovData"
            error.method = "fetch"
            ErrorDialog(error).show()


# -------------------------------
# StarChart Fetcher
# -------------------------------

with st.expander("StarChart", expanded=False):
    col_left, col_right = st.columns(2)

    with col_left:
        starchart_query = st.text_area(
            label="Query",
            value="",
            height=150,
            key="starchart_query"
        )

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            starchart_submit = st.button("Submit", key="starchart_submit")

        with btn_col2:
            starchart_clear = st.button("Clear", key="starchart_clear")

    with col_right:
        starchart_output = st.empty()

    if starchart_clear:
        st.session_state["starchart_query"] = ""
        starchart_output.empty()

    if starchart_submit:
        try:
            # API key (if required) is expected via sidebar / environment
            fetcher = StarChart()

            result = fetcher.fetch(starchart_query)

            if not result:
                starchart_output.info("No results returned.")
            else:
                starchart_output.text_area(
                    label="Results",
                    value=str(result),
                    height=300
                )

        except Exception as exc:
            error = Error(exc)
            error.module = "app"
            error.cause = "StarChart"
            error.method = "fetch"
            ErrorDialog(error).show()


# -------------------------------
# StarChart Fetcher
# -------------------------------

with st.expander("StarChart", expanded=False):
    col_left, col_right = st.columns(2)

    with col_left:
        starchart_query = st.text_area(
            label="Query",
            value="",
            height=150,
            key="starchart_query"
        )

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            starchart_submit = st.button("Submit", key="starchart_submit")

        with btn_col2:
            starchart_clear = st.button("Clear", key="starchart_clear")

    with col_right:
        starchart_output = st.empty()

    if starchart_clear:
        st.session_state["starchart_query"] = ""
        starchart_output.empty()

    if starchart_submit:
        try:
            # API key (if required) is expected via sidebar / environment
            fetcher = StarChart()

            result = fetcher.fetch(starchart_query)

            if not result:
                starchart_output.info("No results returned.")
            else:
                starchart_output.text_area(
                    label="Results",
                    value=str(result),
                    height=300
                )

        except Exception as exc:
            error = Error(exc)
            error.module = "app"
            error.cause = "StarChart"
            error.method = "fetch"
            ErrorDialog(error).show()
		    
# -------------------------------
# InternetArchive Fetcher
# -------------------------------

with st.expander("InternetArchive", expanded=False):
    col_left, col_right = st.columns(2)

    with col_left:
        ia_query = st.text_area(
            label="Query",
            value="",
            height=150,
            key="internetarchive_query"
        )

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            ia_submit = st.button("Submit", key="internetarchive_submit")

        with btn_col2:
            ia_clear = st.button("Clear", key="internetarchive_clear")

    with col_right:
        ia_output = st.empty()

    if ia_clear:
        st.session_state["internetarchive_query"] = ""
        ia_output.empty()

    if ia_submit:
        try:
            # API key (if required) is expected via sidebar / environment
            fetcher = InternetArchive()

            result = fetcher.fetch(ia_query)

            if not result:
                ia_output.info("No results returned.")
            else:
                ia_output.text_area(
                    label="Results",
                    value=str(result),
                    height=300
                )

        except Exception as exc:
            error = Error(exc)
            error.module = "app"
            error.cause = "InternetArchive"
            error.method = "fetch"
            ErrorDialog(error).show()

# -------------------------------
# OpenWeather Fetcher
# -------------------------------

with st.expander("OpenWeather", expanded=False):
    col_left, col_right = st.columns(2)

    with col_left:
        openweather_query = st.text_area(
            label="Location",
            value="",
            height=150,
            key="openweather_query"
        )

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            openweather_submit = st.button("Submit", key="openweather_submit")

        with btn_col2:
            openweather_clear = st.button("Clear", key="openweather_clear")

    with col_right:
        openweather_output = st.empty()

    if openweather_clear:
        st.session_state["openweather_query"] = ""
        openweather_output.empty()

    if openweather_submit:
        try:
            # API key (if required) is expected via sidebar / environment
            fetcher = OpenWeather()

            result = fetcher.fetch(openweather_query)

            if not result:
                openweather_output.info("No results returned.")
            else:
                openweather_output.text_area(
                    label="Results",
                    value=str(result),
                    height=300
                )

        except Exception as exc:
            error = Error(exc)
            error.module = "app"
            error.cause = "OpenWeather"
            error.method = "fetch"
            ErrorDialog(error).show()




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

# ======================================================================================
# CHAT TAB
# ======================================================================================

with tab_chat:
	st.subheader( "Chat" )
	
	col1, col2 = st.columns( 2 )
	with col1:
		provider = st.selectbox( "Provider", [ "OpenAI",
		                                       "Groq",
		                                       "Gemini" ] )
	with col2:
		model = st.text_input( "Model" )
	
	col3, col4 = st.columns( 2 )
	with col3:
		temperature = st.slider( "Temperature", 0.0, 2.0, 0.7, 0.05 )
	with col4:
		max_tokens = st.number_input( "Max Tokens", 1, 32768, 1024 )
	
	history_key = f"chat_{provider}"
	st.session_state.setdefault( history_key, [ ] )
	
	for msg in st.session_state[ history_key ]:
		with st.chat_message( msg[ "role" ] ):
			st.markdown( msg[ "content" ] )
	
	user_input = st.chat_input( "Send" )
	if user_input:
		st.session_state[ history_key ].append(
			{
					"role": "user",
					"content": user_input }
		)
		with st.chat_message( "assistant" ):
			st.markdown( "…" )
