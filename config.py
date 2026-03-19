'''
  ******************************************************************************************
      Assembly:                Foo
      Filename:                config.py
      Author:                  Terry Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="config.py" company="Terry D. Eppler">

	     config.py
	     Copyright ©  2024  Terry Eppler

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
    config.py
  </summary>
  ******************************************************************************************
  '''
import os
from pathlib import Path

# ------ ENVIRONMENT API KEYS  -------------------
ACCESS_DRIVER = r'DRIVER={ Microsoft Access Driver (*.mdb, *.accdb) };DBQ='
CLAUDE_API_KEY = os.getenv( 'CLAUDE_API_KEY' )
CONGRESS_API_KEY = os.getenv( 'CONGRESS_API_KEY' )
GEOAPIFY_API_KEY = os.getenv( 'GEOAPIFY_API_KEY' )
GEOCODING_API_KEY = os.getenv( 'GEOCODING_API_KEY' )
GEMINI_API_KEY = os.getenv( 'GEMINI_API_KEY' )
GOOGLE_API_KEY = os.getenv( 'GOOGLE_API_KEY' )
GOOGLE_CSE_ID = os.getenv( 'GOOGLE_CSE_ID' )
GOOGLE_CLOUD_PROJECT_ID = os.getenv( 'GOOGLE_CLOUD_PROJECT_ID' )
GOOGLE_CLOUD_LOCATION = os.getenv( 'GOOGLE_CLOUD_LOCATION' )
GOVINFO_API_KEY = os.getenv( 'GOVINFO_API_KEY' )
GOOGLE_GENAI_USE_VERTEXAI = os.getenv( 'GOOGLE_GENAI_USE_VERTEXAI' )
GOOGLE_WEATHER_API_KEY = os.getenv( 'GOOGLE_WEATHER_API_KEY' )
GOOGLE_ACCOUNT_FILE = os.getenv( 'GOOGLE_ACCOUNT_CREDENTIALS' )
GOOGLE_DRIVE_TOKEN_PATH = os.getenv( 'GOOGLE_DRIVE_TOKEN_PATH' )
GOOGLE_DRIVE_FOLDER_ID = os.getenv( 'GOOGLE_DRIVE_FOLDER_ID' )
HUGGINGFACE_API_KEY = os.getenv( 'HUGGINGFACE_API_KEY' )
IPINFO_API_KEY = os.getenv( 'IPINFO_API_KEY' )
OPENAI_API_KEY = os.getenv( 'OPENAI_API_KEY' )
PINECONE_API_KEY = os.getenv( 'PINECONE_API_KEY' )
LANGSMITH_API_KEY = os.getenv( 'LANGSMITH_API_KEY' )
LLAMAINDEX_API_KEY = os.getenv( 'LLAMAINDEX_API_KEY' )
LLAMACLOUD_API_KEY = os.getenv( 'LLAMACLOUD_API_KEY' )
MISTRAL_API_KEY = os.getenv( 'MISTRAL_API_KEY' )
NASA_API_KEY = os.getenv( 'NASA_API_KEY' )
NASA_EARTHDATA_TOKEN = os.getenv( 'NASA_EARTHDATA_TOKEN' )
NEWS_API_KEY = os.getenv( 'NEWSAPI_API_KEY' )
THENEWS_API_KEY = os.getenv( 'THENEWSAPI_API_KEY' )
WEATHERAPI_API_KEY = os.getenv( 'WEATHERAPI_API_KEY' )
XAI_API_KEY = os.getenv( 'XAI_API_KEY' )
O365_CLIENT_ID = os.getenv( 'O365_CLIENT_ID ' )
O365_CLIENT_SECRET = os.getenv( 'O365_CLIENT_SECRET ' )

# ---------------- CONSTANTS -----------------------
APP_TITLE = 'Foo'
BLUE_DIVIDER = "<div style='height:2px;align:left;background:#0078FC;margin:6px 0 10px 0;'></div>"
SQLSERVER_DRIVER = r'DRIVER={ ODBC Driver 17 for SQL Server };SERVER=.\SQLExpress;'
BASE_DIR = Path( __file__ ).resolve( ).parent
DB_PATH = BASE_DIR / 'stores' / 'sqlite' / 'datamodels' / 'Data.db'
AGENTS ='''Mozilla/5.0 Windows NT 10.0; Win64; x64; AppleWebKit/537.36 (KHTML, like Gecko)
		Chrome/124.0 Safari/537.36'''
SKYMAP_TOKEN = '''06f556f517061802aab305e26066233926a41785fddafd2867d5dc6d6a917d7b5edd56e8d57766aa3
		7cb6d16dff82d6ccae1625233b27b05483fd534173bcae659e7edf7b083bb18b7786d03e874d921374dec9287626047f7e49637b701bf9420faee5cadffa46b1501c47366d9693e'''
FAVICON = r'resources/images/favicon.ico'
LOGO = r'resources/images/foo_logo.png'
DB = r'stores/sqlite/datamodels/Data.db'
MODES = [ 'Data Loading', 'Data Scraping', 'Data Retrieval',
          'Data Generation', 'Geospatial Data', 'Data Management' ]

MODE_MAP = \
{
		'Loading': 'Data Loading',
		'Scraping': 'Data Scraping',
		'Retrieval': 'Data Collections & Public Archives',
		'Geospatial': 'Weather & Geospatial Information',
		'Population': 'Health & Population Data',
		'Environmental': 'Environmental Information',
		'Astronomical': 'Physics & Astronomical Data',
		'Generation': 'AI Generation',
		'Management': 'Data Management'
 }

CHUNKABLE_LOADERS = {
		'TextLoader': [ 'chars', 'tokens' ],
		'CsvLoader': [ 'chars' ],
		'PdfLoader': [ 'chars' ],
		'ExcelLoader': [ 'chars' ],
		'WordLoader': [ 'chars' ],
		'MarkdownLoader': [ 'chars' ],
		'HtmlLoader': [ 'chars' ],
		'JsonLoader': [ 'chars' ],
		'PowerPointLoader': [ 'chars' ],
}

REQUIRED_CORPORA = [
		'brown',
		'gutenberg',
		'reuters',
		'webtext',
		'inaugural',
		'state_union',
		'punkt',
		'stopwords',
]

SESSION_STATE_DEFAULTS = {
		# ------------ Ingestion
		'documents': None,
		'raw_documents': None,
		'active_loader': None,
		# ------------ Input
		'raw_text': None,
		'raw_tokens': None,
		'raw_text_view': None,
		# Processing
		'parser': None,
		'processed_text': None,
		'processed_text_view': None,
		# ------------ Performance
		'start_time': None,
		'end_time': None,
		'total_time': None,
		# ------------ Tokenization / Vocabulary
		'tokens': None,
		'vocabulary': None,
		'token_counts': None,
		'df_synsets': None,
		# ------------ SQLite / Excel
		'active_table': None,
		# ------------ Chunking
		'lines': None,
		'chunks': None,
		'chunk_modes': None,
		'chunked_documents': None,
		# ------------ Embeddings
		'embedder': None,
		'embeddings': None,
		'embedding_provider': None,
		'embedding_model': None,
		'embedding_source': None,
		'embedding_documents': None,
		'df_embedding_input': None,
		'df_embedding_output': None,
		# ------------ Retrieval / Search
		'search_results': None,
		# ------------ DataFrames
		'df_frequency': None,
		'df_tables': None,
		'df_schema': None,
		'df_preview': None,
		'df_count': None,
		'df_chunks': None,
		# ------------ Data
		'data_connection': None,
		# ------------ Sidebar / API Keys
		'api_keys': {
				'openai_api_key': None,
				'groq_api_key': None,
				'google_api_key': None,
				'pinecone_api_key': None,
				'google_credentials_path_api_key': None,
		},
		# ------------ XML Loader (explicit contract)
		'xml_loader': None,
		'xml_documents': None,
		'xml_split_documents': None,
		'xml_tree_loaded': None,
		'xml_namespaces': None,
		'xml_xpath_results': None,
		# ------------ WordNet Caches
		'wordnet_synsets_sig': None,
		'df_wordnet_synsets': None,
		'df_wordnet_lemmas': None,
}

# ----------------- Models

GPT_MODELS = [ 'gpt-5.4', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano',
               'gpt-5.1', 'gpt-5.2', 'gpt-4.1' ]

GEMINI_MODELS = [ 'gemini-2.5-flash', 'gemini-2.5-flash-lite',
                  'gemini-2.5-flash-lite' ]

GROK_MODELS = [ 'grok-4-1-fast-reasoning', 'grok-4-fast-reasoning', 'grok-4',
                'grok-code-fast-1', 'grok-3-mini', 'grok-2-image-1212' ]

CLAUDE_MODELS = [ 'claude-opus-4-6', 'claude-sonnet-4-6',
                  'claude-haiku-4-5' ]

MISTRAL_MODELS = [ 'mistral-large-latest', 'mistral-medium-latest',
                   'mistral-small-latest', 'mistral-ocr-latest'  ]

# ------------- API DEFINITIONS ------------------

ARXIV = r'''arXiv is a free distribution service and an open-access archive for nearly 2.4 million
		scholarly articles in the fields of physics, mathematics, computer science, quantitative
		biology, quantitative finance, statistics, electrical engineering and systems science, and
		economics. Materials on this site are not peer-reviewed by arXiv.
		
		https://docs.langchain.com/oss/python/integrations/retrievers/arxiv
'''

GOOGLE_DRIVE = r'''Google Drive is a file storage and synchronization service developed by Google
		
		https://docs.langchain.com/oss/python/integrations/retrievers/google_drive
'''

WIKIPEDIA = r'''A multilingual free online encyclopedia written and maintained by a community of
		volunteers, known as Wikipedians, through open collaboration and using a wiki-based editing
		system called MediaWiki. Wikipedia is the largest and most-read reference work in history.
		
		https://docs.langchain.com/oss/python/integrations/retrievers/wikipedia
'''

PUBMED = r'''The National Center for Biotechnology Information, National Library of Medicine
		comprises more than 35 million citations for biomedical literature from MEDLINE,
		life science journals, and online books. Citations may include links to full text content
		from PubMed Central and publisher web sites.
		
		https://docs.langchain.com/oss/python/integrations/retrievers/pubmed
'''

THENEWS = r'''An API to provide global news from thousands of sources with exceptional
		response times adding over 1 million articles weekly.
		
		https://www.thenewsapi.com/documentation
'''

GOOGLE_CSE = r'''The Cse Service is the endpoint that returns the requested searches.
		You must identify a particular search engine to use in your request
		(using the cx query parameter) as well as the search query (using the q query parameter).
		In addition, you should provide a developer key (using the key query parameter).
		
		https://developers.google.com/custom-search/v1/reference/rest
'''

GOOGLE_WEATHER = r'''The Weather API lets you request real-time, hyperlocal weather data for
		locations around the world. Weather information includes temperature, precipitation,
		humidity, and more.

		For a location at a given latitude and longitude, the API provides endpoints that let you query:

		Current conditions: The current weather conditions.
		Hourly forecast: Up to 240 hours of forecasted conditions for all elements.
		Daily forecast: Up to 10 days of forecasted conditions for all elements.
		Hourly history: Up to 24 hours of cached past conditions for all elements
		
		https://developers.google.com/maps/documentation/weather/overview
'''

US_NAVAL_OBSERVATORY = r'''Provides access to APIs from the US Naval Observatory's Celestial Navigation Data for
		Assumed Position and Time:  this data service provides all the astronomical information
		necessary to plot navigational lines of position from observations of the altitudes of
		celestial bodies. Simply fill in the form below and click on the "Get Data" button at
		the end of the form.

		The output table gives both almanac data and altitude corrections for each celestial body
		that is above the horizon at the place and time that you specify. Sea-level observations
		are assumed. The almanac data consist of Greenwich hour angle (GHA), declination (Dec),
		computed altitude (Hc), and computed azimuth (Zn). The altitude corrections consist of
		atmospheric refraction (Refr), semidiameter (SD), parallax in altitude (PA), and the sum of
		the altitude corrections (Sum = Refr + SD + PA). The SD and PA values are zero for stars.
		The SD values are non-zero only for the Sun and Moon; for all other objects, it is assumed
		that the center of light is observed.

		The assumed position that you enter below can be your best estimate of your actual
		location (e.g., your DR position); there is no need to round the coordinate values,
		since all data is computed specifically for the exact position you provide without
		any table lookup.

		https://aa.usno.navy.mil/data/api
'''

OPEN_SCIENCE = r'''Provides access to APIs from NASA's Open Science Data Repostitory (OSDR).
		NASA OSDR provides a RESTful Application Programming Interface (API) to its
		full-text search, data file retrieval, and metadata retrieval capabilities.
		The API provides a choice of standard web output formats,
		either JavaScript Object Notation (JSON) or Hyper Text Markup Language (HTML),
		of query results.
		
		The Data File API returns metadata on data files associated with dataset(s),
		including the location of these files for download via https. The Metadata API returns
		entire sets of metadata for input study dataset accession numbers. The Search API can be
		used to search dataset metadata by keywords and/or metadata. It can also be used to provide
		search of three other omics databases: the National Institutes of Health (NIH) /
		National Center for Biotechnology Information's (NCBI) Gene Expression Omnibus (GEO);
		the European Bioinformatics Institute's (EBI) Proteomics Identification (PRIDE);
		the Argonne National Laboratory's (ANL);
		Metagenomics Rapid Annotations using Subsystems Technology (MG-RAST).
		
		https://science.nasa.gov/biological-physical/data/osdr/
'''

GOV_INFO = r'''The GovInfo Link Service provides services for developers and webmasters to access
        content and metadata on GovInfo. Current and planned services include a link service,
        list service, and search service.
        
		The link service is used to create embedded links to content and metadata on GovInfo and is
		currently enabled for the collections below. The collection code is listed in parenthesis
		after each collection name, and the available queries are listed below each collection.
		More information about each query is provided on the individual collection page.
		
		https://www.govinfo.gov/link-docs/
'''

CONGRESS = r'''Submit queries against the Congressional Research Service's (CRS) Appropriation
		Status Table.
		
		https://api.congress.gov/
'''

INTERNET_ARCHIVE = r'''The Internet Archive, a 501(c)(3) non-profit, is building a digital library
		of Internet sites and other cultural artifacts in digital form. Like a paper library,
		we provide free access to researchers, historians, scholars, people with print disabilities,
		and the general public. Our mission is to provide Universal Access to All Knowledge.
		
		https://help.archive.org/help/search-a-basic-guide/
'''

THE_SATELLITE_CENTER = r'''Provides access to APIs from NASA's Satellite Situation Center Web (SSCWeb) service
		that is operated jointly by the NASA/GSFC Space Physics Data Facility (SPDF) and the
		National Space Science Data Center (NSSDC) to support a range of NASA science programs
		and to fulfill key international NASA responsibilities including those of NSSDC and the
		World Data Center-A for Rockets and Satellites.
		
		The software and associated database of SSCWeb together form a system to cast geocentric
		spacecraft location information into a framework of (empirical) geophysical regions and
		mappings of spacecraft locations along lines of the Earth's magnetic field.
		
		This capability is one key to mission science planning (both single missions and
		coordinated observations of multiple spacecraft with ground-based investigations) and to
		subsequent multi-mission data analysis.
		
		https://sscweb.gsfc.nasa.gov/index.html
'''

NEAR_BY_OBJECTS = r'''Provides access to APIs from JPL’s SSD (Solar System Dynamics) and CNEOS
		(Center for Near-Earth Object Studies) API (Application Program Interface) service.
		This service provides an interface to machine-readable data (JSON-format) related to SSD
		and CNEOS.
		
		https://ssd-api.jpl.nasa.gov/doc/
'''

ASTRONOMY_CATALOG = r'''Access to the Open Astronomy Catalog API (OACAPI) offers a lightweight,
		simple way to access data available via the api.
		
		The pattern for the API is one of the domains listed above followed by
		
			/OBJECT/QUANTITY/ATTRIBUTE?ARGUMENT1=VALUE1&ARGUMENT2=VALUE2&...
		
		where OBJECT is set to a transient's name, QUANTITY is set to a desired
		quantity to retrieve (e.g. redshift), ATTRIBUTE is a property of that quantity,
		and the ARGUMENT variables allow to user to filter data based upon various
		attribute values. The ARGUMENT variables can either be used to guarantee that
		a certain attribute appears in the returned results (e.g. adding &time&e_magnitude to
		the query will guarantee that each returned item has a time and e_magnitude attribute),
		or used to filter via a simple equality such as telescope=HST
		(which would only return QUANTITY objects where the telescope attribute equals "HST"),
		or they can be more powerful for certain filter attributes
		(examples being ra and dec for performing cone searches).
		
		https://astrocats.space/
'''

ASTRO_QUERY = r'''Access to the astropy package that contains key functionality and common tools needed for
		performing astronomy and astrophysics with Python. It is at the core of the Astropy Project,
		which aims to enable the community to develop a robust ecosystem of affiliated packages
		covering a broad range of needs for astronomical research, data processing, and data analysis.
		
		https://github.com/astropy/astroquery
'''

OPEN_WEATHER = r'''
		Provides forecast weather retrieval by location name using the Open-Meteo
		Geocoding API and Open-Meteo Forecast API.

		This class is forecast-only by design and intentionally excludes archive /
		historical date-based retrieval so it does not overlap with the separate
		HistoricalWeather class.

		Referenced API Requirements:
		----------------------------
		Geocoding API:
			- Endpoint: https://geocoding-api.open-meteo.com/v1/search
			- Required parameter: name
			- Optional parameter: count

		Forecast API:
			- Endpoint: https://api.open-meteo.com/v1/forecast
			- Required parameters: latitude, longitude
			- Optional parameters used here:
				- current
				- hourly
				- daily
				- timezone
				- forecast_days
				- past_days
				- temperature_unit
				- wind_speed_unit
				- precipitation_unit
				
		https://open-meteo.com/en/docs
'''

HISTORICAL_WEATHER = r'''Provides historical weather retrieval by location name and date using the
		Open-Meteo Geocoding API and Open-Meteo Historical Weather API.

		This class is intentionally designed around the actual user-facing need in
		the Foo fetcher expander: enter a location and a date, resolve that location
		to coordinates, then retrieve historical weather for that date.

		Referenced API Requirements:
		----------------------------
		Geocoding API:
			- Endpoint: https://geocoding-api.open-meteo.com/v1/search
			- Required parameter: name
			- Optional parameter: count

		Historical Weather API:
			- Endpoint: https://archive-api.open-meteo.com/v1/archive
			- Required parameters: latitude, longitude, start_date, end_date
			- Optional parameters used here:
				- timezone
				- daily
				- hourly
				- temperature_unit
				- wind_speed_unit
				- precipitation_unit
		
		https://open-meteo.com/en/docs
'''

EARTH_OBSERVATORY = r'''NASA Earth Observatory's Natural Event Tracker (EONET) allows users to access imagery,
				often in near real-time (NRT), of natural events such as dust storms, forest fires, and
				tropical cyclones—empowering people all across the planet to locate, track, and potentially
				prepare for and manage events that affect communities in their paths.
				Version 3 API for events, categories, sources, and layers.
		
				This class is aligned to the current documented EONET v3 API and supports:
				- events
				- categories
				- sources
				- layers
				
				https://eonet.gsfc.nasa.gov/docs/v3
'''

SPACE_WEATHER = r'''NASA DONKI (Space Weather Database Of Notifications, Knowledge, Information) is
		is a comprehensive on-line tool for space weather forecasters, scientists, and the general
		space science community. DONKI chronicles the daily interpretations of space weather observations,
		analysis, models, forecasts, and notifications provided by the Space Weather Research Center (SWRC),
		comprehensive knowledge-base search functionality to support anomaly resolution and space
		science research, intelligent linkages, relationships, cause-and-effects between space weather
		activities and comprehensive webservice API access to information stored in DONKI.
		
		https://api.nasa.gov/
'''

NEAR_BY_OBJECTS = r''''SSD (Solar System Dynamics) and CNEOS (Center for Near-Earth Object Studies)
		API (Application Program Interface) service. This service provides an interface to
		machine-readable data (JSON-format) related to SSD and CNEOS.
		
		https://ssd-api.jpl.nasa.gov/
'''

SKY_MAP = r'''Provides static and link-based star chart generation using the SKY-MAP.ORG
		XML API, Site Linker, and Image Generator interfaces.
		
		Referenced API Requirements:
		----------------------------
		XML Search:
			- Endpoint: https://server1.sky-map.org/search

		Site Linker:
			- Endpoint: https://www.sky-map.org/

		Image Generator:
			- Endpoint: https://server2.sky-map.org/map

		https://docs.gammapy.org/2.0.1/api-reference/maps.html
'''

OPEN_SCIENCE = r'''NASA’s Open Science Data Repository (OSDR) enables the reuse of comprehensive,
		multi-modal space life science data—including omics, physiological, phenotypic, behavioral,
		and environmental telemetry—to advance basic and applied research as well as operational
		outcomes for human space exploration.
		
		Documentation here (https://science.nasa.gov/biological-physical/data/osdr/)
'''

GROK = r'''
'''

CHATGPT = r'''
'''

GEMINI = r'''
'''

MISTRAL = r'''
'''

CLAUDE = r'''
'''

# -------- GENERATION PARAMETER DEFINITIONS -------------------

TEMPERATURE = r'''Optional. A number between 0 and 2. Higher values like 0.8 will make the output
		more random, while lower values like 0.2 will make it more focused and deterministic'''

TOP_P = r'''Optional. The maximum cumulative probability of tokens to consider when sampling.
		The model uses combined Top-k and Top-p (nucleus) sampling. Tokens are sorted based on
		their assigned probabilities so that only the most likely tokens are considered.
		Top-k sampling directly limits the maximum number of tokens to consider,
		while Nucleus sampling limits the number of tokens based on the cumulative probability.'''

TOP_K = r'''Optional. The maximum number of tokens to consider when sampling. Gemini models use
		Top-p (nucleus) sampling or a combination of Top-k and nucleus sampling. Top-k sampling considers
		the set of topK most probable tokens. Models running with nucleus sampling don't allow topK setting.
		Note: The default value varies by Model and is specified by theModel.top_p attribute returned
		from the getModel function. An empty topK attribute indicates that the model doesn't apply
		top-k sampling and doesn't allow setting topK on requests.'''

PRESENCE_PENALTY = r'''Optional. Presence penalty applied to the next token's logprobs
		if the token has already been seen in the response. This penalty is binary on/off
		and not dependant on the number of times the token is used (after the first).'''

FREQUENCY_PENALTY = r'''Optional. Frequency penalty applied to the next token's logprobs,
		multiplied by the number of times each token has been seen in the respponse so far.
		A positive penalty will discourage the use of tokens that have already been used,
		proportional to the number of times the token has been used: The more a token is used,
		the more difficult it is for the model to use that token again increasing
		the vocabulary of responses.'''

MAX_OUTPUT_TOKENS = r'''Optional. The maximum number of tokens used in generating output content'''

ALLOWED_DOMAINS = r'''Optional. The allowed domains used in generating output content and
		grounding of generated content to reduce halucinations.'''

STOP_SEQUENCE = r'''Optional. Up to 4 string sequences where the API will stop generating further tokens.'''

STORE = 'Optional. Whether to maintain state from turn to turn, preserving reasoning and tool context '

STREAM = 'Optional. Whether to return the generated respose in asynchronous chunks'

TOOLS = '''Optional. An array of tools the model may call while generating a response. You can specify which
		tool to use by setting the tool_choice parameter. Used by the Reponses API
		and Reasoning models'''

INCLUDE = r'''Optional. Specifies additional output data to include in the model response enabling reasoning
			items to be used in multi-turn conversations when using the Responses API statelessly
			and Reasoning models.
			'''

REASONING = r'''Optional. Reasoning models introduce reasoning tokens in addition to input and output tokens.
				The models use these reasoning tokens to “think,” breaking down the prompt and
				considering multiple approaches to generating a response. After generating reasoning tokens,
				the model produces an answer as visible completion tokens and discards
				the reasoning tokens from its context. Used by the Reasoning models'''