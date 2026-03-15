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
from typing import Optional, List, Dict

# ------ ENVIRONMENT API KEYS  -------------------
ACCESS_DRIVER = r'DRIVER={ Microsoft Access Driver (*.mdb, *.accdb) };DBQ='
CLAUDE_API_KEY = os.getenv( 'CLAUDE_API_KEY' )
CONGRESS_API_KEY = os.getenv( 'CONGRESS_API_KEY' )
GEOAPIFY_API_KEY = os.getenv( 'GEOAPIFY_API_KEY' )
GEOCODING_API_KEY = os.getenv( 'GEOCODING_API_KEY' )
GEMINI_API_KEY = os.getenv( 'GEMINI_API_KEY' )
GOOGLE_API_KEY = os.getenv( 'GOOGLE_API_KEY' )
GOOGLE_CSE_ID = os.getenv( 'GOOGLE_CSE_ID' )
GOOGLE_CLOUD_PROJECT = os.getenv( 'GOOGLE_CLOUD_PROJECT' )
GOVINFO_API_KEY = os.getenv( 'GOVINFO_API_KEY' )
GOOGLE_CLOUD_LOCATION = os.getenv( 'GOOGLE_CLOUD_LOCATION' )
GOOGLE_GENAI_USE_VERTEXAI = os.getenv( 'GOOGLE_GENAI_USE_VERTEXAI' )
GOOGLE_PROJECT_ID = os.getenv( 'GOOGLE_PROJECT_ID' )
GOOGLE_WEATHER_API_KEY = os.getenv( 'GOOGLE_WEATHER_API_KEY' )
GOOGLE_ACCOUNT_FILE = os.getenv( 'GOOGLE_ACCOUNT_FILE' )
GOOGLE_DRIVE_TOKEN_PATH = os.getenv( 'GOOGLE_DRIVE_TOKEN_PATH' )
GOOGLE_DRIVE_FOLDER_ID = os.getenv( 'GOOGLE_DRIVE_FOLDER_ID' )
GROQ_API_KEY = os.getenv( 'GROQ_API_KEY' )
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
MODES = [ 'Document Loading', 'Web Scrapping', 'Data Collection',
          'Generative AI', 'Satellite Data', 'Data Management' ]




# ------------- API DEFINITIONS ------------------

ARXIV = r'''arXiv is a free distribution service and an open-access archive for nearly 2.4 million
		scholarly articles in the fields of physics, mathematics, computer science, quantitative
		biology, quantitative finance, statistics, electrical engineering and systems science, and
		economics. Materials on this site are not peer-reviewed by arXiv.
		
		
		Documentation here  (https://docs.langchain.com/oss/javascript/integrations/retrievers/arxiv-retriever)
'''

GOOGLE_DRIVE = r'''Google Drive is a file storage and synchronization service developed by Google

		Documentation here  (https://docs.langchain.com/oss/python/integrations/document_loaders/google_drive)
'''

WIKIPEDIA = r'''Wikipedia is a free, multilingual online encyclopedia created and maintained by
			a global community of volunteers through open collaboration. Offering over 55 million
			articles in over 300 languages.
			
			Documentation here (https://wikipedia-api.readthedocs.io/en/latest/wikipediaapi/api.html)
'''

THENEWS = r'''An API to provide global news from thousands of sources with exceptional
		response times adding over 1 million articles weekly.
		
		Documentation here (https://www.thenewsapi.com/documentation)
'''

GOOGLE_CSE = r'''The Cse Service is the endpoint that returns the requested searches.
		You must identify a particular search engine to use in your request
		(using the cx query parameter) as well as the search query (using the q query parameter).
		In addition, you should provide a developer key (using the key query parameter).
		
		Documentation here (https://developers.google.com/custom-search/v1/cse)
'''

GOOGLE_WEATHER = r'''The Weather API lets you request real-time, hyperlocal weather data for
		locations around the world. Weather information includes temperature, precipitation,
		humidity, and more.

		For a location at a given latitude and longitude, the API provides endpoints that let you query:

		Current conditions: The current weather conditions.
		Hourly forecast: Up to 240 hours of forecasted conditions for all elements.
		Daily forecast: Up to 10 days of forecasted conditions for all elements.
		Hourly history: Up to 24 hours of cached past conditions for all elements
		
		Documentation here (https://developers.google.com/maps/documentation/weather/overview)
'''

USNO = r'''Provides access to APIs from the US Naval Observatory's Celestial Navigation Data for
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

		Documentation here (https://aa.usno.navy.mil/data/api)
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
		
		Documentation here (https://science.nasa.gov/biological-physical/data/osdr/)
'''

GOV_INFO = r'''The GovInfo Link Service provides services for developers and webmasters to access
		content and metadata on GovInfo. Current and planned services include a link service,
		list service, and search service.
		
		Documentation here (https://www.govinfo.gov/link-docs/)
'''

CONGRESS = r'''Provides  service that can be used to query the GovInfo search engine and return results
		that are the equivalent to what is returned by the main user interface.

		You can use field operators, such as congress, published date, branch, and others to construct
		complex queries that will return only matching documents.
		
		Documentation here (https://api.congress.gov/)
'''

INTERNET_ARCHIVE = r'''The Internet Archive, a 501(c)(3) non-profit, is building a digital library
		of Internet sites and other cultural artifacts in digital form. Like a paper library,
		we provide free access to researchers, historians, scholars, people with print disabilities,
		and the general public. Our mission is to provide Universal Access to All Knowledge.
		
		Documentation here (https://help.archive.org/help/search-a-basic-guide/)
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
		
		Documentation here (https://sscweb.gsfc.nasa.gov/index.html)
'''

NEAR_BY_OBJECTS = r'''Provides access to APIs from JPL’s SSD (Solar System Dynamics) and CNEOS
		(Center for Near-Earth Object Studies) API (Application Program Interface) service.
		This service provides an interface to machine-readable data (JSON-format) related to SSD
		and CNEOS.
		
		Documentation here (https://ssd-api.jpl.nasa.gov/doc/)
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
		
		Documnetation here (https://astrocats.space/)
'''

ASTRO_QUERY = r'''Access to the astropy package that contains key functionality and common tools needed for
		performing astronomy and astrophysics with Python. It is at the core of the Astropy Project,
		which aims to enable the community to develop a robust ecosystem of affiliated packages
		covering a broad range of needs for astronomical research, data processing, and data analysis.
		
		Documentation here (https://github.com/astropy/astroquery)
'''