'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                name.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="guro.py" company="Terry D. Eppler">

	     name.py
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
    name.py
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

import base64
import config as cfg
import astroquery
import uuid
import streamlit as st
import os

from typing import Dict, List

# ---------------------------------------------------------------------------
# Foo imports (engine only – no UI coupling)
# ---------------------------------------------------------------------------
import config

# ---------------------------------------------------------------------------
# Workspace imports (render-only contracts)
# ---------------------------------------------------------------------------
from workspaces.web_extraction import render as render_web_extraction
from workspaces.search_retrieval import render as render_search
from workspaces.document_ingestion import render as render_documents

# ============================================================================
# Session State Initialization
# ============================================================================

def initialize_session( ) -> None:
	"""
	Purpose:
		Initialize all required session state keys exactly once.

	Returns:
		None
	"""
	
	if "session_id" not in st.session_state:
		st.session_state.session_id = str( uuid.uuid4( ) )
	
	if "active_workspace" not in st.session_state:
		st.session_state.active_workspace = "Web Extraction"
	
	if "runs" not in st.session_state:
		st.session_state.runs = [ ]
	
	if "results" not in st.session_state:
		st.session_state.results = { }
	
	if "documents" not in st.session_state:
		st.session_state.documents = { }
	
	if "diagnostics" not in st.session_state:
		st.session_state.diagnostics = [ ]

# ============================================================================
# Provider Status Helpers
# ============================================================================

def provider_status( ) -> Dict[ str, bool ]:
	"""
	Purpose:
		Determine which external providers are configured based on Foo config.

	Returns:
		Dict[str, bool]: Provider name → configured flag
	"""
	
	return {
			"OpenAI": bool( config.OPENAI_API_KEY ),
			"Gemini": bool( config.GEMINI_API_KEY ),
			"Groq": bool( config.GROQ_API_KEY ),
			"Google CSE": bool( config.GOOGLE_CSE_ID ),
			"News API": bool( config.NEWS_API_KEY ),
			"ArXiv": True,  # No key required
	}

# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar_logo() -> None:
    """
    Render the Foo logo centered at the top of the sidebar.
    """

    logo_path = os.path.join("resources", "images", "foo_logo.png")

    if not os.path.exists(logo_path):
        return

    with open(logo_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.sidebar.markdown(
        f"""
        <div style="display: flex; justify-content: left; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{encoded}" width="70"/>
        </div>
        """,
        unsafe_allow_html=True,
    )
		
render_sidebar_logo( )

def render_sidebar( ) -> None:
	"""
	Purpose:
		Render the persistent sidebar controls.
	"""
	st.sidebar.title( "Control Room" )
	# Workspace selector
	workspace = st.sidebar.radio(
		"Workspace",
		[
				"Web Extraction",
				"Search & Retrieval",
				"Document Ingestion",
		],
		index=0,
	)
	
	st.session_state.active_workspace = workspace
	
	st.sidebar.divider( )
	
	# Provider health
	st.sidebar.subheader( "Provider Status" )
	
	status = provider_status( )
	for name, ok in status.items( ):
		st.sidebar.markdown(
			f"- **{name}**: {'Configured' if ok else 'Missing'}"
		)
	
	st.sidebar.divider( )
	
	# Session controls
	if st.sidebar.button( "Clear Session" ):
		for key in list( st.session_state.keys( ) ):
			del st.session_state[ key ]
		st.experimental_rerun( )

# ============================================================================
# Workspace Router
# ============================================================================

def render_workspace( ) -> None:
	"""
	Purpose:
		Dispatch rendering to the active workspace.
	"""
	
	workspace = st.session_state.active_workspace
	
	if workspace == "Web Extraction":
		render_web_extraction( st.session_state )
	elif workspace == "Search & Retrieval":
		render_search( st.session_state )
	elif workspace == "Document Ingestion":
		render_documents( st.session_state )
	else:
		st.error( "Unknown workspace selected." )

# ============================================================================
# Main Entry
# ============================================================================

def main( ) -> None:
	"""
	Purpose:
		Application entry point.
	"""
	
	st.set_page_config(
		page_title="foo",
		layout="wide",
		page_icon=cfg.FAVICON
	)
	
	initialize_session( )
	render_sidebar( )
	render_workspace( )

if __name__ == "__main__":
	main( )
