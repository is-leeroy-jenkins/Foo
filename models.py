'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                models.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="models.py" company="Terry D. Eppler">

	     Boo is a df analysis tool integrating various Generative GPT, GptText-Processing, and
	     Machine-Learning algorithms for federal analysts.
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
    models.py
  </summary>
  ******************************************************************************************
  '''
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class Prompt( BaseModel ):
	'''

		Purpose:
		--------
		Represents a structured “system prompt” or instruction bundle used to steer an LLM call.
		This model is intended to capture the canonical components you pass into Boo when you
		want to track prompts as first-class objects (versioning, variables, and provenance).

		Attributes:
		----------
		instructions: Optional[ str ]
			The primary instruction block (typically the system message content).

		context: Optional[ str ]
			Optional background context provided to the model (policies, references, etc.).

		output_indicator: Optional[ str ]
			A short indicator describing the desired output style/format (e.g., "json", "table").

		input_data: Optional[ str ]
			Optional data payload embedded into the prompt (small inputs, examples, etc.).

		id: Optional[ str ]
			Optional identifier for tracking prompts (e.g., GUID, hash, or friendly name).

		version: Optional[ str ]
			Optional version string for prompt management and experimentation.

		format: Optional[ str ]
			Optional format label describing the prompt template type (e.g., "chat", "completion").

		variables: Optional[ List[ str ] ]
			Optional list of placeholder variables referenced by the prompt template.

		question: Optional[ str ]
			Optional question or user query associated with the prompt.

	'''
	instructions: Optional[ str ]
	id: Optional[ str ]
	version: Optional[ str ]
	format: Optional[ str ]
	question: Optional[ str ]

class File( BaseModel ):
	'''

		Purpose:
		--------
		Represents a file-like object returned by an API (uploaded artifacts, generated files,
		or tool outputs). This is intentionally permissive: Boo only needs the common metadata.

		Attributes:
		----------
		filename: Optional[ str ]
			The original or assigned filename.

		bytes: Optional[ int ]
			The size of the file in bytes, if provided.

		created_at: Optional[ int ]
			Unix timestamp of creation (seconds), if provided.

		expires_at: Optional[ int ]
			Unix timestamp when the file expires, if the upstream supports expiring artifacts.

		id: Optional[ str ]
			Unique file identifier in the upstream system.

		object: Optional[ str ]
			Object discriminator from the upstream API (e.g., "file").

		purpose: Optional[ str ]
			Intended purpose for the file in the upstream system (e.g., "assistants", "fine-tune").

	'''
	filename: Optional[ str ]
	bytes: Optional[ int ]
	created_at: Optional[ int ]
	expires_at: Optional[ int ]
	id: Optional[ str ]
	object: Optional[ str ]
	purpose: Optional[ str ]

class Error( BaseModel ):
	'''

		Purpose:
		--------
		Represents an error object returned by an upstream API. Boo stores errors in structured
		form so UI and logging layers can surface details without brittle string parsing.

		Attributes:
		----------
		code: Optional[ str ]
			A short error code when available (e.g., "invalid_request_error").

		message: Optional[ str ]
			Human-readable error message.

	'''
	code: Optional[ str ]
	message: Optional[ str ]

class Reasoning( BaseModel ):
	'''

		Purpose:
		--------
		Represents reasoning configuration and/or summary data surfaced by an upstream model.
		Boo keeps this structured so callers can persist “reasoning metadata” without coupling
		to a specific vendor response shape.

		Attributes:
		----------
		effort: Optional[ str ]
			A label describing reasoning effort (when supported), e.g., "low", "medium", "high".

		summary: Optional[ str ]
			A short summary of reasoning (when the upstream provides it).

	'''
	effort: Optional[ str ]
	summary: Optional[ str ]

class Document( BaseModel ):
	'''

		Purpose:
		--------
		Represents a generic “document-like” structured output. Boo uses this for demos and
		for workflows where the model produces a multi-field narrative artifact with concepts.

		Attributes:
		----------
		invented_year: Optional[ int ]
			Example field used in some structured-output prompts; can be repurposed.

		summary: Optional[ str ]
			High-level summary of the document.

		inventors: Optional[ List[ str ] ]
			Example list field used in structured-output prompts.

		description: Optional[ str ]
			Long-form description.

		concepts: Optional[ List[ Concept ] ]
			List of extracted or described concepts.

	'''
	summary: Optional[ str ]
	description: Optional[ str ]

class Message( BaseModel ):
	'''

		Purpose:
		--------
		Represents a chat message-like object used by Boo to normalize conversational state.
		This is intentionally general to support both “input messages” and “output messages”.

		Attributes:
		----------
		content: str
			Message content payload. Boo treats this as required for operational messages.

		role: str
			Message role (e.g., "system", "user", "assistant", "tool").

		type: Optional[ str ]
			Optional discriminator if an upstream system emits typed message objects.

		instructions: Optional[ str ]
			Optional per-message instruction string (used in some orchestration patterns).

		data: Optional[ Dict ]
			Optional message metadata or additional structured payload.

	'''
	content: Optional[ str ]
	role: Optional[ str ]
	type: Optional[ str ]
	data: Optional[ Dict ]

class Location( BaseModel ):
	'''

		Purpose:
		--------
		Represents a high-level user location descriptor used by web search or other tools.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for location objects.

		city: Optional[ str ]
			City name.

		country: Optional[ str ]
			Country name or code.

		region: Optional[ str ]
			State/province/region.

		timezone: Optional[ str ]
		IANA timezone string when known.

	'''
	type: Optional[ str ]
	city: Optional[ str ]
	country: Optional[ str ]
	region: Optional[ str ]
	timezone: Optional[ str ]

class GeoCoordinates( BaseModel ):
	'''

		Purpose:
		--------
		Represents a latitude/longitude coordinate pair, optionally with a timezone. This is
		useful for tools like web search, maps, or proximity-based retrieval.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for geocoordinate objects.

		latitude: Optional[ float ]
			Latitude in decimal degrees.

		longitude: Optional[ float ]
			Longitude in decimal degrees.

		timezone: Optional[ str ]
			IANA timezone string when known.

	'''
	type: Optional[ str ]
	latitude: Optional[ float ]
	longitude: Optional[ float ]
	timezone: Optional[ str ]

class Forecast( BaseModel ):
	'''

		Purpose:
		--------
		Represents a simplified weather forecast payload returned by a tool or model.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the object.

		temperature: Optional[ int ]
			Temperature value (units depend on the tool/provider).

		precipitation: Optional[ int ]
			Precipitation percentage or amount (provider-specific).

		sky_conditions: Optional[ str ]
			Text description such as "clear", "cloudy", "rain".

	'''
	type: Optional[ str ]
	temperature: Optional[ int ]
	precipitation: Optional[ int ]
	sky_conditions: Optional[ str ]

class Directions( BaseModel ):
	'''

		Purpose:
		--------
		Represents a simplified directions/route payload returned by a mapping tool.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the object.

		route: Optional[ Any ]
			Route representation (provider-specific). Frequently this is a string/polyline or
			a structured list of steps.

	'''
	type: Optional[ str ]
	route: Optional[ Any ]
	
class SkyCoordinates( BaseModel ):
	'''

		Purpose:
		--------
		Represents right ascension / declination coordinate pairs used in astronomy-oriented
		structured outputs.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the object.

		declination: Optional[ float ]
			Declination in decimal degrees.

		right_ascension: Optional[ float ]
			Right ascension in decimal degrees or hours (provider-specific).

	'''
	type: Optional[ str ]
	declination: Optional[ float ]
	right_ascension: Optional[ float ]

class Tool( BaseModel ):
	'''

		Purpose:
		--------
		Represents a tool/function descriptor for tool-calling. Boo uses this to keep “tool
		specification” structured (name, description, JSON schema parameters, strictness).

		Attributes:
		----------
		name: Optional[ str ]
			Function name as exposed to the model.

		type: Optional[ str ]
			Type discriminator (commonly "function") when used in tool lists.

		description: Optional[ str ]
			Human-readable description of the tool/function.

		parameters: Optional[ Dict[ str, Any ] ]
			JSON Schema-like parameters object describing accepted inputs.

		strict: Optional[ bool ]
			Whether the upstream should strictly validate arguments against the schema.

	'''
	name: Optional[ str ]
	type: Optional[ str ]
	description: Optional[ str ]

class Function( Tool ):
	'''

		Purpose:
		--------
		Represents a tool/function descriptor for tool-calling. Boo uses this to keep “tool
		specification” structured (name, description, JSON schema parameters, strictness).

		Attributes:
		----------
		name: Optional[ str ]
			Function name as exposed to the model.

		type: Optional[ str ]
			Type discriminator (commonly "function") when used in tool lists.

		description: Optional[ str ]
			Human-readable description of the tool/function.

		parameters: Optional[ Dict[ str, Any ] ]
			JSON Schema-like parameters object describing accepted inputs.

		strict: Optional[ bool ]
			Whether the upstream should strictly validate arguments against the schema.

	'''
	parameters: Optional[ Dict[ str, Any ] ]
	strict: Optional[ bool ]

class FileSearch( Tool ):
	'''

		Purpose:
		--------
		Represents configuration for a file-search tool invocation. Boo uses this to keep tool
		config structured and to support serialization/rehydration of tool configurations.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the tool (commonly "file_search").

		vector_store_ids: Optional[ List[ str ] ]
			Vector store identifiers available to the search tool.

		max_num_results: Optional[ int ]
			Maximum number of results to return.

		filters: Optional[ Dict[ str, Any ] ]
			Optional filter object (metadata filters, namespace filters, etc.).

	'''
	vector_store_ids: Optional[ List[ str ] ]
	max_num_results: Optional[ int ]
	filters: Optional[ Dict[ str, Any ] ]

class WebSearch( Tool ):
	'''

		Purpose:
		--------
		Represents configuration for a web-search tool invocation.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the tool (commonly "web_search").

		search_context_size: Optional[ str ]
			Desired context size (vendor-specific; common values are "low", "medium", "high").

		user_location: Optional[ Any ]
			Optional location descriptor to bias search results. This may be a Location,
			GeoCoordinates, or a vendor-specific object.

	'''
	type: Optional[ str ]
	search_context_size: Optional[ str ]
	user_location: Optional[ Any ]

class ComputerUse( Tool ):
	'''

		Purpose:
		--------
		Represents configuration for a computer-use tool invocation (UI automation / virtual
		display sessions).

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the tool (commonly "computer_use").

		display_height: Optional[ int ]
			Height (pixels) of the virtual display.

		display_width: Optional[ int ]
			Width (pixels) of the virtual display.

		environment: Optional[ str ]
			Environment label (e.g., "browser", "desktop") when supported by the tool provider.

	'''
	type: Optional[ str ]
	display_height: Optional[ int ]
	display_width: Optional[ int ]
	environment: Optional[ str ]
