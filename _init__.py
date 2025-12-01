'''
    ******************************************************************************************
      Assembly:                Foo
      Filename:                budget_fiscal_year.py
      Author:                  Terry D. Eppler
      Created:                 08-26-2025

      Last Modified By:        Terry D. Eppler
      Last Modified On:        08-26-2025
    ******************************************************************************************
    <copyright file="init.py" company="Terry D. Eppler">

         Budget tempus Year Tools

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
        Schetchy
    </summary>
    ******************************************************************************************
'''
from __future__ import annotations
import glob
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional
from typing import Tuple
import chromadb
import numpy as np
from boogr import Error, ErrorDialog  # type: ignore
from chromadb.config import Settings
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools.base import Tool
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

try:
	from langchain_core.documents import Document
except Exception:
	from langchain.schema import Document

from langchain_community.document_loaders import (
	CSVLoader,
	Docx2txtLoader,
	PyPDFLoader,
	UnstructuredExcelLoader,
	WebBaseLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Fetch( ):
	"""
	
		Purpose:
		---------
		Provides a unified conversational system with explicit methods for
		querying structured data (SQL), unstructured documents, or free-form
		chat with an OpenAI LLM. Each method is deterministic and isolates
		a specific capability.
	
		Parameters:
		-----------
		db_uri (str):
		URI string for the SQLite database connection.
		doc_paths (List[str]):
		File paths to documents (txt, pdf, csv, html) for ingestion.
		model (str, optional):
		OpenAI model to use (default: 'gpt-4o-mini').
		temperature (float, optional):
		Sampling temperature for the LLM (default: 0.8).
	
		Attributes:
		----------
		model (str): OpenAI model identifier.
		temperature (float): Temperature setting for sampling.
		llm (ChatOpenAI): Instantiated OpenAI-compatible chat model.
		db_uri (str): SQLite database URI.
		doc_paths (List[str]): Paths to local document sources.
		memory (ConversationBufferMemory): LangChain conversation buffer.
		sql_tool (Optional[Tool]): SQL query tool.
		doc_tool (Optional[Tool]): Vector document retrieval tool.
		api_tools (List[Tool]): List of custom API tools.
		agent (AgentExecutor): LangChain multi-tool agent.
		__tools (List[Tool]): Consolidated tool list used by agent.
		documents (List[str]): Cached document source text or metadata.
		db_toolkit (Optional[object]): SQLDatabaseToolkit instance.
		database (Optional[object]): Underlying SQLAlchemy database.
		loader (Optional[object]): Last-used document loader.
		tool (Optional[object]): Active retrieval tool.
		extension (Optional[str]): File extension for routing.
		
	"""
	model: str
	temperature: float
	llm: ChatOpenAI
	db_uri: str
	doc_paths: List[ str ]
	memory: ConversationBufferMemory
	sql_tool: Optional[ Tool ]
	doc_tool: Optional[ Tool ]
	api_tools: List[ Tool ]
	agent: AgentExecutor
	__tools: List[ Tool ]
	documents: List[ str ]
	db_toolkit: Optional[ object ]
	database: Optional[ object ]
	loader: Optional[ object ]
	tool: Optional[ object ]
	extension: Optional[ str ]
	answer: Optional[ Dict ]
	sources: Optional[ Dict[ str, str ] ]
	
	def __init__( self, db_uri: str, doc_paths: List[ str ], model: str = 'gpt-4o-mini',
			temperature: float = 0.8 ):
		"""
		
			Purpose:
			--------
			Initializes the Fetch system and configures tools for SQL,
			document retrieval, and conversational use.
	
			Parameters:
			-----------
			db_uri (str): Path or URI to SQLite database.
			doc_paths (List[str]): Files to be processed for retrieval.
			model (str): LLM model name (default: gpt-4o-mini).
			temperature (float): Sampling diversity (default: 0.8).
	
			Returns:
			-----------
			None
			
		"""
		self.model = model
		self.temperature = temperature
		self.llm = ChatOpenAI( model=self.model, temperature=self.temperature, streaming=True )
		self.db_uri = db_uri
		self.doc_paths = doc_paths
		self.memory = ConversationBufferMemory( memory_key='chat_history', return_messages=True )
		self.sql_tool = self._init_sql_tool( )
		self.doc_tool = self._init_doc_tool( )
		self.api_tools = self._init_api_tools( )
		self.documents = [ ]
		self.db_toolkit = None
		self.database = None
		self.loader = None
		self.tool = None
		self.extension = None
		self.answer = { }
		self.__tools = [ t for t in [ self.sql_tool,
		                              self.doc_tool ] + self.api_tools if t is not None ]
		self.agent = initialize_agent( tools=self.__tools, llm=self.llm, memory=self.memory,
			agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True )
	
	def query_sql( self, question: str ) -> str | None:
		"""
		
			Purpose:
				Answer a question using ONLY the SQL database tool.
	
			Parameters:
				question (str): Natural language SQL-like question.
	
			Returns:
				str: Answer from the SQL query tool.
				
		"""
		try:
			throw_if( 'question', question )
			return self.sql_tool.func( question )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Fetch'
			exception.method = 'query_sql(self, question)'
			error = ErrorDialog( exception )
			error.show( )
	
	def query_docs( self, question: str, with_sources: bool=False ) -> str | None:
		"""
		
			Purpose:
			-----------
			Answer a question using ONLY the document retrieval tool.
	
			Parameters:
			-----------
			question (str):
			Natural language question grounded in the loaded documents.
			with_sources (bool):
			If True, returns sources alongside the answer.
	
			Returns:
			-----------
			str: Response from the document retriever. Includes sources if available.
			
		"""
		try:
			throw_if( 'question', question )
			if with_sources:
				if self.doc_chain_with_sources is None:
					raise RuntimeError( 'Document chain with sources is not available' )
				
				result = self.doc_chain_with_sources( {'question': question } )
				if 'answer' not in result or 'sources' not in result:
					raise RuntimeError( 'Malformed response from doc_chain_with_sources' )
				
				answer = result[ 'answer' ]
				sources = result[ 'sources' ]
				
				if sources:
					return f'{answer}\n\nSOURCES:\n{sources}'
				return answer
			else:
				return self.doc_tool.func( question )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Fetch'
			exception.method = 'query_docs(self, question, with_sources)'
			error = ErrorDialog( exception )
			error.show( )
	
	def query_chat( self, prompt: str ) -> str | None:
		"""
		
			Purpose:
			-----------
			Send a general-purpose prompt directly to the LLM without using
			tools, but with full memory context.
	
			Parameters:
			-----------
			prompt (str): User message for free-form reasoning.
	
			Returns:
			-----------
			str: LLM-generated conversational response.
			
		"""
		try:
			throw_if( 'prompt', prompt )
			return self.llm.invoke( prompt ).content
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Fetch'
			exception.method = 'query_chat(self, prompt)'
			error = ErrorDialog( exception )
			error.show( )

class SQLite:
	"""

		Purpose:
		-----------
		Manages persistent storage and retrieval of text chunks and their embedding vectors
		using a local SQLite database.

		Parameters:
		-----------
		db_path (str):
		Path to the SQLite database file.

		Attributes:
		-----------
		db_path (str): File path for the SQLite connection.
		conn (sqlite3.Connection): SQLite database connection.
		cursor (sqlite3.Cursor): Cursor used for SQL execution.

	"""
	db_path: str
	connection: sqlite3.Connection
	cursor: sqlite3.Cursor
	
	def __init__( self, db_path: str = "./embeddings.db" ) -> None:
		"""

			Purpose:
				Establishes a SQLite connection and initializes the embeddings table schema.

			Parameters:
				db_path (str): Path to the SQLite file used for storage.

			Returns:
				None

		"""
		self.db_path = db_path
		self.connection = sqlite3.connect( self.db_path )
		self.cursor = self.connection.cursor( )
		self.create( )
	
	def create( self ) -> None:
		"""

			Purpose:
			-------
			
			Creates the 'embeddings' table with appropriate schema if it does not already exist.

			Returns:
			None

		"""
		try:
			sql = """CREATE TABLE IF NOT EXISTS embeddings
                     (
                         id          INTEGER PRIMARY KEY AUTOINCREMENT,
                         source_file TEXT    NOT NULL,
                         chunk_index INTEGER NOT NULL,
                         chunk_text  TEXT    NOT NULL,
                         embedding   TEXT    NOT NULL,
                         created_at  TEXT DEFAULT CURRENT_TIMESTAMP
                     )""" 
			self.cursor.execute( sql )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'SQLie'
			exception.method = 'create( self ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def insert( self, source_file: str, index: int,
			text: str, embedding: np.ndarray ) -> None:
		"""

			Purpose:
				Inserts a single embedding record with metadata into the database.

			Parameters:
				source_file (str): Name or path of the source document.
				index (int): Ordinal position of the chunk.
				text (str): Cleaned text of the chunk.
				embedding (np.ndarray): Vector representation of the chunk.

			Returns:
				None

		"""
		try:
			throw_if( 'source_file', source_file )
			throw_if( 'index', index )
			throw_if( 'text', text )
			throw_if( 'embedding', embedding )
			vector_str = json.dumps( embedding.tolist( ) )
			sql = """INSERT INTO embeddings (source_file, chunk_index, chunk_text,  embedding)
			         VALUES (?, ?, ?, ?)"""
			self.cursor.execute( sql, (source_file, index, text, vector_str) )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'SQLite'
			exception.method = 'insert( )'
			error = ErrorDialog( exception )
			error.show( )
	
	def insert_many( self, source_file: str, chunks: List[ str ], vectors: np.ndarray ) -> None:
		"""

			Purpose:
				Batch inserts multiple chunks and their embeddings into the database.

			Parameters:
				source_file (str): Name or path of the source document.
				chunks (List[str]): List of cleaned text chunks.
				vectors (np.ndarray): Matrix of embedding vectors.

			Returns:
				None

		"""
		try:
			records = [ (source_file, i, chunks[ i ], json.dumps( vectors[ i ].tolist( ) ))
			            for i in range( len( chunks ) ) ]
			sql_insert = """INSERT INTO embeddings (source_file, chunk_index, chunk_text, embedding)
			         VALUES (?, ?, ?, ?)"""
			self.cursor.executemany( sql_insert, records )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'SQLite'
			exception.method = 'insert_many( )'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_all( self ) -> Tuple[ List[ str ], np.ndarray ] | None:
		'''

			Purpose:
			--------
			Retrieves all text chunks and their embeddings from the database.

			Returns:
			--------
			Tuple[List[str], np.ndarray]:
			List of texts and matrix of embeddings.

		'''
		try:
			self.cursor.execute( 'SELECT chunk_text, embedding FROM embeddings' )
			rows = self.cursor.fetchall( )
			texts, vectors = [ ], [ ]
			for text, emb in rows:
				texts.append( text )
				vectors.append( np.array( json.loads( emb ) ) )
			return texts, np.array( vectors )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'SQLite'
			exception.method = 'fetch_all( self ) -> Tuple[ List[ str ], np.ndarray ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_by_file( self, file: str ) -> Tuple[ List[ str ], np.ndarray ] | None:
		'''

			Purpose:
			Retrieves all records associated with a specific file.

			Parameters:
			file (str): Identifier of the source file.

			Returns:
			Tuple[List[str], np.ndarray]:
			Filtered texts and embeddings.

		'''
		try:
			self.cursor.execute( '''
                                 SELECT chunk_text, embedding
                                 FROM embeddings
                                 WHERE source_file = ?
			                     ''', (file,) )
			rows = self.cursor.fetchall( )
			texts, vectors = [ ], [ ]
			for text, emb in rows:
				texts.append( text )
				vectors.append( np.array( json.loads( emb ) ) )
			return texts, np.array( vectors )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'SQLite'
			exception.method = 'fetch_by_file( self, file: str ) -> Tuple[ List[ str ], np.ndarray ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def delete_by_file( self, file: str ) -> None:
		'''

			Purpose:
			Deletes all embeddings associated with a given source file.

			Parameters:
			file (str): Source file whose records are to be deleted.

			Returns:
			None

		'''
		try:
			self.cursor.execute( 'DELETE FROM embeddings WHERE source_file = ?', (file,) )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'SQLite'
			exception.method = 'delete_by_file( self, file: str ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def update_embedding_by_id( self, row_id: int, new_embedding: np.ndarray ) -> None:
		'''

			Purpose:
			Updates an embedding vector in the database by record ID.

			Parameters:
			row_id (int): ID of the record to update.
			new_embedding (np.ndarray): New embedding vector.

			Returns:
			None

		'''
		try:
			vector_str = json.dumps( new_embedding.tolist( ) )
			self.cursor.execute( '''
                                 UPDATE embeddings
                                 SET embedding = ?
                                 WHERE id = ?
			                     ''', (vector_str, row_id) )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'SQLite'
			exception.method = 'update_embedding_by_id( )'
			error = ErrorDialog( exception )
			error.show( )
	
	def count( self ) -> int | None:
		'''

			Purpose:
			Returns the total number of records stored in the embeddings table.

			Parameters:
			None

			Returns:
			int: Number of rows in the table.

		'''
		try:
			self.cursor.execute( 'SELECT COUNT(*) FROM embeddings' )
			return self.cursor.fetchone( )[ 0 ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'SQLite'
			exception.method = 'count( ) -> int'
			error = ErrorDialog( exception )
			error.show( )
	
	def purge_all( self ) -> None:
		'''

			Purpose:
			Deletes all data from the embeddings table without altering the schema.

			Parameters:
			None

			Returns:
			None

		'''
		try:
			self.cursor.execute( 'DELETE FROM embeddings' )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'SQLite'
			exception.method = 'purge_all( )'
			error = ErrorDialog( exception )
			error.show( )
	
	def close( self ) -> None:
		'''

			Purpose:
			---------
			Closes the database connection.

			Parameters:
			----------
			None

			Returns:
			------
			None

		'''
		try:
			self.connection.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'SQLite'
			exception.method = 'close( )'
			error = ErrorDialog( exception )
			error.show( )

class Chroma:
	'''

		Purpose:
		--------
		Provides persistent storage and retrieval of sentence-level embeddings using ChromaDB.
		Supports adding documents, semantic querying, deletion by ID, and disk persistence.

		Parameters:
		-----------
		persist_path (str):  Filesystem path for storing ChromaDB collections.
		collection_name (str):  Logical name of the vector store collection.

		Attributes:
		---------
		client (chromadb.Client): Instantiated Chroma client.
		collection (chromadb.Collection): Vector collection used for insert and query.

	'''
	client: chromadb.Client
	collection: chromadb.Collection
	
	def __init__( self, path: st = './chroma', collection: str = 'embeddings' ) -> None:
		'''

			Purpose:
			-------
			Initializes a persistent Chroma vector database collection.

			Parameters:
			-----------
			persist_path (str): Directory to persist collection to disk.
			collection_name (str): Identifier for the Chroma collection.

			Returns:
			--------
			None

		'''
		self.client = chromadb.Client( Settings( persist_directory=path, anonymized_telemetry=False ) )
		self.collection = self.client.get_or_create_collection( name=collection )
	
	def add( self, ids: List[ str ], texts: List[ str ], embeddings: List[ List[ float ] ],
			metadatas: Optional[ List[ dict ] ] = None ) -> None:
		'''

			Purpose:
			---------
			Adds documents, embeddings, and optional metadata to the vector store.

			Parameters:
			----------
			ids (List[str]): Unique identifiers for each record.
			texts (List[str]): Corresponding document strings.
			embeddings (List[List[float]]): Vector representations of documents.
			metadatas (Optional[List[dict]]): Optional metadata for filtering or tagging.

			Returns:
			------
			None

		'''
		try:
			self.collection.add( documents=texts, embeddings=embeddings, ids=ids,
				metadatas=metadatas )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Chroma'
			exception.method = 'add( )'
			error = ErrorDialog( exception )
			error.show( )
	
	def query( self, text: List[ str ], num: int = 5, where: Optional[ dict ] = None ) -> List[ str  ] | None:
		'''

			Purpose:
			--------
			Performs similarity-based vector search using provided queries.

			Parameters:
			-----------
			query_texts (List[str]): List of queries to run.
			n_results (int): Number of top matches to return.
			where (Optional[dict]): Optional metadata filter to apply.

			Returns:
			--------
			List[str]: Most relevant documents based on vector similarity.

		'''
		try:
			result = self.collection.query( query_texts=text, n_results=num, where=where or { } )
			return result.get( 'documents', [ ] )[ 0 ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Chroma'
			exception.method = 'query( )'
			error = ErrorDialog( exception )
			error.show( )
	
	def delete( self, ids: List[ str ] ) -> None:
		'''

			Purpose:
			Deletes one or more records from the collection by document ID.

			Parameters:
			ids (List[str]): List of unique document IDs to delete.

			Returns:
			None

		'''
		try:
			self.collection.delete( ids=ids )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Chroma'
			exception.method = 'delete( )'
			error = ErrorDialog( exception )
			error.show( )
	
	def count( self ) -> int | None:
		'''

			Purpose:
			Returns the total number of records in the collection.

			Returns:
			int: Row count of stored vectors.

		'''
		try:
			return self.collection.count( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Chroma'
			exception.method = 'count( )'
			error = ErrorDialog( exception )
			error.show( )
	
	def clear( self ) -> None:
		'''

			Purpose:
			Deletes all documents from the collection.

			Parameters:
			None

			Returns:
			None

		'''
		try:
			self.collection.delete( where={ } )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Chroma'
			exception.method = 'clear( )'
			error = ErrorDialog( exception )
			error.show( )
	
	def persist( self ) -> None:
		'''

			Purpose:
			Saves the current state of the collection to disk.

			Returns:
			None

		'''
		try:
			self.client.persist( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Chroma'
			exception.method = 'persist'
			error = ErrorDialog( exception )
			error.show( )

class Loader( ):
	'''
	
		Purpose:
		--------
		Base class providing shared utilities for concrete loader wrappers.
		Encapsulates file validation, path resolution, and document splitting.
		
		Attributes:
		----------
		documents - List[ Document ]
		file_path -  str
		pattern -  str
		expanded - List[ str ]
		candidates - List[ str ]
		resolved - List[ str ]
		splitter - RecursiveCharacterTextSplitter
		chunk_size - int
		overlap_amount - int
		
		Methods:
		-------
		_ensure_existing_file( self, path: str ) -> str
		_resolve_paths( self, pattern: str ) -> List[ str ]
		_split_documents( self, docs: List[ Document ], chunk: int=1000, overlap: int=200 ) ->
		List[ Document ]
		
	'''
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	pattern: Optional[ str ]
	expanded: Optional[ List[ str ] ]
	candidates: Optional[ List[ str ] ]
	resolved: Optional[ List[ str ] ]
	loader: Optional[ RecursiveCharacterTextSplitter ]
	chunk_size: Optional[ int ]
	overlap_amount: Optional[ int ]
	
	def __init__( self ) -> None:
		self.documents = [ ]
		self.candidates = [ ]
		self.resolved = [ ]
		self.expanded = [ ]
		self.file_path = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None

	def verify_exists( self, path: str ) -> str | None:
		'''
		
			Purpose:
			--------
			Ensure the given file path exists.
	
			Parameters:
			-----------
			path (str): Path to a file on disk.
	
			Returns:
			--------
			str: The validated file path.
			
		'''
		try:
			throw_if( 'path', path )
			self.file_path = path
			if not os.path.isfile( self.file_path ):
				raise FileNotFoundError( f'File not found: {self.file_path}' )
			else:
				self.file_path = path
			return self.file_path
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Loader'
			exception.method = '_ensure_existing_file( self, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def _resolve_paths( self, pattern: str ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Normalize a string glob pattern or a list of paths to a list of real file paths.
	
			Parameters:
			-----------
			pattern (str | List[str]): Path pattern or list of file paths.
	
			Returns:
			--------
			List[str]: Validated list of file paths.
			
		'''
		try:
			throw_if( 'pattern', pattern )
			self.candidates.append( pattern )
			for p in self.candidates:
				if os.path.isfile( p ):
					self.resolved.append( p )
				else:
					for m in glob.glob( p ):
						if os.path.isfile( m ):
							self.resolved.append( m )
			
			if not self.resolved:
				raise FileNotFoundError( f'No files matched or existed for input: {pattern}' )
			return sorted( set( self.resolved ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Loader'
			exception.method = '_resolve_paths( self, pattern: str ) -> List[ str ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def _split_documents( self, documents: List[ Document ], chunk: int=1000 , overlap: int=200 ) \
			-> \
	List[ Document ] | None:
		'''
		
			Purpose:
			--------
			Split long Document objects into smaller chunks for better token management.
	
			Parameters:
			-----------
			docs (List[Document]): Input LangChain Document objects.
			chunk_size (int): Max characters in each chunk.
			chunk_overlap (int): Overlapping characters between chunks.
	
			Returns:
			--------
			List[Document]: Re-chunked list of Document objects.
			
		'''
		try:
			throw_if( 'docs', documents )
			self.documents = documents
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.loader = RecursiveCharacterTextSplitter( chunk_size=self.chunk_size,
				chunk_overlap=self.overlap_amount )
			return self.loader.split_documents( documents )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Loader'
			exception.method = ('_split_documents( self, docs: List[ Document ], chunk: int=1000, '
			                    'overlap: int=200 ) -> List[ Document ]')
			error = ErrorDialog( exception )
			error.show( )

class CsvLoader( Loader ):
	'''
	
		Purpose:
		--------
		Provides CSVLoader functionality to parse CSV files into Document objects.
		
		
		Attributes:
		----------
		documents: List[ Document ],
		file_path: str,
		pattern: str,
		expanded: List[ str ],
		candidates: List[ str ],
		resolved: List[ str ],
		splitter: RecursiveCharacterTextSplitter,
		chunk_size: int,
		overlap_amount: int,
		
		Methods:
		-------
		_ensure_existing_file( self, path: str ) -> str,
		_resolve_paths( self, pattern: str ) -> List[ str ],
		_split_documents( self, docs: List[ Document ], chunk: int=1000, overlap: int=200 ) ->
		List[ Document ],
		load( self, path: str, encoding: Optional[ str ]=None,
			csv_args: Optional[ Dict[ str, Any ] ]=None,
			source_column: Optional[ str ]=None ) -> List[ Document ]
		
	'''
	loader: Optional[ CSVLoader ]
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	pattern: Optional[ List[ str ] ]
	expanded: Optional[ List[ str ] ]
	candidates: Optional[ List[ str ] ]
	resolved: Optional[ List[ str ] ]
	encoding: Optional[ str ]
	csv_args: Optional[ Dict[ str, Any ] ]
	source_column: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.encoding = None
		self.csv_args = None
		self.source_column = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None
	
	def load( self, path: str, encoding: Optional[ str ], csv_args: Optional[ Dict[ str, Any ] ],
			source_column: Optional[ str ] ) -> List[ Document ] | None:
		'''
		
			Purpose:
			--------
			Load a CSV file into LangChain Document objects.
	
			Parameters:
			-----------
			path (str): Path to the CSV file.
			encoding (Optional[str]): File encoding (e.g., 'utf-8') if known.
			csv_args (Optional[Dict[str, Any]]): Additional CSV parsing arguments.
			source_column (Optional[str]): Column name used for source attribution.
	
			Returns:
			--------
			List[Document]: List of LangChain Document objects parsed from the CSV.
			
		'''
		try:
			self.file_path = self.verify_exists( path )
			self.encoding = encoding
			self.csv_args = csv_args
			self.source_column = source_column
			self.loader = CSVLoader( file_path=self.file_path, encoding=self.encoding,
				csv_args=self.csv_args, source_column=self.source_column )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'CSV'
			exception.method = 'loader( )'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, size: int=1000 , amount: int=200 ) -> List[ Document ] | None:
		'''
		
			Purpose:
			--------
			Split loaded CSV documents into smaller text chunks.
	
			Parameters:
			-----------
			chunk_size (int): Maximum number of characters per chunk.
			chunk_overlap (int): Number of overlapping characters between chunks.
	
			Returns:
			--------
			List[Document]: List of split Document chunks.
		
		'''
		try:
			throw_if( 'docs', self.documents )
			self.chunk_size = size
			self.overlap_amount = amount
			_documents = self._split_documents( documents=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return _documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'CSV'
			exception.method = 'split( self, size: int=1000, amount: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class WebLoader( Loader ):
	'''
	
		Purpose:
		--------
		Provides LangChain's WebBaseLoader functionality to retrieve 
		and parse HTML content into Document objects.
		
	'''
	loader: Optional[ WebBaseLoader ]
	urls: Optional[ List[ str ] ]
	documents: Optional[ List[ Document ] ]
	file_path: Optional[ str ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.urls = None
		self.documents = None
		self.file_path = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None

	def load( self, urls: List[ str ] ) -> List[ Document ] | None:
		'''
		
			Purpose:
			--------
			Load one or more web pages and convert to Document objects.
	
			Parameters:
			-----------
			urls (str | List[str]): A single URL string or list of URL strings.
	
			Returns:
			--------
			List[Document]: Parsed Document objects from fetched HTML content.
			
		'''
		try:
			throw_if( 'urls', urls )
			self.urls = urls
			self.loader = WebBaseLoader( web_path=self.urls )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Web'
			exception.method = 'load( self, urls: List[ str ] ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000 , overlap: int=200 ) -> List[ Document ] | None:
		'''
		
			Purpose:
			--------
			Split loaded web documents into smaller chunks for better LLM processing.
	
			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlap between chunks in characters.
	
			Returns:
			--------
			List[Document]: Chunked Document objects.
			
		'''
		try:
			throw_if( 'docs', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			_documents = self._split_documents( documents=self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return _documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Web'
			exception.method = 'split( self, chunk: int=1000 , overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class PdfLoader( Loader ):
	'''
	
		Purpose:
		--------
		Wrap LangChain's PyPDFLoader to extract and chunk PDF documents.
	
	'''
	loader: Optional[ PyPDFLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_pathpath = None
		self.documents = [ ]
		self.file_path = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None

	def __repr__( self ) -> str:
		return f'PDF(path={self.path!r}, docs={len( self.docs or [ ] )})'
	
	def load( self, path: str ) -> List[ Document ] | None:
		'''
			
			
			Purpose:
			--------
			Load a PDF file and convert its contents into LangChain Document objects.
	
			Parameters:
			-----------
			path (str): File path to a PDF document.
	
			Returns:
			--------
			List[Document]: List of parsed Document objects from the PDF.
			
			
		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = PyPDFLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.docs
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'PDF'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''
		
		
			Purpose:
			--------
			Split loaded PDF documents into smaller chunks for efficient processing.
	
			Parameters:
			-----------
			chunk_size (int): Max characters allowed per chunk.
			chunk_overlap (int): Characters that overlap between consecutive chunks.
	
			Returns:
			--------
			List[Document]: Chunked list of Document objects.
			
			
		'''
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self._split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.docs
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'PDF'
			exception.method = 'split( self, chunk: int=1000 , overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class ExcelLoader( Loader ):
	'''
	
	
		Purpose:
		--------
		Wrap LangChain's UnstructuredExcelLoader to parse Excel spreadsheets into documents.
		
		
	'''
	loader: Optional[ UnstructuredExcelLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None

	def __repr__( self ) -> str:
		return f'Excel(path={self.path!r}, docs={len( self.docs or [ ] )})'
	
	def load( self, path: str, mode: str='elements', headers: bool=True ) -> List[ Document ] | None:
		'''
		
		
			Purpose:
			--------
			Load and convert Excel data into LangChain Document objects.
	
			Parameters:
			-----------
			path (str): File path to the Excel spreadsheet.
			mode (str): Extraction mode, either 'elements' or 'paged'.
			include_headers (bool): Whether to include column headers in parsing.
	
			Returns:
			--------
			List[Document]: List of parsed Document objects from Excel content.
			
		
		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = UnstructuredExcelLoader( file_path=self.file_path, mode=mode,
				include_headers=headers )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Excel'
			exception.method = 'load( self, path: str, mode: str=elements, include_headers: bool=True ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''
		
		
			Purpose:
			--------
			Split loaded Excel documents into manageable chunks.
	
			Parameters:
			-----------
			chunk_size (int): Maximum characters per chunk.
			chunk_overlap (int): Characters overlapping between chunks.
	
			Returns:
			--------
			List[Document]: Chunked and cleaned list of Document objects.
			
		
		'''
		try:
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self._split_documents( self.documents, chunk=self.chunk_size,
				overlap=overlap )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Excel'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class WordLoader( Loader ):
	'''
	
	
		Purpose:
		--------
		Wrap LangChain's Docx2txtLoader to convert Word .docx files into Document objects.
		
	
	'''
	loader: Optional[ Docx2txtLoader ]
	file_path: Optional[ str ]
	documents: Optional[ List[ Document ] ]
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.documents = None
		self.file_path = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None

	def __repr__( self ) -> str:
		return f'Word(path={self.path!r}, docs={len( self.docs or [ ] )})'
	
	def load( self, path: str ) -> List[ Document ] | None:
		'''
		
		
			Purpose:
			--------
			Load the contents of a Word .docx file into LangChain Document objects.
	
			Parameters:
			-----------
			path (str): File path to the .docx document.
	
			Returns:
			--------
			List[Document]: Parsed Document list from Word file.
			
		
		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = Docx2txtLoader( self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Word'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''
		
		
			Purpose:
			--------
			Split Word documents into text chunks suitable for LLM processing.
	
			Parameters:
			-----------
			chunk_size (int): Maximum characters per chunk.
			chunk_overlap (int): Overlap between chunks in characters.
	
			Returns:
			--------
			List[Document]: Chunked list of Document objects.
			
			
		'''
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self._split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.docs
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Word'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class MarkLoader( Loader ):
	'''
	
	
		Purpose:
		--------
		Wrap LangChain's UnstructuredMarkdownLoader to parse Markdown files into Document objects.
		
		
	'''
	loader: Optional[ UnstructuredMarkdownLoader ]
	file_path: str | None
	documents: List[ Document ] | None
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = [ ]
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None

	def __repr__( self ) -> str:
		return f'Markdown(path={self.path!r}, docs={len( self.docs or [ ] )})'
	
	def load( self, path: str ) -> List[ Document ] | None:
		'''
		
			Purpose:
			--------
			Load a Markdown (.md) file into LangChain Document objects.
	
			Parameters:
			-----------
			path (str): File path to the Markdown file.
	
			Returns:
			--------
			List[Document]: List of parsed Document objects from the Markdown file.
			
		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = UnstructuredMarkdownLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Markdown'
			exception.method = 'load( self, path: str ) -> List[ Document ] '
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''
		
			Purpose:
			--------
			Split loaded Markdown content into text chunks for LLM consumption.
	
			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Number of characters that overlap between chunks.
	
			Returns:
			--------
			List[Document]: Split Document chunks from the original Markdown content.
			
		'''
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self._split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'Markdown'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )

class HtmlLoader( Loader ):
	'''
		
		Purpose:
		--------
		Provides the UnstructuredHTMLLoader's functionality 
		to parse HTML files into Document objects.
		
	'''
	loader: Optional[ UnstructuredHTMLLoader ]
	file_path: str | None
	documents: List[ Document ] | None
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.file_path = None
		self.documents = None
		self.pattern = None
		self.chunk_size = None
		self.overlap_amount = None
		self.loader = None

	def __repr__( self ) -> str:
		return f'HTML(path={self.path!r}, docs={len( self.documents or [ ] )})'
	
	def load( self, path: str ) -> List[ Document ] | None:
		'''
		
			Purpose:
			--------
			Load an HTML file and convert its contents into LangChain Document objects.
	
			Parameters:
			-----------
			path (str): Path to the HTML (.html or .htm) file.
	
			Returns:
			--------
			List[Document]: List of Document objects parsed from HTML content.
			
		'''
		try:
			throw_if( 'path', path )
			self.file_path = self.verify_exists( path )
			self.loader = UnstructuredHTMLLoader( file_path=self.file_path )
			self.documents = self.loader.load( )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'HTML'
			exception.method = 'load( self, path: str ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ] | None:
		'''
		
			Purpose:
			--------
			Split loaded HTML documents into manageable text chunks.
	
			Parameters:
			-----------
			chunk_size (int): Max characters per chunk.
			chunk_overlap (int): Overlapping characters between chunks.
	
			Returns:
			--------
			List[Document]: Chunked list of LangChain Document objects.
			
		'''
		try:
			throw_if( 'documents', self.documents )
			self.chunk_size = chunk
			self.overlap_amount = overlap
			self.documents = self._split_documents( self.documents, chunk=self.chunk_size,
				overlap=self.overlap_amount )
			return self.documents
		except Exception as e:
			exception = Error( e )
			exception.module = 'Foo'
			exception.cause = 'HTML'
			exception.method = 'split( self, chunk: int=1000, overlap: int=200 ) -> List[ Document ]'
			error = ErrorDialog( exception )
			error.show( )
