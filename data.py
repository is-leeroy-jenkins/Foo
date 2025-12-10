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
import sqlite3
from boogr import Error, ErrorDialog
from boogr.enums import Source, SQL, ParamStyle, Provider
import chromadb
from chromadb.config import Settings
import config as cfg

def throw_if( name: str, value: Any ) -> None:
	'''

		Purpose:
		-----------
		Simple guard which raises ValueError when `value` is falsy (None, empty).

		Parameters:
		-----------
		name (str): Variable name used in the raised message.
		value (Any): Value to validate.

		Returns:
		-----------
		None: Raises ValueError when `value` is falsy.

	'''
	if value is None:
		raise ValueError( f"Argument '{name}' cannot be empty!" )

class DB( ):
	'''
	
		Constructor:
		------------
		DbConfig( source: Source, provider: Provider=Provider.SQLite )
	
		Purpose:
		---------
		Class provides list of Budget Execution tables across two databases

	'''
	provider: Optional[ Provider ]
	source: Optional[ Source ]
	table_name: Optional[ str ]
	path: Optional[ str ]
	driver: Optional[ str ]

	def __init__( self  ):
		self.provider = None
		self.source = None
		self.table_name = None
		self.path = None
		self.driver = None

	def __dir__( self ) -> list[ str ]:
		'''
		Retunes a list[ str ] of member names.
		'''
		return [ 'source',
		         'provider',
		         'table_name',
		         'get_driver_info',
		         'path',
		         'adriver',
		         'access_path',
		         'get_data_path',
		         'get_connection_string' ]

	@property
	def driver_info( self ) -> str:
		'''

		'''
		try:
			if self.provider.name == 'Access':
				self.driver = cfg.ACCESS_DRIVER
				return self.driver
			elif self.provider.name == 'SqlServer':
				self.driver = cfg.SQLSERVER_DRIVER
				return self.driver
			elif self.provider.name == 'SqlServer':
				return self.sqlserver_driver
			else:
				return cfg.BASEDIR
		except Exception as e:
			_exc = Error( e )
			_exc.cause = 'DB'
			_exc.method = 'getdriver_info( self )'
			_error = ErrorDialog( _exc )
			_error.show( )

	@property
	def data_path( self ) -> str:
		'''
	
			Purpose:
			--------
	
			Parameters:
			--------
	
			Returns:
			--------

		'''
		try:
			if self.provider.name == 'SQLite':
				self.path = cfg.BASEDIR + r'\stores\sqlite'
				return self.path
			elif self.provider.name == 'Access':
				self.path = cfg.BASEDIR + r'\stores\access'
				return self.path
			elif self.provider.name == 'SqlServer':
				self.path = cfg.BASEDIR + r'\stores\sqlserver'
				return self.path
			else:
				self.path = cfg.BASEDIR + r'\stores\sqlite'
				return self.path
		except Exception as e:
			_exc = Error( e )
			_exc.cause = 'DB'
			_exc.method = 'get_data_path( self )'
			_error = ErrorDialog( _exc )
			_error.show( )

	@property
	def connection_string( self ) -> str:
		'''

			Purpose:
			--------
	
			Parameters:
			--------
	
			Returns:
			--------

		'''
		try:
			_path = self.data_path
			if self.provider.name == Provider.Access.name:
				return self.driver_info + _path
			elif self.provider.name == Provider.SqlServer.name:
				return r'DRIVER={ ODBC Driver 17 for SQL Server };Server=.\SQLExpress;' \
					+ f'AttachDBFileName={_path}' \
					+ f'DATABASE={_path}Trusted_Connection=yes;'
			else:
				return f'{_path} '
		except Exception as e:
			_exc = Error( e )
			_exc.cause = 'DB'
			_exc.method = 'get_connection_string( self )'
			_error = ErrorDialog( _exc )
			_error.show( )
			
class SQLite( DB ):
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
	provider: Optional[ Provider ]
	path: Optional[ str ]
	connection: sqlite3.Connection
	cursor: sqlite3.Cursor
	column_names: List[ str ]
	data_types: List[ str ]
	
	def __init__( self ) -> None:
		"""

			Purpose:
			--------
			Establishes a SQLite connection and initializes the embeddings table schema.

			Parameters:
			--------
			db_path (str): Path to the SQLite file used for storage.

			Returns:
			--------
			None

		"""
		super( ).__init__( self )
		self.provider = Provider.SQLite
		self.path = os.getcwd( ) + r'\stores\sqlite\datamodels\Data.db'
		self.connection = sqlite3.connect( self.db_path )
		self.cursor = self.connection.cursor( )
	
	def create_table( self, name: str, cols: List[ str ], types: List[ str ] ) -> None:
		"""

			Purpose:
			-------

			Creates the 'embeddings' table with appropriate schema if it does not already exist.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'name', name )
			throw_if( 'cols', cols )
			throw_if( 'types', types )
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
			exception.cause = 'SQLite'
			exception.method = 'create( self ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def insert( self, source_file: str, index: int,
			text: str, embedding: np.ndarray ) -> None:
		"""

			Purpose:
			----------
			Inserts a single embedding record with metadata into the database.

			Parameters:
			------------
			source_file (str): Name or path of the source document.
			index (int): Ordinal position of the chunk.
			text (str): Cleaned text of the chunk.
			embedding (np.ndarray): Vector representation of the chunk.

			Returns:
			--------
			None

		"""
		try:
			throw_if( 'source_file', source_file )
			throw_if( 'index', index )
			throw_if( 'text', text )
			throw_if( 'embedding', embedding )
			vector_str = json.dumps( embedding.tolist( ) )
			sql = """INSERT INTO embeddings ( source_file, chunk_index, chunk_text, embedding )
                     VALUES ( ?, ?, ?, ? )"""
			self.cursor.execute( sql, ( source_file, index, text, vector_str ) )
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
			-------
			Batch inserts multiple chunks and their embeddings into the database.

			Parameters:
			-----------
			source_file (str): Name or path of the source document.
			chunks (List[str]): List of cleaned text chunks.
			vectors (np.ndarray): Matrix of embedding vectors.

			Returns:
			--------
			None

		"""
		try:
			records = [ ( source_file, i, chunks[ i ], json.dumps( vectors[ i ].tolist( ) ) )
			            for i in range( len( chunks ) ) ]
			sql_insert = """INSERT INTO embeddings ( source_file, chunk_index, chunk_text, embedding )
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
			--------
			Retrieves all records associated with a specific file.

			Parameters:
			--------
			file (str): Identifier of the source file.

			Returns:
			--------
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
			--------
			Deletes all embeddings associated with a given source file.

			Parameters:
			--------
			file (str): Source file whose records are to be deleted.

			Returns:
			--------
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
			--------
			Updates an embedding vector in the database by record ID.

			Parameters:
			--------
			row_id (int): ID of the record to update.
			new_embedding (np.ndarray): New embedding vector.

			Returns:
			--------
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
			--------
			Returns the total number of records stored in the embeddings table.

			Parameters:
			--------
			None

			Returns:
			--------
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
			--------
			Deletes all data from the embeddings table without altering the schema.

			Parameters:
			--------
			None

			Returns:
			--------
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
			if self.connection is not None:
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
	
	def query( self, text: List[ str ], num: int = 5, where: Optional[ dict ] = None ) -> List[ str ] | None:
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
			--------
			Deletes one or more records from the collection by document ID.

			Parameters:
			--------
			ids (List[str]): List of unique document IDs to delete.

			Returns:
			--------
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
			--------
			Returns the total number of records in the collection.

			Returns:
			--------
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
			--------
			Deletes all documents from the collection.

			Parameters:
			--------
			None

			Returns:
			--------
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
			--------
			Saves the current state of the collection to disk.

			Returns:
			--------
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
