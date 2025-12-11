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
	column_names: Optional[ List[ str ] ]
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

		Class providing CRUD
		operations for a SQLite database.

		Methods:
			- create_table: Creates a df with specified schema.
			- insert: Inserts a record into a df.
			- fetch_all: Fetches all rows from a df.
			- fetch_one: Fetches a single record matching the query.
			- update: Updates rows that match a given condition.
			- delete: Deletes rows that match a given condition.
			- close: Closes the database connection.

	"""
	db_path: Optional[ str ]
	connection: Optional[ Connection ]
	cursor: Optional[ Cursor ]
	file_path: Optional[ str ]
	where: Optional[ str ]
	file_name: Optional[ str ]
	table_name: Optional[ str ]
	placeholders: Optional[ List[ str ] ]
	column_names: Optional[ List[ str ] ]
	params: Optional[ Tuple ]
	tables: Optional[ List ]
	
	def __init__( self ):
		"""

			Pupose:
			-------
			Initializes the connection to the SQLite database.


		"""
		self.db_path = r'stores\sqlite\datamodels\Data.db'
		self.connection = sqlite3.connect( self.db_path )
		self.cursor = self.connection.cursor( )
		self.file_path = None
		self.where = None
		self.pairs = None
		self.sql = None
		self.file_name = None
		self.table_name = None
		self.placeholders = [ str ]
		self.params = ( )
		self.column_names = [ str ]
		self.tables = [ ]
	
	def __dir__( self ):
		return [ 'db_path',
		         'connection',
		         'cursor',
		         'path',
		         'where',
		         'pairs',
		         'sql',
		         'file_name',
		         'table_name',
		         'placeholders',
		         'columns',
		         'params',
		         'column_names',
		         'tables',
		         'close',
		         'import_excel',
		         'delete',
		         'update',
		         'insert',
		         'create_table',
		         'fetch_one',
		         'fetch_all' ]
	
	def create( self ) -> None:
		"""

			Purpose:
			Creates the 'embeddings' table with appropriate schema if it does not already exist.

			Returns:
			None

		"""
		try:
			self.cursor.execute( """
             CREATE TABLE IF NOT EXISTS embeddings
             (
                 id          INTEGER PRIMARY KEY AUTOINCREMENT,
                 source_file TEXT    NOT NULL,
                 chunk_index INTEGER NOT NULL,
                 chunk_text  TEXT    NOT NULL,
                 embedding   TEXT    NOT NULL,
                 created_at  TEXT DEFAULT CURRENT_TIMESTAMP
             )""" )
			
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'create( self ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def create_table( self, sql: str ) -> None:
		"""

			Purpose:
			--------
			Creates a df using a provided SQL statement.

			Parameters:
			-----------
			sql (str): The CREATE TABLE SQL statement.

		"""
		try:
			throw_if( 'sql', sql )
			self.sql = sql
			self.cursor.execute( self.sql )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'create_table( self, sql: str ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def insert( self, table: str, columns: List[ str ], values: Tuple[ Any, ... ] ) -> None:
		"""

			Purpose:
			--------
			Inserts a new record into a df.

			Parameter:
			--------
			table (str): The name of the df.
			columns (List[str]): Column names.
			values (Tuple): Corresponding target_values.

		"""
		try:
			throw_if( 'table', table )
			throw_if( 'columns', columns )
			throw_if( 'values', values )
			self.placeholders = ', '.join( '?' for _ in values )
			col_names = ', '.join( columns )
			self.sql = f'INSERT INTO {table} ({col_names}) VALUES ({self.placeholders})'
			self.cursor.execute( self.sql, values )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = ('insert( self, df: str, columns: List[ str ], '
			                    'target_values: Tuple[ Any, ... ] ) -> None')
			error = ErrorDialog( exception )
			error.show( )
	
	def insert_many( self, source_file: str, chunks: List[ str ], vectors: np.ndarray ) -> None:
		"""

			Purpose:
			--------
			Batch inserts multiple chunks and their embeddings into the database.

			Parameters:
			--------
			source_file (str): Name or path of the source document.
			chunks (List[str]): List of cleaned text chunks.
			vectors (np.ndarray): Matrix of embedding vectors.

			Returns:
			--------
				None

		"""
		try:
			throw_if( 'source_file', source_file )
			throw_if( 'chuncks', chunks )
			throw_if( 'vectors', vectors )
			records = [ (source_file, i, chunks[ i ], json.dumps( vectors[ i ].tolist( ) ))
			            for i in range( len( chunks ) ) ]
			
			self.sql = f''' INSERT INTO {self.table_name} ({self.file_name}, chunk_index,
					chunk_text, embedding) VALUES (?, ?, ?, ?) '''
			self.cursor.executemany( self.sql, records )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'insert_many'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_all( self, table: str ) -> List[ Tuple ] | None:
		"""

			Purpose:
			--------
			Retrieves all rows from a df.

			Parameters:
			--------
			table (str): The name of the df.

			Returns:
			--------
			List[Tuple]: List of rows.

		"""
		try:
			throw_if( 'table', table )
			self.sql = f'SELECT * FROM {table}'
			self.cursor.execute( self.sql )
			return self.cursor.fetchall( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'fetch_all( self, df: str ) -> List[ Tuple ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def fetch_one( self, table: str, where: str, params: Tuple[ Any, ... ] ) -> Tuple | None:
		"""

			Purpose:
			--------
			Retrieves a single row matching a WHERE clause.

			Parameters:
			--------
			table (str): Table name.
			where (str): WHERE clause (excluding 'WHERE').
			params (Tuple): Parameters for the clause.

			Returns:
			--------
			Optional[Tuple]: The fetched row or None.

		"""
		try:
			throw_if( 'params', params )
			throw_if( 'where', where )
			throw_if( 'table', table )
			self.table_name = table
			self.where = where
			self.sql = f'SELECT * FROM {self.table_name} WHERE {self.where} LIMIT 1'
			self.cursor.execute( self.sql, self.params )
			return self.cursor.fetchone( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = (
					'fetch_one( self, df: str, where: str, params: Tuple[ Any, ... ] ) -> '
					'Optional[ Tuple ]')
			error = ErrorDialog( exception )
			error.show( )
	
	def update( self, table: str, pairs: str, where: str, params: Tuple[ Any, ... ] ) -> None:
		"""

			Purpose:
			--------
			Updates rows in a df.

			Parameters:
			--------
			table (str): Table name.
			pairs (str): SET clause with placeholders.
			where (str): WHERE clause with placeholders.
			params (Tuple): Parameters for both clauses.

		"""
		try:
			throw_if( 'pairs', pairs )
			throw_if( 'params', params )
			throw_if( 'where', where )
			throw_if( 'table', table )
			self.table_name = table
			self.where = where
			self.params = params
			self.sql = f'UPDATE {self.table_name} SET {pairs} WHERE {self.where}'
			self.cursor.execute( self.sql, params )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = (
					'update( self, df: str, pairs: str, where: str, params: Tuple[ Any, '
					'... ] ) -> None')
			error = ErrorDialog( exception )
			error.show( )
	
	def delete( self, table: str, where: str, params: Tuple[ Any, ... ] ) -> None:
		"""

			Purpose:
			--------
			Deletes row matching the given WHERE clause.

			Parameters:
			--------
			table (str): Table name.
			where (str): WHERE clause (excluding 'WHERE').
			params (Tuple): Parameters for clause.

		"""
		try:
			throw_if( 'where', where )
			throw_if( 'table', table )
			throw_if( 'params', params )
			self.table_name = table
			self.where = where
			self.params = params
			self.sql = f"DELETE FROM {self.table_name} WHERE {self.where}"
			self.cursor.execute( self.sql, self.params )
			self.connection.commit( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'delete( self, df: str, where: str, params: Tuple[ Any] )->None'
			error = ErrorDialog( exception )
			error.show( )
	
	def import_excel( self, path: str ) -> None:
		"""

			Purpose:
			--------
			Reads all worksheets from an Excel file into pandas DataFrames and
			stores each as a df in the SQLite database.

			Parameters:
			--------
			path (str): Path to the Excel workbook.

		"""
		try:
			throw_if( 'path', path )
			self.file_path = path
			self.file_name = os.path.basename( self.file_path )
			_excel = pd.ExcelFile( self.file_path )
			for _sheet in _excel.sheet_names:
				_df = _excel.parse( _sheet )
				_df.to_sql( _sheet, self.connection, if_exists='replace', index=False )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'import_excel( self, path: str ) -> None'
			error = ErrorDialog( exception )
			error.show( )
	
	def close( self ) -> None:
		"""

			Purpose:
			--------
			Closes the database connection.

		"""
		try:
			if self.connection is not None:
				self.connection.close( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'data'
			exception.cause = 'SQLite'
			exception.method = 'close( self ) -> None'
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
