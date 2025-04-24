# -*- coding: utf-8 -*-

import datetime
import time
import io
import os
import csv
import sys
import psycopg2
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import text
import mylogging

sys.stdout.reconfigure(line_buffering=True)

# Pour la table Markets, utilisé aussi dans le constructeur
# mid, nom, alias, prefix boursorama, symbol SWS
initial_markets_data = (
    (1, "New York", "nyse", "", "NYSE", ""),
    (2, "London Stock Exchange", "lse", "1u*.L", "LSE", ""),
    (3, "Bourse de Milan", "milano", "1g", "", ""),
    (4, "Mercados Espanoles", "mercados", "FF55-", "", ""),
    (5, "Amsterdam", "amsterdam", "1rA", "", "Amsterdam"),
    (6, "Paris", "paris", "1rP", "ENXTPA", "Paris"),
    (7, "Deutsche Borse", "xetra", "1z", "", ""),
    (8, "Bruxelle", "bruxelle", "FF11_", "", "Brussels"),
    (9, "Australie", "asx", "", "ASX", ""),
    (100, "International", "int", "", "", ""),  # should be last one
)


def _psql_insert_copy(table, conn, keys, data_iter):  # method used by df_write
    """
    Execute SQL statement inserting data
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_sql.html

    Parameters
    ----------
    table : pandas.io.sql.SQLTable
    conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
    keys : list of str
        Column names
    data_iter : Iterable that iterates the values to be inserted
    """
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = io.StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ", ".join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = "{}.{}".format(table.schema, table.name)
        else:
            table_name = table.name

        sql = "COPY {} ({}) FROM STDIN WITH CSV".format(table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)


class TimescaleStockMarketModel:
    """ Bourse model with TimeScaleDB persistence."""

    def __init__(self, database, user=None, host=None, password=None, port=None, remove_all=False):
        """Create a TimescaleStockMarketModel

        database -- The name of the persistence database.
        user     -- Username to connect with to the database. Same as the
                    database name by default.
        remove_all -- REMOVE ALL DATA from the database
        """
        # Force stdout to flush for better logging
        print("DASHBOARD: Initializing TimescaleStockMarketModel", flush=True)
        
        self.__database = database
        self.__user = user or database
        self.__host = host or 'localhost'
        self.__port = port or 5432
        self.__password = password or ''
        self.__squash = False

        # Create connection string and engine
        conn_string = f"postgresql://{self.__user}:{self.__password}@{self.__host}:{self.__port}/{self.__database}"
        print(f"DASHBOARD: Using connection string: {conn_string.replace(self.__password, '***')}", flush=True)
        
        self.__engine = sqlalchemy.create_engine(conn_string)
        self.engine = self.__engine  # For backwards compatibility
        
        # For backwards compatibility
        try:
            self.conn = self.__engine.connect()
            print("DASHBOARD: SQLAlchemy connection successful", flush=True)
        except Exception as e:
            print(f"DASHBOARD ERROR: SQLAlchemy connection failed: {e}", flush=True)
            raise
            
        # markets
        self.market_id = {a: i+1 for i,
                          a in enumerate([m[2] for m in initial_markets_data])}
        self.market_id2sws = {
            i+1: w for i, w in enumerate([m[4] for m in initial_markets_data])}
        for i, w in self.market_id2sws.items():
            if w == "":
                self.market_id2sws[i] = None

        # Setup logger
        print(f"DASHBOARD: Setting up logger", flush=True)
        self.logger = mylogging.getLogger(__name__, filename="/tmp/bourse.log")
        
        # Connect to database
        try:
            print(f"DASHBOARD: Connecting to database", flush=True)
            self.connection = self._connect_to_database()
            print(f"DASHBOARD: Database connection established", flush=True)
        except Exception as e:
            print(f"DASHBOARD ERROR: Database connection failed: {e}", flush=True)
            raise

        # Setup database
        self.logger.info(
            "Setup database generates an error if it exists already, it's ok")
        if remove_all:
            self._purge_database()
        self._setup_database()

    def _connect_to_database(self, retry_limit=5, retry_delay=1):
        """
            With a SQL server running in a Docker, it can take time to connect if all
            services are started in the same time.
        """
        for attempt in range(retry_limit):
            try:
                print(f"DASHBOARD: Connection attempt {attempt+1}/{retry_limit}", flush=True)
                connection = psycopg2.connect(
                    database=self.__database,
                    user=self.__user,
                    host=self.__host,
                    password=self.__password,
                )
                print(f"DASHBOARD: Connection successful on attempt {attempt+1}", flush=True)
                return connection
            except Exception as e:
                print(f"DASHBOARD: Connection attempt {attempt+1} failed: {e}", flush=True)
                time.sleep(retry_delay)
        
        error_msg = f"Failed to connect to database after {retry_limit} attempts"
        print(f"DASHBOARD ERROR: {error_msg}", flush=True)
        raise Exception(error_msg)

    def _create_sequence(self, sequence_name, commit=False):
        """Create a sequence in the database."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"CREATE SEQUENCE {sequence_name};")
            if commit:
                self.connection.commit()
        except Exception as e:
            print(f"Error creating sequence: {e}", flush=True)
            self.connection.rollback()  # Rollback the current transaction

    def _drop_sequence(self, sequence_name, commit=False):
        """Drop a sequence from the database."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"DROP SEQUENCE IF EXISTS {sequence_name};")
            if commit:
                self.connection.commit()
        except Exception as e:
            print(f"Error dropping sequence: {e}", flush=True)
            self.connection.rollback()  # Rollback the current transaction

    def _create_table(self, table_name, columns_definition, commit=False):
        """Create a table in the database."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"CREATE TABLE {table_name} ({columns_definition});")
            if commit:
                self.connection.commit()
        except Exception as e:
            print(f"Error creating table: {e}", flush=True)
            self.connection.rollback()  # Rollback the current transaction

    def _drop_table(self, table_name, commit=False):
        """Drop a table from the database."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
            if commit:
                self.connection.commit()
        except Exception as e:
            print(f"Error dropping table: {e}", flush=True)
            self.connection.rollback()  # Rollback the current transaction

    def _create_hypertable(self, table_name, time_column, commit=False):
        """Create a hypertable in the database."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                f"SELECT create_hypertable('{table_name}', '{time_column}');"
            )
            if commit:
                self.connection.commit()
        except Exception as e:
            print(f"Error creating hypertable: {e}", flush=True)
            self.connection.rollback()  # Rollback the current transaction

    def _drop_hypertable(self, table_name, commit=False):
        """Drop a hypertable from the database."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"SELECT drop_hypertable('{table_name}');")
            if commit:
                self.connection.commit()
        except Exception as e:
            print(f"Error dropping hypertable: {e}", flush=True)
            self.connection.rollback()  # Rollback the current transaction

    def _create_index(self, table_name, index_name, columns, commit=False):
        """Create an index in the database."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"CREATE INDEX {index_name} ON {table_name} ({columns});")
            if commit:
                self.connection.commit()
        except Exception as e:
            print(f"Error creating index: {e}", flush=True)
            self.connection.rollback()  # Rollback the current transaction

    def _drop_index(self, index_name, commit=False):
        """Drop an index from the database."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"DROP INDEX IF EXISTS {index_name};")
            if commit:
                self.connection.commit()
        except Exception as e:
            print(f"Error dropping index: {e}", flush=True)
            self.connection.rollback()  # Rollback the current transaction

    def _insert_data(self, table_name, data, commit=False):
        """Insert data into a table in the database."""
        cursor = self.connection.cursor()
        try:
            for row in data:
                cursor.execute(f"INSERT INTO {table_name} VALUES %s;", (row,))
            if commit:
                self.connection.commit()
        except Exception as e:
            print(f"Error inserting data: {e}", flush=True)
            self.connection.rollback()  # Rollback the current transaction

    def _setup_database(self):
        """Setup the database schema."""
        print("DASHBOARD: Setting up database tables", flush=True)
        try:
            if len(self.df_query("select id from markets")) == 0:
                print("DASHBOARD: Creating database tables", flush=True)
                # Create sequences
                self._create_sequence("market_id_seq")
                self._create_sequence("company_id_seq")

                # Create tables
                # boursorama : exchange prefix for boursorama
                # sws : exchange name for Simply Wall Street
                self._create_table(
                    "markets",
                    ''' id SMALLINT PRIMARY KEY DEFAULT nextval('market_id_seq'), 
                        name VARCHAR, 
                        alias VARCHAR,
                        boursorama VARCHAR,
                        sws VARCHAR,
                        euronext VARCHAR
                    '''
                )
                self._create_table(
                    "companies",
                    """ id SMALLINT PRIMARY KEY DEFAULT nextval('company_id_seq'), 
                        name VARCHAR,
                        mid SMALLINT DEFAULT 0,
                        symbol VARCHAR, 
                        isin CHAR(12),
                        boursorama VARCHAR, 
                        euronext VARCHAR, 
                        pea BOOLEAN DEFAULT FALSE, 
                        sector1 VARCHAR,
                        sector2 VARCHAR,
                        sector3 VARCHAR
                    """
                )
                self._create_table(
                    "stocks",
                    ''' date TIMESTAMPTZ, 
                        cid SMALLINT, 
                        value FLOAT4, 
                        volume FLOAT4
                    '''
                )
                self._create_table(
                    "daystocks",
                    ''' date TIMESTAMPTZ, 
                        cid SMALLINT, 
                        open FLOAT4,
                        close FLOAT4, 
                        high FLOAT4, 
                        low FLOAT4, 
                        volume FLOAT4, 
                        mean FLOAT4, 
                        std FLOAT4
                    '''
                )
                self._create_table("file_done", "name VARCHAR PRIMARY KEY")
                self._create_table(
                    "tags", "name VARCHAR PRIMARY KEY, value VARCHAR")
                self._create_table("error_dates", "date TIMESTAMPTZ")

                # Create hypertables
                self._create_hypertable("stocks", "date")
                self._create_hypertable("daystocks", "date")

                # Create indexes
                self._create_index(
                    "stocks", "idx_cid_stocks", "cid, date DESC")
                self._create_index(
                    "daystocks", "idx_cid_daystocks", "cid, date DESC")

                # Insert initial market data
                self._insert_data("markets", initial_markets_data)
                self.connection.commit()
                print("DASHBOARD: Database tables created successfully", flush=True)
            else:
                print("DASHBOARD: Database tables already exist", flush=True)
        except Exception as e:
            self.logger.exception("SQL error: %s" % e)
            print(f"DASHBOARD ERROR: Database setup error: {e}", flush=True)
            self.connection.rollback()

    def _purge_database(self):
        print("DASHBOARD: Purging database (removing all tables)", flush=True)
        self._drop_table("markets")
        self._drop_table("companies")
        self._drop_table("stocks")
        self._drop_table("daystocks")
        self._drop_table("file_done")
        self._drop_table("tags")
        self._drop_table("error_dates")

        self._drop_sequence("market_id_seq")
        self._drop_sequence("company_id_seq")

        self._drop_index("stocks")
        self._drop_index("daystocks")
        self.commit()
        print("DASHBOARD: Database purged successfully", flush=True)

    # ------------------------------ public methods --------------------------------

    def execute(self, query, args=None, cursor=None, commit=False):
        """Send a Postgres SQL command. No return"""
        if args is None:
            pretty = query
        else:
            pretty = '%s %% %r' % (query, args)
        self.logger.debug('SQL: QUERY: %s' % pretty)
        if cursor is None:
            cursor = self.connection.cursor()
        try:
            cursor.execute(query, args)
            if commit:
                self.commit()
            return cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Exception with execute: {e}")
            print(f"DASHBOARD ERROR: Query execution error: {e}", flush=True)
            if self.connection:
                self.connection.rollback()

    def df_write(self, df, table, args=None, commit=False, if_exists="append",
                 index=False, index_label=None, chunksize=100, dtype=None, method=_psql_insert_copy):
        """Write a Pandas dataframe to the Postgres SQL database

        :param query:
        :param args: arguments for the query
        :param commit: do a commit after writing
        :param other args: see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_sql.html
        """
        self.logger.debug("df_write")
        df.to_sql(
            table,
            con=self.__engine,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            chunksize=chunksize,
            dtype=dtype,
            method=method,
        )
        if commit:
            self.commit()

    # general query methods

    def raw_query(self, query, args=None, cursor=None):
        """Return a tuple from a Postgres SQL query"""
        if args is None:
            pretty = query
        else:
            pretty = '%s %% %r' % (query, args)
        self.logger.debug('SQL: QUERY: %s' % pretty)
        if cursor is None:
            cursor = self.connection.cursor()
        try:
            cursor.execute(query, args)
            query = query.strip().upper()
            if query.startswith('SELECT'):
                return cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Exception with raw_query: {e}")
            print(f"DASHBOARD ERROR: Raw query error: {e}", flush=True)
            if self.connection:
                self.connection.rollback()

    def df_query(self, query, args=None, index_col=None, coerce_float=True, params=None,
                 parse_dates=None, columns=None, chunksize=None, dtype=None):
        '''Returns a Pandas dataframe from a Postgres SQL query

        :param query:
        :param args: arguments for the query
        :param index_col: index column of the DataFrame
        :param other args: see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html
        :return: a dataframe
        '''
        if args is not None:
            query = query % args
        self.logger.debug('df_query: %s' % query)
        try:
            res = pd.read_sql(query, self.__engine, index_col=index_col, coerce_float=coerce_float,
                              params=params, parse_dates=parse_dates, columns=columns,
                              chunksize=chunksize, dtype=dtype)
        except Exception as e:
            self.logger.error(e)
            print(f"DASHBOARD ERROR: DataFrame query error: {e}", flush=True)
            res = pd.DataFrame()
        return res

    # system methods

    def commit(self):
        if not self.__squash:
            self.connection.commit()

    def get_companies(self):
        """Get list of all company names/symbols in the database"""
        print("DASHBOARD: Retrieving company list from database...", flush=True)
        
        # First try with JOIN
        query = """
        SELECT DISTINCT c.name 
        FROM companies c
        JOIN stocks s ON s.cid = c.id
        ORDER BY c.name
        """
        
        try:
            result = pd.read_sql(query, self.__engine)
            if not result.empty:
                companies = result['name'].tolist()
                print(f"DASHBOARD: Found {len(companies)} companies with JOIN query", flush=True)
                return companies
            
            # No results? Try without the JOIN
            print("DASHBOARD: No companies found with JOIN query, trying direct company table query", flush=True)
            query = "SELECT name FROM companies ORDER BY name"
            result = pd.read_sql(query, self.__engine)
            if not result.empty:
                companies = result['name'].tolist()
                print(f"DASHBOARD: Found {len(companies)} companies from companies table", flush=True)
                return companies
            print("DASHBOARD WARNING: No companies found in database at all", flush=True)
            return []
        except Exception as e:
            print(f"DASHBOARD ERROR retrieving companies: {str(e)}", flush=True)
            return []

    def get_date_range(self):
        """Get the min and max dates in the stocks table"""
        print("DASHBOARD: Getting date range", flush=True)
        try:
            query = "SELECT MIN(date) as min_date, MAX(date) as max_date FROM stocks"
            result = pd.read_sql(query, self.__engine)
            min_date = result['min_date'].iloc[0].strftime('%Y-%m-%d')
            max_date = result['max_date'].iloc[0].strftime('%Y-%m-%d')
            print(f"DASHBOARD: Date range from {min_date} to {max_date}", flush=True)
            return min_date, max_date
        except Exception as e:
            print(f"DASHBOARD ERROR getting date range: {e}", flush=True)
            # Return default values
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            last_year = (datetime.datetime.now() -
                         datetime.timedelta(days=1200)).strftime('%Y-%m-%d')
            return last_year, today

    def get_company_data(self, company_name, start_date, end_date):
        """Get stock data for a specific company between start_date and end_date"""
        print(f"DASHBOARD: Getting data for {company_name} from {start_date} to {end_date}", flush=True)
        
        # First, check if the company exists
        check_query = f"SELECT id FROM companies WHERE name = '{company_name}'"
        try:
            check_result = pd.read_sql(check_query, self.__engine)
            if check_result.empty:
                print(f"DASHBOARD WARNING: Company '{company_name}' not found", flush=True)
                return pd.DataFrame()
            company_id = check_result['id'].iloc[0]
            print(f"DASHBOARD: Found company '{company_name}' with ID {company_id}", flush=True)
        except Exception as e:
            print(f"DASHBOARD ERROR checking company: {str(e)}", flush=True)
            return pd.DataFrame()
        
        # Check both daystocks and stocks tables
        try:
            # First try daystocks table which would have OHLC data
            query = f"""
            SELECT d.date as timestamp, d.open, d.high, d.low, d.close, d.volume 
            FROM daystocks d
            WHERE d.cid = {company_id}
            AND d.date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY d.date
            """
            
            df = pd.read_sql(query, self.__engine)
            
            # If daystocks is empty, try the stocks table
            if df.empty:
                print(f"DASHBOARD: No daystock data found for {company_name}, trying stocks table", flush=True)
                
                # Get raw stocks data and reshape it into OHLC format
                stocks_query = f"""
                SELECT s.date as timestamp, s.value, s.volume
                FROM stocks s
                WHERE s.cid = {company_id}
                AND s.date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY s.date
                """
                
                stocks_df = pd.read_sql(stocks_query, self.__engine)
                
                if not stocks_df.empty:
                    print(f"DASHBOARD: Found {len(stocks_df)} stock records for {company_name}", flush=True)
                    
                    # Convert timestamp to datetime if it's not already
                    stocks_df['timestamp'] = pd.to_datetime(stocks_df['timestamp'])
                    
                    # Resample to daily and create OHLC format
                    stocks_df.set_index('timestamp', inplace=True)
                    
                    # Group by day
                    daily_df = stocks_df.groupby(pd.Grouper(freq='D')).agg({
                        'value': ['first', 'max', 'min', 'last', 'mean'],
                        'volume': 'sum'
                    })
                    
                    # Flatten MultiIndex columns
                    daily_df.columns = ['open', 'high', 'low', 'close', 'mean', 'volume']
                    
                    # Remove days with no data
                    daily_df = daily_df.dropna(subset=['close'])
                    
                    return daily_df
                else:
                    print(f"DASHBOARD WARNING: No data found for {company_name} in date range", flush=True)
                    return pd.DataFrame()
            
            print(f"DASHBOARD: Retrieved {len(df)} records for {company_name}", flush=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"DASHBOARD ERROR retrieving data for {company_name}: {str(e)}", flush=True)
            return pd.DataFrame()

    def get_database_stats(self):
        """Get basic database statistics for diagnostic purposes"""
        stats = {}
        
        print("\n===== DATABASE DIAGNOSTIC STARTED =====", flush=True)
        print(f"Connection string: postgresql://{self.__user}:***@{self.__host}:{self.__port}/{self.__database}", flush=True)
        
        try:
            # Test basic connectivity
            print("Testing basic connection...", flush=True)
            self.__engine.connect().close()
            print("Basic connection works", flush=True)
            
            # Check tables exist
            print("\nChecking if required tables exist...", flush=True)
            tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
            tables = pd.read_sql(tables_query, self.__engine)
            print(f"Tables found: {', '.join(tables['table_name'].tolist())}", flush=True)
            
            # Count companies
            print("\nCounting companies...", flush=True)
            query = "SELECT COUNT(*) FROM companies"
            result = pd.read_sql(query, self.__engine)
            stats['company_count'] = result.iloc[0, 0]
            print(f"Total companies in database: {stats['company_count']}", flush=True)
            
            # Sample company records
            print("\nSampling company records...", flush=True)
            sample_query = "SELECT id, name FROM companies LIMIT 3"
            samples = pd.read_sql(sample_query, self.__engine)
            if not samples.empty:
                for _, row in samples.iterrows():
                    print(f"  Company ID: {row['id']}, Name: {row['name']}", flush=True)
            else:
                print("  No companies found", flush=True)
            
            # Count companies with stock data
            print("\nCounting companies with stock data...", flush=True)
            query = "SELECT COUNT(DISTINCT cid) FROM stocks"
            result = pd.read_sql(query, self.__engine)
            stats['active_companies'] = result.iloc[0, 0]
            print(f"Companies with stock data: {stats['active_companies']}", flush=True)
            
            # Count stocks
            print("\nCounting stock records...", flush=True)
            query = "SELECT COUNT(*) FROM stocks"
            result = pd.read_sql(query, self.__engine)
            stats['stock_count'] = result.iloc[0, 0]
            print(f"Total stock records: {stats['stock_count']}", flush=True)
            
            # Get table structure
            print("\nChecking database schema...", flush=True)
            query = """
            SELECT table_name, column_name, data_type 
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
            """
            schema = pd.read_sql(query, self.__engine)
            for table in schema['table_name'].unique():
                cols = schema[schema['table_name'] == table]
                print(f"  Table '{table}' columns: {', '.join(cols['column_name'].tolist())}", flush=True)
            
            print("\n===== DATABASE DIAGNOSTIC COMPLETED SUCCESSFULLY =====", flush=True)
            return stats
        except Exception as e:
            print(f"\nDIAGNOSTIC ERROR: {str(e)}", flush=True)
            print("\n===== DATABASE DIAGNOSTIC FAILED =====", flush=True)
            return {"Error": str(e)}
        
    def execute_query(self, query):
        """Execute a SQL query and return the results.
        
        For SELECT queries, returns a pandas DataFrame.
        For other queries (INSERT, UPDATE, DELETE), returns the number of rows affected.
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            pd.DataFrame or int: Query results as DataFrame or number of rows affected
        """
        print(f"DASHBOARD: Executing SQL query: {query[:100]}...", flush=True)
        
        try:
            # Check if this is a SELECT query
            is_select = query.strip().lower().startswith('select')
            
            if is_select:
                # For SELECT queries, return a DataFrame
                result = pd.read_sql(query, self.__engine)
                print(f"DASHBOARD: Query returned {len(result)} rows", flush=True)
                return result
            else:
                # For non-SELECT queries (INSERT, UPDATE, DELETE), execute using connection
                with self.__engine.connect() as connection:
                    from sqlalchemy import text
                    result = connection.execute(text(query))
                    # Commit the transaction
                    connection.commit()
                    print(f"DASHBOARD: Query affected {result.rowcount} rows", flush=True)
                    return result.rowcount
                    
        except Exception as e:
            print(f"DASHBOARD ERROR executing query: {str(e)}", flush=True)
            raise e

# 
# main
#

if __name__ == "__main__":
    import doctest
    # timescaleDB should run, possibly in Docker
    # db = TimescaleStockMarketModel("bourse", "ricou", "localhost", "monmdp")
    import timescaledb_model as tsdb
    db = tsdb.TimescaleStockMarketModel(
        'bourse', 'ricou', 'db', 'monmdp')  # inside docker
    
    # Run a diagnostic test
    print("Running diagnostic test...")
    db.get_database_stats()
    
    doctest.testmod()