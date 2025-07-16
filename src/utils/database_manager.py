"""
Shared DuckDB database management utilities
Centralised database connection logic to prevent locking issues
"""

import duckdb
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import time

from .exceptions import DatabaseError
from .logging_config import get_logger

logger = get_logger('database')


class DatabaseManager:
    """
    Centralised database connection manager
    Handles connection lifecycle and prevents locking issues
    """
    
    def __init__(self, db_path: str):
        """
        Initialise database manager
        
        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        self._ensure_db_directory_exists()
        logger.info(f"Initialised DatabaseManager for: {self.db_path}")
    
    def _ensure_db_directory_exists(self) -> None:
        """Ensure database directory exists"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections
        Ensures connections are properly closed to prevent locking
        
        Usage:
            with db_manager.get_connection() as conn:
                result = conn.execute("SELECT * FROM table").fetchdf()
        """
        conn = None
        try:
            logger.debug(f"Opening connection to {self.db_path}")
            conn = duckdb.connect(str(self.db_path))
            yield conn
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                try:
                    conn.close()
                    logger.debug("Database connection closed")
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")
    
    def execute_query(self, query: str, params: Optional[List] = None) -> Any:
        """
        Execute a query and return results
        
        Args:
            query: SQL query to execute
            params: Query parameters as a list (optional)
            
        Returns:
            Query results
        """
        with self.get_connection() as conn:
            try:
                logger.debug(f"Executing query: {query[:100]}...")
                if params:
                    result = conn.execute(query, params).fetchdf()
                else:
                    result = conn.execute(query).fetchdf()
                logger.debug(f"Query returned {len(result)} rows")
                return result
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise DatabaseError(f"Query failed: {e}", query=query)
    
    def execute_non_query(self, query: str, params: Optional[List] = None) -> None:
        """
        Execute a non-query statement (CREATE, INSERT, etc.)
        
        Args:
            query: SQL statement to execute
            params: Query parameters as a list (optional)
        """
        with self.get_connection() as conn:
            try:
                logger.debug(f"Executing statement: {query[:100]}...")
                if params:
                    conn.execute(query, params)
                else:
                    conn.execute(query)
                logger.debug("Statement executed successfully")
            except Exception as e:
                logger.error(f"Statement execution failed: {e}")
                raise DatabaseError(f"Statement failed: {e}", query=query)
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database
        
        Args:
            table_name: Name of table to check
            
        Returns:
            True if table exists, False otherwise
        """
        try:
            query = """
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_name = ? AND table_type = 'BASE TABLE'
            """
            result = self.execute_query(query, [table_name])
            return result.iloc[0]['count'] > 0
        except Exception as e:
            logger.warning(f"Error checking if table exists: {e}")
            return False
    
    def get_table_count(self, table_name: str) -> int:
        """
        Get row count for a table
        
        Args:
            table_name: Name of table
            
        Returns:
            Number of rows in table
        """
        try:
            query = f'SELECT COUNT(*) as count FROM "{table_name}"'
            result = self.execute_query(query)
            return int(result.iloc[0]['count'])
        except Exception as e:
            logger.error(f"Error getting count for table {table_name}: {e}")
            raise DatabaseError(f"Failed to get count for table {table_name}: {e}", table_name=table_name)
    
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """
        Get schema information for a table
        
        Args:
            table_name: Name of table
            
        Returns:
            Dictionary mapping column names to data types
        """
        try:
            query = f'DESCRIBE "{table_name}"'
            result = self.execute_query(query)
            
            schema = {}
            for _, row in result.iterrows():
                schema[row['column_name']] = row['column_type']
            
            return schema
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {e}")
            raise DatabaseError(f"Failed to get schema for table {table_name}: {e}", table_name=table_name)
    
    def list_tables(self) -> List[str]:
        """
        Get list of all tables in database
        
        Returns:
            List of table names
        """
        try:
            query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_type = 'BASE TABLE'
            ORDER BY table_name
            """
            result = self.execute_query(query)
            return result['table_name'].tolist()
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            raise DatabaseError(f"Failed to list tables: {e}")
    
    def drop_table(self, table_name: str, if_exists: bool = True) -> None:
        """
        Drop a table from the database
        
        Args:
            table_name: Name of table to drop
            if_exists: Whether to use IF EXISTS clause
        """
        try:
            if_exists_clause = "IF EXISTS" if if_exists else ""
            query = f'DROP TABLE {if_exists_clause} "{table_name}"'
            self.execute_non_query(query)
            logger.info(f"Dropped table: {table_name}")
        except Exception as e:
            logger.error(f"Error dropping table {table_name}: {e}")
            raise DatabaseError(f"Failed to drop table {table_name}: {e}", table_name=table_name)
    
    def clear_all_tables(self) -> None:
        """
        Drop all tables in the database
        Useful for clean slate operations
        """
        try:
            tables = self.list_tables()
            
            if not tables:
                logger.info("No tables found - database is already clean")
                return
            
            logger.info(f"Clearing {len(tables)} tables from database")
            
            for table_name in tables:
                self.drop_table(table_name, if_exists=True)
                logger.debug(f"Dropped table: {table_name}")
            
            logger.info("All tables cleared from database")
            
        except Exception as e:
            logger.error(f"Error clearing tables: {e}")
            raise DatabaseError(f"Failed to clear all tables: {e}")
    
    def create_table_from_dataframe(self, df, table_name: str, 
                                   if_exists: str = 'replace') -> None:
        """
        Create table from pandas DataFrame
        
        Args:
            df: pandas DataFrame
            table_name: Name for the new table
            if_exists: What to do if table exists ('replace', 'append', 'fail')
        """
        with self.get_connection() as conn:
            try:
                logger.info(f"Creating table '{table_name}' from DataFrame with {len(df)} rows")
                
                # Handle if_exists logic
                if if_exists == 'replace' and self.table_exists(table_name):
                    self.drop_table(table_name)
                elif if_exists == 'fail' and self.table_exists(table_name):
                    raise DatabaseError(f"Table {table_name} already exists and if_exists='fail'")
                
                # Register DataFrame as temporary table
                temp_table_name = f"temp_{table_name}_{int(time.time())}"
                conn.register(temp_table_name, df)
                
                # Create permanent table from temporary table
                create_query = f'CREATE TABLE "{table_name}" AS SELECT * FROM {temp_table_name}'
                conn.execute(create_query)
                
                logger.info(f"Successfully created table '{table_name}' with {len(df)} rows")
                
            except Exception as e:
                logger.error(f"Error creating table {table_name}: {e}")
                raise DatabaseError(f"Failed to create table {table_name}: {e}", table_name=table_name)
    
    def insert_dataframe(self, df, table_name: str) -> None:
        """
        Insert DataFrame data into existing table
        
        Args:
            df: pandas DataFrame
            table_name: Name of existing table
        """
        with self.get_connection() as conn:
            try:
                logger.info(f"Inserting {len(df)} rows into table '{table_name}'")
                
                if not self.table_exists(table_name):
                    raise DatabaseError(f"Table {table_name} does not exist")
                
                # Register DataFrame as temporary table
                temp_table_name = f"temp_insert_{table_name}_{int(time.time())}"
                conn.register(temp_table_name, df)
                
                # Insert data
                insert_query = f'INSERT INTO "{table_name}" SELECT * FROM {temp_table_name}'
                conn.execute(insert_query)
                
                logger.info(f"Successfully inserted {len(df)} rows into '{table_name}'")
                
            except Exception as e:
                logger.error(f"Error inserting into table {table_name}: {e}")
                raise DatabaseError(f"Failed to insert into table {table_name}: {e}", table_name=table_name)
    
    def get_sample_data(self, table_name: str, n_rows: int = 5) -> Any:
        """
        Get sample data from a table
        
        Args:
            table_name: Name of table
            n_rows: Number of rows to sample
            
        Returns:
            DataFrame with sample data
        """
        try:
            query = f'SELECT * FROM "{table_name}" LIMIT {n_rows}'
            return self.execute_query(query)
        except Exception as e:
            logger.error(f"Error getting sample data from {table_name}: {e}")
            raise DatabaseError(f"Failed to get sample data from {table_name}: {e}", table_name=table_name)
    
    def validate_connection(self) -> bool:
        """
        Validate that database connection works
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            with self.get_connection() as conn:
                conn.execute("SELECT 1").fetchone()
            logger.info("Database connection validation successful")
            return True
        except Exception as e:
            logger.error(f"Database connection validation failed: {e}")
            return False
    
    def get_database_size(self) -> float:
        """
        Get database file size in MB
        
        Returns:
            Database size in MB
        """
        try:
            if self.db_path.exists():
                size_bytes = self.db_path.stat().st_size
                size_mb = size_bytes / (1024 * 1024)
                return round(size_mb, 2)
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Could not get database size: {e}")
            return 0.0

def create_database_manager(db_path: str) -> DatabaseManager:
    """
    Factory function to create a DatabaseManager instance
    
    Args:
        db_path: Path to database file
        
    Returns:
        DatabaseManager instance
    """
    return DatabaseManager(db_path)
                