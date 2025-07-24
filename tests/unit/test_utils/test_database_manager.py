"""
Unit tests for DatabaseManager class
Tests database connection management and operations
"""
import pytest
import pandas as pd
import duckdb
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import tempfile
import sqlite3

from src.utils.database_manager import DatabaseManager, create_database_manager
from src.utils.exceptions import DatabaseError


class TestDatabaseManager:
    """Test suite for DatabaseManager functionality"""
    
    # ========================================
    # Initialisation Tests
    # ========================================
    
    def test_init_with_valid_path_creates_manager(self, test_temp_dir):
        """Test successful DatabaseManager initialisation"""
        # Arrange
        db_path = test_temp_dir / "test.db"
        
        # Act
        db_manager = DatabaseManager(str(db_path))
        
    def test_create_table_from_dataframe_creates_table_successfully(self, test_temp_dir, sample_dataframe):
        """Test creating table from DataFrame"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "df_test.db"))
        
        # Act
        db_manager.create_table_from_dataframe(sample_dataframe, 'df_table')
        
        # Assert
        assert db_manager.table_exists('df_table')
        count = db_manager.get_table_count('df_table')
        assert count == len(sample_dataframe)
    
    def test_create_table_from_dataframe_with_replace_overwrites_existing(self, test_temp_dir, sample_dataframe):
        """Test creating table with replace option overwrites existing table"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "replace_test.db"))
        
        # Create initial table
        small_df = pd.DataFrame({'id': [1, 2]})
        db_manager.create_table_from_dataframe(small_df, 'replace_table')
        assert db_manager.get_table_count('replace_table') == 2
        
        # Act - Replace with larger DataFrame
        db_manager.create_table_from_dataframe(sample_dataframe, 'replace_table', if_exists='replace')
        
        # Assert
        count = db_manager.get_table_count('replace_table')
        assert count == len(sample_dataframe)
    
    def test_create_table_from_dataframe_with_fail_raises_error_if_exists(self, test_temp_dir, sample_dataframe):
        """Test creating table with fail option raises error if table exists"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "fail_test.db"))
        db_manager.create_table_from_dataframe(sample_dataframe, 'fail_table')
        
        # Act & Assert
        with pytest.raises(DatabaseError, match="already exists"):
            db_manager.create_table_from_dataframe(sample_dataframe, 'fail_table', if_exists='fail')
    
    def test_insert_dataframe_adds_data_to_existing_table(self, test_temp_dir, sample_dataframe):
        """Test inserting DataFrame data into existing table"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "insert_test.db"))
        
        # Create table with initial data
        initial_df = sample_dataframe.iloc[:1]  # First row only
        db_manager.create_table_from_dataframe(initial_df, 'insert_table')
        assert db_manager.get_table_count('insert_table') == 1
        
        # Act - Insert remaining data
        remaining_df = sample_dataframe.iloc[1:]  # Remaining rows
        db_manager.insert_dataframe(remaining_df, 'insert_table')
        
        # Assert
        count = db_manager.get_table_count('insert_table')
        assert count == len(sample_dataframe)
    
    def test_insert_dataframe_with_missing_table_raises_error(self, test_temp_dir, sample_dataframe):
        """Test inserting into missing table raises error"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "missing_insert_test.db"))
        
        # Act & Assert
        with pytest.raises(DatabaseError, match="does not exist"):
            db_manager.insert_dataframe(sample_dataframe, 'missing_table')
    
    def test_get_sample_data_returns_limited_rows(self, test_temp_dir, sample_dataframe):
        """Test getting sample data from table"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "sample_test.db"))
        db_manager.create_table_from_dataframe(sample_dataframe, 'sample_table')
        
        # Act
        sample_data = db_manager.get_sample_data('sample_table', n_rows=2)
        
        # Assert
        assert isinstance(sample_data, pd.DataFrame)
        assert len(sample_data) == 2
        assert list(sample_data.columns) == list(sample_dataframe.columns)
    
    def test_get_sample_data_with_missing_table_raises_error(self, test_temp_dir):
        """Test getting sample data from missing table raises error"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "missing_sample_test.db"))
        
        # Act & Assert
        with pytest.raises(DatabaseError, match="Failed to get sample data"):
            db_manager.get_sample_data('missing_table')
    
    # ========================================
    # Validation and Utility Tests
    # ========================================
    
    def test_validate_connection_with_working_database_returns_true(self, test_temp_dir):
        """Test connection validation with working database"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "validate_test.db"))
        
        # Act
        is_valid = db_manager.validate_connection()
        
        # Assert
        assert is_valid is True
    
    @patch('src.utils.database_manager.duckdb.connect')
    def test_validate_connection_with_broken_database_returns_false(self, mock_connect, test_temp_dir):
        """Test connection validation with broken database"""
        # Arrange
        mock_connect.side_effect = Exception("Connection failed")
        db_manager = DatabaseManager(str(test_temp_dir / "broken_validate_test.db"))
        
        # Act
        is_valid = db_manager.validate_connection()
        
        # Assert
        assert is_valid is False
    
    def test_get_database_size_returns_file_size(self, test_temp_dir, sample_dataframe):
        """Test getting database file size"""
        # Arrange
        db_path = test_temp_dir / "size_test.db"
        db_manager = DatabaseManager(str(db_path))
        
        # Create some data to ensure file has size
        db_manager.create_table_from_dataframe(sample_dataframe, 'size_table')
        
        # Act
        size_mb = db_manager.get_database_size()
        
        # Assert
        assert isinstance(size_mb, float)
        assert size_mb >= 0
        # Database should have some size after creating table
        assert size_mb > 0
    
    def test_get_database_size_with_missing_file_returns_zero(self, test_temp_dir):
        """Test getting size of non-existent database file"""
        # Arrange
        db_path = test_temp_dir / "missing_size_test.db"
        db_manager = DatabaseManager(str(db_path))
        # Don't create any tables or perform operations
        
        # Act
        size_mb = db_manager.get_database_size()
        
        # Assert
        assert size_mb == 0.0
    
    # ========================================
    # Error Handling Tests
    # ========================================
    
    def test_database_error_includes_context_information(self, test_temp_dir):
        """Test that DatabaseError includes helpful context"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "error_context_test.db"))
        
        # Act & Assert
        try:
            db_manager.execute_query("INVALID SQL")
        except DatabaseError as e:
            assert hasattr(e, 'query')
            assert "INVALID SQL" in str(e.query) if e.query else True
    
    def test_table_operations_with_special_characters_in_names(self, test_temp_dir):
        """Test table operations with special characters in table names"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "special_chars_test.db"))
        table_name = "table-with-hyphens"
        
        # Act
        db_manager.execute_non_query(f'CREATE TABLE "{table_name}" (id INTEGER)')
        
        # Assert
        assert db_manager.table_exists(table_name)
        count = db_manager.get_table_count(table_name)
        assert count == 0
        
        # Cleanup
        db_manager.drop_table(table_name)
        assert not db_manager.table_exists(table_name)


# ========================================
# Factory Function Tests
# ========================================

class TestCreateDatabaseManager:
    """Test suite for the factory function"""
    
    def test_create_database_manager_returns_database_manager_instance(self, test_temp_dir):
        """Test factory function creates DatabaseManager instance"""
        # Arrange
        db_path = str(test_temp_dir / "factory_test.db")
        
        # Act
        db_manager = create_database_manager(db_path)
        
        # Assert
        assert isinstance(db_manager, DatabaseManager)
        assert str(db_manager.db_path) == db_path
    
    def test_create_database_manager_with_memory_database(self):
        """Test factory function with in-memory database"""
        # Act
        db_manager = create_database_manager(":memory:")
        
        # Assert
        assert isinstance(db_manager, DatabaseManager)
        assert db_manager.validate_connection()


# ========================================
# Integration-style Tests
# ========================================

class TestDatabaseManagerIntegration:
    """Integration-style tests for complete workflows"""
    
    def test_complete_table_lifecycle(self, test_temp_dir, sample_dataframe):
        """Test complete table creation, population, and cleanup lifecycle"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "lifecycle_test.db"))
        table_name = "lifecycle_table"
        
        # Act & Assert - Create table
        db_manager.create_table_from_dataframe(sample_dataframe, table_name)
        assert db_manager.table_exists(table_name)
        
        # Act & Assert - Verify data
        count = db_manager.get_table_count(table_name)
        assert count == len(sample_dataframe)
        
        # Act & Assert - Get sample data
        sample_data = db_manager.get_sample_data(table_name, n_rows=2)
        assert len(sample_data) == 2
        
        # Act & Assert - Get schema
        schema = db_manager.get_table_schema(table_name)
        assert len(schema) == len(sample_dataframe.columns)
        
        # Act & Assert - Drop table
        db_manager.drop_table(table_name)
        assert not db_manager.table_exists(table_name)
    
    def test_multiple_table_operations(self, test_temp_dir, sample_dataframe):
        """Test operations with multiple tables"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "multi_table_test.db"))
        
        # Create multiple tables
        for i in range(3):
            table_name = f"table_{i}"
            test_df = sample_dataframe.copy()
            test_df['table_id'] = i
            db_manager.create_table_from_dataframe(test_df, table_name)
        
        # Act
        tables = db_manager.list_tables()
        
        # Assert
        assert len(tables) == 3
        for i in range(3):
            assert f"table_{i}" in tables
            count = db_manager.get_table_count(f"table_{i}")
            assert count == len(sample_dataframe)
        
        # Cleanup all tables
        db_manager.clear_all_tables()
        assert len(db_manager.list_tables()) == 0
    
    def test_concurrent_connection_usage(self, test_temp_dir):
        """Test that multiple connection contexts work correctly"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "concurrent_test.db"))
        
        # Act - Use multiple connection contexts
        with db_manager.get_connection() as conn1:
            conn1.execute("CREATE TABLE concurrent_test (id INTEGER)")
            
        with db_manager.get_connection() as conn2:
            conn2.execute("INSERT INTO concurrent_test VALUES (1)")
            
        with db_manager.get_connection() as conn3:
            result = conn3.execute("SELECT COUNT(*) FROM concurrent_test").fetchone()
            count = result[0]
        
        # Assert
        assert count == 1
        assert db_manager.db_path.parent.exists()  # Directory should be created
    
    def test_init_creates_parent_directories(self, test_temp_dir):
        """Test that parent directories are created if they don't exist"""
        # Arrange
        nested_db_path = test_temp_dir / "nested" / "path" / "test.db"
        
        # Act
        db_manager = DatabaseManager(str(nested_db_path))
        
        # Assert
        assert nested_db_path.parent.exists()
        assert db_manager.db_path == nested_db_path
    
    # ========================================
    # Connection Management Tests
    # ========================================
    
    def test_get_connection_provides_working_connection(self, test_temp_dir):
        """Test that connection context manager provides working connection"""
        # Arrange
        db_path = test_temp_dir / "connection_test.db"
        db_manager = DatabaseManager(str(db_path))
        
        # Act & Assert
        with db_manager.get_connection() as conn:
            assert conn is not None
            # Test basic query
            result = conn.execute("SELECT 1 as test").fetchone()
            assert result[0] == 1
    
    def test_get_connection_closes_connection_properly(self, test_temp_dir):
        """Test that connections are properly closed"""
        # Arrange
        db_path = test_temp_dir / "close_test.db"
        db_manager = DatabaseManager(str(db_path))
        
        # Act
        conn_ref = None
        with db_manager.get_connection() as conn:
            conn_ref = conn
            conn.execute("SELECT 1")
        
        # Assert - Connection should be closed after context exit
        # Note: DuckDB doesn't have a simple "is_closed" property, 
        # but attempting operations on closed connection should fail
        with pytest.raises(Exception):
            conn_ref.execute("SELECT 1")
    
    @patch('src.utils.database_manager.duckdb.connect')
    def test_get_connection_handles_connection_errors(self, mock_connect, test_temp_dir):
        """Test handling of connection errors"""
        # Arrange
        mock_connect.side_effect = Exception("Connection failed")
        db_path = test_temp_dir / "error_test.db"
        db_manager = DatabaseManager(str(db_path))
        
        # Act & Assert
        with pytest.raises(DatabaseError, match="Database operation failed"):
            with db_manager.get_connection():
                pass
    
    # ========================================
    # Query Execution Tests
    # ========================================
    
    def test_execute_query_with_simple_query_returns_dataframe(self, test_temp_dir):
        """Test executing simple query"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "query_test.db"))
        
        # Act
        result = db_manager.execute_query("SELECT 1 as number, 'test' as text")
        
        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]['number'] == 1
        assert result.iloc[0]['text'] == 'test'
    
    def test_execute_query_with_parameters_handles_correctly(self, test_temp_dir):
        """Test executing query with parameters"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "param_test.db"))
        
        # Act
        result = db_manager.execute_query("SELECT ? as value", [42])
        
        # Assert
        assert len(result) == 1
        assert result.iloc[0]['value'] == 42
    
    def test_execute_query_with_invalid_sql_raises_database_error(self, test_temp_dir):
        """Test that invalid SQL raises DatabaseError"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "invalid_test.db"))
        
        # Act & Assert
        with pytest.raises(DatabaseError, match="Query failed"):
            db_manager.execute_query("INVALID SQL SYNTAX")
    
    def test_execute_non_query_creates_table_successfully(self, test_temp_dir):
        """Test executing non-query statements like CREATE TABLE"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "create_test.db"))
        create_sql = """
        CREATE TABLE test_table (
            id INTEGER,
            name VARCHAR
        )
        """
        
        # Act
        db_manager.execute_non_query(create_sql)
        
        # Assert - Table should exist
        assert db_manager.table_exists('test_table')
    
    def test_execute_non_query_with_invalid_sql_raises_database_error(self, test_temp_dir):
        """Test that invalid non-query SQL raises DatabaseError"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "invalid_nonquery_test.db"))
        
        # Act & Assert
        with pytest.raises(DatabaseError, match="Statement failed"):
            db_manager.execute_non_query("CREATE INVALID TABLE")
    
    # ========================================
    # Table Management Tests
    # ========================================
    
    def test_table_exists_with_existing_table_returns_true(self, test_temp_dir):
        """Test table existence check for existing table"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "exists_test.db"))
        db_manager.execute_non_query("CREATE TABLE existing_table (id INTEGER)")
        
        # Act
        exists = db_manager.table_exists('existing_table')
        
        # Assert
        assert bool(exists) is True
    
    def test_table_exists_with_missing_table_returns_false(self, test_temp_dir):
        """Test table existence check for missing table"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "missing_test.db"))
        
        # Act
        exists = db_manager.table_exists('missing_table')
        
        # Assert
        assert bool(exists) is False
    
    def test_get_table_count_returns_correct_count(self, test_temp_dir):
        """Test getting row count from table"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "count_test.db"))
        db_manager.execute_non_query("CREATE TABLE count_table (id INTEGER)")
        db_manager.execute_non_query("INSERT INTO count_table VALUES (1), (2), (3)")
        
        # Act
        count = db_manager.get_table_count('count_table')
        
        # Assert
        assert count == 3
    
    def test_get_table_count_with_missing_table_raises_error(self, test_temp_dir):
        """Test getting count from missing table raises error"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "missing_count_test.db"))
        
        # Act & Assert
        with pytest.raises(DatabaseError, match="Failed to get count"):
            db_manager.get_table_count('missing_table')
    
    def test_get_table_schema_returns_column_info(self, test_temp_dir):
        """Test getting table schema information"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "schema_test.db"))
        create_sql = """
        CREATE TABLE schema_table (
            id INTEGER,
            name VARCHAR,
            score DOUBLE
        )
        """
        db_manager.execute_non_query(create_sql)
        
        # Act
        schema = db_manager.get_table_schema('schema_table')
        
        # Assert
        assert isinstance(schema, dict)
        assert 'id' in schema
        assert 'name' in schema
        assert 'score' in schema
    
    def test_list_tables_returns_table_names(self, test_temp_dir):
        """Test listing all tables in database"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "list_test.db"))
        db_manager.execute_non_query("CREATE TABLE table1 (id INTEGER)")
        db_manager.execute_non_query("CREATE TABLE table2 (id INTEGER)")
        
        # Act
        tables = db_manager.list_tables()
        
        # Assert
        assert isinstance(tables, list)
        assert 'table1' in tables
        assert 'table2' in tables
    
    def test_drop_table_removes_table(self, test_temp_dir):
        """Test dropping table removes it from database"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "drop_test.db"))
        db_manager.execute_non_query("CREATE TABLE drop_me (id INTEGER)")
        assert db_manager.table_exists('drop_me')
        
        # Act
        db_manager.drop_table('drop_me')
        
        # Assert
        assert not db_manager.table_exists('drop_me')
    
    def test_drop_table_with_if_exists_false_and_missing_table_raises_error(self, test_temp_dir):
        """Test dropping missing table without IF EXISTS raises error"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "drop_missing_test.db"))
        
        # Act & Assert
        with pytest.raises(DatabaseError, match="Failed to drop table"):
            db_manager.drop_table('missing_table', if_exists=False)
    
    def test_clear_all_tables_removes_all_tables(self, test_temp_dir):
        """Test clearing all tables from database"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "clear_test.db"))
        db_manager.execute_non_query("CREATE TABLE table1 (id INTEGER)")
        db_manager.execute_non_query("CREATE TABLE table2 (id INTEGER)")
        assert len(db_manager.list_tables()) == 2
        
        # Act
        db_manager.clear_all_tables()
        
        # Assert
        assert len(db_manager.list_tables()) == 0
    
    # ========================================
    # DataFrame Operations Tests
    # ========================================
    
    def test_create_table_from_dataframe_creates_table_successfully(self, test_temp_dir, sample_dataframe):
        """Test creating table from DataFrame"""
        # Arrange
        db_manager = DatabaseManager(str(test_temp_dir / "df_test.db"))
        
        # Act
        db_manager.create_table_from_dataframe(sample_dataframe, 'df_table')
        
        # Assert
        assert db_manager.table_exists('df_table')
        count = db_manager.get_table_count('df_table')
        assert count == len(sample_dataframe)