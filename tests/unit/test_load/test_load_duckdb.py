"""
Unit tests for DataLoader class
Tests data loading operations, database management, and error handling
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, Mock, mock_open, call
import time

from src.load.load_duckdb import DataLoader
from src.config.config_manager import ConfigManager
from src.utils.exceptions import DatabaseError
from src.utils.database_manager import DatabaseManager
from src.extract.base_extract import DatasetMetadata


def mock_schema_file_read():
    """Helper function to mock schema file reading"""
    return patch.object(DataLoader, '_read_schema_file', return_value="CREATE TABLE test (id INTEGER);")


class TestDataLoaderInitialisation:
    """Test suite for DataLoader initialisation"""
    
    def test_init_with_valid_config_manager_succeeds(self, mock_config_manager_success):
        """Test successful DataLoader initialisation with valid config"""
        # Arrange & Act
        loader = DataLoader(mock_config_manager_success)
        
        # Assert
        assert loader.config == mock_config_manager_success
        assert hasattr(loader, 'logger')
        assert hasattr(loader, 'db_manager')
        assert isinstance(loader.db_manager, DatabaseManager)
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_init_creates_database_manager_with_correct_path(self, mock_db_class, mock_config_manager_success):
        """Test that DatabaseManager is created with correct database path"""
        # Arrange
        expected_db_path = ":memory:"
        mock_config_manager_success.get_database_path.return_value = expected_db_path
        
        # Act
        loader = DataLoader(mock_config_manager_success)
        
        # Assert
        mock_db_class.assert_called_once_with(expected_db_path)
        mock_config_manager_success.get_database_path.assert_called_once()
    
    def test_init_sets_up_logger_correctly(self, mock_config_manager_success):
        """Test that logger is properly configured"""
        # Arrange & Act
        loader = DataLoader(mock_config_manager_success)
        
        # Assert
        assert loader.logger is not None
        assert hasattr(loader.logger, 'info')
        assert hasattr(loader.logger, 'error')
        assert hasattr(loader.logger, 'warning')


class TestDataLoaderDatabaseConnection:
    """Test suite for database connection management"""
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_load_datasets_validates_database_connection_success(self, mock_db_class, mock_config_manager_success, normalised_data_single_table):
        """Test successful database connection validation"""
        # Arrange
        mock_db = Mock()
        mock_db.validate_connection.return_value = True
        mock_db.list_tables.return_value = []
        mock_db.clear_all_tables.return_value = None
        mock_db.execute_non_query.return_value = None
        mock_db.insert_dataframe.return_value = None
        mock_db.get_table_count.return_value = 3
        mock_db.get_sample_data.return_value = pd.DataFrame({'name': ['Test']})
        mock_db.get_database_size.return_value = 1.0
        mock_db_class.return_value = mock_db
        
        # Mock schema file reading to avoid file system dependencies
        with mock_schema_file_read():
            loader = DataLoader(mock_config_manager_success)
            
            # Act
            result = loader.load_datasets(normalised_data_single_table)
            
            # Assert
            mock_db.validate_connection.assert_called_once()
            assert len(result) == 1
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_load_datasets_with_connection_failure_raises_database_error(self, mock_db_class, mock_config_manager_success, normalised_data_single_table):
        """Test that connection failure raises DatabaseError"""
        # Arrange
        mock_db = Mock()
        mock_db.validate_connection.return_value = False
        mock_db_class.return_value = mock_db
        
        loader = DataLoader(mock_config_manager_success)
        
        # Act & Assert
        with pytest.raises(DatabaseError, match="Cannot connect to database"):
            loader.load_datasets(normalised_data_single_table)


class TestDataLoaderSchemaManagement:
    """Test suite for schema file reading and table creation"""
    
    def test_read_schema_file_with_valid_file_returns_content(self, mock_config_manager_success, temp_schema_file):
        """Test reading valid schema file"""
        # Arrange
        loader = DataLoader(mock_config_manager_success)
        
        # Act
        content = loader._read_schema_file(str(temp_schema_file))
        
        # Assert
        assert isinstance(content, str)
        assert len(content) > 0
        assert 'CREATE TABLE' in content
        assert '{table_name}' in content
    
    def test_read_schema_file_with_missing_file_raises_database_error(self, mock_config_manager_success):
        """Test reading missing schema file raises DatabaseError"""
        # Arrange
        loader = DataLoader(mock_config_manager_success)
        missing_file = "/path/that/does/not/exist.sql"
        
        # Act & Assert
        with pytest.raises(DatabaseError, match="Failed to read schema file"):
            loader._read_schema_file(missing_file)
    
    def test_read_schema_file_with_empty_file_raises_database_error(self, mock_config_manager_success, temp_schema_dir):
        """Test reading empty schema file raises DatabaseError"""
        # Arrange
        loader = DataLoader(mock_config_manager_success)
        empty_file = temp_schema_dir / "empty.sql"
        empty_file.touch()  # Create empty file
        
        # Act & Assert
        with pytest.raises(DatabaseError, match="Schema file is empty"):
            loader._read_schema_file(str(empty_file))
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_create_table_schema_with_valid_schema_succeeds(self, mock_db_class, mock_config_manager_success, temp_schema_file):
        """Test table schema creation with valid schema file"""
        # Arrange
        mock_db = Mock()
        mock_db.execute_non_query.return_value = None
        mock_db_class.return_value = mock_db
        
        mock_config_manager_success.get_schema_for_sheet.return_value = "test_schema"
        mock_config_manager_success.get_schema_file_path.return_value = str(temp_schema_file)
        
        loader = DataLoader(mock_config_manager_success)
        
        # Act
        loader._create_table_schema("test_table", "test_sheet")
        
        # Assert
        mock_db.execute_non_query.assert_called_once()
        # Verify that table name was substituted in schema
        call_args = mock_db.execute_non_query.call_args[0][0]
        assert "test_table" in call_args
        assert "{table_name}" not in call_args
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_create_table_schema_with_database_error_raises_database_error(self, mock_db_class, mock_config_manager_success, temp_schema_file):
        """Test table schema creation with database error"""
        # Arrange
        mock_db = Mock()
        mock_db.execute_non_query.side_effect = Exception("SQL execution failed")
        mock_db_class.return_value = mock_db
        
        mock_config_manager_success.get_schema_for_sheet.return_value = "test_schema"
        mock_config_manager_success.get_schema_file_path.return_value = str(temp_schema_file)
        
        loader = DataLoader(mock_config_manager_success)
        
        # Act & Assert
        with pytest.raises(DatabaseError, match="Failed to create table schema"):
            loader._create_table_schema("test_table", "test_sheet")


class TestDataLoaderDataLoading:
    """Test suite for data loading operations"""
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_load_datasets_with_single_dataset_returns_success(self, mock_db_class, mock_config_manager_success, normalised_data_single_table):
        """Test loading single dataset successfully"""
        # Arrange
        mock_db = Mock()
        mock_db.validate_connection.return_value = True
        mock_db.list_tables.return_value = []
        mock_db.clear_all_tables.return_value = None
        mock_db.execute_non_query.return_value = None
        mock_db.insert_dataframe.return_value = None
        mock_db.get_table_count.return_value = 3  # Match expected row count
        mock_db.get_sample_data.return_value = pd.DataFrame({'name': ['Test']})
        mock_db.get_database_size.return_value = 1.0
        mock_db_class.return_value = mock_db
        
        # Mock schema file reading to avoid file system dependencies
        with mock_schema_file_read():
            loader = DataLoader(mock_config_manager_success)
            
            # Act
            result = loader.load_datasets(normalised_data_single_table)
            
            # Assert
            assert isinstance(result, list)
            assert len(result) == 1
            assert 'test_table' in result
            mock_db.insert_dataframe.assert_called_once()
            mock_db.get_table_count.assert_called_once()
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_load_datasets_with_multiple_datasets_loads_all(self, mock_db_class, mock_config_manager_success, normalised_data_multiple_tables):
        """Test loading multiple datasets successfully"""
        # Arrange
        mock_db = Mock()
        mock_db.validate_connection.return_value = True
        mock_db.list_tables.return_value = []
        mock_db.clear_all_tables.return_value = None
        mock_db.execute_non_query.return_value = None
        mock_db.insert_dataframe.return_value = None
        # Return correct counts for each dataset
        mock_db.get_table_count.side_effect = [2, 3]  # CS has 2 rows, Engineering has 3
        mock_db.get_sample_data.return_value = pd.DataFrame({'name': ['Test']})
        mock_db.get_database_size.return_value = 2.0
        mock_db_class.return_value = mock_db
        
        # Mock schema file reading to avoid file system dependencies
        with mock_schema_file_read():
            loader = DataLoader(mock_config_manager_success)
            
            # Act
            result = loader.load_datasets(normalised_data_multiple_tables)
            
            # Assert
            assert len(result) == 2
            assert 'computer_science_feb_highly_cited_only' in result
            assert 'engineering_feb_incites_researchers' in result
            assert mock_db.insert_dataframe.call_count == 2
            assert mock_db.get_table_count.call_count == 2
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_load_datasets_with_empty_dataset_raises_database_error(self, mock_db_class, mock_config_manager_success, normalised_data_empty_table):
        """Test loading empty dataset raises DatabaseError"""
        # Arrange
        mock_db = Mock()
        mock_db.validate_connection.return_value = True
        mock_db.list_tables.return_value = []
        mock_db.clear_all_tables.return_value = None
        mock_db_class.return_value = mock_db
        
        loader = DataLoader(mock_config_manager_success)
        
        # Act & Assert
        with pytest.raises(DatabaseError, match="No tables were successfully loaded"):
            loader.load_datasets(normalised_data_empty_table)
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_load_dataframe_to_table_with_valid_data_succeeds(self, mock_db_class, mock_config_manager_success, simple_normalised_dataset):
        """Test loading DataFrame to table with valid data"""
        # Arrange
        mock_db = Mock()
        mock_db.insert_dataframe.return_value = None
        mock_db_class.return_value = mock_db
        
        loader = DataLoader(mock_config_manager_success)
        df = simple_normalised_dataset['dataframe']
        
        # Act
        loader._load_dataframe_to_table(df, 'test_table')
        
        # Assert
        mock_db.insert_dataframe.assert_called_once_with(df, 'test_table')
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_load_dataframe_to_table_with_none_dataframe_raises_database_error(self, mock_db_class, mock_config_manager_success):
        """Test loading None DataFrame raises DatabaseError"""
        # Arrange
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        
        loader = DataLoader(mock_config_manager_success)
        
        # Act & Assert
        with pytest.raises(DatabaseError, match="No data to load"):
            loader._load_dataframe_to_table(None, 'test_table')
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_load_dataframe_to_table_with_insert_failure_raises_database_error(self, mock_db_class, mock_config_manager_success, simple_normalised_dataset):
        """Test loading DataFrame with insert failure raises DatabaseError"""
        # Arrange
        mock_db = Mock()
        mock_db.insert_dataframe.side_effect = Exception("Insert failed")
        mock_db_class.return_value = mock_db
        
        loader = DataLoader(mock_config_manager_success)
        df = simple_normalised_dataset['dataframe']
        
        # Act & Assert
        with pytest.raises(DatabaseError, match="Failed to load data into table"):
            loader._load_dataframe_to_table(df, 'test_table')


class TestDataLoaderValidation:
    """Test suite for data validation operations"""
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_validate_row_count_with_matching_counts_succeeds(self, mock_db_class, mock_config_manager_success):
        """Test row count validation with matching counts"""
        # Arrange
        mock_db = Mock()
        mock_db.get_table_count.return_value = 100
        mock_db_class.return_value = mock_db
        
        loader = DataLoader(mock_config_manager_success)
        
        # Act & Assert
        # Should not raise exception
        loader._validate_row_count(100, 'test_table')
        mock_db.get_table_count.assert_called_once_with('test_table')
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_validate_row_count_with_mismatched_counts_raises_database_error(self, mock_db_class, mock_config_manager_success):
        """Test row count validation with mismatched counts raises DatabaseError"""
        # Arrange
        mock_db = Mock()
        mock_db.get_table_count.return_value = 50
        mock_db_class.return_value = mock_db
        
        loader = DataLoader(mock_config_manager_success)
        
        # Act & Assert
        with pytest.raises(DatabaseError, match="Row count mismatch"):
            loader._validate_row_count(100, 'test_table')
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_validate_row_count_with_database_error_raises_database_error(self, mock_db_class, mock_config_manager_success):
        """Test row count validation with database error"""
        # Arrange
        mock_db = Mock()
        mock_db.get_table_count.side_effect = Exception("Database error")
        mock_db_class.return_value = mock_db
        
        loader = DataLoader(mock_config_manager_success)
        
        # Act & Assert
        with pytest.raises(DatabaseError, match="Failed to validate row count"):
            loader._validate_row_count(100, 'test_table')


class TestDataLoaderTableManagement:
    """Test suite for table management operations"""
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_clear_existing_tables_with_existing_tables_clears_all(self, mock_db_class, mock_config_manager_success):
        """Test clearing existing tables when tables exist"""
        # Arrange
        mock_db = Mock()
        mock_db.list_tables.return_value = ['table1', 'table2', 'table3']
        mock_db.clear_all_tables.return_value = None
        mock_db_class.return_value = mock_db
        
        loader = DataLoader(mock_config_manager_success)
        
        # Act
        loader._clear_existing_tables()
        
        # Assert
        mock_db.list_tables.assert_called_once()
        mock_db.clear_all_tables.assert_called_once()
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_clear_existing_tables_with_no_tables_skips_clearing(self, mock_db_class, mock_config_manager_success):
        """Test clearing when no existing tables"""
        # Arrange
        mock_db = Mock()
        mock_db.list_tables.return_value = []
        mock_db_class.return_value = mock_db
        
        loader = DataLoader(mock_config_manager_success)
        
        # Act
        loader._clear_existing_tables()
        
        # Assert
        mock_db.list_tables.assert_called_once()
        mock_db.clear_all_tables.assert_not_called()


class TestDataLoaderErrorHandling:
    """Test suite for error handling scenarios"""
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_load_datasets_with_schema_creation_failure_continues_processing(self, mock_db_class, mock_config_manager_success, normalised_data_multiple_tables):
        """Test loading continues when schema creation fails for some tables"""
        # Arrange
        mock_db = Mock()
        mock_db.validate_connection.return_value = True
        mock_db.list_tables.return_value = []
        mock_db.clear_all_tables.return_value = None
        # First schema creation succeeds, second fails
        mock_db.execute_non_query.side_effect = [None, Exception("Schema creation failed")]
        mock_db.insert_dataframe.return_value = None
        mock_db.get_table_count.return_value = 2
        mock_db.get_sample_data.return_value = pd.DataFrame({'name': ['Test']})
        mock_db.get_database_size.return_value = 1.0
        mock_db_class.return_value = mock_db
        
        # Mock schema file reading for the first successful call
        with mock_schema_file_read():
            loader = DataLoader(mock_config_manager_success)
            
            # Act
            result = loader.load_datasets(normalised_data_multiple_tables)
            
            # Assert
            # Should have 1 successful load despite 1 failure
            assert len(result) == 1
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_load_datasets_with_insert_failure_continues_processing(self, mock_db_class, mock_config_manager_success, normalised_data_multiple_tables):
        """Test loading continues when data insertion fails for some tables"""
        # Arrange
        mock_db = Mock()
        mock_db.validate_connection.return_value = True
        mock_db.list_tables.return_value = []
        mock_db.clear_all_tables.return_value = None
        mock_db.execute_non_query.return_value = None
        # First insert succeeds, second fails
        mock_db.insert_dataframe.side_effect = [None, Exception("Insert failed")]
        mock_db.get_table_count.return_value = 2
        mock_db.get_sample_data.return_value = pd.DataFrame({'name': ['Test']})
        mock_db.get_database_size.return_value = 1.0
        mock_db_class.return_value = mock_db
        
        with mock_schema_file_read():
            loader = DataLoader(mock_config_manager_success)
            
            # Act
            result = loader.load_datasets(normalised_data_multiple_tables)
            
            # Assert
            # Should have 1 successful load despite 1 failure
            assert len(result) == 1


class TestDataLoaderPerformanceAndLogging:
    """Test suite for performance monitoring and logging"""
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_load_datasets_logs_performance_metrics(self, mock_db_class, mock_config_manager_success, normalised_data_single_table):
        """Test that performance metrics are logged during loading"""
        # Arrange
        mock_db = Mock()
        mock_db.validate_connection.return_value = True
        mock_db.list_tables.return_value = []
        mock_db.clear_all_tables.return_value = None
        mock_db.execute_non_query.return_value = None
        mock_db.insert_dataframe.return_value = None
        mock_db.get_table_count.return_value = 3
        mock_db.get_sample_data.return_value = pd.DataFrame({'name': ['Test']})
        mock_db.get_database_size.return_value = 1.0
        mock_db_class.return_value = mock_db
        
        with mock_schema_file_read():
            loader = DataLoader(mock_config_manager_success)
            mock_logger = Mock()
            loader.logger = mock_logger
            
            # Act
            result = loader.load_datasets(normalised_data_single_table)
            
            # Assert
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any('Starting data loading' in msg for msg in info_calls)
            assert any('Data loading complete' in msg for msg in info_calls)
            assert any('Successfully loaded' in msg for msg in info_calls)


class TestDataLoaderUtilityMethods:
    """Test suite for utility and summary methods"""
    
    @patch('src.load.load_duckdb.DatabaseManager')
    def test_get_loaded_tables_info_returns_comprehensive_info(self, mock_db_class, mock_config_manager_success):
        """Test getting comprehensive information about loaded tables"""
        # Arrange
        mock_db = Mock()
        mock_db.list_tables.return_value = ['table1', 'table2']
        mock_db.get_table_count.side_effect = [10, 20]
        mock_db.get_table_schema.side_effect = [
            {'name': 'VARCHAR', 'times_cited': 'INTEGER'},
            {'name': 'VARCHAR', 'times_cited': 'INTEGER', 'esi_field': 'VARCHAR'}
        ]
        mock_db_class.return_value = mock_db
        
        loader = DataLoader(mock_config_manager_success)
        
        # Act
        result = loader.get_loaded_tables_info()
        
        # Assert
        assert isinstance(result, dict)
        assert 'table1' in result
        assert 'table2' in result
        assert result['table1']['row_count'] == 10
        assert result['table2']['row_count'] == 20
        assert result['table1']['column_count'] == 2
        assert result['table2']['column_count'] == 3


class TestDataLoaderEndToEndIntegration:
    """End-to-end integration tests using real components where possible"""
    
    def test_load_datasets_with_real_config_manager_and_temp_database(self, valid_config_file, temp_load_workspace):
        """Test loading with real ConfigManager and temporary database"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Create test data with ALL 13 columns in the EXACT order of your actual schema
        test_df = pd.DataFrame({
            'name': ['Dr. Real Test', 'Prof. Integration Test', 'Dr. End to End'],
            'percent_docs_cited': [85.0, 90.0, 87.0],
            'web_of_science_documents': [120, 180, 150],
            'rank': [1, 2, 3],
            'times_cited': [1000, 1500, 1200],
            'affiliation': ['University A', 'University B', 'University C'],
            'web_of_science_researcherid': ['A-1234-2024', 'B-5678-2024', 'C-9012-2024'],
            'category_normalised_citation_impact': [1.0, 1.3, 1.1],
            'orcid': ['0000-0000-0000-0001', '0000-0000-0000-0002', '0000-0000-0000-0003'],
            'highly_cited_papers': [10, 15, 12],
            'hot_papers': [2, 3, 2],
            'esi_field': ['Engineering', 'Computer Science', 'Mathematics'],
            'indicative_cross_field_score': [1.0, 1.5, 1.2]
        })
        
        test_metadata = DatasetMetadata(
            source_file="/test/real_test.xlsx",
            subject="integration_test",
            period="test",
            sheet_name="Test Sheet",
            normalised_sheet_name="highly_cited_only",
            table_name="integration_test_test_highly_cited_only",
            row_count=3,
            columns_mapped={'name': 'Name'},
            processing_timestamp="2024-01-15T10:30:45.123456",
            extraction_duration_seconds=1.0,
            normalisation_duration_seconds=0.5
        )
        
        test_data = {
            'integration_test_table': {
                'dataframe': test_df,
                'metadata': test_metadata
            }
        }
        
        # Override database path to use temporary location
        temp_db_path = temp_load_workspace / "integration_test.db"
        with patch.object(config_manager, 'get_database_path', return_value=str(temp_db_path)):
            loader = DataLoader(config_manager)
            
            # Act
            result = loader.load_datasets(test_data)
            
            # Assert
            assert len(result) == 1
            assert temp_db_path.exists()
            
            # Verify database contains expected data
            validation_result = loader.validate_loaded_data(result)
            assert validation_result['is_valid'] is True
            
            # Get summary to verify data integrity
            summary = loader.get_database_summary()
            assert summary['total_tables'] == 1
            assert summary['total_rows'] > 0
    
    def test_load_datasets_with_schema_file_integration(self, valid_config_file, temp_load_workspace, temp_schema_file):
        """Test loading with real schema file integration"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        temp_db_path = temp_load_workspace / "schema_integration_test.db"
        
        # Create test data with ALL 13 columns in the EXACT order of your actual schema
        test_df = pd.DataFrame({
            'name': ['Dr. Real Test', 'Prof. Integration Test', 'Dr. End to End'],
            'percent_docs_cited': [85.0, 90.0, 87.0],
            'web_of_science_documents': [120, 180, 150],
            'rank': [1, 2, 3],
            'times_cited': [1000, 1500, 1200],
            'affiliation': ['University A', 'University B', 'University C'],
            'web_of_science_researcherid': ['A-1234-2024', 'B-5678-2024', 'C-9012-2024'],
            'category_normalised_citation_impact': [1.0, 1.3, 1.1],
            'orcid': ['0000-0000-0000-0001', '0000-0000-0000-0002', '0000-0000-0000-0003'],
            'highly_cited_papers': [10, 15, 12],
            'hot_papers': [2, 3, 2],
            'esi_field': ['Engineering', 'Computer Science', 'Mathematics'],
            'indicative_cross_field_score': [1.0, 1.5, 1.2]
        })
        
        test_metadata = DatasetMetadata(
            source_file="/test/schema_test.xlsx",
            subject="schema_test",
            period="test",
            sheet_name="Test Sheet",
            normalised_sheet_name="highly_cited_only",
            table_name="schema_test_test_highly_cited_only",
            row_count=2,
            columns_mapped={'name': 'Name'},
            processing_timestamp="2024-01-15T10:30:45.123456",
            extraction_duration_seconds=1.0,
            normalisation_duration_seconds=0.5
        )
        
        test_data = {
            'schema_test_table': {
                'dataframe': test_df,
                'metadata': test_metadata
            }
        }
        
        with patch.object(config_manager, 'get_database_path', return_value=str(temp_db_path)), \
             patch.object(config_manager, 'get_schema_file_path', return_value=str(temp_schema_file)):
            
            loader = DataLoader(config_manager)
            
            # Act
            result = loader.load_datasets(test_data)
            
            # Assert
            assert len(result) == 1
            
            # Verify schema was applied correctly by checking table structure
            tables_info = loader.get_loaded_tables_info()
            table_name = list(result)[0]
            assert table_name in tables_info
            assert tables_info[table_name]['column_count'] > 0
