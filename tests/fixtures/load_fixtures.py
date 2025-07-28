"""
Load-specific test fixtures and factories
Provides test data structures, mock databases, and utilities for load testing
"""
import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock
from datetime import datetime

from src.extract.base_extract import DatasetMetadata


class LoadDataFactory:
    """Factory for creating test data for load testing"""
    
    @staticmethod
    def create_normalised_dataset_simple() -> Dict[str, Any]:
        """Create simple normalised dataset for load testing"""
        # DataFrame with ALL 13 columns in the exact order of your actual schema
        df = pd.DataFrame({
            'name': ['Dr. John Smith', 'Prof. Jane Doe', 'Dr. Bob Wilson'],
            'percent_docs_cited': [85.5, 72.3, 91.2],
            'web_of_science_documents': [100, 80, 150],
            'rank': [1, 2, 3],
            'times_cited': [1250, 890, 2100],
            'affiliation': ['University A', 'University B', 'University C'],
            'web_of_science_researcherid': ['A-1234-2024', 'B-5678-2024', 'C-9012-2024'],
            'category_normalised_citation_impact': [1.1, 0.9, 1.3],
            'orcid': ['0000-0000-0000-0001', '0000-0000-0000-0002', '0000-0000-0000-0003'],
            'highly_cited_papers': [15, 8, 25],
            'hot_papers': [2, 1, 4],
            'esi_field': ['Engineering', 'Computer Science', 'Mathematics'],
            'indicative_cross_field_score': [1.2, 0.8, 1.8]
        })
        
        metadata = DatasetMetadata(
            source_file="/test/path/test_file.xlsx",
            subject="computer_science",
            period="feb",
            sheet_name="Highly Cited only",
            normalised_sheet_name="highly_cited_only",
            table_name="computer_science_feb_highly_cited_only",
            row_count=len(df),
            columns_mapped={
                'name': 'Name',
                'times_cited': 'Times Cited',
                'highly_cited_papers': 'Highly Cited Papers',
                'esi_field': 'ESI Field'
            },
            processing_timestamp="2024-01-15T10:30:45.123456",
            extraction_duration_seconds=2.5,
            normalisation_duration_seconds=1.2
        )
        
        return {
            'dataframe': df,
            'metadata': metadata
        }
    
    @staticmethod
    def create_normalised_dataset_with_nulls() -> Dict[str, Any]:
        """Create normalised dataset with null values for testing data handling"""
        df = pd.DataFrame({
            'name': ['Dr. Valid Name', 'Dr. Another Valid'],
            'percent_docs_cited': [88.5, 76.2],
            'web_of_science_documents': [120, None],  # One null value
            'rank': [1, 2],
            'times_cited': [1000, None],  # One null value
            'affiliation': ['University X', 'University Y'],
            'web_of_science_researcherid': ['X-1234-2024', 'Y-5678-2024'],
            'category_normalised_citation_impact': [1.0, 0.9],
            'orcid': ['0000-0000-0000-0004', '0000-0000-0000-0005'],
            'highly_cited_papers': [10, 5],
            'hot_papers': [1, 0],
            'esi_field': ['Engineering', 'Mathematics'],
            'indicative_cross_field_score': [1.0, 0.8]
        })
        
        metadata = DatasetMetadata(
            source_file="/test/path/test_with_nulls.xlsx",
            subject="engineering",
            period="july",
            sheet_name="Incites Researchers",
            normalised_sheet_name="incites_researchers",
            table_name="engineering_july_incites_researchers",
            row_count=len(df),
            columns_mapped={'name': 'Name', 'times_cited': 'Times Cited'},
            processing_timestamp="2024-01-15T10:35:45.123456",
            extraction_duration_seconds=1.8,
            normalisation_duration_seconds=0.9
        )
        
        return {
            'dataframe': df,
            'metadata': metadata
        }
    
    @staticmethod
    def create_empty_normalised_dataset() -> Dict[str, Any]:
        """Create empty normalised dataset for error testing"""
        # Empty DataFrame with all 13 column names in correct order
        df = pd.DataFrame(columns=[
            'name', 'percent_docs_cited', 'web_of_science_documents', 'rank',
            'times_cited', 'affiliation', 'web_of_science_researcherid',
            'category_normalised_citation_impact', 'orcid', 'highly_cited_papers',
            'hot_papers', 'esi_field', 'indicative_cross_field_score'
        ])
        
        metadata = DatasetMetadata(
            source_file="/test/path/empty_file.xlsx",
            subject="empty",
            period="test",
            sheet_name="Empty Sheet",
            normalised_sheet_name="empty_sheet",
            table_name="empty_test_empty_sheet",
            row_count=0,
            columns_mapped={},
            processing_timestamp="2024-01-15T10:40:45.123456",
            extraction_duration_seconds=0.1,
            normalisation_duration_seconds=0.1
        )
        
        return {
            'dataframe': df,
            'metadata': metadata
        }
    
    @staticmethod
    def create_multiple_normalised_datasets() -> Dict[str, Dict[str, Any]]:
        """Create multiple datasets for batch loading tests"""
        datasets = {}
        
        # Dataset 1: Computer Science with all 13 columns in correct order
        cs_df = pd.DataFrame({
            'name': ['Dr. CS One', 'Dr. CS Two'],
            'percent_docs_cited': [90.0, 85.5],
            'web_of_science_documents': [150, 120],
            'rank': [1, 2],
            'times_cited': [1500, 1200],
            'affiliation': ['University CS1', 'University CS2'],
            'web_of_science_researcherid': ['CS1-2024-001', 'CS2-2024-002'],
            'category_normalised_citation_impact': [1.2, 1.0],
            'orcid': ['0000-0000-0000-1001', '0000-0000-0000-1002'],
            'highly_cited_papers': [20, 15],
            'hot_papers': [3, 2],
            'esi_field': ['Computer Science', 'Computer Science'],
            'indicative_cross_field_score': [1.5, 1.2]
        })
        
        cs_metadata = DatasetMetadata(
            source_file="/test/path/cs_file.xlsx",
            subject="computer_science",
            period="feb",
            sheet_name="Highly Cited only",
            normalised_sheet_name="highly_cited_only",
            table_name="computer_science_feb_highly_cited_only",
            row_count=len(cs_df),
            columns_mapped={'name': 'Name', 'times_cited': 'Times Cited'},
            processing_timestamp="2024-01-15T10:30:45.123456",
            extraction_duration_seconds=2.0,
            normalisation_duration_seconds=1.0
        )
        
        datasets['computer_science_feb_highly_cited_only'] = {
            'dataframe': cs_df,
            'metadata': cs_metadata
        }
        
        # Dataset 2: Engineering with all 13 columns in correct order
        eng_df = pd.DataFrame({
            'name': ['Dr. Eng One', 'Dr. Eng Two', 'Dr. Eng Three'],
            'percent_docs_cited': [92.0, 88.5, 84.0],
            'web_of_science_documents': [200, 180, 160],
            'rank': [1, 2, 3],
            'times_cited': [2000, 1800, 1600],
            'affiliation': ['University Eng1', 'University Eng2', 'University Eng3'],
            'web_of_science_researcherid': ['ENG1-2024-001', 'ENG2-2024-002', 'ENG3-2024-003'],
            'category_normalised_citation_impact': [1.5, 1.3, 1.1],
            'orcid': ['0000-0000-0000-2001', '0000-0000-0000-2002', '0000-0000-0000-2003'],
            'highly_cited_papers': [25, 22, 18],
            'hot_papers': [4, 3, 2],
            'esi_field': ['Engineering', 'Engineering', 'Engineering'],
            'indicative_cross_field_score': [2.0, 1.8, 1.6]
        })
        
        eng_metadata = DatasetMetadata(
            source_file="/test/path/eng_file.xlsx",
            subject="engineering",
            period="feb",
            sheet_name="Incites Researchers",
            normalised_sheet_name="incites_researchers",
            table_name="engineering_feb_incites_researchers",
            row_count=len(eng_df),
            columns_mapped={'name': 'Name', 'times_cited': 'Times Cited'},
            processing_timestamp="2024-01-15T10:32:45.123456",
            extraction_duration_seconds=2.2,
            normalisation_duration_seconds=1.1
        )
        
        datasets['engineering_feb_incites_researchers'] = {
            'dataframe': eng_df,
            'metadata': eng_metadata
        }
        
        return datasets
    
    @staticmethod
    def create_large_normalised_dataset() -> Dict[str, Any]:
        """Create large dataset for performance testing"""
        # Create 1000 rows of test data with all 13 columns
        names = [f'Dr. Researcher {i:04d}' for i in range(1000)]
        percent_docs_cited = [75.0 + (i % 25) for i in range(1000)]  # 75-99%
        web_of_science_documents = [50 + i for i in range(1000)]  # 50-1049 documents
        ranks = [1 + (i % 100) for i in range(1000)]  # Ranks 1-100
        times_cited = [100 + i for i in range(1000)]  # 100-1099 citations
        affiliations = [f'University {i % 100:03d}' for i in range(1000)]  # 100 different universities
        researcherids = [f'RID-{i:04d}-2024' for i in range(1000)]
        citation_impacts = [1.0 + (i % 100) / 100 for i in range(1000)]  # 1.0-1.99
        orcids = [f'0000-0000-{(i//100):04d}-{(i%100):04d}' for i in range(1000)]
        highly_cited_papers = [5 + (i % 20) for i in range(1000)]  # 5-24 papers
        hot_papers = [i % 5 for i in range(1000)]  # 0-4 papers
        
        # Distribute across 4 ESI fields evenly
        esi_fields = (['Engineering'] * 250 + 
                    ['Computer Science'] * 250 + 
                    ['Mathematics'] * 250 + 
                    ['Physics'] * 250)
        
        cross_field_scores = [1.0 + (i % 100) / 100 for i in range(1000)]  # 1.0-1.99
        
        df = pd.DataFrame({
            'name': names,
            'percent_docs_cited': percent_docs_cited,
            'web_of_science_documents': web_of_science_documents,
            'rank': ranks,
            'times_cited': times_cited,
            'affiliation': affiliations,
            'web_of_science_researcherid': researcherids,
            'category_normalised_citation_impact': citation_impacts,
            'orcid': orcids,
            'highly_cited_papers': highly_cited_papers,
            'hot_papers': hot_papers,
            'esi_field': esi_fields,
            'indicative_cross_field_score': cross_field_scores
        })
        
        metadata = DatasetMetadata(
            source_file="/test/path/large_file.xlsx",
            subject="large_dataset",
            period="performance_test",
            sheet_name="Large Dataset",
            normalised_sheet_name="large_dataset",
            table_name="large_dataset_performance_test_large_dataset",
            row_count=len(df),
            columns_mapped={'name': 'Name', 'times_cited': 'Times Cited'},
            processing_timestamp="2024-01-15T10:45:45.123456",
            extraction_duration_seconds=5.0,
            normalisation_duration_seconds=2.5
        )
        
        return {
            'dataframe': df,
            'metadata': metadata
        }

class MockDatabaseFactory:
    """Factory for creating mock database objects and scenarios"""
    
    @staticmethod
    def create_mock_database_manager_success():
        """Create mock database manager that succeeds for all operations"""
        mock_db = Mock()
        mock_db.validate_connection.return_value = True
        mock_db.list_tables.return_value = []
        mock_db.clear_all_tables.return_value = None
        mock_db.execute_non_query.return_value = None
        mock_db.insert_dataframe.return_value = None
        mock_db.get_table_count.return_value = 3  # Default successful count
        mock_db.get_sample_data.return_value = pd.DataFrame({'name': ['Sample'], 'times_cited': [100]})
        mock_db.get_table_schema.return_value = {'name': 'VARCHAR', 'times_cited': 'INTEGER'}
        mock_db.get_database_size.return_value = 1.5  # 1.5 MB
        mock_db.db_path = Path("/test/path/test.db")
        return mock_db
    
    @staticmethod
    def create_mock_database_manager_connection_failure():
        """Create mock database manager that fails connection"""
        mock_db = Mock()
        mock_db.validate_connection.return_value = False
        return mock_db
    
    @staticmethod
    def create_mock_database_manager_partial_failure():
        """Create mock database manager that succeeds for some operations, fails for others"""
        mock_db = Mock()
        mock_db.validate_connection.return_value = True
        mock_db.list_tables.return_value = []
        mock_db.clear_all_tables.return_value = None
        
        # First call succeeds, second call fails
        mock_db.execute_non_query.side_effect = [None, Exception("Schema creation failed")]
        mock_db.insert_dataframe.side_effect = Exception("Insert failed")
        
        return mock_db
    
    @staticmethod
    def create_mock_database_manager_with_existing_tables():
        """Create mock database manager with existing tables"""
        mock_db = Mock()
        mock_db.validate_connection.return_value = True
        mock_db.list_tables.return_value = ['existing_table_1', 'existing_table_2']
        mock_db.clear_all_tables.return_value = None
        mock_db.execute_non_query.return_value = None
        mock_db.insert_dataframe.return_value = None
        mock_db.get_table_count.return_value = 5
        mock_db.get_sample_data.return_value = pd.DataFrame({'name': ['Existing'], 'times_cited': [200]})
        mock_db.get_table_schema.return_value = {'name': 'VARCHAR', 'times_cited': 'INTEGER'}
        mock_db.get_database_size.return_value = 2.0
        mock_db.db_path = Path("/test/path/existing.db")
        return mock_db
    
    @staticmethod
    def create_mock_database_manager_row_count_mismatch():
        """Create mock database manager that returns wrong row count"""
        mock_db = Mock()
        mock_db.validate_connection.return_value = True
        mock_db.list_tables.return_value = []
        mock_db.clear_all_tables.return_value = None
        mock_db.execute_non_query.return_value = None
        mock_db.insert_dataframe.return_value = None
        mock_db.get_table_count.return_value = 999  # Wrong count
        mock_db.get_sample_data.return_value = pd.DataFrame({'name': ['Sample'], 'times_cited': [100]})
        mock_db.get_table_schema.return_value = {'name': 'VARCHAR', 'times_cited': 'INTEGER'}
        mock_db.get_database_size.return_value = 1.0
        mock_db.db_path = Path("/test/path/mismatch.db")
        return mock_db


class SchemaFileFactory:
    """Factory for creating test schema files"""
    
    @staticmethod
    def create_valid_schema_content() -> str:
        """Create valid SQL schema content that matches your actual schema"""
        return '''
    CREATE TABLE IF NOT EXISTS "{table_name}" (
        name VARCHAR NOT NULL,
        percent_docs_cited DOUBLE,
        web_of_science_documents INTEGER,
        rank INTEGER,
        times_cited INTEGER,
        affiliation VARCHAR,
        web_of_science_researcherid VARCHAR,
        category_normalised_citation_impact DOUBLE,
        orcid VARCHAR,
        highly_cited_papers INTEGER,
        hot_papers INTEGER,
        esi_field VARCHAR NOT NULL,
        indicative_cross_field_score DOUBLE
    );
    '''
    
    @staticmethod
    def create_invalid_schema_content() -> str:
        """Create invalid SQL schema content"""
        return '''
CREATE TABLE IF NOT EXISTS "{table_name}" (
    name VARCHAR NOT NULL,
    invalid_syntax_here <<<>>> ERROR
);
'''
    
    @staticmethod
    def create_minimal_schema_content() -> str:
        """Create minimal valid SQL schema"""
        return '''
CREATE TABLE IF NOT EXISTS "{table_name}" (
    name VARCHAR NOT NULL,
    times_cited INTEGER
);
'''
    
    @staticmethod
    def create_schema_without_placeholder() -> str:
        """Create schema without table name placeholder"""
        return '''
CREATE TABLE IF NOT EXISTS "hardcoded_table" (
    name VARCHAR NOT NULL,
    times_cited INTEGER
);
'''


class MockConfigFactory:
    """Factory for creating mock configuration objects for load testing"""
    
    @staticmethod
    def create_mock_config_manager_success():
        """Create mock config manager that provides valid configuration"""
        mock_config = Mock()
        mock_config.get_database_path.return_value = ":memory:"
        mock_config.get_schema_for_sheet.return_value = "hcr_default"
        mock_config.get_schema_file_path.return_value = "/test/schema/hcr_default.sql"
        mock_config.get_critical_columns.return_value = ['name', 'times_cited', 'highly_cited_papers', 'esi_field']
        mock_config.get_column_mapping.return_value = {
            'name': 'Name',
            'times_cited': 'Times Cited',
            'highly_cited_papers': 'Highly Cited Papers',
            'esi_field': 'ESI Field'
        }
        return mock_config
    
    @staticmethod
    def create_mock_config_manager_missing_schema():
        """Create mock config manager with missing schema file"""
        mock_config = Mock()
        mock_config.get_database_path.return_value = ":memory:"
        mock_config.get_schema_for_sheet.return_value = "missing_schema"
        mock_config.get_schema_file_path.return_value = "/test/schema/missing.sql"
        mock_config.get_critical_columns.return_value = ['name', 'times_cited']
        mock_config.get_column_mapping.return_value = {'name': 'Name', 'times_cited': 'Times Cited'}
        return mock_config


# ========================================
# Pytest Fixtures
# ========================================

@pytest.fixture
def load_data_factory():
    """Provide load data factory for tests"""
    return LoadDataFactory

@pytest.fixture
def mock_database_factory():
    """Provide mock database factory for tests"""
    return MockDatabaseFactory

@pytest.fixture
def schema_file_factory():
    """Provide schema file factory for tests"""
    return SchemaFileFactory

@pytest.fixture
def mock_config_factory():
    """Provide mock config factory for tests"""
    return MockConfigFactory

@pytest.fixture
def simple_normalised_dataset(load_data_factory):
    """Provide simple normalised dataset for testing"""
    return load_data_factory.create_normalised_dataset_simple()

@pytest.fixture
def normalised_dataset_with_nulls(load_data_factory):
    """Provide normalised dataset with null values"""
    return load_data_factory.create_normalised_dataset_with_nulls()

@pytest.fixture
def empty_normalised_dataset(load_data_factory):
    """Provide empty normalised dataset"""
    return load_data_factory.create_empty_normalised_dataset()

@pytest.fixture
def multiple_normalised_datasets(load_data_factory):
    """Provide multiple normalised datasets"""
    return load_data_factory.create_multiple_normalised_datasets()

@pytest.fixture
def large_normalised_dataset(load_data_factory):
    """Provide large dataset for performance testing"""
    return load_data_factory.create_large_normalised_dataset()

@pytest.fixture
def mock_database_manager_success(mock_database_factory):
    """Provide mock database manager that succeeds"""
    return mock_database_factory.create_mock_database_manager_success()

@pytest.fixture
def mock_database_manager_connection_failure(mock_database_factory):
    """Provide mock database manager with connection failure"""
    return mock_database_factory.create_mock_database_manager_connection_failure()

@pytest.fixture
def mock_database_manager_partial_failure(mock_database_factory):
    """Provide mock database manager with partial failures"""
    return mock_database_factory.create_mock_database_manager_partial_failure()

@pytest.fixture
def mock_database_manager_with_existing_tables(mock_database_factory):
    """Provide mock database manager with existing tables"""
    return mock_database_factory.create_mock_database_manager_with_existing_tables()

@pytest.fixture
def mock_database_manager_row_count_mismatch(mock_database_factory):
    """Provide mock database manager with row count mismatch"""
    return mock_database_factory.create_mock_database_manager_row_count_mismatch()

@pytest.fixture
def mock_config_manager_success(mock_config_factory):
    """Provide mock config manager that succeeds"""
    return mock_config_factory.create_mock_config_manager_success()

@pytest.fixture
def mock_config_manager_missing_schema(mock_config_factory):
    """Provide mock config manager with missing schema"""
    return mock_config_factory.create_mock_config_manager_missing_schema()

@pytest.fixture
def temp_schema_file(temp_schema_dir, schema_file_factory):
    """Create temporary schema file for testing"""
    schema_content = schema_file_factory.create_valid_schema_content()
    schema_file = temp_schema_dir / "test_schema.sql"
    with open(schema_file, 'w') as f:
        f.write(schema_content)
    return schema_file

@pytest.fixture
def invalid_schema_file(temp_schema_dir, schema_file_factory):
    """Create invalid schema file for testing"""
    schema_content = schema_file_factory.create_invalid_schema_content()
    schema_file = temp_schema_dir / "invalid_schema.sql"
    with open(schema_file, 'w') as f:
        f.write(schema_content)
    return schema_file

@pytest.fixture
def minimal_schema_file(temp_schema_dir, schema_file_factory):
    """Create minimal schema file for testing"""
    schema_content = schema_file_factory.create_minimal_schema_content()
    schema_file = temp_schema_dir / "minimal_schema.sql"
    with open(schema_file, 'w') as f:
        f.write(schema_content)
    return schema_file

@pytest.fixture
def temp_load_workspace(test_temp_dir):
    """Create temporary workspace for load testing"""
    load_workspace = test_temp_dir / "load_tests"
    load_workspace.mkdir(exist_ok=True)
    return load_workspace

@pytest.fixture
def normalised_data_single_table(simple_normalised_dataset):
    """Provide single table normalised data structure expected by DataLoader"""
    return {
        'test_table': simple_normalised_dataset
    }

@pytest.fixture
def normalised_data_multiple_tables(multiple_normalised_datasets):
    """Provide multiple table normalised data structure expected by DataLoader"""
    return multiple_normalised_datasets

@pytest.fixture
def normalised_data_empty_table(empty_normalised_dataset):
    """Provide empty table normalised data structure for error testing"""
    return {
        'empty_table': empty_normalised_dataset
    }
