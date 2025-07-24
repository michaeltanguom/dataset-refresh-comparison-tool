"""
Unit tests for DataExtractor abstract base class
Tests abstract method enforcement and shared functionality
"""
import pytest
from unittest.mock import Mock, patch
from abc import ABC

from src.extract.base_extract import DataExtractor, DatasetMetadata
from src.config.config_manager import ConfigManager
from src.utils.exceptions import ExtractionError


class TestDataExtractor:
    """Test suite for DataExtractor abstract base class"""
    
    # ========================================
    # Abstract Class Enforcement Tests  
    # ========================================
    
    def test_data_extractor_cannot_be_instantiated_directly(self, valid_config_file):
        """Test that DataExtractor ABC cannot be instantiated directly"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Act & Assert
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            DataExtractor(config_manager)
    
    def test_concrete_implementation_requires_extract_files_method(self, valid_config_file):
        """Test that concrete implementations must implement extract_files method"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        class IncompleteExtractor(DataExtractor):
            pass  # Missing extract_files implementation
        
        # Act & Assert
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteExtractor(config_manager)
    
    def test_concrete_implementation_with_extract_files_succeeds(self, valid_config_file):
        """Test that concrete implementation with extract_files method works"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        class ValidExtractor(DataExtractor):
            def extract_files(self, folder_path: str, period_name: str):
                return {}
        
        # Act
        extractor = ValidExtractor(config_manager)
        
        # Assert
        assert extractor.config == config_manager
        assert hasattr(extractor, 'extract_files')
    
    # ========================================
    # Table Name Generation Tests
    # ========================================
    
    def test_generate_table_name_with_valid_inputs_returns_normalised_name(self, valid_config_file):
        """Test table name generation with valid inputs"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        class TestExtractor(DataExtractor):
            def extract_files(self, folder_path: str, period_name: str):
                return {}
        
        extractor = TestExtractor(config_manager)
        
        # Act
        table_name = extractor.generate_table_name(
            subject="Computer Science",
            period_name="February 2024", 
            sheet_name="Highly Cited Only"
        )
        
        # Assert
        assert table_name == "computer_science_february_2024_highly_cited_only"
    
    def test_generate_table_name_with_special_characters_normalises_correctly(self, valid_config_file):
        """Test table name generation handles special characters"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        class TestExtractor(DataExtractor):
            def extract_files(self, folder_path: str, period_name: str):
                return {}
        
        extractor = TestExtractor(config_manager)
        
        # Act
        table_name = extractor.generate_table_name(
            subject="Bio & Chemistry",
            period_name="Q1-2024", 
            sheet_name="Top 10% Papers"
        )
        
        # Assert
        assert table_name == "bio_chemistry_q1_2024_top_10_papers"
    
    def test_generate_table_name_with_whitespace_trims_correctly(self, valid_config_file):
        """Test table name generation handles whitespace"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        class TestExtractor(DataExtractor):
            def extract_files(self, folder_path: str, period_name: str):
                return {}
        
        extractor = TestExtractor(config_manager)
        
        # Act
        table_name = extractor.generate_table_name(
            subject="  Engineering  ",
            period_name=" July ", 
            sheet_name="  Researchers  "
        )
        
        # Assert
        assert table_name == "engineering_july_researchers"


class TestDatasetMetadata:
    """Test suite for DatasetMetadata dataclass"""
    
    def test_dataset_metadata_creation_with_all_fields(self):
        """Test DatasetMetadata creation with all fields"""
        # Act
        metadata = DatasetMetadata(
            source_file="/path/to/file.xlsx",
            subject="computer_science",
            period="feb",
            sheet_name="Highly Cited only",
            normalised_sheet_name="highly_cited_only",
            table_name="computer_science_feb_highly_cited_only",
            row_count=100,
            columns_mapped={'name': 'Name', 'times_cited': 'Times Cited'},
            processing_timestamp="2024-01-15T10:30:45.123456",
            extraction_duration_seconds=2.5,
            normalisation_duration_seconds=1.2
        )
        
        # Assert
        assert metadata.source_file == "/path/to/file.xlsx"
        assert metadata.subject == "computer_science"
        assert metadata.period == "feb"
        assert metadata.sheet_name == "Highly Cited only"
        assert metadata.normalised_sheet_name == "highly_cited_only"
        assert metadata.table_name == "computer_science_feb_highly_cited_only"
        assert metadata.row_count == 100
        assert metadata.columns_mapped == {'name': 'Name', 'times_cited': 'Times Cited'}
        assert metadata.processing_timestamp == "2024-01-15T10:30:45.123456"
        assert metadata.extraction_duration_seconds == 2.5
        assert metadata.normalisation_duration_seconds == 1.2
    
    def test_dataset_metadata_creation_with_minimal_fields(self):
        """Test DatasetMetadata creation with minimal required fields"""
        # Act
        metadata = DatasetMetadata(
            source_file="/path/to/file.xlsx",
            subject="test",
            period="feb",
            sheet_name="Sheet1",
            normalised_sheet_name="sheet1",
            table_name="test_feb_sheet1",
            row_count=0,
            columns_mapped={},
            processing_timestamp="2024-01-15T10:30:45.123456",
            extraction_duration_seconds=0.0
        )
        
        # Assert
        assert metadata.source_file == "/path/to/file.xlsx"
        assert metadata.normalisation_duration_seconds == 0.0  # Default value
    
    def test_dataset_metadata_is_dataclass(self):
        """Test that DatasetMetadata is properly configured as dataclass"""
        # Assert
        assert hasattr(DatasetMetadata, '__dataclass_fields__')
        assert 'source_file' in DatasetMetadata.__dataclass_fields__
        assert 'row_count' in DatasetMetadata.__dataclass_fields__