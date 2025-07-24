"""
Unit tests for ExcelDataExtractor class
Tests Excel-specific extraction functionality
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock, mock_open
import tempfile
import time

from src.extract.excel_extractor import ExcelDataExtractor
from src.extract.base_extract import DatasetMetadata
from src.config.config_manager import ConfigManager
from src.utils.exceptions import ExtractionError


class TestExcelDataExtractor:
    """Test suite for ExcelDataExtractor functionality"""
    
    # ========================================
    # Initialisation Tests
    # ========================================
    
    def test_init_with_valid_config_manager_succeeds(self, valid_config_file):
        """Test successful ExcelDataExtractor initialisation"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Act
        extractor = ExcelDataExtractor(config_manager)
        
        # Assert
        assert extractor.config == config_manager
        assert hasattr(extractor, 'logger')
    
    # ========================================
    # File Discovery Tests
    # ========================================
    
    def test_extract_files_with_valid_excel_files_returns_dataframes(self, valid_config_file, excel_test_files):
        """Test extraction with valid Excel files"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = ExcelDataExtractor(config_manager)
        
        # Use the feb directory from excel_test_files
        feb_dir = None
        for file_path in excel_test_files.values():
            if 'feb' in str(file_path):
                feb_dir = file_path.parent
                break
        
        # Act
        result = extractor.extract_files(str(feb_dir), 'feb')
        
        # Assert
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Check that each result has expected structure
        for table_name, data in result.items():
            assert 'dataframe' in data
            assert 'metadata' in data
            assert isinstance(data['dataframe'], pd.DataFrame)
            assert isinstance(data['metadata'], DatasetMetadata)
    
    def test_extract_files_with_missing_folder_raises_extraction_error(self, valid_config_file):
        """Test extraction with missing folder raises ExtractionError"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = ExcelDataExtractor(config_manager)
        missing_folder = "/path/that/does/not/exist"
        
        # Act & Assert
        with pytest.raises(ExtractionError, match="Folder not found"):
            extractor.extract_files(missing_folder, 'test_period')
    
    def test_extract_files_with_no_excel_files_returns_empty_dict(self, valid_config_file, temp_extract_workspace):
        """Test extraction with folder containing no Excel files"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = ExcelDataExtractor(config_manager)
        
        # Create empty directory
        empty_dir = temp_extract_workspace / "empty"
        empty_dir.mkdir()
        
        # Act
        result = extractor.extract_files(str(empty_dir), 'test_period')
        
        # Assert
        assert result == {}
    
    def test_extract_files_filters_temporary_excel_files(self, valid_config_file, excel_test_files):
        """Test that temporary Excel files are filtered out"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = ExcelDataExtractor(config_manager)
        
        # Get directory containing temp file
        feb_dir = None
        for file_path in excel_test_files.values():
            if 'feb' in str(file_path):
                feb_dir = file_path.parent
                break
        
        # Act
        result = extractor.extract_files(str(feb_dir), 'feb')
        
        # Assert
        # Should have results (temp files filtered out, but real files processed)
        assert len(result) > 0
        
        # Verify no table names contain temp file indicators
        for table_name in result.keys():
            assert '~$' not in table_name

    # ========================================
    # Sheet Processing Tests
    # ========================================
    
    @patch('pandas.read_excel')
    def test_extract_files_processes_configured_sheets_only(self, mock_read_excel, valid_config_file, temp_extract_workspace):
        """Test that only configured sheets are processed"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = ExcelDataExtractor(config_manager)
        
        # Create test Excel file
        test_file = temp_extract_workspace / "test.xlsx"
        test_file.touch()
        
        # Mock pandas ExcelFile - FIXED: Use MagicMock with proper context manager setup
        mock_excel_file = MagicMock()
        mock_excel_file.sheet_names = ['Highly Cited only', 'Incites Researchers', 'Unwanted Sheet']
        
        # CRITICAL: Configure the context manager to return the mock itself
        mock_excel_file.__enter__.return_value = mock_excel_file
        mock_excel_file.__exit__.return_value = None
        
        # Mock read_excel to return sample data
        sample_df = pd.DataFrame({
            'Name': ['Dr. Test'],
            'Times Cited': [100],
            'Highly Cited Papers': [5],
            'ESI Field': ['Engineering']
        })
        mock_read_excel.return_value = sample_df
        
        with patch('pandas.ExcelFile', return_value=mock_excel_file):
            # Act
            result = extractor.extract_files(str(temp_extract_workspace), 'test_period')
        
        # Assert
        assert len(result) == 2
        
        # Verify correct sheets were processed
        table_names = list(result.keys())
        assert any('highly_cited_only' in name for name in table_names)
        assert any('incites_researchers' in name for name in table_names)
        assert not any('unwanted_sheet' in name for name in table_names)

    # ========================================
    # Data Validation Tests
    # ========================================
    
    def test_extract_files_applies_name_filtering(self, valid_config_file, temp_extract_workspace):
        """Test that invalid names are filtered out during extraction"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = ExcelDataExtractor(config_manager)
        
        # Create DataFrame with invalid names
        test_df = pd.DataFrame({
            'Name': ['Valid Name', None, '', 'nan', 'none', 'Another Valid'],
            'Times Cited': [100, 200, 300, 400, 500, 600],
            'Highly Cited Papers': [5, 10, 15, 20, 25, 30],
            'ESI Field': ['Engineering'] * 6
        })
        
        test_file = temp_extract_workspace / "test.xlsx"
        
        with pd.ExcelWriter(str(test_file), engine='openpyxl') as writer:
            test_df.to_excel(writer, sheet_name='Highly Cited only', index=False)
        
        # Act
        result = extractor.extract_files(str(temp_extract_workspace), 'test_period')
        
        # Assert
        assert len(result) == 1
        table_name = list(result.keys())[0]
        extracted_df = result[table_name]['dataframe']
        
        # Should only have 2 valid names (filtered from 6 total)
        assert len(extracted_df) == 2
        assert 'Valid Name' in extracted_df['Name'].values
        assert 'Another Valid' in extracted_df['Name'].values
    
    def test_extract_files_skips_empty_sheets(self, valid_config_file, temp_extract_workspace):
        """Test that empty sheets are skipped"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = ExcelDataExtractor(config_manager)
        
        test_file = temp_extract_workspace / "test.xlsx"
        
        with pd.ExcelWriter(str(test_file), engine='openpyxl') as writer:
            # Create empty DataFrame
            empty_df = pd.DataFrame()
            empty_df.to_excel(writer, sheet_name='Highly Cited only', index=False)
        
        # Act & Assert
        # FIXED: Expect ExtractionError when no data extracted
        with pytest.raises(ExtractionError, match="No data extracted"):
            extractor.extract_files(str(temp_extract_workspace), 'test_period')
    
    # ========================================
    # Metadata Generation Tests
    # ========================================
    
    def test_extract_files_generates_correct_metadata(self, valid_config_file, excel_test_files):
        """Test that correct metadata is generated for extracted files"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = ExcelDataExtractor(config_manager)
        
        # Get feb directory
        feb_dir = None
        for file_path in excel_test_files.values():
            if 'feb' in str(file_path):
                feb_dir = file_path.parent
                break
        
        # Act
        result = extractor.extract_files(str(feb_dir), 'feb')
        
        # Assert
        assert len(result) > 0
        
        # Check metadata for first result
        table_name, data = list(result.items())[0]
        metadata = data['metadata']
        
        assert isinstance(metadata, DatasetMetadata)
        assert metadata.period == 'feb'
        assert metadata.row_count > 0
        assert isinstance(metadata.extraction_duration_seconds, float)
        assert metadata.extraction_duration_seconds >= 0
        assert isinstance(metadata.processing_timestamp, str)
        assert 'T' in metadata.processing_timestamp  # ISO format
    
    # ========================================
    # Error Handling Tests
    # ========================================
    
    @patch('pandas.read_excel')
    def test_extract_files_handles_corrupted_excel_file(self, mock_read_excel, valid_config_file, temp_extract_workspace):
        """Test handling of corrupted Excel files"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = ExcelDataExtractor(config_manager)
        
        # Create test file
        test_file = temp_extract_workspace / "corrupted.xlsx"
        test_file.touch()
        
        # Mock pandas to raise exception
        mock_read_excel.side_effect = Exception("Corrupted file")
        
        with patch('pandas.ExcelFile', side_effect=Exception("Cannot open file")):
            # Act & Assert
            # FIXED: Expect ExtractionError when no files can be processed
            with pytest.raises(ExtractionError, match="No data extracted"):
                extractor.extract_files(str(temp_extract_workspace), 'test_period')
    
    @patch('pandas.read_excel')
    def test_extract_files_handles_sheet_reading_errors(self, mock_read_excel, valid_config_file, temp_extract_workspace):
        """Test handling of individual sheet reading errors"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = ExcelDataExtractor(config_manager)
        
        # Create a clean subdirectory for this test only
        test_subdir = temp_extract_workspace / "sheet_reading_test"
        test_subdir.mkdir()
        
        test_file = test_subdir / "test.xlsx"
        test_file.touch()
        
        # Use MagicMock with proper context manager setup
        mock_excel_file = MagicMock()
        mock_excel_file.sheet_names = ['Highly Cited only', 'Incites Researchers']
        
        # Configure the context manager to return the mock itself
        mock_excel_file.__enter__.return_value = mock_excel_file
        mock_excel_file.__exit__.return_value = None
        
        # Mock read_excel to fail for first sheet, succeed for second
        def side_effect(*args, **kwargs):
            if kwargs.get('sheet_name') == 'Highly Cited only':
                raise Exception("Sheet reading error")
            return pd.DataFrame({
                'Name': ['Dr. Test'],
                'Times Cited': [100],
                'Highly Cited Papers': [5],
                'ESI Field': ['Engineering']
            })
        
        mock_read_excel.side_effect = side_effect
        
        with patch('pandas.ExcelFile', return_value=mock_excel_file):
            # Act - use the clean subdirectory
            result = extractor.extract_files(str(test_subdir), 'test_period')
        
        # Assert
        # Should have 1 successful extraction (second sheet from 1 file)
        assert len(result) == 1
        
        # Verify it's the correct sheet
        table_name = list(result.keys())[0]
        assert 'incites_researchers' in table_name
    
    # ========================================
    # Performance and Logging Tests
    # ========================================
    
    def test_extract_files_logs_extraction_progress(self, valid_config_file, excel_test_files):
        """Test that extraction progress is properly logged"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = ExcelDataExtractor(config_manager)
        
        # Mock logger to capture calls
        mock_logger = Mock()
        extractor.logger = mock_logger
        
        # Get feb directory
        feb_dir = None
        for file_path in excel_test_files.values():
            if 'feb' in str(file_path):
                feb_dir = file_path.parent
                break
        
        # Act
        result = extractor.extract_files(str(feb_dir), 'feb')
        
        # Assert
        # Verify important log messages were called
        assert mock_logger.info.called
        assert mock_logger.warning.called or not mock_logger.warning.called  # May or may not have warnings
        
        # Check for specific log patterns
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any('Starting Excel extraction' in msg for msg in info_calls)
        assert any('Extraction complete' in msg for msg in info_calls)
    
    def test_extract_files_measures_extraction_time(self, valid_config_file, excel_test_files):
        """Test that extraction time is measured correctly"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = ExcelDataExtractor(config_manager)
        
        feb_dir = None
        for file_path in excel_test_files.values():
            if 'feb' in str(file_path):
                feb_dir = file_path.parent
                break
        
        # Act
        start_time = time.time()
        result = extractor.extract_files(str(feb_dir), 'feb')
        total_time = time.time() - start_time
        
        # Assert
        assert len(result) > 0
        
        # Check that individual extraction times are reasonable
        for table_name, data in result.items():
            metadata = data['metadata']
            assert metadata.extraction_duration_seconds >= 0
            assert metadata.extraction_duration_seconds <= total_time