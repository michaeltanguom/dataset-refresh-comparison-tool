"""
Unit tests for JSONDataExtractor class
Tests JSON comparison report extraction functionality
"""
import pytest
import json
from pathlib import Path
from unittest.mock import patch, Mock, mock_open

from src.extract.json_extractor import JSONDataExtractor
from src.extract.base_extract import DatasetMetadata
from src.config.config_manager import ConfigManager
from src.utils.exceptions import ExtractionError


class TestJSONDataExtractor:
    """Test suite for JSONDataExtractor functionality"""
    
    # ========================================
    # Initialisation Tests
    # ========================================
    
    def test_init_with_valid_config_manager_succeeds(self, valid_config_file):
        """Test successful JSONDataExtractor initialisation"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Act
        extractor = JSONDataExtractor(config_manager)
        
        # Assert
        assert extractor.config == config_manager
        assert hasattr(extractor, 'logger')
    
    # ========================================
    # File Discovery Tests
    # ========================================
    
    def test_extract_files_with_valid_json_reports_returns_data(self, valid_config_file, json_test_files):
        """Test extraction with valid JSON reports"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = JSONDataExtractor(config_manager)
        
        # Get the comparison_reports directory
        reports_dir = None
        for file_path in json_test_files.values():
            if 'comparison_reports' in str(file_path):
                # Get the comparison_reports directory (parent of subdirectories)
                parts = file_path.parts
                reports_index = parts.index('comparison_reports')
                reports_dir = Path(*parts[:reports_index + 1])
                break
        
        # Act
        result = extractor.extract_files(str(reports_dir), 'test_period')
        
        # Assert
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Check structure of results
        for table_name, data in result.items():
            assert 'json_data' in data
            assert 'metadata' in data
            assert isinstance(data['json_data'], dict)
            assert isinstance(data['metadata'], DatasetMetadata)
    
    def test_extract_files_with_missing_folder_raises_extraction_error(self, valid_config_file):
        """Test extraction with missing folder raises ExtractionError"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = JSONDataExtractor(config_manager)
        missing_folder = "/path/that/does/not/exist"
        
        # Act & Assert
        with pytest.raises(ExtractionError, match="Reports folder not found"):
            extractor.extract_files(missing_folder, 'test_period')
    
    def test_extract_files_with_no_json_files_returns_empty_dict(self, valid_config_file, tmp_path):
        """Test extraction with folder containing no JSON files"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = JSONDataExtractor(config_manager)
        
        # Create directory structure but no JSON files
        reports_dir = tmp_path / "comparison_reports"
        subdir = reports_dir / "empty_type"
        subdir.mkdir(parents=True)
        
        # Act
        result = extractor.extract_files(str(reports_dir), 'test_period')
        
        # Assert
        assert result == {}
    
    # ========================================
    # JSON Validation Tests
    # ========================================
    
    def test_load_and_validate_json_with_valid_file_succeeds(self, valid_config_file, sample_comparison_report, temp_extract_workspace):
        """Test JSON loading and validation with valid file"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = JSONDataExtractor(config_manager)
        
        # Create valid JSON file
        json_file = temp_extract_workspace / "test_report.json"
        with open(json_file, 'w') as f:
            json.dump(sample_comparison_report, f)
        
        # Act
        result = extractor._load_and_validate_json(json_file)
        
        # Assert
        assert isinstance(result, dict)
        assert result['comparison_id'] == sample_comparison_report['comparison_id']
        assert 'summary_statistics' in result
        assert 'researcher_changes' in result
    
    def test_load_and_validate_json_with_missing_required_fields_raises_error(self, valid_config_file, temp_extract_workspace):
        """Test JSON validation with missing required fields"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = JSONDataExtractor(config_manager)
        
        # Create JSON with missing required fields
        invalid_json = {"incomplete": "data"}
        json_file = temp_extract_workspace / "invalid_report.json"
        with open(json_file, 'w') as f:
            json.dump(invalid_json, f)
        
        # Act & Assert
        with pytest.raises(ExtractionError, match="missing required fields"):
            extractor._load_and_validate_json(json_file)
    
    def test_load_and_validate_json_with_invalid_json_raises_error(self, valid_config_file, temp_extract_workspace):
        """Test JSON loading with invalid JSON format"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = JSONDataExtractor(config_manager)
        
        # Create invalid JSON file
        json_file = temp_extract_workspace / "invalid.json"
        with open(json_file, 'w') as f:
            f.write('{"invalid": json content}')  # Invalid JSON
        
        # Act & Assert
        with pytest.raises(ExtractionError, match="Invalid JSON format"):
            extractor._load_and_validate_json(json_file)
    
    # ========================================
    # Metadata Parsing Tests
    # ========================================
    
    def test_parse_report_metadata_with_standard_filename_succeeds(self, valid_config_file):
        """Test metadata parsing with standard filename format"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = JSONDataExtractor(config_manager)
        
        # Create test file path
        test_path = Path("comparison_reports/highly_cited_only/computer_science_highly_cited_only_comparison_report.json")
        
        # Act
        subject, sheet_type = extractor._parse_report_metadata(test_path)
        
        # Assert
        assert subject == "computer_science"
        assert sheet_type == "highly_cited_only"
    
    def test_parse_report_metadata_with_complex_filename_succeeds(self, valid_config_file):
        """Test metadata parsing with complex filename"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = JSONDataExtractor(config_manager)
        
        # Create test file path
        test_path = Path("comparison_reports/incites_researchers/biology_biochemistry_incites_researchers_comparison_report.json")
        
        # Act
        subject, sheet_type = extractor._parse_report_metadata(test_path)
        
        # Assert
        assert subject == "biology_biochemistry"
        assert sheet_type == "incites_researchers"
    
    def test_parse_report_metadata_with_malformed_filename_uses_fallback(self, valid_config_file):
        """Test metadata parsing with malformed filename uses fallback"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = JSONDataExtractor(config_manager)
        
        # Create test file path with malformed name
        test_path = Path("comparison_reports/some_type/malformed_filename.json")
        
        # Act
        subject, sheet_type = extractor._parse_report_metadata(test_path)
        
        # Assert
        # Should fall back to filename without extension
        assert subject == "malformed_filename"
        assert sheet_type == "some_type"
    
    # ========================================
    # Researcher Count Calculation Tests
    # ========================================
    
    def test_calculate_researcher_count_with_complete_data_returns_correct_count(self, valid_config_file, sample_comparison_report):
        """Test researcher count calculation with complete data"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = JSONDataExtractor(config_manager)
        
        # Act
        count = extractor._calculate_researcher_count(sample_comparison_report)
        
        # Assert
        # Should sum all researcher categories
        expected_count = (
            len(sample_comparison_report['researcher_changes']) +
            len(sample_comparison_report['researchers_unchanged']) +
            len(sample_comparison_report['researchers_only_in_dataset_1']) +
            len(sample_comparison_report['researchers_only_in_dataset_2'])
        )
        assert count == expected_count
    
    def test_calculate_researcher_count_with_missing_sections_returns_partial_count(self, valid_config_file):
        """Test researcher count calculation with missing sections"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = JSONDataExtractor(config_manager)
        
        # Create partial report data
        partial_report = {
            "researcher_changes": [{"name": "Dr. A"}, {"name": "Dr. B"}],
            "researchers_unchanged": [{"name": "Dr. C"}]
            # Missing other sections
        }
        
        # Act
        count = extractor._calculate_researcher_count(partial_report)
        
        # Assert
        assert count == 3  # Only counts existing sections
    
    def test_calculate_researcher_count_with_invalid_data_returns_zero(self, valid_config_file):
        """Test researcher count calculation with invalid data"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = JSONDataExtractor(config_manager)
        
        # Create invalid report data
        invalid_report = {"invalid": "structure"}
        
        # Act
        count = extractor._calculate_researcher_count(invalid_report)
        
        # Assert
        assert count == 0
    
    # ========================================
    # Metadata Generation Tests
    # ========================================
    
    def test_extract_files_generates_correct_metadata(self, valid_config_file, json_test_files):
        """Test that correct metadata is generated for extracted JSON files"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = JSONDataExtractor(config_manager)
        
        # Get reports directory
        reports_dir = None
        for file_path in json_test_files.values():
            if 'comparison_reports' in str(file_path):
                parts = file_path.parts
                reports_index = parts.index('comparison_reports')
                reports_dir = Path(*parts[:reports_index + 1])
                break
        
        # Act
        result = extractor.extract_files(str(reports_dir), 'test_period')
        
        # Assert
        assert len(result) > 0
        
        # Check metadata for first result
        table_name, data = list(result.items())[0]
        metadata = data['metadata']
        
        assert isinstance(metadata, DatasetMetadata)
        assert metadata.period == 'test_period'
        assert metadata.row_count >= 0
        assert isinstance(metadata.extraction_duration_seconds, float)
        assert metadata.extraction_duration_seconds >= 0
        assert isinstance(metadata.processing_timestamp, str)
        assert 'T' in metadata.processing_timestamp  # ISO format
    
    # ========================================
    # Error Handling Tests
    # ========================================
    
    def test_extract_files_handles_corrupted_json_file(self, valid_config_file, tmp_path):
        """Test handling of corrupted JSON files"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = JSONDataExtractor(config_manager)
        
        # Create directory structure with corrupted JSON
        reports_dir = tmp_path / "comparison_reports"
        subdir = reports_dir / "test_type"
        subdir.mkdir(parents=True)
        
        corrupted_file = subdir / "corrupted_comparison_report.json"
        with open(corrupted_file, 'w') as f:
            f.write('{"corrupted": json}')  # Invalid JSON
        
        # Act & Assert
        # Should raise ExtractionError when no files can be processed (matches actual behavior)
        with pytest.raises(ExtractionError, match="No JSON reports successfully extracted"):
            extractor.extract_files(str(reports_dir), 'test_period')
    
    def test_extract_files_continues_after_single_file_error(self, valid_config_file, tmp_path, sample_comparison_report):
        """Test that extraction continues after single file error"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = JSONDataExtractor(config_manager)
        
        # Create directory with one good and one bad file
        reports_dir = tmp_path / "comparison_reports"
        subdir = reports_dir / "test_type"
        subdir.mkdir(parents=True)
        
        # Good file
        good_file = subdir / "good_comparison_report.json"
        with open(good_file, 'w') as f:
            json.dump(sample_comparison_report, f)
        
        # Bad file
        bad_file = subdir / "bad_comparison_report.json"
        with open(bad_file, 'w') as f:
            f.write('invalid json')
        
        # Act
        result = extractor.extract_files(str(reports_dir), 'test_period')
        
        # Assert
        # Should have 1 successful extraction
        assert len(result) == 1
        
        # Verify it's the good file
        table_name = list(result.keys())[0]
        assert 'good' in table_name


class TestJSONDataExtractorTableNameGeneration:
    """Test suite for JSON extractor table name generation"""
    
    def test_generate_table_name_follows_pattern(self, valid_config_file):
        """Test that table names follow expected pattern"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = JSONDataExtractor(config_manager)
        
        # Act
        table_name = extractor.generate_table_name(
            subject="computer_science",
            period_name="test_period",
            sheet_name="highly_cited_only"
        )
        
        # Assert
        assert table_name == "computer_science_test_period_highly_cited_only"
    
    def test_generate_table_name_normalises_components(self, valid_config_file):
        """Test that table name components are properly normalised"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        extractor = JSONDataExtractor(config_manager)
        
        # Act
        table_name = extractor.generate_table_name(
            subject="Biology & Biochemistry",
            period_name="Q1 2024",
            sheet_name="Incites Researchers"
        )
        
        # Assert
        assert table_name == "biology_biochemistry_q1_2024_incites_researchers"