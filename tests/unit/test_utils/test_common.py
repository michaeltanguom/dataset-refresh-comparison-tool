"""
Unit tests for common utilities module
Tests shared functions used across the pipeline
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock

from src.utils.common import (
    normalise_text,
    generate_timestamp,
    safe_convert_numeric,
    validate_file_path,
    clean_dataframe_columns,
    get_sample_values,
    log_dataframe_info,
    create_directory_if_not_exists,
    format_number_with_commas,
    calculate_percentage_change,
    is_valid_esi_field
)


class TestNormaliseText:
    """Test suite for text normalisation functionality"""
    
    def test_normalise_text_basic_cases(self):
        """Test normalisation with basic cases"""
        test_cases = [
            ('Highly Cited Papers', 'highly_cited_papers'),
            ('ESI Field', 'esi_field'),
            ('  Spaces  Around  ', 'spaces_around'),
            ('Special-Chars@Here!', 'special_chars_here'),
            ('', ''),
            ('   ', ''),
        ]
        
        for input_text, expected in test_cases:
            result = normalise_text(input_text)
            assert result == expected, f"Input: '{input_text}' -> Expected: '{expected}', Got: '{result}'"
    
    def test_normalise_text_with_none_returns_empty_string(self):
        """Test normalisation with None input"""
        result = normalise_text(None)
        assert result == ""


class TestSafeConvertNumeric:
    """Test suite for safe numeric conversion"""
    
    def test_safe_convert_numeric_basic_integer_conversion(self):
        """Test basic integer conversions"""
        test_cases = [
            ('123', int, 123),
            ('123.0', int, 123),
            (123, int, 123),
            (123.7, int, 123),
        ]
        
        for value, target_type, expected in test_cases:
            result = safe_convert_numeric(value, target_type)
            assert result == expected
    
    def test_safe_convert_numeric_with_invalid_values_returns_default(self):
        """Test conversion with invalid values"""
        assert safe_convert_numeric('invalid', int) == 0
        assert safe_convert_numeric('invalid', float) == 0.0
        assert safe_convert_numeric(None, int) == 0
        assert safe_convert_numeric('invalid', int, default=-1) == -1


class TestValidateFilePath:
    """Test suite for file path validation"""
    
    def test_validate_file_path_with_temp_file(self, tmp_path):
        """Test validation with temporary files"""
        # Create a test file
        test_file = tmp_path / "test_file.txt"
        test_file.write_text("test content")
        
        # Test existing file
        assert validate_file_path(str(test_file), must_exist=True) is True
        
        # Test non-existing file with existing parent
        non_existing = tmp_path / "non_existing.txt"
        assert validate_file_path(str(non_existing), must_exist=False) is True
        
        # Test non-existing file with must_exist=True
        assert validate_file_path(str(non_existing), must_exist=True) is False


class TestCleanDataFrameColumns:
    """Test suite for DataFrame column cleaning"""
    
    def test_clean_dataframe_columns_with_normal_dataframe(self, sample_dataframe):
        """Test cleaning with normal DataFrame"""
        cleaned_df = clean_dataframe_columns(sample_dataframe)
        assert list(cleaned_df.columns) == list(sample_dataframe.columns)
    
    def test_clean_dataframe_columns_with_empty_dataframe(self, empty_dataframe):
        """Test cleaning with empty DataFrame"""
        result = clean_dataframe_columns(empty_dataframe)
        assert result.empty


class TestGetSampleValues:
    """Test suite for getting sample values"""
    
    def test_get_sample_values_basic_functionality(self, sample_dataframe):
        """Test basic sample value retrieval"""
        samples = get_sample_values(sample_dataframe, 'Name', n_samples=2)
        assert isinstance(samples, list)
        assert len(samples) <= 2
        assert all(sample in sample_dataframe['Name'].values for sample in samples)
    
    def test_get_sample_values_with_missing_column(self, sample_dataframe):
        """Test with non-existent column"""
        samples = get_sample_values(sample_dataframe, 'NonExistentColumn')
        assert samples == []


class TestLogDataFrameInfo:
    """Test suite for DataFrame logging"""
    
    def test_log_dataframe_info_with_valid_dataframe(self, sample_dataframe, mock_logger):
        """Test logging of valid DataFrame"""
        log_dataframe_info(sample_dataframe, 'test_df', mock_logger)
        
        assert mock_logger.has_logged('info', 'test_df')
        assert mock_logger.has_logged('info', 'Shape:')
    
    def test_log_dataframe_info_with_empty_dataframe(self, empty_dataframe, mock_logger):
        """Test logging of empty DataFrame"""
        log_dataframe_info(empty_dataframe, 'empty_df', mock_logger)
        
        assert mock_logger.has_logged('warning', 'empty_df')


class TestCreateDirectoryIfNotExists:
    """Test suite for directory creation"""
    
    def test_create_directory_if_not_exists_creates_new_directory(self, tmp_path):
        """Test creation of new directory"""
        new_dir = tmp_path / 'new_directory'
        assert not new_dir.exists()
        
        result = create_directory_if_not_exists(str(new_dir))
        
        assert result is True
        assert new_dir.exists()


class TestFormatNumberWithCommas:
    """Test suite for number formatting"""
    
    def test_format_number_with_commas_basic_cases(self):
        """Test basic number formatting"""
        test_cases = [
            (1234567, '1,234,567'),
            (1000, '1,000'),
            (999, '999'),
            (0, '0'),
            (-1234567, '-1,234,567')
        ]
        
        for number, expected in test_cases:
            result = format_number_with_commas(number)
            assert result == expected


class TestCalculatePercentageChange:
    """Test suite for percentage change calculations"""
    
    def test_calculate_percentage_change_basic_cases(self, percentage_test_cases):
        """Test basic percentage change calculations"""
        for case in percentage_test_cases:
            result = calculate_percentage_change(case['old'], case['new'])
            if case['expected'] == float('inf'):
                assert result == float('inf')
            else:
                assert abs(result - case['expected']) < 0.001, f"Expected {case['expected']}, got {result} for old={case['old']}, new={case['new']}"
    
    def test_calculate_percentage_change_with_zero_old_value(self):
        """Test percentage change when old value is zero"""
        result = calculate_percentage_change(0, 100)
        assert result == float('inf')


class TestIsValidESIField:
    """Test suite for ESI field validation"""
    
    def test_is_valid_esi_field_with_valid_fields(self):
        """Test with valid ESI fields"""
        valid_fields = [
            'Engineering',
            'Computer Science',
            'Mathematics',
            'Biology & Biochemistry'
        ]
        
        for field in valid_fields:
            assert is_valid_esi_field(field) is True, f"Field '{field}' should be valid"
    
    def test_is_valid_esi_field_with_invalid_fields(self):
        """Test with invalid ESI fields"""
        invalid_fields = [
            '',
            '   ',
            'a',      # Too short
            '123',    # Purely numeric
            'nan',
            'none',
            None
        ]
        
        for field in invalid_fields:
            assert is_valid_esi_field(field) is False, f"Field '{field}' should be invalid"


class TestGenerateTimestamp:
    """Test suite for timestamp generation"""
    
    def test_generate_timestamp_returns_iso_format(self):
        """Test basic timestamp generation"""
        timestamp = generate_timestamp()
        
        assert isinstance(timestamp, str)
        assert 'T' in timestamp  # ISO format
        assert ':' in timestamp  # Time components
    
    @patch('src.utils.common.datetime')
    def test_generate_timestamp_with_mocked_datetime(self, mock_datetime):
        """Test timestamp with mocked datetime"""
        from datetime import datetime
        fixed_datetime = datetime(2024, 1, 15, 10, 30, 45, 123456)
        mock_datetime.now.return_value = fixed_datetime
        
        timestamp = generate_timestamp()
        assert timestamp == "2024-01-15T10:30:45.123456"