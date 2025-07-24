"""
Unit tests for DataNormaliser and ESIFieldNormaliser classes
Tests normalisation, column mapping, and ESI field standardisation
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime

from src.extract.light_transform import DataNormaliser, ESIFieldNormaliser, TransformationMetadata
from src.config.config_manager import ConfigManager
from src.utils.exceptions import NormalisationError, ValidationError


class TestESIFieldNormaliser:
    """Test suite for ESIFieldNormaliser class"""
    
    # ========================================
    # ESI Field Normalisation Tests
    # ========================================
    
    def test_normalise_esi_field_with_valid_canonical_fields_returns_correct_values(self, esi_field_test_cases):
        """Test ESI field normalisation with valid canonical fields"""
        # Act & Assert
        for input_field, expected_output in esi_field_test_cases['valid_mappings']:
            result = ESIFieldNormaliser.normalise_esi_field(input_field)
            assert result == expected_output, f"Input: '{input_field}' -> Expected: '{expected_output}', Got: '{result}'"
    
    def test_normalise_esi_field_is_case_insensitive(self, esi_field_test_cases):
        """Test that ESI field normalisation is case insensitive"""
        # Act & Assert
        for input_field, expected_output in esi_field_test_cases['case_insensitive']:
            result = ESIFieldNormaliser.normalise_esi_field(input_field)
            assert result == expected_output, f"Input: '{input_field}' -> Expected: '{expected_output}', Got: '{result}'"
    
    def test_normalise_esi_field_handles_whitespace(self, esi_field_test_cases):
        """Test that ESI field normalisation handles whitespace correctly"""
        # Act & Assert
        for input_field, expected_output in esi_field_test_cases['whitespace_handling']:
            result = ESIFieldNormaliser.normalise_esi_field(input_field)
            assert result == expected_output, f"Input: '{input_field}' -> Expected: '{expected_output}', Got: '{result}'"
    
    def test_normalise_esi_field_with_unknown_fields_returns_original(self, esi_field_test_cases):
        """Test that unknown ESI fields return original value"""
        # Act & Assert
        for input_field, expected_output in esi_field_test_cases['unknown_fields']:
            result = ESIFieldNormaliser.normalise_esi_field(input_field)
            assert result == expected_output, f"Input: '{input_field}' -> Expected: '{expected_output}', Got: '{result}'"
    
    def test_normalise_esi_field_with_none_returns_empty_string(self):
        """Test that None input returns empty string"""
        # Act
        result = ESIFieldNormaliser.normalise_esi_field(None)
        
        # Assert
        assert result == ""
    
    def test_canonical_esi_fields_constant_contains_expected_fields(self):
        """Test that canonical ESI fields constant contains expected fields"""
        # Assert
        canonical_fields = ESIFieldNormaliser.CANONICAL_ESI_FIELDS
        
        # Check some expected fields
        expected_fields = [
            'agricultural sciences',
            'biology_biochemistry',
            'chemistry',
            'computer science',
            'engineering',
            'physics'
        ]
        
        for field in expected_fields:
            assert field in canonical_fields, f"Expected field '{field}' not found in canonical fields"
        
        # Check that values are properly formatted
        for key, value in canonical_fields.items():
            assert isinstance(value, str)
            assert len(value) > 0
            assert value[0].isupper()  # Should be title case


class TestDataNormaliser:
    """Test suite for DataNormaliser class"""
    
    # ========================================
    # Initialisation Tests
    # ========================================
    
    def test_init_with_valid_config_manager_succeeds(self, valid_config_file):
        """Test successful DataNormaliser initialisation"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Act
        normaliser = DataNormaliser(config_manager)
        
        # Assert
        assert normaliser.config == config_manager
        assert hasattr(normaliser, 'column_mapping')
        assert hasattr(normaliser, 'column_variants')
        assert hasattr(normaliser, 'critical_columns')
        assert hasattr(normaliser, 'validation_rules')
        assert hasattr(normaliser, 'column_lookup')
        assert hasattr(normaliser, 'esi_normalisation_stats')
    
    def test_create_column_lookup_builds_comprehensive_mapping(self, valid_config_file):
        """Test that column lookup is built correctly"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        normaliser = DataNormaliser(config_manager)
        
        # Act
        # Column lookup is created during init
        lookup = normaliser.column_lookup
        
        # Assert
        assert isinstance(lookup, dict)
        assert len(lookup) > 0
        
        # Check that main mappings are included
        assert 'Name' in lookup
        assert lookup['Name'] == 'name'
        
        # Check that normalised versions are included
        assert 'times_cited' in lookup
        assert lookup['times_cited'] == 'times_cited'
    
    # ========================================
    # Dataset Normalisation Tests
    # ========================================
    
    def test_normalise_datasets_with_valid_data_returns_normalised_datasets(self, valid_config_file, sample_excel_dataframe, sample_dataset_metadata):
        """Test normalisation of valid datasets"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        normaliser = DataNormaliser(config_manager)
        
        # Create extracted data structure
        extracted_data = {
            'test_table': {
                'dataframe': sample_excel_dataframe,
                'metadata': sample_dataset_metadata
            }
        }
        
        # Act
        result = normaliser.normalise_datasets(extracted_data)
        
        # Assert
        assert isinstance(result, dict)
        assert 'test_table' in result
        assert 'dataframe' in result['test_table']
        assert 'metadata' in result['test_table']
        assert 'transformation_metadata' in result['test_table']
        
        # Check transformation metadata
        transform_metadata = result['test_table']['transformation_metadata']
        assert isinstance(transform_metadata, TransformationMetadata)
        assert isinstance(transform_metadata.columns_mapped, dict)
        assert transform_metadata.normalisation_duration_seconds >= 0
    
    def test_normalise_datasets_applies_column_mapping(self, valid_config_file, sample_dataset_metadata):
        """Test that column mapping is applied correctly"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        normaliser = DataNormaliser(config_manager)
        
        # Create DataFrame with original column names
        test_df = pd.DataFrame({
            'Name': ['Dr. Test'],
            'Times Cited': [100],
            'Highly Cited Papers': [5],
            'ESI Field': ['Engineering']
        })
        
        extracted_data = {
            'test_table': {
                'dataframe': test_df,
                'metadata': sample_dataset_metadata
            }
        }
        
        # Act
        result = normaliser.normalise_datasets(extracted_data)
        
        # Assert
        normalised_df = result['test_table']['dataframe']
        
        # Check that columns are now normalised
        expected_columns = ['name', 'times_cited', 'highly_cited_papers', 'esi_field']
        for col in expected_columns:
            assert col in normalised_df.columns, f"Expected column '{col}' not found"
        
        # Check that original columns are not present
        original_columns = ['Name', 'Times Cited', 'Highly Cited Papers', 'ESI Field']
        for col in original_columns:
            assert col not in normalised_df.columns, f"Original column '{col}' should be mapped"
    
    def test_normalise_datasets_applies_esi_field_normalisation(self, valid_config_file, sample_dataset_metadata):
        """Test that ESI field normalisation is applied"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        normaliser = DataNormaliser(config_manager)
        
        # Create DataFrame with unnormalised ESI fields
        test_df = pd.DataFrame({
            'Name': ['Dr. Test1', 'Dr. Test2'],
            'Times Cited': [100, 200],
            'Highly Cited Papers': [5, 10],
            'ESI Field': ['computer science', 'ENGINEERING']  # Unnormalised
        })
        
        extracted_data = {
            'test_table': {
                'dataframe': test_df,
                'metadata': sample_dataset_metadata
            }
        }
        
        # Act
        result = normaliser.normalise_datasets(extracted_data)
        
        # Assert
        normalised_df = result['test_table']['dataframe']
        
        # Check that ESI fields are normalised
        esi_values = normalised_df['esi_field'].tolist()
        assert 'Computer Science' in esi_values
        assert 'Engineering' in esi_values
        
        # Check transformation metadata
        transform_metadata = result['test_table']['transformation_metadata']
        assert transform_metadata.esi_fields_normalised >= 0
    
    # ========================================
    # Column Mapping Tests
    # ========================================
    
    def test_normalise_dataframe_with_perfect_column_match(self, valid_config_file, column_mapping_test_scenarios):
        """Test normalisation with perfect column match"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        normaliser = DataNormaliser(config_manager)
        
        scenario = column_mapping_test_scenarios['perfect_match']
        test_df = pd.DataFrame({col: [1, 2, 3] for col in scenario['input_columns']})
        
        # Act
        normalised_df, mapping_applied = normaliser._normalise_dataframe(test_df, 'test_sheet')
        
        # Assert
        # Check that all expected columns are mapped
        for expected_col in scenario['expected_mapped']:
            assert expected_col in normalised_df.columns
        
        # Check that mapping was applied
        assert len(mapping_applied) == len(scenario['expected_mapped'])
    
    def test_normalise_dataframe_with_variant_matching(self, valid_config_file):
        """Test normalisation with column variants"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        normaliser = DataNormaliser(config_manager)
        
        # Create DataFrame with standard column names that should map
        test_df = pd.DataFrame({
            'Name': ['Dr. Test'],
            'Times Cited': [100],
            'Highly Cited Papers': [5],
            'ESI Field': ['Engineering']
        })
        
        # Act
        normalised_df, mapping_applied = normaliser._normalise_dataframe(test_df, 'test_sheet')
        
        # Assert
        # Should map to standard column names
        expected_columns = ['name', 'times_cited', 'highly_cited_papers', 'esi_field']
        for col in expected_columns:
            assert col in normalised_df.columns
    
    def test_normalise_dataframe_with_missing_critical_columns_raises_error(self, valid_config_file):
        """Test that missing critical columns raise NormalisationError"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        normaliser = DataNormaliser(config_manager)
        
        # Create DataFrame missing critical columns
        test_df = pd.DataFrame({
            'Name': ['Dr. Test'],
            'Some Other Column': [100]
            # Missing times_cited, highly_cited_papers, esi_field
        })
        
        # Act & Assert
        with pytest.raises(NormalisationError, match="Missing critical columns"):
            normaliser._normalise_dataframe(test_df, 'test_sheet')
    
    def test_normalise_dataframe_adds_missing_optional_columns(self, valid_config_file):
        """Test that missing optional columns are added with defaults"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        normaliser = DataNormaliser(config_manager)
        
        # Create DataFrame with only critical columns
        test_df = pd.DataFrame({
            'Name': ['Dr. Test'],
            'Times Cited': [100],
            'Highly Cited Papers': [5],
            'ESI Field': ['Engineering']
        })
        
        # Act
        normalised_df, mapping_applied = normaliser._normalise_dataframe(test_df, 'test_sheet')
        
        # Assert
        # Should have all expected columns from column_mapping
        expected_columns = set(config_manager.get_column_mapping().keys())
        actual_columns = set(normalised_df.columns)
        
        assert expected_columns.issubset(actual_columns)
        
        # Missing columns should have default empty values
        for col in expected_columns - set(['name', 'times_cited', 'highly_cited_papers', 'esi_field']):
            if col in normalised_df.columns:
                # Should be empty string by default
                assert all(normalised_df[col] == "")
    
    # ========================================
    # ESI Field Normalisation Tests
    # ========================================
    
    def test_normalise_esi_fields_in_dataframe_with_esi_column_normalises_values(self, valid_config_file):
        """Test ESI field normalisation in DataFrame with ESI column"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        normaliser = DataNormaliser(config_manager)
        
        # Create DataFrame with already-mapped columns and ESI field
        test_df = pd.DataFrame({
            'name': ['Dr. Test1', 'Dr. Test2'],
            'times_cited': [100, 200],
            'esi_field': ['computer science', 'ENGINEERING']  # Unnormalised
        })
        
        # Act
        normalised_df, fields_normalised = normaliser._normalise_esi_fields_in_dataframe(test_df, 'test_df')
        
        # Assert
        assert fields_normalised > 0
        assert 'Computer Science' in normalised_df['esi_field'].values
        assert 'Engineering' in normalised_df['esi_field'].values
    
    def test_normalise_esi_fields_in_dataframe_without_esi_column_returns_unchanged(self, valid_config_file):
        """Test ESI field normalisation with DataFrame without ESI column"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        normaliser = DataNormaliser(config_manager)
        
        # Create DataFrame without ESI field
        test_df = pd.DataFrame({
            'name': ['Dr. Test'],
            'times_cited': [100],
            'other_column': ['data']
        })
        
        # Act
        normalised_df, fields_normalised = normaliser._normalise_esi_fields_in_dataframe(test_df, 'test_df')
        
        # Assert
        assert fields_normalised == 0
        assert normalised_df.equals(test_df)  # Should be unchanged
    
    def test_find_esi_field_columns_identifies_correct_columns(self, valid_config_file):
        """Test identification of ESI field columns"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        normaliser = DataNormaliser(config_manager)
        
        # Test with standard ESI column name
        test_df1 = pd.DataFrame({
            'name': ['Dr. Test'],
            'esi_field': ['Engineering'],
            'other_column': ['data']
        })
        
        # Act
        esi_columns1 = normaliser._find_esi_field_columns(test_df1)
        
        # Assert
        assert 'esi_field' in esi_columns1
        assert len(esi_columns1) == 1
        
        # Test with alternative ESI column name
        test_df2 = pd.DataFrame({
            'name': ['Dr. Test'],
            'ESI Field': ['Engineering'],
            'other_column': ['data']
        })
        
        esi_columns2 = normaliser._find_esi_field_columns(test_df2)
        assert 'ESI Field' in esi_columns2
    
    # ========================================
    # Validation Tests
    # ========================================
    
    def test_validate_transformation_with_valid_data_succeeds(self, valid_config_file):
        """Test transformation validation with valid data"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        normaliser = DataNormaliser(config_manager)
        
        # Create valid original and normalised DataFrames
        original_df = pd.DataFrame({
            'Name': ['Dr. Test'],
            'Times Cited': [100],
            'Highly Cited Papers': [5],
            'ESI Field': ['Engineering']
        })
        
        normalised_df = pd.DataFrame({
            'name': ['Dr. Test'],
            'times_cited': [100],
            'highly_cited_papers': [5],
            'esi_field': ['Engineering']
        })
        
        mapping_applied = {
            'name': 'Name',
            'times_cited': 'Times Cited',
            'highly_cited_papers': 'Highly Cited Papers',
            'esi_field': 'ESI Field'
        }
        
        # Act & Assert
        # Should not raise exception
        normaliser._validate_transformation(original_df, normalised_df, mapping_applied, 'test_sheet')
    
    def test_validate_transformation_with_empty_dataframe_raises_error(self, valid_config_file):
        """Test validation with empty normalised DataFrame"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        normaliser = DataNormaliser(config_manager)
        
        original_df = pd.DataFrame({'Name': ['Dr. Test']})
        empty_df = pd.DataFrame()
        mapping_applied = {}
        
        # Act & Assert
        with pytest.raises(ValidationError, match="is empty"):
            normaliser._validate_transformation(original_df, empty_df, mapping_applied, 'test_sheet')
    
    # ========================================
    # Summary and Statistics Tests
    # ========================================
    
    def test_get_normalisation_summary_returns_comprehensive_info(self, valid_config_file):
        """Test normalisation summary generation"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        normaliser = DataNormaliser(config_manager)
        
        # Act
        summary = normaliser.get_normalisation_summary()
        
        # Assert
        assert isinstance(summary, dict)
        assert 'dataframes_processed' in summary
        assert 'fields_normalised' in summary
        assert 'normalisation_errors' in summary
        assert 'canonical_esi_fields' in summary
        assert 'total_canonical_fields' in summary
        assert 'normalisation_timestamp' in summary
        
        # Check data types
        assert isinstance(summary['dataframes_processed'], int)
        assert isinstance(summary['fields_normalised'], int)
        assert isinstance(summary['normalisation_errors'], int)
        assert isinstance(summary['canonical_esi_fields'], list)
        assert isinstance(summary['total_canonical_fields'], int)
        assert isinstance(summary['normalisation_timestamp'], str)


class TestTransformationMetadata:
    """Test suite for TransformationMetadata dataclass"""
    
    def test_transformation_metadata_creation_with_all_fields(self):
        """Test TransformationMetadata creation with all fields"""
        # Act
        metadata = TransformationMetadata(
            columns_mapped={'name': 'Name', 'times_cited': 'Times Cited'},
            normalisation_duration_seconds=1.5,
            esi_fields_normalised=3,
            transformation_timestamp="2024-01-15T10:30:45.123456"
        )
        
        # Assert
        assert metadata.columns_mapped == {'name': 'Name', 'times_cited': 'Times Cited'}
        assert metadata.normalisation_duration_seconds == 1.5
        assert metadata.esi_fields_normalised == 3
        assert metadata.transformation_timestamp == "2024-01-15T10:30:45.123456"
    
    def test_transformation_metadata_creation_with_defaults(self):
        """Test TransformationMetadata creation with default values"""
        # Act
        metadata = TransformationMetadata(
            columns_mapped={'name': 'Name'}
        )
        
        # Assert
        assert metadata.columns_mapped == {'name': 'Name'}
        assert metadata.normalisation_duration_seconds == 0.0
        assert metadata.esi_fields_normalised == 0
        assert metadata.transformation_timestamp == ""
    
    def test_transformation_metadata_is_dataclass(self):
        """Test that TransformationMetadata is properly configured as dataclass"""
        # Assert
        assert hasattr(TransformationMetadata, '__dataclass_fields__')
        assert 'columns_mapped' in TransformationMetadata.__dataclass_fields__
        assert 'normalisation_duration_seconds' in TransformationMetadata.__dataclass_fields__