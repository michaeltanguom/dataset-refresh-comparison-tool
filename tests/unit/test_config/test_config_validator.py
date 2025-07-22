"""
Unit tests for ConfigValidator class
Tests comprehensive configuration validation logic
"""
import pytest
from unittest.mock import Mock, patch
import yaml

from src.config.config_validator import ConfigValidator, validate_config_file
from src.config.config_manager import ConfigManager
from src.utils.exceptions import ConfigurationError

class TestConfigValidator:
    """Test suite for ConfigValidator functionality"""
    
    # ========================================
    # Validator Initialisation Tests
    # ========================================
    
    def test_init_with_valid_config_manager_succeeds(self, valid_config_file):
        """Test successful validator initialisation"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Act
        validator = ConfigValidator(config_manager)
        
        # Assert
        assert validator.config == config_manager
        assert 'passed' in validator.validation_results
        assert 'warnings' in validator.validation_results
        assert 'errors' in validator.validation_results
    
    # ========================================
    # Schema Validation Tests
    # ========================================
    
    def test_validate_schema_configuration_with_valid_schema(self, valid_config_file, test_schema_file):
        """Test schema validation with valid schema files"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Mock schema folder and file paths
        with patch.object(config_manager, 'get_schema_config') as mock_schema_config, \
             patch.object(config_manager, 'get_schema_file_path') as mock_schema_path:
            
            mock_schema_config.return_value = {
                'schema_folder': str(test_schema_file.parent),
                'default_schema': 'hcr_default'
            }
            mock_schema_path.return_value = str(test_schema_file)
            
            validator = ConfigValidator(config_manager)
            
            # Act
            validator._validate_schema_configuration()
            
            # Assert
            passed_checks = [result['message'] for result in validator.validation_results['passed']]
            assert any('Schema folder exists' in msg for msg in passed_checks)
            assert any('Default schema file is valid' in msg for msg in passed_checks)
    
    # ========================================
    # Column Mapping Validation Tests
    # ========================================
    
    def test_validate_column_mapping_completeness_with_all_critical_mapped(self, valid_config_file):
        """Test column mapping validation when all critical columns are mapped"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        validator = ConfigValidator(config_manager)
        
        # Act
        validator._validate_column_mapping_completeness()
        
        # Assert
        passed_messages = [result['message'] for result in validator.validation_results['passed']]
        assert any('critical columns are mapped' in msg for msg in passed_messages)
    
    def test_validate_column_mapping_with_missing_critical_columns(self, temp_config_dir):
        """Test column mapping validation with missing critical columns"""
        # Arrange - Create a minimal valid config first
        config_with_missing_mapping = {
            'data_sources': {
                'dataset_1': {'folder': 'test1', 'period_name': 'period1'},
                'dataset_2': {'folder': 'test2', 'period_name': 'period2'}
            },
            'database': {'path': ':memory:'},
            'sheets_to_process': ['Test Sheet'],
            'column_mapping': {'name': 'Name'},  # Only has 'name'
            'critical_columns': ['name', 'missing_column']  # But requires 'missing_column'
        }
        
        config_file = temp_config_dir / "invalid_mapping.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_with_missing_mapping, f)
        
        # Act & Assert - This should fail at ConfigManager level
        with pytest.raises(ConfigurationError, match="Missing column mappings for critical columns"):
            ConfigManager(str(config_file))
    
    # ========================================
    # Comprehensive Validation Tests
    # ========================================
    
    def test_validate_all_with_valid_config_returns_valid_result(self, valid_config_file, test_schema_file, temp_data_dir):
        """Test complete validation with valid configuration"""
        # Arrange
        # Create required directories
        (temp_data_dir / "test_data" / "feb").mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "test_data" / "july").mkdir(parents=True, exist_ok=True)
        
        config_manager = ConfigManager(str(valid_config_file))
        
        # Mock paths to use our temp directories
        with patch.object(config_manager, 'get_data_source_config') as mock_get_config, \
            patch.object(config_manager, 'get_schema_config') as mock_schema_config, \
            patch.object(config_manager, 'get_schema_file_path') as mock_schema_path:
            
            mock_get_config.side_effect = lambda key: {
                'folder': str(temp_data_dir / "test_data" / ("feb" if key == "dataset_1" else "july")),
                'period_name': 'feb' if key == 'dataset_1' else 'july'
            }
            mock_schema_config.return_value = {
                'schema_folder': str(test_schema_file.parent),
                'default_schema': 'hcr_default'
            }
            mock_schema_path.return_value = str(test_schema_file)
            
            validator = ConfigValidator(config_manager)
            
            # Act
            results = validator.validate_all()
            
            # Assert
            assert results['is_valid'] is True
            assert results['errors'] == 0
            assert results['passed'] > 0
            assert 'total_checks' in results
    
    # ========================================
    # Recommendations Tests
    # ========================================
    
    def test_get_recommendations_returns_helpful_suggestions(self, valid_config_file):
        """Test recommendation generation"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        validator = ConfigValidator(config_manager)
        
        # Add some warnings to trigger recommendations
        validator.validation_results['warnings'].append({
            'check': 'test_check',
            'message': 'Excel files not found'
        })
        
        # Act
        recommendations = validator.get_recommendations()
        
        # Assert
        assert isinstance(recommendations, list)
        if recommendations:  # If there are recommendations
            assert all(isinstance(rec, str) for rec in recommendations)
    
    # ========================================
    # Validation Report Tests
    # ========================================
    
    def test_generate_validation_report_creates_readable_output(self, valid_config_file, test_schema_file):
        """Test validation report generation"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        with patch.object(config_manager, 'get_schema_config') as mock_schema_config, \
             patch.object(config_manager, 'get_schema_file_path') as mock_schema_path:
            
            mock_schema_config.return_value = {
                'schema_folder': str(test_schema_file.parent),
                'default_schema': 'hcr_default'
            }
            mock_schema_path.return_value = str(test_schema_file)
            
            validator = ConfigValidator(config_manager)
            
            # Act
            report = validator.generate_validation_report()
            
            # Assert
            assert isinstance(report, str)
            assert 'CONFIGURATION VALIDATION REPORT' in report
            assert 'Total checks:' in report

# ========================================
# Standalone Function Tests
# ========================================

class TestValidateConfigFile:
    """Test the standalone validate_config_file function"""
    
    def test_validate_config_file_with_valid_file_returns_valid(self, valid_config_file):
        """Test standalone validation function with valid file"""
        # Act
        result = validate_config_file(str(valid_config_file))
        
        # Assert
        assert 'is_valid' in result
        assert 'total_checks' in result
    
    def test_validate_config_file_with_missing_file_returns_error(self):
        """Test standalone validation function with missing file"""
        # Act
        result = validate_config_file('nonexistent_file.yaml')
        
        # Assert
        assert result['is_valid'] is False
        assert result['errors'] == 1