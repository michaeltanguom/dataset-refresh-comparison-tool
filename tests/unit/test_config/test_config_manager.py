"""
Unit tests for ConfigManager class
Tests configuration loading, validation, and data retrieval
"""
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from src.config.config_manager import ConfigManager, ValidationRule
from src.utils.exceptions import ConfigurationError

class TestConfigManager:
    """Test suite for ConfigManager functionality"""
    
    # ========================================
    # Configuration Loading Tests
    # ========================================
    
    def test_init_with_valid_config_file_succeeds(self, valid_config_file):
        """Test successful initialisation with valid config file"""
        # Act
        config_manager = ConfigManager(str(valid_config_file))
        
        # Assert
        assert config_manager.config is not None
        assert config_manager.config_path == valid_config_file
    
    def test_init_with_missing_config_file_raises_error(self, temp_config_dir):
        """Test that missing config file raises ConfigurationError"""
        # Arrange
        missing_file = temp_config_dir / "missing.yaml"
        
        # Act & Assert
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            ConfigManager(str(missing_file))
    
    def test_init_with_empty_config_file_raises_error(self, temp_config_dir):
        """Test that empty config file raises ConfigurationError"""
        # Arrange
        empty_file = temp_config_dir / "empty.yaml"
        empty_file.touch()  # Create empty file
        
        # Act & Assert
        with pytest.raises(ConfigurationError, match="Configuration file is empty"):
            ConfigManager(str(empty_file))
    
    def test_init_with_invalid_yaml_raises_error(self, temp_config_dir):
        """Test that invalid YAML raises ConfigurationError"""
        # Arrange
        invalid_file = temp_config_dir / "invalid.yaml"
        with open(invalid_file, 'w') as f:
            f.write("invalid: yaml: content: [")  # Invalid YAML
        
        # Act & Assert
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            ConfigManager(str(invalid_file))
    
    # ========================================
    # Configuration Validation Tests
    # ========================================
    
    @pytest.mark.parametrize("missing_section", [
        'data_sources',
        'database', 
        'sheets_to_process',
        'column_mapping',
        'critical_columns'
    ])
    def test_init_with_missing_required_section_raises_error(self, temp_config_dir, config_factory, missing_section):
        """Test that missing required sections raise ConfigurationError"""
        # Arrange
        invalid_config = config_factory.create_invalid_config_missing_section(missing_section)
        config_file = temp_config_dir / "invalid_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Act & Assert
        with pytest.raises(ConfigurationError, match=f"Missing required config section: {missing_section}"):
            ConfigManager(str(config_file))
    
    def test_init_validates_data_sources_structure(self, temp_config_dir, minimal_config):
        """Test validation of data sources structure"""
        # Arrange - Create config with missing period_name (caught first)
        invalid_config = minimal_config.copy()
        invalid_config['data_sources'] = {
            'dataset_1': {'folder': 'test'},  # Missing period_name
            'dataset_2': {'folder': 'test2', 'period_name': 'period2'}
        }
        
        config_file = temp_config_dir / "invalid_data_sources.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Act & Assert
        with pytest.raises(ConfigurationError, match="Missing period_name in dataset_1"):
            ConfigManager(str(config_file))
        
    # ========================================
    # Data Retrieval Tests
    # ========================================
    
    def test_get_data_source_config_returns_correct_values(self, valid_config_file):
        """Test retrieval of data source configuration"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Act
        dataset_1_config = config_manager.get_data_source_config('dataset_1')
        dataset_2_config = config_manager.get_data_source_config('dataset_2')
        
        # Assert
        assert dataset_1_config['folder'] == 'test_data/feb'
        assert dataset_1_config['period_name'] == 'feb'
        assert dataset_2_config['folder'] == 'test_data/july'
        assert dataset_2_config['period_name'] == 'july'
    
    def test_get_data_source_config_with_invalid_key_raises_error(self, valid_config_file):
        """Test that invalid dataset key raises ConfigurationError"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Act & Assert
        with pytest.raises(ConfigurationError, match="Dataset configuration not found: invalid_dataset"):
            config_manager.get_data_source_config('invalid_dataset')
    
    def test_get_database_path_returns_correct_value(self, valid_config_file):
        """Test database path retrieval"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Act
        db_path = config_manager.get_database_path()
        
        # Assert
        assert db_path == 'test_comparison.db'
    
    def test_get_sheets_to_process_returns_list(self, valid_config_file):
        """Test sheets to process retrieval"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Act
        sheets = config_manager.get_sheets_to_process()
        
        # Assert
        assert isinstance(sheets, list)
        assert 'Highly Cited only' in sheets
        assert 'Incites Researchers' in sheets
    
    def test_get_column_mapping_returns_dict(self, valid_config_file):
        """Test column mapping retrieval"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Act
        mapping = config_manager.get_column_mapping()
        
        # Assert
        assert isinstance(mapping, dict)
        assert mapping['name'] == 'Name'
        assert mapping['times_cited'] == 'Times Cited'
    
    def test_get_critical_columns_returns_list(self, valid_config_file):
        """Test critical columns retrieval"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Act
        critical_cols = config_manager.get_critical_columns()
        
        # Assert
        assert isinstance(critical_cols, list)
        assert 'name' in critical_cols
        assert 'times_cited' in critical_cols
    
    # ========================================
    # Validation Rules Tests
    # ========================================
    
    def test_get_validation_rules_returns_validation_rule_objects(self, valid_config_file):
        """Test validation rules are properly converted to ValidationRule objects"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Act
        rules = config_manager.get_validation_rules()
        
        # Assert
        assert isinstance(rules, dict)
        assert 'times_cited' in rules
        assert isinstance(rules['times_cited'], ValidationRule)
        assert rules['times_cited'].min_value == 50
        assert rules['times_cited'].max_value == 100000
        assert rules['times_cited'].required is True
    
    # ========================================
    # Schema Configuration Tests
    # ========================================
    
    def test_get_schema_for_sheet_returns_correct_schema(self, valid_config_file):
        """Test schema selection for different sheet types"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Act
        schema_name = config_manager.get_schema_for_sheet('highly_cited_only')
        
        # Assert
        assert schema_name == 'hcr_default'
    
    def test_get_schema_file_path_returns_correct_path(self, valid_config_file):
        """Test schema file path generation"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Act
        schema_path = config_manager.get_schema_file_path('hcr_default')
        
        # Assert
        assert schema_path == 'schema/hcr_default.sql'
    
    # ========================================
    # HTML Generation Configuration Tests
    # ========================================
    
    def test_get_html_generation_config_returns_correct_values(self, valid_config_file):
        """Test HTML generation configuration retrieval"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Act
        html_config = config_manager.get_html_generation_config()
        
        # Assert
        assert html_config['enabled'] is True
        assert html_config['config_path'] == 'config/html_generator_config.yaml'
    
    def test_load_html_config_with_missing_file_returns_default(self, valid_config_file):
        """Test HTML config loading falls back to defaults when file missing"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Act
        html_config = config_manager.load_html_config('nonexistent_file.yaml')
        
        # Assert
        assert 'html_generation' in html_config
        assert html_config['html_generation']['input_source'] == 'comparison_reports'
    
    # ========================================
    # Configuration Summary Tests
    # ========================================
    
    def test_get_config_summary_returns_comprehensive_info(self, valid_config_file):
        """Test configuration summary generation"""
        # Arrange
        config_manager = ConfigManager(str(valid_config_file))
        
        # Act
        summary = config_manager.get_config_summary()
        
        # Assert
        assert 'config_file' in summary
        assert 'dataset_1_folder' in summary
        assert 'dataset_2_folder' in summary
        assert 'database_path' in summary
        assert 'sheets_to_process' in summary
        assert isinstance(summary['critical_columns_count'], int)
        assert isinstance(summary['column_mappings_count'], int)
    
    # ========================================
    # Output Configuration Tests
    # ========================================
    
    def test_get_output_folder_name_with_custom_mapping(self, temp_config_dir, valid_config):
        """Test custom output folder name mapping"""
        # Arrange
        # Add custom folder mapping to config
        valid_config['output']['folder_names'] = {
            'highly_cited_only': 'custom_hco_folder'
        }
        
        config_file = temp_config_dir / "custom_output_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(valid_config, f)
        
        config_manager = ConfigManager(str(config_file))
        
        # Act
        folder_name = config_manager.get_output_folder_name('highly_cited_only')
        default_folder = config_manager.get_output_folder_name('unmapped_sheet')
        
        # Assert
        assert folder_name == 'custom_hco_folder'
        assert default_folder == 'unmapped_sheet'