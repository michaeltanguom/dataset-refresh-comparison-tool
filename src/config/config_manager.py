"""
Configuration management for the dataset comparison pipeline
Centralised configuration loading and validation
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..utils.exceptions import ConfigurationError
from ..utils.logging_config import get_logger
from ..utils.common import normalise_text, validate_file_path

logger = get_logger('config')


@dataclass
class ValidationRule:
    """Configuration for validation rules"""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    required: bool = False
    min_length: Optional[int] = None
    data_type: Optional[type] = None


class ConfigManager:
    """
    Manages configuration loading and validation
    Single responsibility: Configuration management only
    """
    
    def __init__(self, config_path: str):
        """
        Initialise with path to YAML config file
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        logger.info(f"Configuration loaded successfully from {self.config_path}")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            if not self.config_path.exists():
                raise ConfigurationError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if not config:
                raise ConfigurationError(f"Configuration file is empty: {self.config_path}")
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file {self.config_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {self.config_path}: {e}")
    
    def _validate_config(self) -> None:
        """Validate required configuration sections exist"""
        required_sections = [
            'data_sources', 
            'database', 
            'sheets_to_process', 
            'column_mapping',
            'critical_columns'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ConfigurationError(f"Missing required config section: {section}")
        
        # Validate data sources structure
        self._validate_data_sources()
        
        # Validate database configuration
        self._validate_database_config()
        
        # Validate column mapping
        self._validate_column_mapping()
        
        # Validate sheets to process
        self._validate_sheets_config()
        
        logger.info("Configuration validation passed")
    
    def _validate_data_sources(self) -> None:
        """Validate data sources configuration"""
        data_sources = self.config['data_sources']
        
        for dataset_key in ['dataset_1', 'dataset_2']:
            if dataset_key not in data_sources:
                raise ConfigurationError(f"Missing dataset configuration: {dataset_key}")
            
            dataset_config = data_sources[dataset_key]
            
            # Check required fields
            required_fields = ['folder', 'period_name']
            for field in required_fields:
                if field not in dataset_config:
                    raise ConfigurationError(f"Missing {field} in {dataset_key}")
            
            # Validate folder path format
            folder_path = dataset_config['folder']
            if not folder_path or not isinstance(folder_path, str):
                raise ConfigurationError(f"Invalid folder path in {dataset_key}: {folder_path}")
            
            # Validate period name
            period_name = dataset_config['period_name']
            if not period_name or not isinstance(period_name, str):
                raise ConfigurationError(f"Invalid period_name in {dataset_key}: {period_name}")
    
    def _validate_database_config(self) -> None:
        """Validate database configuration"""
        db_config = self.config['database']
        
        if 'path' not in db_config:
            raise ConfigurationError("Missing database path in configuration")
        
        db_path = db_config['path']
        if not db_path or not isinstance(db_path, str):
            raise ConfigurationError(f"Invalid database path: {db_path}")
    
    def _validate_column_mapping(self) -> None:
        """Validate column mapping configuration"""
        column_mapping = self.config['column_mapping']
        
        if not column_mapping or not isinstance(column_mapping, dict):
            raise ConfigurationError("Column mapping must be a non-empty dictionary")
        
        # Check that critical columns are mapped
        critical_columns = self.config['critical_columns']
        missing_mappings = []
        
        for critical_col in critical_columns:
            if critical_col not in column_mapping:
                missing_mappings.append(critical_col)
        
        if missing_mappings:
            raise ConfigurationError(f"Missing column mappings for critical columns: {missing_mappings}")
    
    def _validate_sheets_config(self) -> None:
        """Validate sheets to process configuration"""
        sheets = self.config['sheets_to_process']
        
        if not sheets or not isinstance(sheets, list):
            raise ConfigurationError("sheets_to_process must be a non-empty list")
        
        if len(sheets) == 0:
            raise ConfigurationError("At least one sheet must be specified in sheets_to_process")
    
    def get_data_source_config(self, dataset_key: str) -> Dict[str, str]:
        """
        Get configuration for specific dataset
        
        Args:
            dataset_key: 'dataset_1' or 'dataset_2'
            
        Returns:
            Dataset configuration dictionary
        """
        if dataset_key not in self.config['data_sources']:
            raise ConfigurationError(f"Dataset configuration not found: {dataset_key}")
        
        return self.config['data_sources'][dataset_key]
    
    def get_database_path(self) -> str:
        """Get database file path"""
        return self.config['database']['path']
    
    def get_sheets_to_process(self) -> List[str]:
        """Get list of sheet names to process"""
        return self.config['sheets_to_process']
    
    def get_column_mapping(self) -> Dict[str, str]:
        """Get column mapping configuration"""
        return self.config['column_mapping']
    
    def get_column_mapping_variants(self) -> Dict[str, List[str]]:
        """Get column mapping variants (for handling spelling differences)"""
        return self.config.get('column_mapping_variants', {})
    
    def get_critical_columns(self) -> List[str]:
        """Get list of critical columns for validation"""
        return self.config['critical_columns']
    
    def get_validation_rules(self) -> Dict[str, ValidationRule]:
        """Get validation rules for columns"""
        rules = {}
        validation_config = self.config.get('validation_rules', {})
        
        for column, rule_config in validation_config.items():
            rules[column] = ValidationRule(
                min_value=rule_config.get('min'),
                max_value=rule_config.get('max'),
                required=rule_config.get('required', False),
                min_length=rule_config.get('min_length'),
                data_type=rule_config.get('data_type')
            )
        
        return rules
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self.config.get('output', {'reports_folder': 'comparison_reports'})
    
    def get_table_naming_config(self) -> Dict[str, str]:
        """Get table naming configuration"""
        return self.config.get('table_naming', {'prefix': 'df'})
    
    def get_output_folder_name(self, normalised_sheet_name: str) -> str:
        """
        Get output folder name for sheet, with configurable overrides
        
        Args:
            normalised_sheet_name: Normalised sheet name
            
        Returns:
            Folder name to use for output
        """
        output_config = self.get_output_config()
        folder_names = output_config.get('folder_names', {})
        
        # Return custom folder name if configured, otherwise use normalised sheet name
        return folder_names.get(normalised_sheet_name, normalised_sheet_name)
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config.get('logging', {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_to_file': True,
            'log_file_path': 'comparison_pipeline.log'
        })
    
    def get_data_cleaning_config(self) -> Dict[str, Any]:
        """Get data cleaning configuration"""
        return self.config.get('data_cleaning', {
            'remove_duplicates': True,
            'handle_nulls': {'strategy': 'fail'},
            'outlier_detection': {'enabled': True}
        })
    
    def get_comparison_config(self) -> Dict[str, Any]:
        """Get comparison configuration"""
        return self.config.get('comparison', {
            'comparison_columns': ['highly_cited_papers', 'indicative_cross_field_score', 'hot_papers', 'times_cited'],
            'float_tolerance': 0.001,
            'include_unchanged': True,
            'fuzzy_matching': {'enabled': False}
        })
    
    def validate_paths(self) -> Dict[str, bool]:
        """
        Validate that configured paths exist or can be created
        
        Returns:
            Dictionary of path validation results
        """
        results = {}
        
        # Check input folders
        for dataset_key in ['dataset_1', 'dataset_2']:
            folder_path = self.get_data_source_config(dataset_key)['folder']
            results[f"{dataset_key}_folder"] = validate_file_path(folder_path, must_exist=True)
        
        # Check output folder (parent should exist)
        output_config = self.get_output_config()
        reports_folder = output_config['reports_folder']
        results['output_folder'] = validate_file_path(reports_folder, must_exist=False)
        
        return results
    
    def get_schema_config(self) -> Dict[str, Any]:
        """Get schema configuration"""
        return self.config.get('schema', {
            'schema_folder': 'schema',
            'default_schema': 'hcr_default',
            'schema_mapping': {}
        })
    
    def get_schema_for_sheet(self, normalised_sheet_name: str) -> str:
        """
        Get schema name for a specific sheet type
        
        Args:
            normalised_sheet_name: Normalised sheet name
            
        Returns:
            Schema name to use
        """
        schema_config = self.get_schema_config()
        schema_mapping = schema_config.get('schema_mapping', {})
        default_schema = schema_config.get('default_schema', 'hcr_default')
        
        return schema_mapping.get(normalised_sheet_name, default_schema)
    
    def get_schema_file_path(self, schema_name: str) -> str:
        """
        Get the full path to a schema SQL file
        
        Args:
            schema_name: Name of the schema (without .sql extension)
            
        Returns:
            Full path to the schema file
        """
        schema_config = self.get_schema_config()
        schema_folder = schema_config.get('schema_folder', 'schema')
        schema_file = f"{schema_name}.sql"
        
        return str(Path(schema_folder) / schema_file)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get summary of configuration for logging/debugging
        
        Returns:
            Summary dictionary
        """
        return {
            'config_file': str(self.config_path),
            'dataset_1_folder': self.get_data_source_config('dataset_1')['folder'],
            'dataset_2_folder': self.get_data_source_config('dataset_2')['folder'],
            'database_path': self.get_database_path(),
            'sheets_to_process': self.get_sheets_to_process(),
            'critical_columns_count': len(self.get_critical_columns()),
            'column_mappings_count': len(self.get_column_mapping()),
            'validation_rules_count': len(self.get_validation_rules()),
            'output_folder': self.get_output_config()['reports_folder'],
            'schema_folder': self.get_schema_config()['schema_folder'],
            'default_schema': self.get_schema_config()['default_schema']
        }
    
    def get_html_generation_config(self) -> Dict[str, Any]:
        """Get HTML generation configuration"""
        return self.config.get('html_generation', {
            'enabled': False,
            'config_path': 'config/html_generator_config.yaml',
            'auto_generate': False
        })

    def load_html_config(self, html_config_path: str) -> Dict[str, Any]:
        """
        Load HTML generator specific configuration
        
        Args:
            html_config_path: Path to HTML config file
            
        Returns:
            HTML configuration dictionary
        """
        try:
            html_config_file = Path(html_config_path)
            
            if not html_config_file.exists():
                return self._get_default_html_config()
            
            with open(html_config_file, 'r', encoding='utf-8') as f:
                html_config = yaml.safe_load(f)
            
            return html_config
            
        except Exception as e:
            return self._get_default_html_config()

    def _get_default_html_config(self) -> Dict[str, Any]:
        """Get default HTML configuration if file not found"""
        return {
            'html_generation': {
                'input_source': 'comparison_reports',
                'output_directory': 'html_reports',
                'template_mapping': {
                    'highly_cited_only': 'research_dashboard',
                    'incites_researchers': 'research_dashboard'
                },
                'default_template': 'research_dashboard'
            },
            'templates': {
                'research_dashboard': {
                    'class': 'ResearchDashboardTemplate',
                    'config': {
                        'title_format': '{dataset_type} - Research Performance Reports',
                        'colour_scheme': 'blue_gradient'
                    }
                }
            },
            'styling': {
                'default_theme': 'modern_blue'
            },
            'output': {
                'file_naming': '{dataset_type}_{template_name}_dashboard.html',
                'include_timestamp': True
            }
        }
