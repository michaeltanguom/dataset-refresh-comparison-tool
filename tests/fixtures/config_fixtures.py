"""
Configuration-specific test fixtures and factories
"""
import pytest
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigFactory:
    """Factory for creating test configurations"""
    
    @staticmethod
    def create_valid_config() -> Dict[str, Any]:
        """Create a minimal valid configuration"""
        return {
            'data_sources': {
                'dataset_1': {
                    'folder': 'test_data/feb',
                    'period_name': 'feb'
                },
                'dataset_2': {
                    'folder': 'test_data/july',
                    'period_name': 'july'
                }
            },
            'database': {
                'path': 'test_comparison.db'
            },
            'schema': {
                'schema_folder': 'schema',
                'default_schema': 'hcr_default',
                'schema_mapping': {
                    'highly_cited_only': 'hcr_default',
                    'incites_researchers': 'hcr_default'
                }
            },
            'sheets_to_process': [
                'Highly Cited only',
                'Incites Researchers'
            ],
            'column_mapping': {
                'name': 'Name',
                'percent_docs_cited': '% Docs Cited',
                'times_cited': 'Times Cited',
                'highly_cited_papers': 'Highly Cited Papers',
                'esi_field': 'ESI Field',
                'indicative_cross_field_score': 'Indicative Cross-Field Score'
            },
            'column_mapping_variants': {
                'category_normalised_citation_impact': [
                    'Category Normalized Citation Impact',
                    'Category Normalised Citation Impact'
                ]
            },
            'critical_columns': [
                'name',
                'times_cited',
                'highly_cited_papers',
                'esi_field'
            ],
            'validation_rules': {
                'times_cited': {
                    'min': 50,
                    'max': 100000,
                    'required': True
                },
                'highly_cited_papers': {
                    'min': 0,
                    'max': 200,
                    'required': True
                },
                'name': {
                    'required': True,
                    'min_length': 2
                }
            },
            'output': {
                'reports_folder': 'comparison_reports'
            },
            'table_naming': {
                'prefix': 'df'
            },
            'data_cleaning': {
                'remove_duplicates': True,
                'handle_nulls': {
                    'strategy': 'fail'
                }
            },
            'comparison': {
                'comparison_columns': [
                    'highly_cited_papers',
                    'indicative_cross_field_score'
                ],
                'float_tolerance': 0.001,
                'include_unchanged': True
            },
            'html_generation': {
                'enabled': True,
                'config_path': 'config/html_generator_config.yaml'
            },
            'logging': {
                'level': 'INFO',
                'log_to_file': False  # Disable file logging in tests
            }
        }
    
    @staticmethod
    def create_minimal_config() -> Dict[str, Any]:
        """Create minimal configuration with only required fields"""
        return {
            'data_sources': {
                'dataset_1': {'folder': 'test1', 'period_name': 'period1'},
                'dataset_2': {'folder': 'test2', 'period_name': 'period2'}
            },
            'database': {'path': ':memory:'},
            'sheets_to_process': ['Test Sheet'],
            'column_mapping': {'name': 'Name'},
            'critical_columns': ['name']
        }
    
    @staticmethod
    def create_invalid_config_missing_section(missing_section: str) -> Dict[str, Any]:
        """Create invalid configuration missing a required section"""
        config = ConfigFactory.create_minimal_config()
        if missing_section in config:
            del config[missing_section]
        return config

@pytest.fixture
def config_factory():
    """Provide configuration factory for tests"""
    return ConfigFactory

@pytest.fixture
def valid_config():
    """Provide a valid test configuration"""
    return ConfigFactory.create_valid_config()

@pytest.fixture
def minimal_config():
    """Provide minimal valid configuration"""
    return ConfigFactory.create_minimal_config()

@pytest.fixture
def valid_config_file(temp_config_dir, valid_config):
    """Create a valid configuration file on disk"""
    config_file = temp_config_dir / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(valid_config, f)
    return config_file

@pytest.fixture
def test_schema_file(temp_schema_dir):
    """Create a test schema file"""
    schema_content = '''
CREATE TABLE IF NOT EXISTS "{table_name}" (
    name VARCHAR NOT NULL,
    times_cited INTEGER,
    highly_cited_papers INTEGER,
    esi_field VARCHAR NOT NULL,
    indicative_cross_field_score DOUBLE
);
'''
    schema_file = temp_schema_dir / "hcr_default.sql"
    with open(schema_file, 'w') as f:
        f.write(schema_content)
    return schema_file