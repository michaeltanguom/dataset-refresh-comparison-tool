"""
Global test configuration and fixtures
"""
import sys
from pathlib import Path
import pytest
import tempfile
import shutil
import yaml
from unittest.mock import Mock

# Add project root to Python path so we can import from src/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import fixtures from fixtures module
from tests.fixtures.config_fixtures import *
from tests.fixtures.utils_fixtures import *
from tests.fixtures.extract_fixtures import *
from tests.fixtures.mock_objects import *

@pytest.fixture(scope="session")
def test_temp_dir():
    """Create a temporary directory for the test session"""
    temp_dir = tempfile.mkdtemp(prefix="dataset_comparison_tests_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def temp_config_dir(test_temp_dir):
    """Create temporary directory for configuration files"""
    config_dir = test_temp_dir / "config"
    config_dir.mkdir(exist_ok=True)
    return config_dir

@pytest.fixture
def temp_data_dir(test_temp_dir):
    """Create temporary directory for test data"""
    data_dir = test_temp_dir / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir

@pytest.fixture
def temp_schema_dir(test_temp_dir):
    """Create temporary directory for schema files"""
    schema_dir = test_temp_dir / "schema"
    schema_dir.mkdir(exist_ok=True)
    return schema_dir

@pytest.fixture
def temp_output_dir(test_temp_dir):
    """Create temporary directory for output files"""
    output_dir = test_temp_dir / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir

# ========================================
# Common Utilities Testing Fixtures
# ========================================

@pytest.fixture
def isolated_test_directory(test_temp_dir):
    """Create isolated directory for file operation tests"""
    isolated_dir = test_temp_dir / "isolated_tests"
    isolated_dir.mkdir(exist_ok=True)
    return isolated_dir

@pytest.fixture
def sample_file_paths(isolated_test_directory):
    """Create sample file paths for validation testing"""
    return {
        'existing_file': isolated_test_directory / "existing.txt",
        'missing_file': isolated_test_directory / "missing.txt",
        'existing_dir': isolated_test_directory / "existing_dir",
        'missing_dir': isolated_test_directory / "missing_dir",
        'nested_file': isolated_test_directory / "parent" / "child" / "nested.txt"
    }

@pytest.fixture
def create_test_files(sample_file_paths):
    """Actually create some test files for validation"""
    # Create existing file
    sample_file_paths['existing_file'].write_text("test content")
    
    # Create existing directory
    sample_file_paths['existing_dir'].mkdir()
    
    # Create parent directories for nested file
    sample_file_paths['nested_file'].parent.mkdir(parents=True)
    
    yield sample_file_paths
    
    # Cleanup is handled by temp directory fixture

@pytest.fixture
def numeric_test_values():
    """Provide various numeric values for testing conversions"""
    return {
        'valid_integers': [0, 1, -1, 100, -100, 999999],
        'valid_floats': [0.0, 1.1, -1.1, 100.5, -100.5, 999.999],
        'string_integers': ['0', '1', '-1', '100', '-100'],
        'string_floats': ['0.0', '1.1', '-1.1', '100.5', '-100.5'],
        'string_float_as_int': ['123.0', '456.0', '-789.0'],
        'invalid_strings': ['abc', 'not_a_number', '12.34.56', ''],
        'none_values': [None, pd.NA, pd.NaT],
        'edge_cases': [float('inf'), float('-inf'), float('nan')]
    }