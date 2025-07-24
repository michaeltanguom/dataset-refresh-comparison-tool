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