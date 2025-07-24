"""
Mock objects for testing external dependencies
Provides mocks for logging, file system operations, etc.
"""
import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path
import tempfile
import shutil


class MockLogger:
    """Mock logger for testing logging functionality"""
    
    def __init__(self):
        self.debug_calls = []
        self.info_calls = []
        self.warning_calls = []
        self.error_calls = []
        self.log_calls = []
    
    def debug(self, message, *args, **kwargs):
        self.debug_calls.append((message, args, kwargs))
    
    def info(self, message, *args, **kwargs):
        self.info_calls.append((message, args, kwargs))
    
    def warning(self, message, *args, **kwargs):
        self.warning_calls.append((message, args, kwargs))
    
    def error(self, message, *args, **kwargs):
        self.error_calls.append((message, args, kwargs))
    
    def log(self, level, message, *args, **kwargs):
        self.log_calls.append((level, message, args, kwargs))
    
    def get_all_calls(self):
        """Get all logged messages for assertions"""
        return {
            'debug': self.debug_calls,
            'info': self.info_calls,
            'warning': self.warning_calls,
            'error': self.error_calls,
            'log': self.log_calls
        }
    
    def has_logged(self, level, message_substring):
        """Check if a message containing substring was logged at level"""
        calls = getattr(self, f'{level}_calls', [])
        return any(message_substring in str(call[0]) for call in calls)


class MockFileSystem:
    """Mock file system for testing file operations"""
    
    def __init__(self):
        self.existing_files = set()
        self.existing_directories = set()
        self.created_directories = []
    
    def add_existing_file(self, file_path: str):
        """Add a file that should exist"""
        self.existing_files.add(str(Path(file_path)))
    
    def add_existing_directory(self, dir_path: str):
        """Add a directory that should exist"""
        self.existing_directories.add(str(Path(dir_path)))
    
    def file_exists(self, file_path: str) -> bool:
        """Check if file exists in mock filesystem"""
        return str(Path(file_path)) in self.existing_files
    
    def directory_exists(self, dir_path: str) -> bool:
        """Check if directory exists in mock filesystem"""
        return str(Path(dir_path)) in self.existing_directories
    
    def create_directory(self, dir_path: str) -> bool:
        """Mock directory creation"""
        self.created_directories.append(str(Path(dir_path)))
        self.existing_directories.add(str(Path(dir_path)))
        return True


@pytest.fixture
def mock_logger():
    """Provide mock logger for testing"""
    return MockLogger()

@pytest.fixture
def mock_filesystem():
    """Provide mock filesystem for testing"""
    return MockFileSystem()

@pytest.fixture
def temp_file_structure(test_temp_dir):
    """Create temporary file structure for testing"""
    # Create test files and directories
    test_files = []
    test_dirs = []
    
    # Create some test files
    test_file_1 = test_temp_dir / "test_file.txt"
    test_file_1.write_text("Test content")
    test_files.append(test_file_1)
    
    # Create some test directories
    test_dir_1 = test_temp_dir / "test_directory"
    test_dir_1.mkdir()
    test_dirs.append(test_dir_1)
    
    # Create nested structure
    nested_dir = test_temp_dir / "parent" / "child"
    nested_dir.mkdir(parents=True)
    test_dirs.extend([test_temp_dir / "parent", nested_dir])
    
    nested_file = nested_dir / "nested_file.txt"
    nested_file.write_text("Nested content")
    test_files.append(nested_file)
    
    return {
        'base_dir': test_temp_dir,
        'files': test_files,
        'directories': test_dirs
    }

@pytest.fixture
def datetime_mock():
    """Mock datetime for consistent timestamp testing"""
    from unittest.mock import patch
    from datetime import datetime
    
    # Fixed datetime for consistent testing
    fixed_datetime = datetime(2024, 1, 15, 10, 30, 45, 123456)
    
    with patch('src.utils.common.datetime') as mock_datetime:
        mock_datetime.now.return_value = fixed_datetime
        yield mock_datetime