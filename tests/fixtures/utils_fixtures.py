"""
Sample data fixtures for testing utilities
Provides test DataFrames and data structures
"""
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, List


@pytest.fixture
def sample_dataframe():
    """Provide a standard sample DataFrame"""
    return pd.DataFrame({
        'Name': ['Dr. John Smith', 'Prof. Jane Doe', 'Dr. Bob Wilson'],
        'Times Cited': [1250, 890, 2100],
        'Highly Cited Papers': [15, 8, 25],
        'ESI Field': ['Engineering', 'Mathematics', 'Computer Science'],
        '% Docs Cited': [85.5, 72.3, 91.2],
        'Indicative Cross-Field Score': [1.2, 0.8, 1.8]
    })

@pytest.fixture
def dataframe_with_nulls():
    """Provide DataFrame with null values"""
    return pd.DataFrame({
        'name': ['Alice', None, 'Charlie'],
        'value': [10, 20, None],
        'category': ['A', 'B', 'C']
    })

@pytest.fixture
def empty_dataframe():
    """Provide empty DataFrame"""
    return pd.DataFrame()

@pytest.fixture
def mock_logger():
    """Provide mock logger for testing"""
    from unittest.mock import Mock
    
    logger = Mock()
    logger.info_calls = []
    logger.warning_calls = []
    
    def mock_info(message, *args, **kwargs):
        logger.info_calls.append((message, args, kwargs))
    
    def mock_warning(message, *args, **kwargs):
        logger.warning_calls.append((message, args, kwargs))
    
    def has_logged(level, substring):
        calls = getattr(logger, f'{level}_calls', [])
        return any(substring in str(call[0]) for call in calls)
    
    logger.info.side_effect = mock_info
    logger.warning.side_effect = mock_warning
    logger.has_logged = has_logged
    
    return logger

@pytest.fixture
def percentage_test_cases():
    """Provide test cases for percentage calculations"""
    return [
        {'old': 100, 'new': 150, 'expected': 0.5},  # 50% increase
        {'old': 100, 'new': 50, 'expected': -0.5},   # 50% decrease
        {'old': 100, 'new': 100, 'expected': 0.0},   # No change
        {'old': 0, 'new': 100, 'expected': float('inf')},  # From zero
        {'old': 0, 'new': 0, 'expected': 0.0},       # Both zero
        {'old': 50, 'new': 0, 'expected': -1.0},     # To zero
        {'old': -100, 'new': -50, 'expected': -0.5},  # Negative values: (-50-(-100))/-100 = -0.5
    ]