"""
Common utilities for the dataset comparison pipeline
Shared functions used across multiple modules
"""

import re
import pandas as pd
from typing import Any, Optional
from datetime import datetime


def normalise_text(text: str) -> str:
    """
    Normalise text to snake_case format
    
    Args:
        text: Input text to normalise
        
    Returns:
        Normalised text in snake_case format
        
    Examples:
        >>> normalise_text("Highly Cited Papers")
        'highly_cited_papers'
        >>> normalise_text("ESI Field")
        'esi_field'
    """
    if not text or pd.isna(text):
        return ""
    
    # Convert to string and basic normalisation
    normalised = str(text).strip().lower()
    
    # Replace spaces and special chars with underscores
    normalised = re.sub(r'[^a-z0-9]+', '_', normalised)
    
    # Clean up underscores - remove leading/trailing and collapse multiple
    normalised = re.sub(r'_+', '_', normalised).strip('_')
    
    return normalised


def generate_timestamp() -> str:
    """
    Generate ISO format timestamp for metadata
    
    Returns:
        Current timestamp in ISO format
    """
    return datetime.now().isoformat()


def safe_convert_numeric(value: Any, target_type: type, default: Optional[Any] = None) -> Any:
    """
    Safely convert value to numeric type with fallback
    
    Args:
        value: Value to convert
        target_type: Target type (int or float)
        default: Default value if conversion fails
        
    Returns:
        Converted value or default
    """
    if pd.isna(value):
        return default if default is not None else (0 if target_type == int else 0.0)
    
    try:
        if target_type == int:
            return int(float(value))  # Convert via float first to handle strings like "123.0"
        elif target_type == float:
            return float(value)
        else:
            return value
    except (ValueError, TypeError):
        return default if default is not None else (0 if target_type == int else 0.0)


def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """
    Validate that a file path is valid and optionally exists
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        
    Returns:
        True if valid, False otherwise
    """
    from pathlib import Path
    
    try:
        path = Path(file_path)
        if must_exist:
            return path.exists()
        else:
            # Check if parent directory exists
            return path.parent.exists()
    except Exception:
        return False


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame column names (strip whitespace, handle None)
    
    Args:
        df: DataFrame to clean
        
    Returns:
        DataFrame with cleaned column names
    """
    if df.empty:
        return df
    
    # Clean column names
    cleaned_columns = []
    for col in df.columns:
        if col is None:
            cleaned_columns.append('unnamed_column')
        else:
            cleaned_columns.append(str(col).strip())
    
    df.columns = cleaned_columns
    return df


def get_sample_values(df: pd.DataFrame, column: str, n_samples: int = 3) -> list:
    """
    Get sample non-null values from a DataFrame column
    
    Args:
        df: DataFrame to sample from
        column: Column name
        n_samples: Number of samples to return
        
    Returns:
        List of sample values
    """
    if column not in df.columns or df.empty:
        return []
    
    # Get non-null values
    non_null_values = df[column].dropna()
    
    if non_null_values.empty:
        return []
    
    # Return up to n_samples
    return non_null_values.head(n_samples).tolist()


def log_dataframe_info(df: pd.DataFrame, name: str, logger) -> None:
    """
    Log basic information about a DataFrame
    
    Args:
        df: DataFrame to log info about
        name: Name identifier for logging
        logger: Logger instance
    """
    if df.empty:
        logger.warning(f"DataFrame '{name}' is empty")
        return
    
    logger.info(f"DataFrame '{name}' info:")
    logger.info(f"  - Shape: {df.shape}")
    logger.info(f"  - Columns: {list(df.columns)}")
    logger.info(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Log sample of data
    if len(df) > 0:
        logger.info(f"  - Sample first row: {df.iloc[0].to_dict()}")


def create_directory_if_not_exists(directory_path: str) -> bool:
    """
    Create directory if it doesn't exist
    
    Args:
        directory_path: Path to directory
        
    Returns:
        True if successful, False otherwise
    """
    from pathlib import Path
    import os
    
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        return False


def format_number_with_commas(number: int) -> str:
    """
    Format large numbers with comma separators
    
    Args:
        number: Number to format
        
    Returns:
        Formatted number string
        
    Examples:
        >>> format_number_with_commas(1234567)
        '1,234,567'
    """
    return f"{number:,}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change (e.g., 0.1 for 10% increase)
    """
    if old_value == 0:
        return float('inf') if new_value > 0 else 0
    
    return (new_value - old_value) / old_value


def is_valid_esi_field(esi_field: str) -> bool:
    """
    Validate that ESI field is a reasonable value
    
    Args:
        esi_field: ESI field value to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not esi_field or pd.isna(esi_field):
        return False
    
    esi_field = str(esi_field).strip()
    
    # Must be at least 2 characters and not obviously invalid
    if len(esi_field) < 2:
        return False
    
    # Should not be numeric only or common invalid values
    invalid_values = {'nan', 'none', 'null', '', 'n/a', 'unknown'}
    if esi_field.lower() in invalid_values:
        return False
    
    # Should not be purely numeric
    if esi_field.isdigit():
        return False
    
    return True
