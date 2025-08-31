"""
NULL handling transformation
Single responsibility: Handle NULL values based on configured strategy
"""

import pandas as pd
import time
from typing import Dict, Any, List

from ...config.config_manager import ConfigManager
from ...utils.exceptions import NormalisationError
from ...utils.logging_config import get_logger


def null_handling_transform(config_path: str, duplicate_removal_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    NULL handling transformation function
    Handles NULL values based on configured strategy
    
    Args:
        config_path: Path to configuration file
        duplicate_removal_results: Results from duplicate removal task
        
    Returns:
        Structured results with NULL-handled DataFrames
    """
    logger = get_logger('light_transform.null_handling')
    start_time = time.time()
    
    config = ConfigManager(config_path)
    null_config = config.get_null_handling_config()
    
    strategy = null_config.get('strategy')
    default_values = null_config.get('default_values', {})
    critical_fields = null_config.get('critical_fields_never_null', ['name', 'esi_field'])
    
    # Transform all datasets
    transformed_data = {}
    records_processed = 0
    records_changed = 0
    
    for table_name, dataset in duplicate_removal_results['transformed_data'].items():
        df = dataset['dataframe']
        metadata = dataset['metadata']
        
        # Handle NULLs
        processed_df, changed_count = _handle_nulls_in_dataframe(
            df, strategy, default_values, critical_fields, table_name
        )
        
        transformed_data[table_name] = {
            'dataframe': processed_df,
            'metadata': metadata
        }
        
        records_processed += len(df)
        records_changed += changed_count
    
    duration = time.time() - start_time
    
    logger.info(f"NULL handling completed: {records_changed} NULL values processed across {len(transformed_data)} datasets in {duration:.2f}s")
    
    return {
        'transformed_data': transformed_data,
        'transformation_summary': {
            'records_processed': records_processed,
            'records_changed': records_changed,
            'transformation_type': 'null_handling',
            'failed_records': [],
            'datasets_processed': len(transformed_data)
        },
        'performance_metrics': {
            'transformation_duration': duration
        }
    }


def _handle_nulls_in_dataframe(df: pd.DataFrame,
                              strategy: str,
                              default_values: Dict[str, Any],
                              critical_fields: List[str],
                              table_name: str) -> tuple:
    """Handle NULL values in DataFrame"""
    logger = get_logger('light_transform.null_handling')
    
    if df.empty:
        return df, 0
    
    # Check critical fields first (these should never be NULL regardless of strategy)
    _validate_critical_fields_not_null(df, critical_fields, table_name)
    
    result_df = df.copy()
    changes_count = 0
    
    if strategy == 'fail':
        # Check for any NULL values and fail if found
        null_counts = result_df.isnull().sum()
        null_columns = null_counts[null_counts > 0]
        
        if not null_columns.empty:
            raise NormalisationError(f"NULL values found in {table_name}: {dict(null_columns)}")
    
    elif strategy == 'skip':
        # Remove rows with ANY NULL values
        initial_count = len(result_df)
        result_df = result_df.dropna()
        changes_count = initial_count - len(result_df)
        
        if changes_count > 0:
            logger.info(f"Removed {changes_count} rows with NULL values from {table_name}")
    
    elif strategy == 'default':
        # Fill NULLs with default values
        null_mask = result_df.isnull()
        
        for column in result_df.columns:
            if column in default_values and null_mask[column].any():
                null_count = null_mask[column].sum()
                result_df[column] = result_df[column].fillna(default_values[column])
                changes_count += null_count
                logger.debug(f"Filled {null_count} NULL values in {column} with default: {default_values[column]}")
        
        # Check for remaining NULLs in non-default columns
        remaining_nulls = result_df.isnull().sum()
        remaining_null_columns = remaining_nulls[remaining_nulls > 0]
        
        if not remaining_null_columns.empty:
            logger.warning(f"Remaining NULL values in {table_name} (no defaults configured): {dict(remaining_null_columns)}")
    
    return result_df, changes_count


def _validate_critical_fields_not_null(df: pd.DataFrame, critical_fields: List[str], table_name: str) -> None:
    """Validate that critical fields have no NULL values"""
    for field in critical_fields:
        if field in df.columns:
            null_count = df[field].isnull().sum()
            if null_count > 0:
                raise NormalisationError(f"Critical field '{field}' has {null_count} NULL values in {table_name}")