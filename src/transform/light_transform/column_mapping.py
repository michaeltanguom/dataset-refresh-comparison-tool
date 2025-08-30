"""
Column mapping transformation
Single responsibility: Map Excel column names to standardised column names
"""

import pandas as pd
import time
from typing import Dict, List, Any, Tuple

from ...config.config_manager import ConfigManager
from ...utils.exceptions import NormalisationError
from ...utils.logging_config import get_logger
from ...utils.common import normalise_text


def column_mapping_transform(config_path: str, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Column mapping transformation function
    Maps Excel column names to standardised column names based on configuration
    
    Args:
        config_path: Path to configuration file
        extraction_results: Results from extraction task
        
    Returns:
        Structured results with column-mapped DataFrames
    """
    logger = get_logger('light_transform.column_mapping')
    start_time = time.time()
    
    config = ConfigManager(config_path)
    column_mapping = config.get_light_transform_column_mapping()
    
    if not column_mapping:
        raise NormalisationError("Column mapping configuration is empty")
    
    # Create lookup table for flexible matching
    column_lookup = _create_column_lookup(column_mapping)
    
    # Transform all datasets
    transformed_data = {}
    records_processed = 0
    
    for table_name, dataset in extraction_results['extracted_data'].items():
        df = dataset['dataframe']
        metadata = dataset['metadata']
        
        # Map columns
        mapped_df = _map_dataframe_columns(df, column_lookup, table_name)
        
        # Ensure all expected columns exist
        final_df = _add_missing_columns(mapped_df, column_mapping)
        
        transformed_data[table_name] = {
            'dataframe': final_df,
            'metadata': metadata
        }
        
        records_processed += len(final_df)
    
    duration = time.time() - start_time
    
    logger.info(f"Column mapping completed: {len(transformed_data)} datasets, {records_processed:,} records in {duration:.2f}s")
    
    return {
        'transformed_data': transformed_data,
        'transformation_summary': {
            'records_processed': records_processed,
            'records_changed': records_processed,
            'transformation_type': 'column_mapping',
            'failed_records': [],
            'datasets_processed': len(transformed_data)
        },
        'performance_metrics': {
            'transformation_duration': duration
        }
    }


def _create_column_lookup(column_mapping: Dict[str, str]) -> Dict[str, str]:
    """Create flexible column lookup table"""
    lookup = {}
    
    for target_col, source_col in column_mapping.items():
        # Add exact matches and common variations
        lookup[source_col] = target_col
        lookup[source_col.upper()] = target_col
        lookup[source_col.lower()] = target_col
        lookup[source_col.title()] = target_col
        lookup[normalise_text(source_col)] = target_col
    
    return lookup


def _map_dataframe_columns(df: pd.DataFrame, column_lookup: Dict[str, str], table_name: str) -> pd.DataFrame:
    """Map DataFrame columns using lookup table"""
    if df.empty:
        return df
    
    mapped_columns = {}
    unmapped_columns = []
    
    for original_col in df.columns:
        clean_col = str(original_col).strip()
        
        if clean_col in column_lookup:
            mapped_columns[original_col] = column_lookup[clean_col]
        else:
            unmapped_columns.append(clean_col)
    
    if unmapped_columns:
        logger = get_logger('light_transform.column_mapping')
        logger.warning(f"Unmapped columns in {table_name}: {unmapped_columns}")
    
    # Rename columns and select only mapped ones
    return df.rename(columns=mapped_columns)[list(mapped_columns.values())]


def _add_missing_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """Add any missing expected columns with default values"""
    expected_columns = set(column_mapping.keys())
    existing_columns = set(df.columns)
    missing_columns = expected_columns - existing_columns
    
    result_df = df.copy()
    
    for col in missing_columns:
        # Add appropriate default value based on column name
        if any(keyword in col for keyword in ['cited', 'papers', 'documents', 'rank']):
            result_df[col] = 0
        elif any(keyword in col for keyword in ['percent', 'score', 'impact']):
            result_df[col] = 0.0
        else:
            result_df[col] = ""
    
    # Return columns in consistent order
    return result_df.reindex(columns=list(column_mapping.keys()))