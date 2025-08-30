"""
Duplicate removal transformation
Single responsibility: Remove duplicate records based on configured columns
"""

import pandas as pd
import time
from typing import Dict, Any, List

from ...config.config_manager import ConfigManager
from ...utils.exceptions import NormalisationError
from ...utils.logging_config import get_logger


def duplicate_removal_transform(config_path: str, esi_normalisation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Duplicate removal transformation function
    Removes duplicate records based on configured check columns
    
    Args:
        config_path: Path to configuration file
        esi_normalisation_results: Results from ESI normalisation task
        
    Returns:
        Structured results with deduplicated DataFrames
    """
    logger = get_logger('light_transform.duplicate_removal')
    start_time = time.time()
    
    config = ConfigManager(config_path)
    duplicate_config = config.get_duplicate_removal_config()
    
    if not duplicate_config.get('enabled', True):
        logger.info("Duplicate removal disabled - passing through unchanged")
        return esi_normalisation_results
    
    check_columns = duplicate_config.get('duplicate_check_columns', ['name', 'esi_field'])
    strategy = duplicate_config.get('strategy', 'keep_first')
    case_sensitive = duplicate_config.get('case_sensitive_matching', False)
    
    # Transform all datasets
    transformed_data = {}
    records_processed = 0
    records_removed = 0
    
    for table_name, dataset in esi_normalisation_results['transformed_data'].items():
        df = dataset['dataframe']
        metadata = dataset['metadata']
        
        # Remove duplicates
        deduplicated_df, removed_count = _remove_duplicates_from_dataframe(
            df, check_columns, strategy, case_sensitive, table_name
        )
        
        transformed_data[table_name] = {
            'dataframe': deduplicated_df,
            'metadata': metadata
        }
        
        records_processed += len(df)
        records_removed += removed_count
    
    duration = time.time() - start_time
    
    logger.info(f"Duplicate removal completed: {records_removed} duplicates removed from {len(transformed_data)} datasets in {duration:.2f}s")
    
    return {
        'transformed_data': transformed_data,
        'transformation_summary': {
            'records_processed': records_processed,
            'records_changed': records_removed,
            'transformation_type': 'duplicate_removal',
            'failed_records': [],
            'datasets_processed': len(transformed_data)
        },
        'performance_metrics': {
            'transformation_duration': duration
        }
    }


def _remove_duplicates_from_dataframe(df: pd.DataFrame, 
                                    check_columns: List[str], 
                                    strategy: str,
                                    case_sensitive: bool,
                                    table_name: str) -> tuple:
    """Remove duplicates from DataFrame"""
    logger = get_logger('light_transform.duplicate_removal')
    
    if df.empty:
        return df, 0
    
    initial_count = len(df)
    
    # Validate check columns exist
    missing_columns = [col for col in check_columns if col not in df.columns]
    if missing_columns:
        raise NormalisationError(f"Duplicate check columns not found in {table_name}: {missing_columns}")
    
    # Prepare DataFrame for duplicate checking
    work_df = df.copy()
    
    # Handle case sensitivity
    if not case_sensitive:
        for col in check_columns:
            if work_df[col].dtype == 'object':  # String columns
                work_df[col] = work_df[col].astype(str).str.lower().str.strip()
    
    # Remove duplicates based on strategy
    if strategy == 'keep_first':
        deduplicated_df = work_df.drop_duplicates(subset=check_columns, keep='first')
    elif strategy == 'keep_last':
        deduplicated_df = work_df.drop_duplicates(subset=check_columns, keep='last')
    else:  # flag_all - remove all duplicates including originals
        duplicated_mask = work_df.duplicated(subset=check_columns, keep=False)
        deduplicated_df = work_df[~duplicated_mask]
    
    # Use original DataFrame with deduplicated indices to preserve original case
    final_df = df.iloc[deduplicated_df.index].reset_index(drop=True)
    
    removed_count = initial_count - len(final_df)
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} duplicates from {table_name}")
    
    return final_df, removed_count