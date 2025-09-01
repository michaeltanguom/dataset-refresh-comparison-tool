"""
Duplicate detection transformation
Single responsibility: Flag duplicate records and add metadata columns to DataFrames
"""

import pandas as pd
import time
from typing import Dict, Any, List, Tuple

from ...config.config_manager import ConfigManager
from ...utils.exceptions import NormalisationError
from ...utils.logging_config import get_logger


def duplicate_detection_transform(config_path: str, esi_normalisation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Duplicate detection transformation function
    Flags duplicate records and adds metadata columns to DataFrames
    
    Args:
        config_path: Path to configuration file
        esi_normalisation_results: Results from ESI normalisation task
        
    Returns:
        Structured results with duplicate-flagged DataFrames
    """
    logger = get_logger('light_transform.duplicate_detection')
    start_time = time.time()
    
    config = ConfigManager(config_path)
    duplicate_config = config.get_duplicate_detection_config()
    
    if not duplicate_config.get('enabled', True):
        logger.info("Duplicate detection disabled - passing through unchanged")
        return esi_normalisation_results
    
    check_columns = duplicate_config.get('duplicate_check_columns', ['name', 'esi_field'])
    case_sensitive = duplicate_config.get('case_sensitive_matching', False)
    
    # Transform all datasets with duplicate flagging
    transformed_data = {}
    records_processed = 0
    total_duplicates_flagged = 0
    duplicate_groups_found = 0
    
    for table_name, dataset in esi_normalisation_results['transformed_data'].items():
        df = dataset['dataframe']
        metadata = dataset['metadata']
        
        # Flag duplicates and get statistics
        flagged_df, duplicates_flagged, groups_found = _flag_duplicates_in_dataframe(
            df, check_columns, case_sensitive, table_name
        )
        
        transformed_data[table_name] = {
            'dataframe': flagged_df,
            'metadata': metadata
        }
        
        records_processed += len(df)
        total_duplicates_flagged += duplicates_flagged
        duplicate_groups_found += groups_found
    
    duration = time.time() - start_time
    
    # Log summary with appropriate level based on findings
    if total_duplicates_flagged > 0:
        logger.warning(f"Duplicate detection completed: {total_duplicates_flagged} duplicate records flagged across {duplicate_groups_found} groups in {len(transformed_data)} datasets")
        logger.warning("MANUAL ACTION REQUIRED: Review flagged duplicates in dashboard before proceeding to production")
        
        # Fail pipeline if configured
        if duplicate_config.get('fail_on_duplicates_found', False):
            raise NormalisationError(f"Pipeline halted: {total_duplicates_flagged} duplicates detected. Review duplicates before proceeding.")
    else:
        logger.info(f"Duplicate detection completed: No duplicates found in {len(transformed_data)} datasets in {duration:.2f}s")
    
    return {
        'transformed_data': transformed_data,
        'transformation_summary': {
            'records_processed': records_processed,
            'records_changed': 0,  # No records removed, only flagged
            'transformation_type': 'duplicate_detection',
            'failed_records': [],
            'datasets_processed': len(transformed_data),
            'duplicates_flagged': total_duplicates_flagged,
            'duplicate_groups': duplicate_groups_found
        },
        'performance_metrics': {
            'transformation_duration': duration
        }
    }


def _flag_duplicates_in_dataframe(df: pd.DataFrame, 
                                check_columns: List[str], 
                                case_sensitive: bool,
                                table_name: str) -> Tuple[pd.DataFrame, int, int]:
    """
    Flag duplicates in DataFrame and add metadata columns
    
    Returns:
        Tuple of (flagged_dataframe, number_of_duplicates_flagged, number_of_duplicate_groups)
    """
    logger = get_logger('light_transform.duplicate_removal')
    
    if df.empty:
        return df, 0, 0
    
    # Validate check columns exist
    missing_columns = [col for col in check_columns if col not in df.columns]
    if missing_columns:
        raise NormalisationError(f"Duplicate check columns not found in {table_name}: {missing_columns}")
    
    # Work with a copy to preserve original data
    work_df = df.copy()
    
    # Add duplicate detection columns
    work_df['is_duplicate'] = False
    work_df['duplicate_profile'] = ''
    work_df['duplicate_group_id'] = ''
    
    # Prepare comparison DataFrame for case sensitivity
    comparison_df = work_df.copy()
    if not case_sensitive:
        for col in check_columns:
            if comparison_df[col].dtype == 'object':  # String columns
                comparison_df[col] = comparison_df[col].astype(str).str.lower().str.strip()
    
    # Find duplicate groups
    duplicate_mask = comparison_df.duplicated(subset=check_columns, keep=False)
    duplicates_found = duplicate_mask.sum()
    
    # Initialise duplicate_groups variable outside the conditional block
    duplicate_groups = 0
    
    if duplicates_found > 0:
        # Create duplicate profiles and group IDs
        duplicate_records = comparison_df[duplicate_mask]
        
        # Group duplicates and assign group IDs
        group_id = 1
        
        for group_values, group_data in duplicate_records.groupby(check_columns):
            if len(group_data) > 1:  # Ensure it's actually a duplicate group
                # Create profile string
                profile_parts = []
                for i, col in enumerate(check_columns):
                    value = group_values[i] if isinstance(group_values, tuple) else group_values
                    profile_parts.append(f"{col}:{value}")
                profile_str = " | ".join(profile_parts)
                
                # Mark all records in this group
                group_indices = group_data.index
                work_df.loc[group_indices, 'is_duplicate'] = True
                work_df.loc[group_indices, 'duplicate_profile'] = profile_str
                work_df.loc[group_indices, 'duplicate_group_id'] = f"DUP_{group_id:04d}"
                
                duplicate_groups += 1
                group_id += 1
                
                # Log each duplicate group for manual review
                logger.warning(f"Duplicate group {group_id-1} in {table_name}: {len(group_data)} records with profile '{profile_str}'")
                
                # Log the actual duplicate records for easier identification
                for idx in group_data.index:
                    original_values = " | ".join([f"{col}:{df.loc[idx, col]}" for col in check_columns])
                    logger.warning(f"  - Row {idx}: {original_values}")
        
        logger.warning(f"Total duplicate summary for {table_name}: {duplicates_found} records across {duplicate_groups} groups require manual review")
    
    return work_df, duplicates_found, duplicate_groups