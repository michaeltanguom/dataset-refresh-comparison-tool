"""
Light transform validation
Single responsibility: Final validation of transformed data
"""

import pandas as pd
import time
from typing import Dict, Any, List

from ...config.config_manager import ConfigManager
from ...utils.exceptions import ValidationError
from ...utils.logging_config import get_logger


def light_transform_validation(config_path: str, null_handling_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Light transform validation function
    Performs final validation of all light transform steps
    
    Args:
        config_path: Path to configuration file
        null_handling_results: Results from NULL handling task
        
    Returns:
        Validated results ready for database loading
    """
    logger = get_logger('light_transform.validation')
    start_time = time.time()
    
    config = ConfigManager(config_path)
    validation_config = config.get_light_transform_validation_config()
    
    if not validation_config.get('enabled', True):
        logger.info("Light transform validation disabled - passing through unchanged")
        return null_handling_results
    
    checks = validation_config.get('checks', [])
    fail_on_error = validation_config.get('fail_on_validation_error', False)
    
    # Validate all datasets
    validated_data = {}
    records_processed = 0
    validation_errors = []
    
    for table_name, dataset in null_handling_results['transformed_data'].items():
        df = dataset['dataframe']
        metadata = dataset['metadata']
        
        # Run validation checks
        dataset_errors = _validate_dataset(df, table_name, checks, config)
        validation_errors.extend(dataset_errors)
        
        # Store validated dataset
        validated_data[table_name] = {
            'dataframe': df,
            'metadata': metadata
        }
        
        records_processed += len(df)
    
    # Handle validation errors
    if validation_errors and fail_on_error:
        error_summary = '; '.join(validation_errors[:5])  # First 5 errors
        if len(validation_errors) > 5:
            error_summary += f" (and {len(validation_errors) - 5} more errors)"
        raise ValidationError(f"Light transform validation failed: {error_summary}")
    
    duration = time.time() - start_time
    
    if validation_errors:
        logger.warning(f"Light transform validation completed with {len(validation_errors)} warnings in {duration:.2f}s")
    else:
        logger.info(f"Light transform validation passed: {len(validated_data)} datasets, {records_processed:,} records in {duration:.2f}s")
    
    return {
        'transformed_data': validated_data,
        'transformation_summary': {
            'records_processed': records_processed,
            'records_changed': 0,  # Validation doesn't change records
            'transformation_type': 'validation',
            'failed_records': validation_errors,
            'datasets_processed': len(validated_data)
        },
        'performance_metrics': {
            'transformation_duration': duration
        }
    }


def _validate_dataset(df: pd.DataFrame, 
                     table_name: str, 
                     checks: List[str], 
                     config: ConfigManager) -> List[str]:
    """Run validation checks on a single dataset"""
    errors = []
    
    for check in checks:
        if check == 'check_required_columns':
            errors.extend(_check_required_columns(df, table_name, config))
        elif check == 'check_data_types':
            errors.extend(_check_data_types(df, table_name))
        elif check == 'check_value_ranges':
            errors.extend(_check_value_ranges(df, table_name, config))
        elif check == 'check_null_constraints':
            errors.extend(_check_null_constraints(df, table_name, config))
        elif check == 'check_duplicate_constraints':
            errors.extend(_check_duplicate_constraints(df, table_name, config))
    
    return errors


def _check_required_columns(df: pd.DataFrame, table_name: str, config: ConfigManager) -> List[str]:
    """Check that all critical columns are present"""
    critical_columns = config.get_critical_columns()
    missing_columns = [col for col in critical_columns if col not in df.columns]
    
    if missing_columns:
        return [f"{table_name}: Missing critical columns: {missing_columns}"]
    
    return []


def _check_data_types(df: pd.DataFrame, table_name: str) -> List[str]:
    """Check basic data types are reasonable"""
    errors = []
    
    # Check numeric columns contain numbers
    numeric_columns = ['times_cited', 'highly_cited_papers', 'hot_papers', 'rank']
    for col in numeric_columns:
        if col in df.columns:
            # Try to convert to numeric and check for failures
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            non_numeric_count = numeric_series.isnull().sum() - df[col].isnull().sum()
            
            if non_numeric_count > 0:
                errors.append(f"{table_name}: Column '{col}' has {non_numeric_count} non-numeric values")
    
    return errors


def _check_value_ranges(df: pd.DataFrame, table_name: str, config: ConfigManager) -> List[str]:
    """Check values are within expected ranges"""
    errors = []
    validation_rules = config.get_validation_rules()
    
    for column, rule in validation_rules.items():
        if column not in df.columns:
            continue
        
        # Check numeric ranges
        if hasattr(rule, 'min_value') and rule.min_value is not None:
            numeric_data = pd.to_numeric(df[column], errors='coerce')
            below_min = numeric_data < rule.min_value
            if below_min.any():
                errors.append(f"{table_name}: {below_min.sum()} values in '{column}' below minimum {rule.min_value}")
        
        if hasattr(rule, 'max_value') and rule.max_value is not None:
            numeric_data = pd.to_numeric(df[column], errors='coerce')
            above_max = numeric_data > rule.max_value
            if above_max.any():
                errors.append(f"{table_name}: {above_max.sum()} values in '{column}' above maximum {rule.max_value}")
    
    return errors


def _check_null_constraints(df: pd.DataFrame, table_name: str, config: ConfigManager) -> List[str]:
    """Check NULL constraints"""
    errors = []
    critical_columns = config.get_critical_columns()
    
    for col in critical_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(f"{table_name}: Critical column '{col}' has {null_count} NULL values")
    
    return errors


def _check_duplicate_constraints(df: pd.DataFrame, table_name: str, config: ConfigManager) -> List[str]:
    """Check for unexpected duplicates"""
    duplicate_config = config.get_duplicate_removal_config()
    check_columns = duplicate_config.get('duplicate_check_columns', ['name', 'esi_field'])
    
    # Check columns exist
    missing_columns = [col for col in check_columns if col not in df.columns]
    if missing_columns:
        return [f"{table_name}: Duplicate check columns missing: {missing_columns}"]
    
    # Check for duplicates
    duplicate_mask = df.duplicated(subset=check_columns, keep=False)
    duplicate_count = duplicate_mask.sum()
    
    if duplicate_count > 0:
        return [f"{table_name}: Found {duplicate_count} duplicate records based on {check_columns}"]
    
    return []