"""
ESI field normalisation transformation
Single responsibility: Standardise ESI field names to canonical format
"""

import pandas as pd
import time
from typing import Dict, Any

from ...config.config_manager import ConfigManager
from ...utils.exceptions import NormalisationError
from ...utils.logging_config import get_logger


def esi_normalisation_transform(config_path: str, column_mapping_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    ESI field normalisation transformation function
    Standardises ESI field names to canonical format
    
    Args:
        config_path: Path to configuration file
        column_mapping_results: Results from column mapping task
        
    Returns:
        Structured results with ESI-normalised DataFrames
    """
    logger = get_logger('light_transform.esi_normalisation')
    start_time = time.time()
    
    config = ConfigManager(config_path)
    esi_config = config.get_esi_normalisation_config()
    
    if not esi_config.get('enabled', True):
        logger.info("ESI normalisation disabled - passing through unchanged")
        return column_mapping_results
    
    canonical_mappings = esi_config.get('canonical_mappings', {})
    if not canonical_mappings:
        raise NormalisationError("ESI canonical mappings are empty")
    
    # Transform all datasets
    transformed_data = {}
    records_processed = 0
    fields_normalised = 0
    
    for table_name, dataset in column_mapping_results['transformed_data'].items():
        df = dataset['dataframe']
        metadata = dataset['metadata']
        
        # Normalise ESI fields
        normalised_df, field_count = _normalise_esi_fields_in_dataframe(df, canonical_mappings)
        
        transformed_data[table_name] = {
            'dataframe': normalised_df,
            'metadata': metadata
        }
        
        records_processed += len(normalised_df)
        fields_normalised += field_count
    
    duration = time.time() - start_time
    
    logger.info(f"ESI normalisation completed: {fields_normalised} fields normalised across {len(transformed_data)} datasets in {duration:.2f}s")
    
    return {
        'transformed_data': transformed_data,
        'transformation_summary': {
            'records_processed': records_processed,
            'records_changed': fields_normalised,
            'transformation_type': 'esi_normalisation',
            'failed_records': [],
            'datasets_processed': len(transformed_data)
        },
        'performance_metrics': {
            'transformation_duration': duration
        }
    }


def _normalise_esi_fields_in_dataframe(df: pd.DataFrame, canonical_mappings: Dict[str, str]) -> tuple:
    """Normalise ESI fields in DataFrame"""
    if df.empty or 'esi_field' not in df.columns:
        return df, 0
    
    result_df = df.copy()
    fields_changed = 0
    
    # Apply ESI field normalisation
    original_values = result_df['esi_field'].fillna('').astype(str)
    normalised_values = original_values.apply(lambda x: _normalise_single_esi_field(x, canonical_mappings))
    
    # Count actual changes
    changes_mask = original_values != normalised_values
    fields_changed = changes_mask.sum()
    
    # Update DataFrame
    result_df['esi_field'] = normalised_values
    
    return result_df, fields_changed


def _normalise_single_esi_field(field_value: str, canonical_mappings: Dict[str, str]) -> str:
    """Normalise a single ESI field value"""
    if not field_value or field_value.strip() == '':
        return field_value
    
    # Clean and normalise for lookup
    clean_field = field_value.strip().lower()
    
    # Direct lookup in canonical mappings
    if clean_field in canonical_mappings:
        return canonical_mappings[clean_field]
    
    # Return original if not found (keep_original strategy)
    return field_value.strip()