"""
Light Transform Module
Handles data normalisation including column mapping and ESI field standardisation
Location: src/extract/light_transform.py
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

from ..config.config_manager import ConfigManager
from ..utils.exceptions import NormalisationError, ValidationError
from ..utils.logging_config import get_logger
from ..utils.common import (
    normalise_text, 
    get_sample_values
)

logger = get_logger('light_transform')


@dataclass
class TransformationMetadata:
    """Metadata about transformation applied to a dataset"""
    columns_mapped: Dict[str, str]
    normalisation_duration_seconds: float = 0.0
    esi_fields_normalised: int = 0
    transformation_timestamp: str = ""


class ESIFieldNormaliser:
    """
    ESI Field Normaliser - handles standardisation of ESI field names
    Integrated into DataNormaliser for clean architecture
    """
    
    # Canonical ESI field names (your exact list)
    CANONICAL_ESI_FIELDS = {
        'agricultural sciences': 'Agricultural Sciences',
        'biology_biochemistry': 'Biology Biochemistry', 
        'chemistry': 'Chemistry',
        'clinical medicine': 'Clinical Medicine',
        'computer science': 'Computer Science',
        'economics and business': 'Economics and Business',
        'engineering': 'Engineering',
        'environment_ecology': 'Environment Ecology',
        'geosciences': 'GeoSciences',
        'immunology': 'Immunology',
        'materials science': 'Materials Science',
        'microbiology': 'Microbiology',
        'molecular biology and genetics': 'Molecular Biology and Genetics',
        'neuroscience and behaviour': 'Neuroscience and Behaviour',
        'pharmacology and toxicology': 'Pharmacology and Toxicology',
        'physics': 'Physics',
        'plant and animal science': 'Plant and Animal Science',
        'psychiatry_psychology': 'Psychiatry Psychology',
        'social sciences': 'Social Sciences',
        'space science': 'Space Science'
    }
    
    @classmethod
    def normalise_esi_field(cls, field_name: str) -> str:
        """
        Normalise ESI field to canonical format
        
        Args:
            field_name: Raw ESI field name
            
        Returns:
            Canonical ESI field name
        """
        if not field_name:
            return ""
        
        # Clean and normalise input
        clean_field = field_name.strip().lower()
        
        # Direct lookup
        if clean_field in cls.CANONICAL_ESI_FIELDS:
            return cls.CANONICAL_ESI_FIELDS[clean_field]
        
        # If no exact match, return original with warning
        logger.warning(f"Unknown ESI field: '{field_name}' - keeping original")
        return field_name.strip()


class DataNormaliser:
    """
    Light Transform: Normalise column names, validate mappings, and standardise ESI fields
    Single responsibility: All data normalisation and transformation
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialise with configuration manager
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.logger = get_logger('normaliser')
        self.column_mapping = self.config.get_column_mapping()
        self.column_variants = self.config.get_column_mapping_variants()
        self.critical_columns = self.config.get_critical_columns()
        self.validation_rules = self.config.get_validation_rules()
        
        # ESI normalisation statistics
        self.esi_normalisation_stats = {
            'dataframes_processed': 0,
            'fields_normalised': 0,
            'normalisation_errors': 0
        }
        
        # Create comprehensive column lookup
        self._create_column_lookup()
        
    def _create_column_lookup(self) -> None:
        """Create comprehensive column lookup including variants"""
        self.column_lookup = {}
        
        # Add main mappings
        for target_col, expected_col in self.column_mapping.items():
            # Add exact match
            self.column_lookup[expected_col] = target_col
            
            # Add normalised version
            norm_expected = normalise_text(expected_col)
            self.column_lookup[norm_expected] = target_col
            
        # Add variants
        for target_col, variants in self.column_variants.items():
            for variant in variants:
                self.column_lookup[variant] = target_col
                norm_variant = normalise_text(variant)
                self.column_lookup[norm_variant] = target_col
        
        self.logger.info(f"Created column lookup with {len(self.column_lookup)} entries")
        
    def normalise_datasets(self, extracted_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        UPDATED: Normalise datasets with proper order: column mapping THEN ESI normalisation
        """
        normalised_data = {}
        
        self.logger.info(f"Starting normalisation of {len(extracted_data)} datasets")
        start_time = time.time()
        
        for table_name, dataset in extracted_data.items():
            dataset_start_time = time.time()
            
            try:
                df = dataset['dataframe']
                metadata = dataset['metadata']
                
                self.logger.info(f"Normalising dataset: {table_name}")
                self.logger.info(f"Original shape: {df.shape}")
                
                # STEP 1: Column normalisation/mapping (existing logic)
                normalised_df, mapping_applied = self._normalise_dataframe(df, metadata.sheet_name)
                
                # STEP 2: ESI field normalisation (AFTER column mapping)
                normalised_df, esi_fields_normalised = self._normalise_esi_fields_in_dataframe(
                    normalised_df, table_name
                )
                
                # STEP 3: Validate the transformation
                self._validate_transformation(df, normalised_df, mapping_applied, metadata.sheet_name)
                
                # Update metadata with transformation information
                dataset_duration = time.time() - dataset_start_time
                
                # Create transformation metadata
                transform_metadata = TransformationMetadata(
                    columns_mapped=mapping_applied,
                    normalisation_duration_seconds=dataset_duration,
                    esi_fields_normalised=esi_fields_normalised,
                    transformation_timestamp=datetime.now().isoformat()
                )
                
                normalised_data[table_name] = {
                    'dataframe': normalised_df,
                    'metadata': metadata,
                    'transformation_metadata': transform_metadata
                }
                
                self.logger.info(f"Successfully normalised {table_name} in {dataset_duration:.2f}s")
                self.logger.info(f"Final shape: {normalised_df.shape}")
                if esi_fields_normalised > 0:
                    self.logger.info(f"ESI fields normalised: {esi_fields_normalised}")
                
            except Exception as e:
                self.logger.error(f"Failed to normalise {table_name}: {e}")
                raise NormalisationError(f"Normalisation failed for {table_name}: {e}", table_name)
        
        total_duration = time.time() - start_time
        self.logger.info(f"Normalisation complete for {len(normalised_data)} datasets in {total_duration:.2f}s")
        
        # Log ESI normalisation summary
        self.logger.info(f"🔧 ESI Normalisation Summary:")
        self.logger.info(f"  - DataFrames processed: {self.esi_normalisation_stats['dataframes_processed']}")
        self.logger.info(f"  - Fields normalised: {self.esi_normalisation_stats['fields_normalised']}")
        self.logger.info(f"  - Normalisation errors: {self.esi_normalisation_stats['normalisation_errors']}")
        
        return normalised_data
    
    def _normalise_esi_fields_in_dataframe(self, df: pd.DataFrame, dataframe_name: str) -> Tuple[pd.DataFrame, int]:
        """
        SIMPLIFIED: Normalise ESI field names in already-mapped DataFrame
        
        Args:
            df: DataFrame with already-mapped column names
            dataframe_name: Name of dataframe for logging purposes
            
        Returns:
            Tuple of (normalised_dataframe, number_of_fields_normalised)
        """
        if df.empty:
            return df, 0
        
        self.logger.debug(f"🔧 Checking for ESI fields in already-mapped DataFrame: {dataframe_name}")
        
        try:
            # Create copy to avoid modifying original
            normalised_df = df.copy()
            fields_normalised_count = 0
            
            # Find ESI field columns (should be exact match after column mapping)
            esi_columns = self._find_esi_field_columns(normalised_df)
            
            if not esi_columns:
                self.logger.debug(f"No ESI field columns found in {dataframe_name} - skipping ESI normalisation")
                return normalised_df, 0
            
            # Normalise each ESI field column
            for col in esi_columns:
                self.logger.debug(f"Processing ESI field column: {col}")
                
                # Handle NaN values and convert to string
                original_values = normalised_df[col].fillna('').astype(str)
                
                # Apply normalisation using ESIFieldNormaliser class
                normalised_values = original_values.apply(
                    lambda x: ESIFieldNormaliser.normalise_esi_field(x) if x.strip() else x
                )
                
                # Count actual changes
                changes_mask = (original_values != normalised_values) & (original_values.str.strip() != '')
                changes_count = changes_mask.sum()
                fields_normalised_count += changes_count
                
                # Log specific changes for debugging (only if there are changes)
                if changes_count > 0:
                    changed_records = normalised_df.loc[changes_mask, [col]].copy()
                    changed_records['original'] = original_values[changes_mask]
                    changed_records['normalised'] = normalised_values[changes_mask]
                    
                    self.logger.debug(f"ESI field changes in {col}:")
                    for idx, row in changed_records.head(3).iterrows():  # Show first 3 changes
                        self.logger.debug(f"  '{row['original']}' → '{row['normalised']}'")
                    
                    if changes_count > 3:
                        self.logger.debug(f"  ... and {changes_count - 3} more changes")
                    
                    # Update the DataFrame
                    normalised_df[col] = normalised_values
                else:
                    self.logger.debug(f"No changes needed in ESI field column: {col}")
            
            # Update statistics
            self.esi_normalisation_stats['dataframes_processed'] += 1
            self.esi_normalisation_stats['fields_normalised'] += fields_normalised_count
            
            if fields_normalised_count > 0:
                self.logger.debug(f"✅ Normalised {fields_normalised_count} ESI field values in {dataframe_name}")
            
            return normalised_df, fields_normalised_count
            
        except Exception as e:
            self.logger.error(f"❌ ESI normalisation failed for {dataframe_name}: {e}")
            self.esi_normalisation_stats['normalisation_errors'] += 1
            return df, 0  # Return original DataFrame if normalisation fails
    
    def _find_esi_field_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Look for exact standardised column name only
        This runs AFTER column mapping, so we only need to check the mapped column name
        """
        # After column mapping, ESI field should be in one of these exact standard names
        standard_esi_column_names = [
            'esi_field',          # Primary standard name (snake_case)
            'ESI Field',          # Alternative standard name
        ]
        
        found_columns = []
        
        for col in df.columns:
            if col in standard_esi_column_names:
                found_columns.append(col)
                self.logger.debug(f"Found mapped ESI field column: {col}")
        
        if found_columns:
            self.logger.debug(f"ESI field columns found: {found_columns}")
        else:
            self.logger.debug(f"No ESI field columns found. Available columns: {list(df.columns)}")
            # Check if there's any column that might be ESI-related for debugging
            potential_esi = [col for col in df.columns if 'esi' in col.lower() or 'field' in col.lower()]
            if potential_esi:
                self.logger.debug(f"Potential ESI-related columns (not exact match): {potential_esi}")
        
        return found_columns
    
    def get_normalisation_summary(self) -> Dict[str, Any]:
        """Get summary of all normalisation activity"""
        return {
            **self.esi_normalisation_stats,
            'canonical_esi_fields': list(ESIFieldNormaliser.CANONICAL_ESI_FIELDS.values()),
            'total_canonical_fields': len(ESIFieldNormaliser.CANONICAL_ESI_FIELDS),
            'normalisation_timestamp': datetime.now().isoformat()
        }

    def _normalise_dataframe(self, df: pd.DataFrame, sheet_name: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Normalise a single DataFrame's column names
        
        Args:
            df: DataFrame to normalise (should already be cleaned in extraction phase)
            sheet_name: Name of sheet for logging
            
        Returns:
            Tuple of (normalised_dataframe, mapping_applied)
        """
        self.logger.info(f"Normalising columns for sheet: {sheet_name}")
        self.logger.info(f"Original columns: {list(df.columns)}")
        
        # Step 1: Map columns using lookup table
        normalised_df = pd.DataFrame()
        mapping_applied = {}
        unmapped_columns = []
        
        for original_col in df.columns:
            # Try exact match first
            if original_col in self.column_lookup:
                target_col = self.column_lookup[original_col]
                normalised_df[target_col] = df[original_col]
                mapping_applied[target_col] = original_col
                self.logger.info(f"Mapped '{original_col}' -> '{target_col}' (exact match)")
                
            else:
                # Try normalised matching
                norm_original = normalise_text(original_col)
                if norm_original in self.column_lookup:
                    target_col = self.column_lookup[norm_original]
                    normalised_df[target_col] = df[original_col]
                    mapping_applied[target_col] = original_col
                    self.logger.info(f"Mapped '{original_col}' -> '{target_col}' (normalised match)")
                    
                else:
                    unmapped_columns.append(original_col)
                    self.logger.warning(f"Unknown column '{original_col}' (normalised: '{norm_original}') - ignored")
        
        # Log mapping results
        self.logger.info(f"Mapped {len(mapping_applied)} columns, ignored {len(unmapped_columns)} columns")
        if unmapped_columns:
            self.logger.info(f"Unmapped columns: {unmapped_columns}")
        
        # Step 2: Check for missing critical columns
        missing_critical = set(self.critical_columns) - set(normalised_df.columns)
        if missing_critical:
            raise NormalisationError(
                f"Missing critical columns in '{sheet_name}': {missing_critical}",
                sheet_name,
                list(missing_critical)
            )
        
        # Step 3: Add missing optional columns with defaults
        all_expected_columns = set(self.column_mapping.keys())
        missing_optional = all_expected_columns - set(normalised_df.columns) - missing_critical
        
        for target_col in missing_optional:
            normalised_df[target_col] = ""  # Default empty value
            self.logger.info(f"Added missing optional column '{target_col}' with default value")
        
        # Step 4: Ensure consistent column order
        column_order = list(self.column_mapping.keys())
        normalised_df = normalised_df.reindex(columns=column_order)
        
        self.logger.info(f"Final normalised columns: {list(normalised_df.columns)}")
        self.logger.info(f"Final row count: {len(normalised_df)}")
        
        return normalised_df, mapping_applied
    
    def _validate_transformation(self, original_df: pd.DataFrame, normalised_df: pd.DataFrame, 
                               mapping_applied: Dict[str, str], sheet_name: str) -> None:
        """
        Validate that the transformation makes sense
        Uses sample-based validation with both config rules and basic sanity checks
        
        Args:
            original_df: Original DataFrame
            normalised_df: Normalised DataFrame
            mapping_applied: Dictionary of applied mappings
            sheet_name: Sheet name for error messages
        """
        self.logger.info(f"Validating transformation for '{sheet_name}'")
        self.logger.info(f"Applied mapping: {mapping_applied}")
        
        if len(normalised_df) == 0:
            raise ValidationError(f"Normalised DataFrame for '{sheet_name}' is empty")
        
        # Collect all validation issues
        all_errors = []
        all_warnings = []
        
        for critical_col in self.critical_columns:
            if critical_col in mapping_applied and critical_col in normalised_df.columns:
                try:
                    # Get sample values for validation
                    sample_values = get_sample_values(normalised_df, critical_col, n_samples=3)
                    
                    if sample_values:
                        # Log sample for audit trail
                        self.logger.info(f"Sample values for {critical_col}: {sample_values}")
                        
                        # Validate using unified approach
                        errors, warnings = self._validate_sample_values(critical_col, sample_values, sheet_name)
                        all_errors.extend(errors)
                        all_warnings.extend(warnings)
                        
                    else:
                        all_warnings.append(f"No sample values found for {critical_col} in '{sheet_name}'")
                        
                except Exception as e:
                    all_warnings.append(f"Error validating {critical_col} in '{sheet_name}': {e}")
        
        # Log all warnings
        for warning in all_warnings:
            self.logger.warning(warning)
        
        # Fail if there are validation errors
        if all_errors:
            error_msg = f"Validation failed for '{sheet_name}': {'; '.join(all_errors)}"
            self.logger.error(error_msg)
            raise ValidationError(error_msg)
        
        self.logger.info(f"Transformation validation completed for '{sheet_name}' - {len(all_warnings)} warnings")
    
    def _validate_sample_values(self, column: str, sample_values: List[Any], sheet_name: str) -> Tuple[List[str], List[str]]:
        """
        Unified validation approach: config rules (errors) + basic sanity checks (warnings)
        
        Args:
            column: Column name
            sample_values: List of sample values to validate
            sheet_name: Sheet name for error messages
            
        Returns:
            Tuple of (error_messages, warning_messages)
        """
        errors = []
        warnings = []
        
        # Primary sample value for strict validation
        primary_sample = sample_values[0] if sample_values else None
        
        if primary_sample is None:
            warnings.append(f"No sample value available for {column}")
            return errors, warnings
        
        # 1. CONFIG-DRIVEN VALIDATION RULES (These cause pipeline failure)
        if column in self.validation_rules:
            rule_errors = self._apply_config_validation_rule(column, primary_sample, sheet_name)
            errors.extend(rule_errors)
        
        # 2. BASIC SANITY CHECKS (These cause warnings only)
        sanity_warnings = self._apply_basic_sanity_checks(column, sample_values, sheet_name)
        warnings.extend(sanity_warnings)
        
        return errors, warnings
    
    def _apply_config_validation_rule(self, column: str, sample_value: Any, sheet_name: str) -> List[str]:
        """
        Apply config-driven validation rules (strict - causes pipeline failure)
        
        Args:
            column: Column name
            sample_value: Sample value to validate
            sheet_name: Sheet name for error messages
            
        Returns:
            List of validation error messages
        """
        errors = []
        rule = self.validation_rules[column]
        
        try:
            # Numeric range validation
            if rule.min_value is not None or rule.max_value is not None:
                try:
                    numeric_value = float(sample_value)
                    
                    if rule.min_value is not None and numeric_value < rule.min_value:
                        errors.append(
                            f"Column '{column}' in '{sheet_name}': "
                            f"value {numeric_value} below minimum {rule.min_value}"
                        )
                    
                    if rule.max_value is not None and numeric_value > rule.max_value:
                        errors.append(
                            f"Column '{column}' in '{sheet_name}': "
                            f"value {numeric_value} above maximum {rule.max_value}"
                        )
                        
                except (ValueError, TypeError):
                    errors.append(
                        f"Column '{column}' in '{sheet_name}': "
                        f"cannot convert '{sample_value}' to numeric (expected numeric type)"
                    )
            
            # String length validation
            if rule.min_length is not None:
                str_value = str(sample_value).strip()
                if len(str_value) < rule.min_length:
                    errors.append(
                        f"Column '{column}' in '{sheet_name}': "
                        f"value '{str_value}' length {len(str_value)} below minimum {rule.min_length}"
                    )
            
            # Required field validation
            if rule.required:
                if sample_value is None or str(sample_value).strip() == '':
                    errors.append(
                        f"Column '{column}' in '{sheet_name}': "
                        f"required field is empty"
                    )
                        
        except Exception as e:
            errors.append(f"Error applying config validation rule for '{column}': {e}")
            
        return errors
    
    def _apply_basic_sanity_checks(self, column: str, sample_values: List[Any], sheet_name: str) -> List[str]:
        """
        Apply basic sanity checks (generates warnings only)
        
        Args:
            column: Column name
            sample_values: List of sample values
            sheet_name: Sheet name for warnings
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        try:
            # Check numeric fields for common issues
            if column in ['times_cited', 'highly_cited_papers', 'hot_papers', 'indicative_cross_field_score']:
                for i, value in enumerate(sample_values[:3]):  # Check first 3 samples
                    try:
                        numeric_value = float(value)
                        
                        # Negative values warning
                        if numeric_value < 0:
                            warnings.append(f"Negative value in {column} sample {i+1}: {numeric_value}")
                        
                        # Suspiciously high values
                        if column in ['times_cited'] and numeric_value > 500000:
                            warnings.append(f"Very high times_cited in sample {i+1}: {numeric_value}")
                        elif column in ['highly_cited_papers'] and numeric_value > 500:
                            warnings.append(f"Very high highly_cited_papers in sample {i+1}: {numeric_value}")
                        elif column in ['hot_papers'] and numeric_value > 200:
                            warnings.append(f"Very high hot_papers in sample {i+1}: {numeric_value}")
                        elif column in ['indicative_cross_field_score'] and numeric_value > 20:
                            warnings.append(f"Very high cross_field_score in sample {i+1}: {numeric_value}")
                            
                    except (ValueError, TypeError):
                        warnings.append(f"Non-numeric value in {column} sample {i+1}: '{value}'")
            
            # Check string fields for common issues
            elif column in ['name', 'esi_field', 'affiliation']:
                for i, value in enumerate(sample_values[:3]):
                    str_value = str(value).strip()
                    
                    # Empty or null-like values
                    if not str_value or str_value.lower() in ['nan', 'none', 'null', 'n/a']:
                        warnings.append(f"Empty/null-like value in {column} sample {i+1}: '{value}'")
                    
                    # Purely numeric values in text fields
                    elif str_value.isdigit():
                        warnings.append(f"Purely numeric value in text field {column} sample {i+1}: '{value}'")
                    
                    # Very short values in key fields
                    elif column == 'name' and len(str_value) < 3:
                        warnings.append(f"Very short name in sample {i+1}: '{str_value}'")
                        
        except Exception as e:
            warnings.append(f"Error in sanity checks for {column}: {e}")
            
        return warnings