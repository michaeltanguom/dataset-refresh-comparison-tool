"""
Extract and Light Transform for Dataset Comparison Pipeline
Handles data extraction from various sources and normalisation of column names
"""

import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

from ..config.config_manager import ConfigManager
from ..utils.exceptions import ExtractionError, NormalisationError, ValidationError
from ..utils.logging_config import get_logger
from ..utils.common import (
    normalise_text, 
    generate_timestamp, 
    clean_dataframe_columns, 
    get_sample_values,
    log_dataframe_info
)

logger = get_logger('extract')


@dataclass
class DatasetMetadata:
    """Metadata about a processed dataset"""
    source_file: str
    subject: str
    period: str
    sheet_name: str
    normalised_sheet_name: str
    table_name: str
    row_count: int
    columns_mapped: Dict[str, str]
    processing_timestamp: str
    extraction_duration_seconds: float
    normalisation_duration_seconds: float = 0.0


class DataExtractor(ABC):
    """
    Abstract base class for data extraction
    Enables extension to different file formats (Excel, JSON, CSV, etc.)
    Single responsibility: Extract raw data from files
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialise with configuration manager
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.logger = get_logger(f'extract.{self.__class__.__name__}')
        
    @abstractmethod
    def extract_files(self, folder_path: str, period_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract files from folder and return DataFrames with metadata
        
        Args:
            folder_path: Path to folder containing files
            period_name: Period identifier (e.g., 'feb', 'july')
            
        Returns:
            Dict mapping table names to {'dataframe': df, 'metadata': metadata}
        """
        pass
    
    def generate_table_name(self, subject: str, period_name: str, sheet_name: str) -> str:
        """Generate standardised table name (without prefix for database tables)"""
        # Normalise all components
        norm_subject = normalise_text(subject)
        norm_period = normalise_text(period_name)
        norm_sheet = normalise_text(sheet_name)
        
        # Return clean table name without df_ prefix (matches POC approach)
        return f"{norm_subject}_{norm_period}_{norm_sheet}"


class ExcelDataExtractor(DataExtractor):
    """
    Excel-specific data extraction
    Single responsibility: Extract data from Excel files only
    """
    
    def extract_files(self, folder_path: str, period_name: str) -> Dict[str, Dict[str, Any]]:
        """Extract Excel files to DataFrames with comprehensive metadata and robust validation"""
        start_time = time.time()
        
        folder_path = Path(folder_path)
        extracted_data = {}
        
        self.logger.info(f"Starting Excel extraction from: {folder_path}")
        
        if not folder_path.exists():
            raise ExtractionError(f"Folder not found: {folder_path}", str(folder_path))
        
        # Get Excel files, filter out temporary files
        all_excel_files = list(folder_path.glob("*.xlsx"))
        excel_files = [f for f in all_excel_files if not f.name.startswith('~$')]
        
        if not excel_files:
            self.logger.warning(f"No Excel files found in {folder_path}")
            if all_excel_files:
                self.logger.info(f"Skipped {len(all_excel_files)} temporary Excel files")
            return extracted_data
        
        self.logger.info(f"Found {len(excel_files)} Excel files to process")
        if len(all_excel_files) > len(excel_files):
            self.logger.info(f"Skipped {len(all_excel_files) - len(excel_files)} temporary Excel files")
        
        sheets_to_process = self.config.get_sheets_to_process()
        self.logger.info(f"Will process sheets: {sheets_to_process}")
        
        successful_extractions = 0
        failed_extractions = 0
        
        for file_path in excel_files:
            file_start_time = time.time()
            
            try:
                # Extract subject from filename (remove .xlsx extension)
                subject = file_path.stem
                self.logger.info(f"Processing file: {subject} ({file_path.name})")
                
                # Process each sheet in the Excel file
                with pd.ExcelFile(file_path) as xl_file:
                    available_sheets = xl_file.sheet_names
                    self.logger.info(f"Available sheets: {available_sheets}")
                    
                    file_successful = 0
                    
                    for sheet_name in available_sheets:
                        # Check if this sheet should be processed
                        if sheet_name in sheets_to_process:
                            sheet_start_time = time.time()
                            
                            try:
                                # Read the sheet
                                self.logger.info(f"Reading sheet: {sheet_name}")
                                df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
                                
                                # === APPLY OLD SYSTEM'S ROBUST VALIDATION ===
                                
                                # FIX 1: Check if DataFrame is empty or has no columns (from old system)
                                if df.empty or len(df.columns) == 0:
                                    self.logger.warning(f"Sheet '{sheet_name}' in {file_path.name} is empty or has no columns - skipping")
                                    continue
                                
                                # FIX 2: Robust column name cleaning (from old system)
                                if len(df.columns) > 0:
                                    # Ensure columns are strings before applying string operations
                                    df.columns = [str(col).strip() if col is not None else 'Unnamed' for col in df.columns]
                                else:
                                    self.logger.warning(f"Sheet '{sheet_name}' in {file_path.name} has no columns to process - skipping")
                                    continue
                                
                                self.logger.info(f"Sheet '{sheet_name}' original columns: {list(df.columns)}")
                                
                                # Clean column names using your existing function
                                df = clean_dataframe_columns(df)
                                
                                # Additional validation after cleaning
                                if df.empty:
                                    self.logger.warning(f"Sheet '{sheet_name}' in {file_path.name} became empty after cleaning - skipping")
                                    continue
                                
                                # FIX 3: Apply old system's name filtering logic HERE (not in load phase)
                                # This ensures consistent row counts throughout the pipeline
                                initial_count = len(df)
                                
                                # Check if 'Name' column exists before filtering
                                if 'Name' in df.columns:
                                    # Apply the exact same filtering logic as the old working system
                                    df = df[df['Name'].notna()]  # Remove NaN
                                    df = df[df['Name'].astype(str).str.strip() != '']  # Remove empty strings
                                    df = df[df['Name'].astype(str) != 'nan']  # Remove string 'nan'
                                    df = df[df['Name'].astype(str).str.lower() != 'none']  # Remove string 'none'
                                    
                                    final_count = len(df)
                                    if initial_count != final_count:
                                        self.logger.info(f"Filtered out {initial_count - final_count} rows with invalid names from '{sheet_name}'")
                                    
                                    # Ensure we still have data after filtering
                                    if final_count == 0:
                                        self.logger.warning(f"Sheet '{sheet_name}' in {file_path.name} has no valid data after filtering - skipping")
                                        continue
                                else:
                                    self.logger.warning(f"Sheet '{sheet_name}' in {file_path.name} has no 'Name' column")
                                    # Continue anyway, but log this issue
                                
                                # Generate table name
                                table_name = self.generate_table_name(subject, period_name, sheet_name)
                                
                                # Calculate extraction time for this sheet
                                sheet_duration = time.time() - sheet_start_time
                                
                                # Create metadata with ACTUAL final row count (after filtering)
                                metadata = DatasetMetadata(
                                    source_file=str(file_path),
                                    subject=subject,
                                    period=period_name,
                                    sheet_name=sheet_name,
                                    normalised_sheet_name=normalise_text(sheet_name),
                                    table_name=table_name,
                                    row_count=len(df),  # This is now the CORRECT count after filtering
                                    columns_mapped={},  # Will be populated during normalisation
                                    processing_timestamp=generate_timestamp(),
                                    extraction_duration_seconds=sheet_duration
                                )
                                
                                # Store DataFrame with metadata
                                extracted_data[table_name] = {
                                    'dataframe': df,
                                    'metadata': metadata
                                }
                                
                                self.logger.info(f"Extracted {len(df)} rows, {len(df.columns)} columns from '{sheet_name}'")
                                self.logger.info(f"Table name: {table_name}")
                                self.logger.info(f"Columns: {list(df.columns)}")
                                
                                # Log sample data for verification
                                if len(df) > 0:
                                    sample_data = df.iloc[0].to_dict()
                                    self.logger.debug(f"Sample row: {sample_data}")
                                
                                file_successful += 1
                                
                            except Exception as e:
                                self.logger.error(f"Failed to extract sheet '{sheet_name}' from {file_path.name}: {e}")
                                failed_extractions += 1
                                continue
                        else:
                            self.logger.debug(f"Skipping sheet '{sheet_name}' - not in sheets_to_process")
                    
                    if file_successful > 0:
                        file_duration = time.time() - file_start_time
                        self.logger.info(f"Successfully processed {file_successful} sheets from {file_path.name} in {file_duration:.2f}s")
                        successful_extractions += file_successful
                    else:
                        self.logger.warning(f"No sheets processed from {file_path.name}")
                            
            except Exception as e:
                self.logger.error(f"Failed to process file {file_path.name}: {e}")
                failed_extractions += 1
                continue
        
        total_duration = time.time() - start_time
        
        self.logger.info(f"Extraction complete: {successful_extractions} successful, {failed_extractions} failed")
        self.logger.info(f"Total extraction time: {total_duration:.2f}s")
        self.logger.info(f"Extracted {len(extracted_data)} datasets")
        
        if len(extracted_data) == 0:
            raise ExtractionError(f"No data extracted from {folder_path}", str(folder_path))
        
        return extracted_data

class DataNormaliser:
    """
    Light Transform: Normalise column names and validate mappings for schema validation
    Single responsibility: Column normalisation and validation only
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
        Normalise all datasets and validate mappings
        
        Args:
            extracted_data: Dict of {table_name: {'dataframe': df, 'metadata': metadata}}
            
        Returns:
            Dict with normalised DataFrames and updated metadata
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
                
                # Perform normalisation
                normalised_df, mapping_applied = self._normalise_dataframe(df, metadata.sheet_name)
                
                # Validate the transformation
                self._validate_transformation(df, normalised_df, mapping_applied, metadata.sheet_name)
                
                # Update metadata with mapping information and timing
                dataset_duration = time.time() - dataset_start_time
                metadata.columns_mapped = mapping_applied
                metadata.normalisation_duration_seconds = dataset_duration
                
                normalised_data[table_name] = {
                    'dataframe': normalised_df,
                    'metadata': metadata
                }
                
                self.logger.info(f"Successfully normalised {table_name} in {dataset_duration:.2f}s")
                self.logger.info(f"Final shape: {normalised_df.shape}")
                
            except Exception as e:
                self.logger.error(f"Failed to normalise {table_name}: {e}")
                raise NormalisationError(f"Normalisation failed for {table_name}: {e}", table_name)
        
        total_duration = time.time() - start_time
        self.logger.info(f"Normalisation complete for {len(normalised_data)} datasets in {total_duration:.2f}s")
        
        return normalised_data
    
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
                            f"value {numeric_value} below minimum {rule.min_value} (THIS WOULD CATCH PAUL LORIGAN ISSUE)"
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


# Future extensibility examples:
# 
# class JSONDataExtractor(DataExtractor):
#     """JSON-specific data extraction"""
#     def extract_files(self, folder_path: str, period_name: str) -> Dict[str, Dict[str, Any]]:
#         # Implementation for JSON files
#         pass
#
# class CSVDataExtractor(DataExtractor):
#     """CSV-specific data extraction"""  
#     def extract_files(self, folder_path: str, period_name: str) -> Dict[str, Dict[str, Any]]:
#         # Implementation for CSV files
#         pass
