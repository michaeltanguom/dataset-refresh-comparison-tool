"""
Data Cleaning for Dataset Comparison Pipeline
Handles data quality checks, type conversion, and data integrity validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
from scipy import stats

from ..config.config_manager import ConfigManager
from ..utils.database_manager import DatabaseManager
from ..utils.exceptions import DataQualityError, DatabaseError
from ..utils.logging_config import get_logger, log_performance_metric
from ..utils.common import format_number_with_commas, safe_convert_numeric, is_valid_esi_field

logger = get_logger('transform.clean')


class DataCleaner:
    """
    Data quality and type conversion for database tables
    Single responsibility: Data cleaning and quality assurance only
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialise with configuration manager
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.logger = get_logger('cleaner')
        
        # Initialise database manager
        db_path = self.config.get_database_path()
        self.db_manager = DatabaseManager(db_path)
        
        # Get cleaning configuration
        self.cleaning_config = self.config.get_data_cleaning_config()
        self.validation_rules = self.config.get_validation_rules()
        self.critical_columns = self.config.get_critical_columns()
        
        self.logger.info(f"Initialised DataCleaner with database: {db_path}")
        
    def clean_all_tables(self, table_names: List[str]) -> List[str]:
        """
        Clean all specified tables
        
        Args:
            table_names: List of table names to clean
            
        Returns:
            List of successfully cleaned table names
        """
        start_time = time.time()
        self.logger.info(f"Starting data cleaning for {len(table_names)} tables")
        
        successfully_cleaned = []
        failed_cleaning = []
        total_rows_processed = 0
        
        for table_name in table_names:
            table_start_time = time.time()
            
            try:
                self.logger.info(f"Cleaning table: {table_name}")
                
                # Get initial row count
                initial_count = self.db_manager.get_table_count(table_name)
                self.logger.info(f"  Initial rows: {format_number_with_commas(initial_count)}")
                
                # Perform cleaning steps
                self._clean_single_table(table_name)
                
                # Get final row count
                final_count = self.db_manager.get_table_count(table_name)
                self.logger.info(f"  Final rows: {format_number_with_commas(final_count)}")
                
                if final_count != initial_count:
                    removed_count = initial_count - final_count
                    self.logger.info(f"  Removed {format_number_with_commas(removed_count)} rows during cleaning")
                
                # Performance metrics
                table_duration = time.time() - table_start_time
                log_performance_metric(self.logger, f"clean_table_{table_name}", table_duration, final_count)
                
                successfully_cleaned.append(table_name)
                total_rows_processed += final_count
                
                self.logger.info(f"Successfully cleaned {table_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to clean table {table_name}: {e}")
                failed_cleaning.append((table_name, str(e)))
                continue
        
        # Summary logging
        total_duration = time.time() - start_time
        
        self.logger.info(f"Data cleaning complete:")
        self.logger.info(f"  Successfully cleaned: {len(successfully_cleaned)} tables")
        self.logger.info(f"  Failed cleaning: {len(failed_cleaning)}")
        self.logger.info(f"  Total rows processed: {format_number_with_commas(total_rows_processed)}")
        self.logger.info(f"  Total duration: {total_duration:.2f}s")
        
        if failed_cleaning:
            self.logger.error("Failed table cleaning:")
            for table_name, error in failed_cleaning:
                self.logger.error(f"  {table_name}: {error}")
        
        if len(successfully_cleaned) == 0:
            raise DataQualityError("No tables were successfully cleaned")
        
        return successfully_cleaned
    
    def _clean_single_table(self, table_name: str) -> None:
        """
        Clean a single table through multiple steps
        
        Args:
            table_name: Name of table to clean
        """
        self.logger.info(f"Starting cleaning steps for {table_name}")
        
        # Step 1: Data type conversion and standardisation
        self._convert_data_types(table_name)
        
        # Step 2: Handle null values
        self._handle_null_values(table_name)
        
        # Step 3: Remove duplicates
        if self.cleaning_config.get('remove_duplicates', True):
            self._remove_duplicates(table_name)
        
        # Step 4: Data integrity validation
        self._validate_data_integrity(table_name)
        
        # Step 5: Cross-field consistency checks
        self._validate_cross_field_consistency(table_name)
        
        # Step 6: Outlier detection (warnings only)
        if self.cleaning_config.get('outlier_detection', {}).get('enabled', False):
            self._detect_outliers(table_name)
        
        self.logger.info(f"Completed all cleaning steps for {table_name}")
    
    def _convert_data_types(self, table_name: str) -> None:
        """
        Convert and standardise data types
        
        Args:
            table_name: Name of table to process
        """
        self.logger.info(f"Converting data types for {table_name}")
        
        try:
            # Get table data
            df = self.db_manager.execute_query(f'SELECT * FROM "{table_name}"')
            
            if df.empty:
                self.logger.warning(f"Table {table_name} is empty - skipping type conversion")
                return
            
            # Define expected data types based on schema
            type_conversions = {
                'name': str,
                'percent_docs_cited': float,
                'web_of_science_documents': int,
                'rank': int,
                'times_cited': int,
                'affiliation': str,
                'web_of_science_researcherid': str,
                'category_normalised_citation_impact': float,
                'orcid': str,
                'highly_cited_papers': int,
                'hot_papers': int,
                'esi_field': str,
                'indicative_cross_field_score': float
            }
            
            conversion_errors = []
            converted_df = df.copy()
            
            for column, target_type in type_conversions.items():
                if column in converted_df.columns:
                    try:
                        if target_type == str:
                            # String conversion with cleaning
                            converted_df[column] = converted_df[column].astype(str).fillna('')
                            converted_df[column] = converted_df[column].str.strip()
                            
                        elif target_type in [int, float]:
                            # Numeric conversion with validation
                            original_values = converted_df[column]
                            converted_values = original_values.apply(
                                lambda x: safe_convert_numeric(x, target_type, default=None)
                            )
                            
                            # Check for conversion failures
                            failed_mask = (converted_values.isna()) & (original_values.notna())
                            failed_count = failed_mask.sum()
                            
                            if failed_count > 0:
                                sample_failures = original_values[failed_mask].head(3).tolist()
                                error_msg = f"Failed to convert {failed_count} values in {column} to {target_type.__name__}"
                                self.logger.error(f"{error_msg}. Sample failures: {sample_failures}")
                                conversion_errors.append(error_msg)
                            
                            # Use converted values, filling NaN with appropriate defaults
                            if target_type == int:
                                converted_df[column] = converted_values.fillna(0).astype(int)
                            else:  # float
                                converted_df[column] = converted_values.fillna(0.0)
                            
                    except Exception as e:
                        error_msg = f"Unexpected error converting {column} to {target_type.__name__}: {e}"
                        conversion_errors.append(error_msg)
                        self.logger.error(error_msg)
            
            # Fail if there are critical conversion errors
            if conversion_errors:
                raise DataQualityError(
                    f"Data type conversion failed for {table_name}: {'; '.join(conversion_errors)}", 
                    table_name
                )
            
            # Update table with converted data
            self._replace_table_data(table_name, converted_df)
            self.logger.info(f"Successfully converted data types for {table_name}")
            
        except Exception as e:
            raise DataQualityError(f"Data type conversion failed for {table_name}: {e}", table_name)
    
    def _handle_null_values(self, table_name: str) -> None:
        """
        Handle null values according to configuration
        
        Args:
            table_name: Name of table to process
        """
        null_strategy = self.cleaning_config.get('handle_nulls', {}).get('strategy', 'fail')
        self.logger.info(f"Handling null values for {table_name} using strategy: {null_strategy}")
        
        try:
            # Check for null values in critical columns
            critical_null_query = f"""
            SELECT COUNT(*) as null_count
            FROM "{table_name}"
            WHERE {' OR '.join([f'"{col}" IS NULL' for col in self.critical_columns if col != 'name'])}
            """
            
            null_result = self.db_manager.execute_query(critical_null_query)
            null_count = null_result.iloc[0]['null_count']
            
            if null_count > 0:
                if null_strategy == 'fail':
                    raise DataQualityError(f"Found {null_count} null values in critical columns for {table_name}", table_name)
                elif null_strategy == 'skip':
                    # Remove rows with null values in critical columns
                    delete_query = f"""
                    DELETE FROM "{table_name}"
                    WHERE {' OR '.join([f'"{col}" IS NULL' for col in self.critical_columns if col != 'name'])}
                    """
                    self.db_manager.execute_non_query(delete_query)
                    self.logger.info(f"Removed {null_count} rows with null critical values from {table_name}")
                elif null_strategy == 'default':
                    # Fill nulls with defaults (already handled in type conversion)
                    self.logger.info(f"Null values in {table_name} will be handled with defaults")
            else:
                self.logger.info(f"No null values found in critical columns for {table_name}")
            
        except Exception as e:
            raise DataQualityError(f"Null value handling failed for {table_name}: {e}", table_name)
    
    def _remove_duplicates(self, table_name: str) -> None:
        """
        Remove duplicate records based on configuration
        
        Args:
            table_name: Name of table to process
        """
        duplicate_columns = self.cleaning_config.get('duplicate_check_columns', ['name', 'esi_field'])
        self.logger.info(f"Checking for duplicates in {table_name} based on columns: {duplicate_columns}")
        
        try:
            # Count duplicates
            duplicate_query = f"""
            SELECT {', '.join([f'"{col}"' for col in duplicate_columns])}, COUNT(*) as count
            FROM "{table_name}"
            GROUP BY {', '.join([f'"{col}"' for col in duplicate_columns])}
            HAVING COUNT(*) > 1
            """
            
            duplicates = self.db_manager.execute_query(duplicate_query)
            
            if not duplicates.empty:
                total_duplicates = duplicates['count'].sum() - len(duplicates)
                self.logger.warning(f"Found {len(duplicates)} duplicate groups with {total_duplicates} duplicate rows in {table_name}")
                
                # Remove duplicates, keeping first occurrence
                columns_str = ', '.join([f'"{col}"' for col in duplicate_columns])
                dedup_query = f"""
                DELETE FROM "{table_name}"
                WHERE rowid NOT IN (
                    SELECT MIN(rowid)
                    FROM "{table_name}"
                    GROUP BY {columns_str}
                )
                """
                
                self.db_manager.execute_non_query(dedup_query)
                self.logger.info(f"Removed {total_duplicates} duplicate rows from {table_name}")
            else:
                self.logger.info(f"No duplicates found in {table_name}")
            
        except Exception as e:
            raise DataQualityError(f"Duplicate removal failed for {table_name}: {e}", table_name)
    
    def _validate_data_integrity(self, table_name: str) -> None:
        """
        Validate data integrity constraints
        
        Args:
            table_name: Name of table to validate
        """
        self.logger.info(f"Validating data integrity for {table_name}")
        
        integrity_issues = []
        
        try:
            # Check for empty names
            empty_names_query = f"""
            SELECT COUNT(*) as count
            FROM "{table_name}"
            WHERE "name" IS NULL OR TRIM("name") = ''
            """
            empty_names_result = self.db_manager.execute_query(empty_names_query)
            empty_names_count = empty_names_result.iloc[0]['count']
            
            if empty_names_count > 0:
                integrity_issues.append(f"{empty_names_count} rows with empty names")
            
            # Check for negative values in numeric fields that should be positive
            positive_fields = ['times_cited', 'highly_cited_papers', 'hot_papers']
            for field in positive_fields:
                negative_query = f"""
                SELECT COUNT(*) as count
                FROM "{table_name}"
                WHERE "{field}" < 0
                """
                negative_result = self.db_manager.execute_query(negative_query)
                negative_count = negative_result.iloc[0]['count']
                
                if negative_count > 0:
                    integrity_issues.append(f"{negative_count} rows with negative {field}")
            
            # Check for invalid ESI fields
            invalid_esi_query = f"""
            SELECT COUNT(*) as count
            FROM "{table_name}"
            WHERE "esi_field" IS NULL OR TRIM("esi_field") = '' OR "esi_field" = 'nan'
            """
            invalid_esi_result = self.db_manager.execute_query(invalid_esi_query)
            invalid_esi_count = invalid_esi_result.iloc[0]['count']
            
            if invalid_esi_count > 0:
                integrity_issues.append(f"{invalid_esi_count} rows with invalid ESI field")
            
            # Log issues but don't fail (these are warnings)
            if integrity_issues:
                for issue in integrity_issues:
                    self.logger.warning(f"Data integrity issue in {table_name}: {issue}")
            else:
                self.logger.info(f"Data integrity validation passed for {table_name}")
            
        except Exception as e:
            raise DataQualityError(f"Data integrity validation failed for {table_name}: {e}", table_name)
    
    def _validate_cross_field_consistency(self, table_name: str) -> None:
        """
        Validate cross-field logic and consistency
        
        Args:
            table_name: Name of table to validate
        """
        self.logger.info(f"Validating cross-field consistency for {table_name}")
        
        consistency_warnings = []
        
        try:
            # Check: If highly_cited_papers > 0, times_cited should be reasonable
            inconsistent_citations_query = f"""
            SELECT COUNT(*) as count
            FROM "{table_name}"
            WHERE "highly_cited_papers" > 0 AND "times_cited" < 100
            """
            inconsistent_result = self.db_manager.execute_query(inconsistent_citations_query)
            inconsistent_count = inconsistent_result.iloc[0]['count']
            
            if inconsistent_count > 0:
                consistency_warnings.append(f"{inconsistent_count} rows with highly cited papers but low total citations")
            
            # Check: Percentage fields should be 0-100
            percentage_fields = ['percent_docs_cited']
            for field in percentage_fields:
                invalid_percentage_query = f"""
                SELECT COUNT(*) as count
                FROM "{table_name}"
                WHERE "{field}" < 0 OR "{field}" > 100
                """
                invalid_result = self.db_manager.execute_query(invalid_percentage_query)
                invalid_count = invalid_result.iloc[0]['count']
                
                if invalid_count > 0:
                    consistency_warnings.append(f"{invalid_count} rows with invalid percentage in {field}")
            
            # Check: Cross-field score should be reasonable relative to other metrics
            unreasonable_score_query = f"""
            SELECT COUNT(*) as count
            FROM "{table_name}"
            WHERE "indicative_cross_field_score" > 5 AND "highly_cited_papers" = 0
            """
            unreasonable_result = self.db_manager.execute_query(unreasonable_score_query)
            unreasonable_count = unreasonable_result.iloc[0]['count']
            
            if unreasonable_count > 0:
                consistency_warnings.append(f"{unreasonable_count} rows with high cross-field score but no highly cited papers")
            
            # Log warnings
            if consistency_warnings:
                for warning in consistency_warnings:
                    self.logger.warning(f"Cross-field consistency issue in {table_name}: {warning}")
            else:
                self.logger.info(f"Cross-field consistency validation passed for {table_name}")
            
        except Exception as e:
            raise DataQualityError(f"Cross-field consistency validation failed for {table_name}: {e}", table_name)
    
    def _detect_outliers(self, table_name: str) -> None:
        """
        Detect outliers in numeric fields (warnings only)
        
        Args:
            table_name: Name of table to check
        """
        outlier_config = self.cleaning_config.get('outlier_detection', {})
        methods = outlier_config.get('methods', ['iqr'])
        z_threshold = outlier_config.get('z_threshold', 3)
        
        self.logger.info(f"Detecting outliers in {table_name} using methods: {methods}")
        
        try:
            # Get numeric data
            numeric_fields = ['times_cited', 'highly_cited_papers', 'hot_papers', 'indicative_cross_field_score']
            
            for field in numeric_fields:
                field_data_query = f'SELECT "{field}" FROM "{table_name}" WHERE "{field}" IS NOT NULL'
                field_data = self.db_manager.execute_query(field_data_query)
                
                if field_data.empty:
                    continue
                
                values = field_data[field].values
                
                outliers_found = []
                
                # IQR method
                if 'iqr' in methods:
                    Q1 = np.percentile(values, 25)
                    Q3 = np.percentile(values, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    iqr_outliers = np.sum((values < lower_bound) | (values > upper_bound))
                    if iqr_outliers > 0:
                        outliers_found.append(f"IQR method: {iqr_outliers} outliers")
                
                # Z-score method
                if 'zscore' in methods:
                    z_scores = np.abs(stats.zscore(values))
                    z_outliers = np.sum(z_scores > z_threshold)
                    if z_outliers > 0:
                        outliers_found.append(f"Z-score method: {z_outliers} outliers")
                
                # Log findings
                if outliers_found:
                    self.logger.warning(f"Outliers detected in {table_name}.{field}: {'; '.join(outliers_found)}")
                
        except Exception as e:
            self.logger.warning(f"Outlier detection failed for {table_name}: {e}")
    
    def _replace_table_data(self, table_name: str, df: pd.DataFrame) -> None:
        """
        Replace table data with cleaned DataFrame
        
        Args:
            table_name: Name of table to update
            df: Cleaned DataFrame
        """
        try:
            # Create temporary table
            temp_table_name = f"{table_name}_temp"
            
            # Drop temp table if exists
            self.db_manager.drop_table(temp_table_name, if_exists=True)
            
            # Create temp table with new data
            self.db_manager.create_table_from_dataframe(df, temp_table_name, if_exists='replace')
            
            # Replace original table
            self.db_manager.drop_table(table_name, if_exists=True)
            
            # Rename temp table to original name
            rename_query = f'ALTER TABLE "{temp_table_name}" RENAME TO "{table_name}"'
            self.db_manager.execute_non_query(rename_query)
            
            self.logger.debug(f"Successfully replaced data in {table_name}")
            
        except Exception as e:
            raise DatabaseError(f"Failed to replace table data for {table_name}: {e}", table_name)
    
    def get_cleaning_summary(self, table_names: List[str]) -> Dict[str, Any]:
        """
        Get summary of cleaning results
        
        Args:
            table_names: List of table names to summarise
            
        Returns:
            Cleaning summary dictionary
        """
        summary = {
            'total_tables': len(table_names),
            'tables_processed': 0,
            'total_rows': 0,
            'cleaning_config': self.cleaning_config,
            'tables': {}
        }
        
        for table_name in table_names:
            try:
                row_count = self.db_manager.get_table_count(table_name)
                summary['tables'][table_name] = {
                    'row_count': row_count,
                    'status': 'cleaned'
                }
                summary['tables_processed'] += 1
                summary['total_rows'] += row_count
                
            except Exception as e:
                summary['tables'][table_name] = {
                    'row_count': 0,
                    'status': 'error',
                    'error': str(e)
                }
        
        return summary