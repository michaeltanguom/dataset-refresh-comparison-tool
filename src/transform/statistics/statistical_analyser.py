"""
SQL-based statistical analysis for research performance data
Computes Z-scores, percentiles, and outlier flags using DuckDB's statistical functions
"""

import time
from typing import Dict, List, Any, Optional
from pathlib import Path

from ...config.config_manager import ConfigManager
from ...utils.database_manager import DatabaseManager
from ...utils.exceptions import StatisticalAnalysisError, DatabaseError
from ...utils.logging_config import get_logger, log_performance_metric
from ...utils.common import format_number_with_commas

logger = get_logger('transform.statistics')


class StatisticalAnalyser:
    """
    High-performance statistical analysis using SQL batch processing
    Enhances existing tables with Z-scores, percentiles, and outlier flags
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialise statistical analyser
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.logger = get_logger('statistical_analyser')
        
        # Initialise database manager
        db_path = self.config.get_database_path()
        self.db_manager = DatabaseManager(db_path)
        
        # Get statistical configuration
        self.statistical_config = self._get_statistical_config()
        
        # Define fields to analyse
        self.numeric_fields = self.statistical_config.get('fields_to_analyse', [
            'times_cited', 'highly_cited_papers', 'hot_papers', 'indicative_cross_field_score'
        ])
        
        self.z_threshold = self.statistical_config.get('z_threshold', 3.0)
        self.methods = self.statistical_config.get('methods', ['iqr', 'zscore'])
        
        self.logger.info(f"Initialised StatisticalAnalyser for {len(self.numeric_fields)} fields")
    
    def _get_statistical_config(self) -> Dict[str, Any]:
        """Get statistical methods configuration with fallback to data_cleaning"""
        try:
            # Try new configuration structure first
            if hasattr(self.config, 'get_statistical_methods_config'):
                return self.config.get_statistical_methods_config()
            
            # Fallback to existing data_cleaning structure
            data_cleaning_config = self.config.get_data_cleaning_config()
            return data_cleaning_config.get('outlier_detection', {
                'enabled': True,
                'z_threshold': 3.0,
                'methods': ['iqr', 'zscore'],
                'append_to_schema': True,
                'compute_percentiles': True
            })
        except Exception as e:
            self.logger.warning(f"Could not load statistical configuration: {e}")
            return {
                'enabled': True,
                'z_threshold': 3.0,
                'methods': ['iqr', 'zscore'],
                'append_to_schema': True,
                'compute_percentiles': True
            }
    
    def analyse_all_tables(self, table_names: List[str]) -> Dict[str, Any]:
        """
        Perform statistical analysis on all tables using batch processing
        
        Args:
            table_names: List of table names to analyse
            
        Returns:
            Analysis results summary
        """
        start_time = time.time()
        self.logger.info(f"Starting statistical analysis for {len(table_names)} tables")
        
        if not self.statistical_config.get('enabled', True):
            self.logger.info("Statistical analysis disabled in configuration")
            return {
                'analysis_status': 'skipped',
                'reason': 'disabled_in_configuration',
                'tables_processed': 0
            }
        
        analysis_results = {
            'analysis_status': 'success',
            'tables_processed': 0,
            'tables_failed': 0,
            'total_records_enhanced': 0,
            'table_summaries': {},
            'errors': []
        }
        
        for table_name in table_names:
            table_start_time = time.time()
            
            try:
                self.logger.info(f"Analysing table: {table_name}")
                
                # Get initial record count
                initial_count = self.db_manager.get_table_count(table_name)
                
                # Enhance table with statistical columns
                self._enhance_table_with_statistics(table_name)
                
                # Validate enhancements
                validation_results = self._validate_statistical_enhancements(table_name)
                
                # Calculate performance metrics
                table_duration = time.time() - table_start_time
                log_performance_metric(
                    self.logger, 
                    f"statistical_analysis_{table_name}", 
                    table_duration, 
                    initial_count
                )
                
                # Store table summary
                analysis_results['table_summaries'][table_name] = {
                    'record_count': initial_count,
                    'fields_enhanced': len(self.numeric_fields),
                    'processing_duration': round(table_duration, 2),
                    'validation_passed': validation_results['is_valid'],
                    'outlier_count': validation_results.get('outlier_count', 0),
                    'statistical_columns_added': validation_results.get('columns_added', 0)
                }
                
                analysis_results['tables_processed'] += 1
                analysis_results['total_records_enhanced'] += initial_count
                
                self.logger.info(f"Successfully enhanced {table_name} with statistical analysis")
                
            except Exception as e:
                error_msg = f"Statistical analysis failed for {table_name}: {e}"
                self.logger.error(error_msg)
                analysis_results['errors'].append(error_msg)
                analysis_results['tables_failed'] += 1
                
                # Fail fast approach - raise exception to stop pipeline
                raise StatisticalAnalysisError(error_msg, table_name)
        
        total_duration = time.time() - start_time
        analysis_results['total_duration'] = round(total_duration, 2)
        
        # Summary logging
        self.logger.info(f"Statistical analysis complete:")
        self.logger.info(f"  Tables processed: {analysis_results['tables_processed']}")
        self.logger.info(f"  Total records enhanced: {format_number_with_commas(analysis_results['total_records_enhanced'])}")
        self.logger.info(f"  Total duration: {total_duration:.2f}s")
        
        if analysis_results['tables_failed'] > 0:
            self.logger.error(f"  Failed tables: {analysis_results['tables_failed']}")
        
        return analysis_results
    
    def _enhance_table_with_statistics(self, table_name: str) -> None:
        """
        Enhance single table with statistical columns using batch SQL processing
        
        Args:
            table_name: Name of table to enhance
        """
        self.logger.info(f"Enhancing {table_name} with statistical columns")
        
        try:
            # Step 1: Add statistical columns to schema
            self._add_statistical_columns(table_name)
            
            # Step 2: Compute and update statistics in batches
            self._compute_statistical_values(table_name)
            
            self.logger.info(f"Successfully enhanced {table_name} with statistical analysis")
            
        except Exception as e:
            raise StatisticalAnalysisError(f"Failed to enhance table {table_name}: {e}", table_name)
    
    def _add_statistical_columns(self, table_name: str) -> None:
        """Add statistical columns to table schema"""
        alter_statements = []
        
        for field in self.numeric_fields:
            # Z-score column
            alter_statements.append(f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS {field}_z_score DOUBLE DEFAULT 0.0')
            
            # Percentile column
            alter_statements.append(f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS {field}_percentile DOUBLE DEFAULT 0.0')
            
            # Outlier flag column
            alter_statements.append(f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS {field}_outlier_flag VARCHAR DEFAULT \'None\'')
        
        # General outlier severity column
        alter_statements.append(f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS outlier_severity VARCHAR DEFAULT \'Normal\'')
        
        # Execute all ALTER statements
        for statement in alter_statements:
            try:
                self.db_manager.execute_non_query(statement)
                self.logger.debug(f"Executed: {statement}")
            except Exception as e:
                self.logger.error(f"Failed to execute ALTER statement: {statement}")
                raise DatabaseError(f"Schema alteration failed: {e}", table_name)
    
    def _compute_statistical_values(self, table_name: str) -> None:
        """
        Compute statistical values for all fields using batch SQL processing
        
        Args:
            table_name: Name of table to process
        """
        self.logger.info(f"Computing statistical values for {table_name}")
        
        # Build comprehensive UPDATE statement for batch processing
        update_statements = []
        
        for field in self.numeric_fields:
            # Create field-specific statistical update
            field_update_sql = self._generate_field_statistical_update(table_name, field)
            update_statements.append(field_update_sql)
        
        # Execute batch updates
        for statement in update_statements:
            try:
                self.db_manager.execute_non_query(statement)
                self.logger.debug(f"Updated statistical values for field in {table_name}")
            except Exception as e:
                self.logger.error(f"Failed to compute statistics: {e}")
                raise StatisticalAnalysisError(f"Statistical computation failed for {table_name}: {e}", table_name)
        
        # Final update for overall outlier severity
        self._update_outlier_severity(table_name)
    
    def _generate_field_statistical_update(self, table_name: str, field: str) -> str:
        """
        Generate SQL for computing statistical values for a specific field
        
        Args:
            table_name: Name of table
            field: Field name to analyse
            
        Returns:
            SQL UPDATE statement
        """
        return f"""
        UPDATE "{table_name}" 
        SET 
            {field}_z_score = CASE 
                WHEN field_stats.std_val > 0 AND {field} IS NOT NULL THEN 
                    ({field} - field_stats.mean_val) / field_stats.std_val
                ELSE 0.0 
            END,
            {field}_percentile = CASE 
                WHEN {field} IS NOT NULL THEN 
                    PERCENT_RANK() OVER (ORDER BY {field}) * 100
                ELSE 0.0 
            END,
            {field}_outlier_flag = CASE 
                WHEN {field} IS NULL THEN 'None'
                WHEN field_stats.std_val > 0 AND ABS(({field} - field_stats.mean_val) / field_stats.std_val) > {self.z_threshold} 
                     AND ({field} < (field_stats.q1 - 1.5 * field_stats.iqr) OR {field} > (field_stats.q3 + 1.5 * field_stats.iqr))
                THEN 'IQR, Z-Score'
                WHEN field_stats.std_val > 0 AND ABS(({field} - field_stats.mean_val) / field_stats.std_val) > {self.z_threshold}
                THEN 'Z-Score'
                WHEN {field} < (field_stats.q1 - 1.5 * field_stats.iqr) OR {field} > (field_stats.q3 + 1.5 * field_stats.iqr)
                THEN 'IQR'
                ELSE 'None'
            END
        FROM (
            SELECT 
                AVG({field}) as mean_val,
                STDDEV({field}) as std_val,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {field}) as q1,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {field}) as q3,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {field}) - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {field}) as iqr
            FROM "{table_name}"
            WHERE {field} IS NOT NULL AND {field} IS NOT NaN
        ) as field_stats
        WHERE {field} IS NOT NULL
        """
    
    def _update_outlier_severity(self, table_name: str) -> None:
        """Update overall outlier severity based on individual field flags"""
        severity_sql = f"""
        UPDATE "{table_name}"
        SET outlier_severity = CASE
            WHEN {' OR '.join([f"{field}_outlier_flag != 'None'" for field in self.numeric_fields])} THEN
                CASE 
                    WHEN {' OR '.join([f"ABS({field}_z_score) >= 3" for field in self.numeric_fields])} THEN 'Extreme'
                    WHEN {' OR '.join([f"ABS({field}_z_score) >= 2" for field in self.numeric_fields])} THEN 'High' 
                    WHEN {' OR '.join([f"ABS({field}_z_score) >= 1.5" for field in self.numeric_fields])} THEN 'Moderate'
                    ELSE 'Low'
                END
            ELSE 'Normal'
        END
        """
        
        self.db_manager.execute_non_query(severity_sql)
    
    def _validate_statistical_enhancements(self, table_name: str) -> Dict[str, Any]:
        """
        Validate statistical enhancements with comprehensive checks
        
        Args:
            table_name: Name of table to validate
            
        Returns:
            Validation results dictionary
        """
        self.logger.info(f"Validating statistical enhancements for {table_name}")
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'columns_added': 0,
            'outlier_count': 0
        }
        
        try:
            # Check 1: Verify statistical columns exist
            schema = self.db_manager.get_table_schema(table_name)
            expected_columns = []
            
            for field in self.numeric_fields:
                expected_columns.extend([
                    f"{field}_z_score",
                    f"{field}_percentile", 
                    f"{field}_outlier_flag"
                ])
            expected_columns.append('outlier_severity')
            
            missing_columns = [col for col in expected_columns if col not in schema]
            if missing_columns:
                validation_results['errors'].append(f"Missing statistical columns: {missing_columns}")
                validation_results['is_valid'] = False
            else:
                validation_results['columns_added'] = len(expected_columns)
            
            # Check 2: Validate Z-scores are not NULL when base values exist
            for field in self.numeric_fields:
                null_z_scores_query = f"""
                SELECT COUNT(*) as count 
                FROM "{table_name}" 
                WHERE {field} IS NOT NULL AND {field}_z_score IS NULL
                """
                result = self.db_manager.execute_query(null_z_scores_query)
                null_count = result.iloc[0]['count'] if not result.empty else 0
                
                if null_count > 0:
                    validation_results['errors'].append(f"Found {null_count} NULL Z-scores for {field}")
                    validation_results['is_valid'] = False
            
            # Check 3: Validate percentiles are within 0-100 range
            for field in self.numeric_fields:
                invalid_percentiles_query = f"""
                SELECT COUNT(*) as count 
                FROM "{table_name}" 
                WHERE {field}_percentile < 0 OR {field}_percentile > 100
                """
                result = self.db_manager.execute_query(invalid_percentiles_query)
                invalid_count = result.iloc[0]['count'] if not result.empty else 0
                
                if invalid_count > 0:
                    validation_results['errors'].append(f"Found {invalid_count} invalid percentiles for {field}")
                    validation_results['is_valid'] = False
            
            # Check 4: Count outliers for reporting
            outlier_count_query = f"""
            SELECT COUNT(*) as count 
            FROM "{table_name}" 
            WHERE outlier_severity != 'Normal'
            """
            result = self.db_manager.execute_query(outlier_count_query)
            validation_results['outlier_count'] = result.iloc[0]['count'] if not result.empty else 0
            
            # Check 5: Validate outlier severity distribution
            severity_distribution_query = f"""
            SELECT outlier_severity, COUNT(*) as count 
            FROM "{table_name}" 
            GROUP BY outlier_severity
            """
            severity_dist = self.db_manager.execute_query(severity_distribution_query)
            
            if not severity_dist.empty:
                total_records = severity_dist['count'].sum()
                outlier_records = severity_dist[severity_dist['outlier_severity'] != 'Normal']['count'].sum()
                outlier_rate = (outlier_records / total_records) * 100 if total_records > 0 else 0
                
                self.logger.info(f"Outlier distribution for {table_name}: {outlier_rate:.1f}% outliers")
                
                if outlier_rate > 20:  # More than 20% outliers might indicate issues
                    validation_results['warnings'].append(f"High outlier rate: {outlier_rate:.1f}%")
            
            # Log validation results
            if validation_results['is_valid']:
                self.logger.info(f"Statistical validation passed for {table_name}")
                self.logger.info(f"  Columns added: {validation_results['columns_added']}")
                self.logger.info(f"  Outliers detected: {validation_results['outlier_count']}")
            else:
                self.logger.error(f"Statistical validation failed for {table_name}")
                for error in validation_results['errors']:
                    self.logger.error(f"  - {error}")
            
            # Log warnings
            for warning in validation_results['warnings']:
                self.logger.warning(f"Statistical validation warning for {table_name}: {warning}")
        
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation process failed: {e}")
            self.logger.error(f"Statistical validation error for {table_name}: {e}")
        
        return validation_results