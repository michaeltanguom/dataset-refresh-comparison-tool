"""
Data Loading for Dataset Comparison Pipeline
Loads normalised DataFrames into DuckDB database tables
"""

import pandas as pd
from typing import Dict, List, Any, Optional
import time
from pathlib import Path

from ..config.config_manager import ConfigManager
from ..utils.database_manager import DatabaseManager
from ..utils.exceptions import DatabaseError
from ..utils.logging_config import get_logger, log_performance_metric
from ..utils.common import format_number_with_commas

logger = get_logger('load')


class DataLoader:
    """
    Load normalised DataFrames into DuckDB database
    Single responsibility: Data loading only
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialise with configuration manager
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.logger = get_logger('loader')
        
        # Initialise database manager
        db_path = self.config.get_database_path()
        self.db_manager = DatabaseManager(db_path)
        
        self.logger.info(f"Initialised DataLoader with database: {db_path}")
        
    def load_datasets(self, normalised_data: Dict[str, Dict]) -> List[str]:
        """
        Load all normalised datasets into database
        
        Args:
            normalised_data: Dict of {table_name: {'dataframe': df, 'metadata': metadata}}
            
        Returns:
            List of successfully created table names
        """
        start_time = time.time()
        self.logger.info(f"Starting data loading for {len(normalised_data)} datasets")
        
        # Validate database connection
        if not self.db_manager.validate_connection():
            raise DatabaseError("Cannot connect to database")
        
        # Clear existing tables for clean slate
        self._clear_existing_tables()
        
        successfully_loaded = []
        failed_loads = []
        total_rows_loaded = 0
        
        for table_name, dataset in normalised_data.items():
            table_start_time = time.time()
            
            try:
                df = dataset['dataframe']
                metadata = dataset['metadata']
                
                # Store expected row count
                expected_rows = len(df)
                
                self.logger.info(f"Loading table: {table_name}")
                self.logger.info(f"  Source: {metadata.source_file}")
                self.logger.info(f"  Sheet: {metadata.sheet_name}")
                self.logger.info(f"  Rows: {expected_rows}")
                self.logger.info(f"  Columns: {len(df.columns)}")
                
                # Create table schema
                self._create_table_schema(table_name, metadata.normalised_sheet_name)
                
                # Load data (single responsibility: just load)
                self._load_dataframe_to_table(df, table_name)
                
                # Validate row count (single responsibility: just validate)
                self._validate_row_count(expected_rows, table_name)
                
                # Performance metrics
                table_duration = time.time() - table_start_time
                log_performance_metric(self.logger, f"load_table_{table_name}", table_duration, expected_rows)
                
                successfully_loaded.append(table_name)
                total_rows_loaded += expected_rows
                
                self.logger.info(f"Successfully loaded {table_name} with {format_number_with_commas(expected_rows)} rows")
                
                # Log sample data for verification
                self._log_sample_data(table_name)
                
            except Exception as e:
                self.logger.error(f"Failed to load table {table_name}: {e}")
                failed_loads.append((table_name, str(e)))
                continue
        
        # Summary logging
        total_duration = time.time() - start_time
        
        self.logger.info(f"Data loading complete:")
        self.logger.info(f"  Successfully loaded: {len(successfully_loaded)} tables")
        self.logger.info(f"  Failed loads: {len(failed_loads)}")
        self.logger.info(f"  Total rows loaded: {format_number_with_commas(total_rows_loaded)}")
        self.logger.info(f"  Total duration: {total_duration:.2f}s")
        
        if failed_loads:
            self.logger.error("Failed table loads:")
            for table_name, error in failed_loads:
                self.logger.error(f"  {table_name}: {error}")
        
        # Log database statistics
        self._log_database_statistics()
        
        if len(successfully_loaded) == 0:
            raise DatabaseError("No tables were successfully loaded")
        
        return successfully_loaded
    
    def _clear_existing_tables(self) -> None:
        """Clear existing tables for idempotent operation"""
        try:
            existing_tables = self.db_manager.list_tables()
            
            if existing_tables:
                self.logger.info(f"Clearing {len(existing_tables)} existing tables for clean slate")
                self.db_manager.clear_all_tables()
                self.logger.info("Existing tables cleared")
            else:
                self.logger.info("No existing tables found - starting with clean database")
                
        except Exception as e:
            self.logger.warning(f"Error clearing existing tables: {e}")
            # Continue anyway - might be first run
    
    def _create_and_load_table(self, df: pd.DataFrame, table_name: str, metadata) -> None:
        """Create table schema and load data"""
        # Create table schema
        self._create_table_schema(table_name, metadata.normalised_sheet_name)
        
        # Load data and get actual count loaded
        actual_loaded_count = self._load_dataframe_to_table(df, table_name)
        
        # Verify load using actual loaded count
        loaded_count = self.db_manager.get_table_count(table_name)
        if loaded_count != actual_loaded_count:
            raise DatabaseError(f"Row count mismatch: expected {actual_loaded_count}, got {loaded_count}")

    def _read_schema_file(self, schema_file_path: str) -> str:
        """
        Read SQL schema from file
        
        Args:
            schema_file_path: Path to schema SQL file
            
        Returns:
            SQL schema content
        """
        try:
            schema_path = Path(schema_file_path)
            
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema file not found: {schema_file_path}")
            
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read().strip()
            
            if not schema_sql:
                raise ValueError(f"Schema file is empty: {schema_file_path}")
            
            self.logger.debug(f"Read schema from {schema_file_path}: {len(schema_sql)} characters")
            return schema_sql
            
        except Exception as e:
            raise DatabaseError(f"Failed to read schema file {schema_file_path}: {e}")

    def _create_table_schema(self, table_name: str, normalised_sheet_name: str) -> None:
        """
        Create table with schema from SQL file
        
        Args:
            table_name: Name of table to create
            normalised_sheet_name: Normalised sheet name to determine schema
        """
        try:
            # Get schema name for this sheet type
            schema_name = self.config.get_schema_for_sheet(normalised_sheet_name)
            schema_file_path = self.config.get_schema_file_path(schema_name)
            
            self.logger.info(f"Creating table '{table_name}' using schema: {schema_name}")
            self.logger.info(f"Schema file: {schema_file_path}")
            
            # Read schema SQL file
            schema_sql = self._read_schema_file(schema_file_path)
            
            # Replace table name placeholder
            schema_sql = schema_sql.format(table_name=table_name)
            
            # Execute schema creation
            self.db_manager.execute_non_query(schema_sql)
            self.logger.info(f"Successfully created table schema for: {table_name}")
            
        except Exception as e:
            raise DatabaseError(f"Failed to create table schema for {table_name}: {e}", table_name)
    
    def _load_dataframe_to_table(self, df: pd.DataFrame, table_name: str) -> None:
        """
        Load DataFrame data into table
        Single responsibility: Just load the data as-is
        
        Args:
            df: DataFrame to load (should already be cleaned and normalised)
            table_name: Target table name
        """
        try:
            # Simple validation - ensure we have data
            if df is None or len(df) == 0:
                raise DatabaseError(f"No data to load for table {table_name}")
            
            # Use database manager to insert data
            self.db_manager.insert_dataframe(df, table_name)
            
            self.logger.debug(f"Loaded {len(df)} rows into {table_name}")
            
        except Exception as e:
            raise DatabaseError(f"Failed to load data into table {table_name}: {e}", table_name)
        
    def _validate_row_count(self, expected_count: int, table_name: str) -> None:
        """
        Validate that the loaded table has the expected number of rows
        Single responsibility: Row count validation only
        
        Args:
            expected_count: Expected number of rows
            table_name: Table to validate
        """
        try:
            loaded_count = self.db_manager.get_table_count(table_name)
            
            if loaded_count != expected_count:
                raise DatabaseError(
                    f"Row count mismatch for {table_name}: expected {expected_count}, got {loaded_count}"
                )
            
            self.logger.debug(f"Row count validation passed for {table_name}: {loaded_count} rows")
            
        except Exception as e:
            if "Row count mismatch" in str(e):
                raise  # Re-raise our validation error
            else:
                raise DatabaseError(f"Failed to validate row count for {table_name}: {e}", table_name)
    
    def _verify_table_schema(self, table_name: str, df: pd.DataFrame) -> None:
        """
        Verify that table schema matches expectations
        
        Args:
            table_name: Table to verify
            df: Original DataFrame for comparison
        """
        try:
            # Get table schema
            schema = self.db_manager.get_table_schema(table_name)
            
            # Check that all expected columns exist
            column_mapping = self.config.get_column_mapping()
            expected_columns = set(column_mapping.keys())
            actual_columns = set(schema.keys())
            
            missing_columns = expected_columns - actual_columns
            extra_columns = actual_columns - expected_columns
            
            if missing_columns:
                self.logger.warning(f"Table {table_name} missing columns: {missing_columns}")
            
            if extra_columns:
                self.logger.info(f"Table {table_name} has extra columns: {extra_columns}")
            
            self.logger.debug(f"Schema verification completed for {table_name}")
            
        except Exception as e:
            self.logger.warning(f"Could not verify schema for {table_name}: {e}")
    
    def _log_sample_data(self, table_name: str, n_rows: int = 3) -> None:
        """
        Log sample data from loaded table for verification
        
        Args:
            table_name: Table to sample
            n_rows: Number of rows to sample
        """
        try:
            sample_data = self.db_manager.get_sample_data(table_name, n_rows)
            
            if not sample_data.empty:
                self.logger.info(f"Sample data from {table_name}:")
                for i, row in sample_data.iterrows():
                    # Log key fields for verification
                    key_fields = {
                        'name': row.get('name', 'N/A'),
                        'times_cited': row.get('times_cited', 'N/A'),
                        'highly_cited_papers': row.get('highly_cited_papers', 'N/A'),
                        'esi_field': row.get('esi_field', 'N/A')
                    }
                    self.logger.info(f"  Row {i+1}: {key_fields}")
            else:
                self.logger.warning(f"No sample data retrieved from {table_name}")
                
        except Exception as e:
            self.logger.warning(f"Could not retrieve sample data from {table_name}: {e}")
    
    def _log_database_statistics(self) -> None:
        """Log overall database statistics"""
        try:
            tables = self.db_manager.list_tables()
            total_rows = 0
            
            self.logger.info("Database statistics:")
            self.logger.info(f"  Total tables: {len(tables)}")
            
            for table in tables:
                try:
                    count = self.db_manager.get_table_count(table)
                    total_rows += count
                    self.logger.info(f"  {table}: {format_number_with_commas(count)} rows")
                except Exception as e:
                    self.logger.warning(f"  {table}: Error getting count - {e}")
            
            self.logger.info(f"  Total rows across all tables: {format_number_with_commas(total_rows)}")
            
            # Database file size
            db_size = self.db_manager.get_database_size()
            self.logger.info(f"  Database file size: {db_size:.2f} MB")
            
        except Exception as e:
            self.logger.warning(f"Could not retrieve database statistics: {e}")
    
    def get_loaded_tables_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all loaded tables
        
        Returns:
            Dictionary with table information
        """
        tables_info = {}
        
        try:
            tables = self.db_manager.list_tables()
            
            for table_name in tables:
                try:
                    row_count = self.db_manager.get_table_count(table_name)
                    schema = self.db_manager.get_table_schema(table_name)
                    
                    tables_info[table_name] = {
                        'row_count': row_count,
                        'column_count': len(schema),
                        'columns': list(schema.keys()),
                        'schema': schema
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error getting info for table {table_name}: {e}")
                    tables_info[table_name] = {
                        'row_count': 0,
                        'column_count': 0,
                        'columns': [],
                        'schema': {},
                        'error': str(e)
                    }
            
        except Exception as e:
            self.logger.error(f"Error getting tables info: {e}")
        
        return tables_info
    
    def validate_loaded_data(self, expected_tables: List[str]) -> Dict[str, Any]:
        """
        Validate that data was loaded correctly
        
        Args:
            expected_tables: List of table names that should exist
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'tables_validated': 0,
            'total_rows_validated': 0
        }
        
        try:
            existing_tables = self.db_manager.list_tables()
            
            # Check that all expected tables exist
            missing_tables = set(expected_tables) - set(existing_tables)
            extra_tables = set(existing_tables) - set(expected_tables)
            
            if missing_tables:
                validation_results['errors'].append(f"Missing expected tables: {missing_tables}")
                validation_results['is_valid'] = False
            
            if extra_tables:
                validation_results['warnings'].append(f"Unexpected tables found: {extra_tables}")
            
            # Validate each expected table
            for table_name in expected_tables:
                if table_name in existing_tables:
                    try:
                        # Check row count
                        row_count = self.db_manager.get_table_count(table_name)
                        
                        if row_count == 0:
                            validation_results['errors'].append(f"Table {table_name} is empty")
                            validation_results['is_valid'] = False
                        else:
                            validation_results['total_rows_validated'] += row_count
                            validation_results['tables_validated'] += 1
                        
                        # Check schema
                        schema = self.db_manager.get_table_schema(table_name)
                        critical_columns = self.config.get_critical_columns()
                        
                        missing_critical = set(critical_columns) - set(schema.keys())
                        if missing_critical:
                            validation_results['errors'].append(f"Table {table_name} missing critical columns: {missing_critical}")
                            validation_results['is_valid'] = False
                        
                        # Sample data validation
                        sample_data = self.db_manager.get_sample_data(table_name, 1)
                        if not sample_data.empty:
                            # Basic sanity checks on sample
                            sample_row = sample_data.iloc[0]
                            
                            # Check that name is not empty
                            if not sample_row.get('name') or str(sample_row.get('name')).strip() == '':
                                validation_results['warnings'].append(f"Table {table_name} has empty names in sample")
                            
                            # Check numeric fields are actually numeric
                            numeric_fields = ['times_cited', 'highly_cited_papers', 'hot_papers']
                            for field in numeric_fields:
                                if field in sample_row:
                                    try:
                                        value = float(sample_row[field])
                                        if value < 0:
                                            validation_results['warnings'].append(f"Table {table_name} has negative {field}: {value}")
                                    except (ValueError, TypeError):
                                        validation_results['warnings'].append(f"Table {table_name} has non-numeric {field}: {sample_row[field]}")
                        
                    except Exception as e:
                        validation_results['errors'].append(f"Error validating table {table_name}: {e}")
                        validation_results['is_valid'] = False
            
            # Log validation results
            if validation_results['is_valid']:
                self.logger.info(f"Data validation PASSED: {validation_results['tables_validated']} tables, {format_number_with_commas(validation_results['total_rows_validated'])} rows")
            else:
                self.logger.error(f"Data validation FAILED: {len(validation_results['errors'])} errors")
            
            if validation_results['warnings']:
                self.logger.warning(f"Data validation warnings: {len(validation_results['warnings'])}")
                for warning in validation_results['warnings']:
                    self.logger.warning(f"  - {warning}")
            
        except Exception as e:
            validation_results['errors'].append(f"Validation process failed: {e}")
            validation_results['is_valid'] = False
            
        return validation_results
    
    def cleanup_database(self) -> None:
        """Clean up database connections"""
        try:
            # Database manager handles connection cleanup automatically
            # Just log that we're done
            self.logger.info("Database cleanup completed")
        except Exception as e:
            self.logger.warning(f"Error during database cleanup: {e}")
    
    def get_database_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive database summary
        
        Returns:
            Database summary dictionary
        """
        summary = {
            'database_path': self.db_manager.db_path,
            'database_size_mb': 0,
            'total_tables': 0,
            'total_rows': 0,
            'tables': {}
        }
        
        try:
            # Basic database info
            summary['database_size_mb'] = self.db_manager.get_database_size()
            
            # Table information
            tables = self.db_manager.list_tables()
            summary['total_tables'] = len(tables)
            
            total_rows = 0
            for table_name in tables:
                try:
                    row_count = self.db_manager.get_table_count(table_name)
                    schema = self.db_manager.get_table_schema(table_name)
                    
                    summary['tables'][table_name] = {
                        'row_count': row_count,
                        'column_count': len(schema),
                        'columns': list(schema.keys())
                    }
                    
                    total_rows += row_count
                    
                except Exception as e:
                    summary['tables'][table_name] = {'error': str(e)}
            
            summary['total_rows'] = total_rows
            
        except Exception as e:
            summary['error'] = str(e)
        
        return summary