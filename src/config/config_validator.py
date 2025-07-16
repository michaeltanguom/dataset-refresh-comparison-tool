"""
Configuration validation for the dataset comparison pipeline
Comprehensive validation of configuration completeness and correctness
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple
import re

from ..utils.exceptions import ConfigurationError
from ..utils.logging_config import get_logger
from ..utils.common import validate_file_path, is_valid_esi_field
from .config_manager import ConfigManager

logger = get_logger('config_validator')


class ConfigValidator:
    """
    Comprehensive configuration validator
    Single responsibility: Configuration validation only
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialise with config manager
        
        Args:
            config_manager: ConfigManager instance to validate
        """
        self.config = config_manager
        self.validation_results = {
            'passed': [],
            'warnings': [],
            'errors': []
        }
    
    def validate_all(self) -> Dict[str, Any]:
        """
        Run all validation checks
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Starting comprehensive configuration validation")
        
        # Reset results
        self.validation_results = {
            'passed': [],
            'warnings': [],
            'errors': []
        }
        
        # Run all validation checks
        self._validate_data_source_paths()
        self._validate_database_configuration()
        self._validate_schema_configuration()
        self._validate_column_mapping_completeness()
        self._validate_validation_rules()
        self._validate_output_configuration()
        self._validate_table_naming()
        self._validate_logging_configuration()
        self._validate_data_cleaning_configuration()
        self._validate_comparison_configuration()
        self._validate_cross_dependencies()
        
        # Summarise results
        total_checks = len(self.validation_results['passed']) + len(self.validation_results['warnings']) + len(self.validation_results['errors'])
        
        results = {
            'total_checks': total_checks,
            'passed': len(self.validation_results['passed']),
            'warnings': len(self.validation_results['warnings']),
            'errors': len(self.validation_results['errors']),
            'is_valid': len(self.validation_results['errors']) == 0,
            'details': self.validation_results
        }
        
        if results['is_valid']:
            logger.info(f"Configuration validation PASSED: {results['passed']} checks passed, {results['warnings']} warnings")
        else:
            logger.error(f"Configuration validation FAILED: {results['errors']} errors, {results['warnings']} warnings")
        
        return results
    
    def _add_result(self, category: str, check_name: str, message: str) -> None:
        """Add validation result"""
        self.validation_results[category].append({
            'check': check_name,
            'message': message
        })
    
    def _validate_data_source_paths(self) -> None:
        """Validate data source folder paths"""
        check_name = "data_source_paths"
        
        try:
            for dataset_key in ['dataset_1', 'dataset_2']:
                config = self.config.get_data_source_config(dataset_key)
                folder_path = config['folder']
                
                if not validate_file_path(folder_path, must_exist=True):
                    self._add_result('errors', check_name, f"{dataset_key} folder does not exist: {folder_path}")
                else:
                    # Check if folder contains Excel files
                    folder = Path(folder_path)
                    excel_files = list(folder.glob("*.xlsx"))
                    excel_files = [f for f in excel_files if not f.name.startswith('~$')]
                    
                    if not excel_files:
                        self._add_result('warnings', check_name, f"{dataset_key} folder contains no Excel files: {folder_path}")
                    else:
                        self._add_result('passed', check_name, f"{dataset_key} folder is valid with {len(excel_files)} Excel files")
                        
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating data source paths: {e}")
    
    def _validate_schema_configuration(self) -> None:
        """Validate schema configuration and files"""
        check_name = "schema_configuration"
        
        try:
            schema_config = self.config.get_schema_config()
            
            # Check schema folder exists
            schema_folder = schema_config.get('schema_folder', 'schema')
            schema_path = Path(schema_folder)
            
            if not schema_path.exists():
                self._add_result('errors', check_name, f"Schema folder does not exist: {schema_folder}")
                return
            
            if not schema_path.is_dir():
                self._add_result('errors', check_name, f"Schema path is not a directory: {schema_folder}")
                return
            
            self._add_result('passed', check_name, f"Schema folder exists: {schema_folder}")
            
            # Check default schema file exists
            default_schema = schema_config.get('default_schema', 'hcr_default')
            default_schema_file = self.config.get_schema_file_path(default_schema)
            
            if not Path(default_schema_file).exists():
                self._add_result('errors', check_name, f"Default schema file not found: {default_schema_file}")
            else:
                # Validate schema file content
                try:
                    with open(default_schema_file, 'r') as f:
                        schema_content = f.read().strip()
                    
                    if not schema_content:
                        self._add_result('errors', check_name, f"Default schema file is empty: {default_schema_file}")
                    elif 'CREATE TABLE' not in schema_content.upper():
                        self._add_result('warnings', check_name, f"Schema file may not contain valid DDL: {default_schema_file}")
                    elif '{table_name}' not in schema_content:
                        self._add_result('warnings', check_name, f"Schema file missing table_name placeholder: {default_schema_file}")
                    else:
                        self._add_result('passed', check_name, f"Default schema file is valid: {default_schema_file}")
                        
                except Exception as e:
                    self._add_result('errors', check_name, f"Cannot read default schema file {default_schema_file}: {e}")
            
            # Check mapped schema files exist
            schema_mapping = schema_config.get('schema_mapping', {})
            sheets_to_process = self.config.get_sheets_to_process()
            
            for sheet in sheets_to_process:
                from ..utils.common import normalise_text
                normalised_sheet = normalise_text(sheet)
                schema_name = self.config.get_schema_for_sheet(normalised_sheet)
                schema_file = self.config.get_schema_file_path(schema_name)
                
                if not Path(schema_file).exists():
                    self._add_result('errors', check_name, f"Schema file for sheet '{sheet}' not found: {schema_file}")
                else:
                    self._add_result('passed', check_name, f"Schema file for sheet '{sheet}' exists: {schema_file}")
            
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating schema configuration: {e}")
    
    def _validate_database_configuration(self) -> None:
        """Validate database configuration"""
        check_name = "database_configuration"
        
        try:
            db_path = self.config.get_database_path()
            
            # Check path format
            if not db_path.endswith('.db'):
                self._add_result('warnings', check_name, f"Database path doesn't end with .db: {db_path}")
            
            # Check if parent directory exists or can be created
            db_file = Path(db_path)
            if not db_file.parent.exists():
                try:
                    db_file.parent.mkdir(parents=True, exist_ok=True)
                    self._add_result('passed', check_name, f"Created database directory: {db_file.parent}")
                except Exception as e:
                    self._add_result('errors', check_name, f"Cannot create database directory {db_file.parent}: {e}")
            else:
                self._add_result('passed', check_name, "Database directory is accessible")
                
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating database configuration: {e}")
    
    def _validate_column_mapping_completeness(self) -> None:
        """Validate column mapping completeness"""
        check_name = "column_mapping_completeness"
        
        try:
            column_mapping = self.config.get_column_mapping()
            critical_columns = self.config.get_critical_columns()
            
            # Check all critical columns are mapped
            missing_mappings = []
            for critical_col in critical_columns:
                if critical_col not in column_mapping:
                    missing_mappings.append(critical_col)
            
            if missing_mappings:
                self._add_result('errors', check_name, f"Critical columns not mapped: {missing_mappings}")
            else:
                self._add_result('passed', check_name, f"All {len(critical_columns)} critical columns are mapped")
            
            # Check for duplicate target columns
            target_columns = list(column_mapping.values())
            duplicates = [col for col in set(target_columns) if target_columns.count(col) > 1]
            
            if duplicates:
                self._add_result('errors', check_name, f"Duplicate target columns in mapping: {duplicates}")
            else:
                self._add_result('passed', check_name, "No duplicate target columns in mapping")
            
            # Validate column name formats
            invalid_names = []
            for norm_col, target_col in column_mapping.items():
                if not re.match(r'^[a-z][a-z0-9_]*$', norm_col):
                    invalid_names.append(norm_col)
            
            if invalid_names:
                self._add_result('warnings', check_name, f"Column names don't follow snake_case convention: {invalid_names}")
            else:
                self._add_result('passed', check_name, "All column names follow snake_case convention")
                
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating column mapping: {e}")
    
    def _validate_validation_rules(self) -> None:
        """Validate validation rules configuration"""
        check_name = "validation_rules"
        
        try:
            validation_rules = self.config.get_validation_rules()
            critical_columns = self.config.get_critical_columns()
            
            # Check that critical columns have validation rules
            missing_rules = []
            for critical_col in critical_columns:
                if critical_col not in validation_rules:
                    missing_rules.append(critical_col)
            
            if missing_rules:
                self._add_result('warnings', check_name, f"Critical columns missing validation rules: {missing_rules}")
            else:
                self._add_result('passed', check_name, "All critical columns have validation rules")
            
            # Validate rule consistency
            for column, rule in validation_rules.items():
                if rule.min_value is not None and rule.max_value is not None:
                    if rule.min_value >= rule.max_value:
                        self._add_result('errors', check_name, f"Invalid range for {column}: min {rule.min_value} >= max {rule.max_value}")
                
                if rule.min_value is not None and rule.min_value < 0:
                    # Only warn for negative minimums in fields that should be positive
                    if column in ['times_cited', 'highly_cited_papers', 'hot_papers']:
                        self._add_result('warnings', check_name, f"Negative minimum value for {column}: {rule.min_value}")
            
            self._add_result('passed', check_name, f"Validation rules configured for {len(validation_rules)} columns")
            
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating validation rules: {e}")
    
    def _validate_output_configuration(self) -> None:
        """Validate output configuration"""
        check_name = "output_configuration"
        
        try:
            output_config = self.config.get_output_config()
            reports_folder = output_config['reports_folder']
            
            # Check if output directory can be created
            if not validate_file_path(reports_folder, must_exist=False):
                self._add_result('errors', check_name, f"Cannot create output directory: {reports_folder}")
            else:
                self._add_result('passed', check_name, f"Output directory is valid: {reports_folder}")
            
            # Check folder name configurations
            folder_names = output_config.get('folder_names', {})
            sheets_to_process = self.config.get_sheets_to_process()
            
            for sheet in sheets_to_process:
                normalised_sheet = self.config.config['table_naming']['prefix'] + '_' + sheet.lower().replace(' ', '_')
                folder_name = self.config.get_output_folder_name(normalised_sheet)
                
                if ' ' in folder_name:
                    self._add_result('warnings', check_name, f"Output folder name contains spaces: {folder_name}")
            
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating output configuration: {e}")
    
    def _validate_table_naming(self) -> None:
        """Validate table naming configuration"""
        check_name = "table_naming"
        
        try:
            table_config = self.config.get_table_naming_config()
            prefix = table_config['prefix']
            
            # Validate prefix format
            if not re.match(r'^[a-z][a-z0-9_]*$', prefix):
                self._add_result('warnings', check_name, f"Table prefix doesn't follow naming convention: {prefix}")
            else:
                self._add_result('passed', check_name, f"Table prefix is valid: {prefix}")
            
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating table naming: {e}")
    
    def _validate_logging_configuration(self) -> None:
        """Validate logging configuration"""
        check_name = "logging_configuration"
        
        try:
            logging_config = self.config.get_logging_config()
            
            # Validate log level
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            log_level = logging_config.get('level', 'INFO').upper()
            
            if log_level not in valid_levels:
                self._add_result('errors', check_name, f"Invalid log level: {log_level}")
            else:
                self._add_result('passed', check_name, f"Log level is valid: {log_level}")
            
            # Check log file path if file logging is enabled
            if logging_config.get('log_to_file', False):
                log_file_path = logging_config.get('log_file_path', 'comparison_pipeline.log')
                if not validate_file_path(log_file_path, must_exist=False):
                    self._add_result('warnings', check_name, f"Cannot create log file: {log_file_path}")
                else:
                    self._add_result('passed', check_name, "Log file path is valid")
            
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating logging configuration: {e}")
    
    def _validate_data_cleaning_configuration(self) -> None:
        """Validate data cleaning configuration"""
        check_name = "data_cleaning_configuration"
        
        try:
            cleaning_config = self.config.get_data_cleaning_config()
            
            # Validate null handling strategy
            null_strategy = cleaning_config.get('handle_nulls', {}).get('strategy', 'fail')
            valid_strategies = ['fail', 'skip', 'default']
            
            if null_strategy not in valid_strategies:
                self._add_result('errors', check_name, f"Invalid null handling strategy: {null_strategy}")
            else:
                self._add_result('passed', check_name, f"Null handling strategy is valid: {null_strategy}")
            
            # Validate outlier detection configuration
            outlier_config = cleaning_config.get('outlier_detection', {})
            if outlier_config.get('enabled', False):
                methods = outlier_config.get('methods', [])
                valid_methods = ['iqr', 'zscore']
                
                invalid_methods = [m for m in methods if m not in valid_methods]
                if invalid_methods:
                    self._add_result('errors', check_name, f"Invalid outlier detection methods: {invalid_methods}")
                else:
                    self._add_result('passed', check_name, f"Outlier detection methods are valid: {methods}")
                
                # Check z-score threshold
                z_threshold = outlier_config.get('z_threshold', 3)
                if not isinstance(z_threshold, (int, float)) or z_threshold <= 0:
                    self._add_result('warnings', check_name, f"Invalid z-score threshold: {z_threshold}")
                else:
                    self._add_result('passed', check_name, f"Z-score threshold is valid: {z_threshold}")
            
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating data cleaning configuration: {e}")
    
    def _validate_comparison_configuration(self) -> None:
        """Validate comparison configuration"""
        check_name = "comparison_configuration"
        
        try:
            comparison_config = self.config.get_comparison_config()
            column_mapping = self.config.get_column_mapping()
            
            # Validate comparison columns exist in mapping
            comparison_columns = comparison_config.get('comparison_columns', [])
            missing_columns = []
            
            for col in comparison_columns:
                if col not in column_mapping:
                    missing_columns.append(col)
            
            if missing_columns:
                self._add_result('errors', check_name, f"Comparison columns not in mapping: {missing_columns}")
            else:
                self._add_result('passed', check_name, f"All {len(comparison_columns)} comparison columns are mapped")
            
            # Validate float tolerance
            float_tolerance = comparison_config.get('float_tolerance', 0.001)
            if not isinstance(float_tolerance, (int, float)) or float_tolerance < 0:
                self._add_result('warnings', check_name, f"Invalid float tolerance: {float_tolerance}")
            else:
                self._add_result('passed', check_name, f"Float tolerance is valid: {float_tolerance}")
            
            # Check fuzzy matching configuration
            fuzzy_config = comparison_config.get('fuzzy_matching', {})
            if fuzzy_config.get('enabled', False):
                self._add_result('warnings', check_name, "Fuzzy matching is enabled but may not be implemented")
            else:
                self._add_result('passed', check_name, "Fuzzy matching is disabled as expected")
            
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating comparison configuration: {e}")
    
    def _validate_cross_dependencies(self) -> None:
        """Validate cross-dependencies between configuration sections"""
        check_name = "cross_dependencies"
        
        try:
            # Check that all sheets to process have corresponding output folder configurations
            sheets_to_process = self.config.get_sheets_to_process()
            output_config = self.config.get_output_config()
            folder_names = output_config.get('folder_names', {})
            
            # Normalise sheet names to check against folder configurations
            for sheet in sheets_to_process:
                from ..utils.common import normalise_text
                normalised_sheet = normalise_text(sheet)
                
                # Check if custom folder name exists or if default will be used
                folder_name = self.config.get_output_folder_name(normalised_sheet)
                
                if normalised_sheet in folder_names:
                    self._add_result('passed', check_name, f"Custom output folder configured for {sheet}: {folder_name}")
                else:
                    self._add_result('passed', check_name, f"Using default output folder for {sheet}: {folder_name}")
            
            # Check that validation rules cover critical columns
            validation_rules = self.config.get_validation_rules()
            critical_columns = self.config.get_critical_columns()
            
            coverage = len([col for col in critical_columns if col in validation_rules])
            coverage_percent = (coverage / len(critical_columns)) * 100 if critical_columns else 0
            
            if coverage_percent < 100:
                self._add_result('warnings', check_name, f"Validation rules only cover {coverage_percent:.0f}% of critical columns")
            else:
                self._add_result('passed', check_name, "Validation rules cover all critical columns")
            
            # Check consistency between data source period names
            dataset_1_config = self.config.get_data_source_config('dataset_1')
            dataset_2_config = self.config.get_data_source_config('dataset_2')
            
            period_1 = dataset_1_config['period_name']
            period_2 = dataset_2_config['period_name']
            
            if period_1 == period_2:
                self._add_result('warnings', check_name, f"Both datasets have same period name: {period_1}")
            else:
                self._add_result('passed', check_name, f"Dataset periods are distinct: {period_1} vs {period_2}")
            
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating cross-dependencies: {e}")
    
    def get_recommendations(self) -> List[str]:
        """
        Get recommendations for improving configuration
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Analyse validation results for recommendations
        warnings = self.validation_results.get('warnings', [])
        
        # Check for common issues and suggest improvements
        if any('Excel files' in w['message'] for w in warnings):
            recommendations.append("Consider adding sample Excel files to input folders for testing")
        
        if any('validation rules' in w['message'] for w in warnings):
            recommendations.append("Add validation rules for all critical columns to improve data quality checking")
        
        if any('folder name contains spaces' in w['message'] for w in warnings):
            recommendations.append("Use underscore-separated folder names instead of spaces for better compatibility")
        
        if any('log file' in w['message'] for w in warnings):
            recommendations.append("Ensure log file directory is writable or disable file logging")
        
        # Suggest optimizations based on configuration
        sheets_count = len(self.config.get_sheets_to_process())
        if sheets_count > 2:
            recommendations.append("Consider processing sheets in parallel for better performance with many sheets")
        
        validation_rules_count = len(self.config.get_validation_rules())
        if validation_rules_count < 3:
            recommendations.append("Add more validation rules to catch data quality issues early")
        
        # Database recommendations
        db_path = self.config.get_database_path()
        if not db_path.startswith('/tmp') and not db_path.startswith('./'):
            recommendations.append("Consider using relative paths for database to improve portability")
        
        return recommendations
    
    def generate_validation_report(self) -> str:
        """
        Generate a human-readable validation report
        
        Returns:
            Formatted validation report string
        """
        results = self.validate_all()
        
        report = []
        report.append("=" * 60)
        report.append("CONFIGURATION VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        if results['is_valid']:
            report.append("✅ CONFIGURATION IS VALID")
        else:
            report.append("❌ CONFIGURATION HAS ERRORS")
        
        report.append(f"   Total checks: {results['total_checks']}")
        report.append(f"   Passed: {results['passed']}")
        report.append(f"   Warnings: {results['warnings']}")
        report.append(f"   Errors: {results['errors']}")
        report.append("")
        
        # Errors
        if results['details']['errors']:
            report.append("🚨 ERRORS (must be fixed):")
            for error in results['details']['errors']:
                report.append(f"   ❌ {error['check']}: {error['message']}")
            report.append("")
        
        # Warnings
        if results['details']['warnings']:
            report.append("⚠️  WARNINGS (should be reviewed):")
            for warning in results['details']['warnings']:
                report.append(f"   ⚠️  {warning['check']}: {warning['message']}")
            report.append("")
        
        # Recommendations
        recommendations = self.get_recommendations()
        if recommendations:
            report.append("💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                report.append(f"   {i}. {rec}")
            report.append("")
        
        # Configuration summary
        config_summary = self.config.get_config_summary()
        report.append("📋 CONFIGURATION SUMMARY:")
        for key, value in config_summary.items():
            report.append(f"   {key}: {value}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def validate_config_file(config_path: str) -> Dict[str, Any]:
    """
    Standalone function to validate a configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validation results dictionary
    """
    try:
        config_manager = ConfigManager(config_path)
        validator = ConfigValidator(config_manager)
        return validator.validate_all()
    except Exception as e:
        return {
            'total_checks': 1,
            'passed': 0,
            'warnings': 0,
            'errors': 1,
            'is_valid': False,
            'details': {
                'passed': [],
                'warnings': [],
                'errors': [{'check': 'config_loading', 'message': str(e)}]
            }
        }