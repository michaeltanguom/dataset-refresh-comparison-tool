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
        Run all validation checks - updated for new light_transform structure
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Starting comprehensive configuration validation with light transform support")
        
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
        
        # NEW: Light transform validation
        self._validate_light_transform_configuration()
        self._validate_light_transform_column_mapping()
        self._validate_esi_normalisation_configuration()
        self._validate_duplicate_detection_configuration()
        self._validate_null_handling_configuration()
        self._validate_light_transform_validation_configuration()
        
        # Existing validations (updated for new structure)
        self._validate_validation_rules()
        self._validate_output_configuration()
        self._validate_table_naming()
        self._validate_logging_configuration()
        self._validate_statistical_methods_configuration()
        self._validate_comparison_configuration()
        self._validate_cross_dependencies_light_transform()
        self._validate_html_generation_configuration()
        
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
    
    def _validate_light_transform_configuration(self) -> None:
        """Validate light transform configuration structure"""
        check_name = "light_transform_configuration"
        
        try:
            light_transform_config = self.config.get_light_transform_config()
            
            # Check required sections exist
            required_sections = ['column_mapping', 'esi_normalisation', 'duplicate_detection', 'null_handling', 'validation']
            
            for section in required_sections:
                if section not in light_transform_config or not light_transform_config[section]:
                    self._add_result('errors', check_name, f"Missing or empty light_transform.{section} configuration")
                else:
                    self._add_result('passed', check_name, f"light_transform.{section} configuration exists")
            
            # Check for legacy configurations that should be migrated
            if 'column_mapping' in self.config.config and 'column_mapping' in light_transform_config:
                self._add_result('warnings', check_name, "Both legacy column_mapping and light_transform.column_mapping exist. Consider removing legacy version.")
            
            if 'column_mapping_variants' in self.config.config:
                self._add_result('warnings', check_name, "Legacy column_mapping_variants found. This feature is deprecated and will be ignored.")
            
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating light_transform configuration: {e}")

    def _validate_light_transform_column_mapping(self) -> None:
        """Validate light transform column mapping completeness"""
        check_name = "light_transform_column_mapping"
        
        try:
            column_mapping = self.config.get_light_transform_column_mapping()
            critical_columns = self.config.get_critical_columns()
            
            if not column_mapping:
                self._add_result('errors', check_name, "Column mapping is empty or missing")
                return
            
            # Check all critical columns are mapped
            missing_mappings = []
            for critical_col in critical_columns:
                if critical_col not in column_mapping:
                    missing_mappings.append(critical_col)
            
            if missing_mappings:
                self._add_result('errors', check_name, f"Critical columns not mapped in light_transform.column_mapping: {missing_mappings}")
            else:
                self._add_result('passed', check_name, f"All {len(critical_columns)} critical columns are mapped in light_transform")
            
            # Check for duplicate target columns
            target_columns = list(column_mapping.values())
            duplicates = [col for col in set(target_columns) if target_columns.count(col) > 1]
            
            if duplicates:
                self._add_result('errors', check_name, f"Duplicate target columns in light_transform.column_mapping: {duplicates}")
            else:
                self._add_result('passed', check_name, "No duplicate target columns in light_transform column mapping")
            
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
            self._add_result('errors', check_name, f"Error validating light_transform column mapping: {e}")

    def _validate_esi_normalisation_configuration(self) -> None:
        """Validate ESI normalisation configuration"""
        check_name = "esi_normalisation_configuration"
        
        try:
            esi_config = self.config.get_esi_normalisation_config()
            
            # Check if enabled
            if not esi_config.get('enabled', True):
                self._add_result('warnings', check_name, "ESI normalisation is disabled")
                return
            
            # Validate canonical mappings
            canonical_mappings = esi_config.get('canonical_mappings', {})
            
            if not canonical_mappings:
                self._add_result('errors', check_name, "ESI canonical mappings are empty")
                return
            
            # Check expected ESI fields are present
            expected_fields = [
                'agricultural sciences', 'biology_biochemistry', 'chemistry', 'clinical medicine',
                'computer science', 'economics and business', 'engineering', 'environment_ecology',
                'geosciences', 'immunology', 'materials science', 'microbiology',
                'molecular biology and genetics', 'neuroscience and behaviour',
                'pharmacology and toxicology', 'physics', 'plant and animal science',
                'psychiatry_psychology', 'social sciences', 'space science'
            ]
            
            missing_fields = [field for field in expected_fields if field not in canonical_mappings]
            if missing_fields:
                self._add_result('warnings', check_name, f"Missing canonical ESI field mappings: {missing_fields}")
            else:
                self._add_result('passed', check_name, f"All {len(expected_fields)} expected ESI field mappings are present")
            
            # Validate unknown field strategy
            unknown_strategy = esi_config.get('unknown_field_strategy', 'keep_original')
            valid_strategies = ['keep_original', 'flag_error', 'use_default']
            
            if unknown_strategy not in valid_strategies:
                self._add_result('errors', check_name, f"Invalid unknown_field_strategy: {unknown_strategy}. Must be one of: {valid_strategies}")
            else:
                self._add_result('passed', check_name, f"Unknown field strategy is valid: {unknown_strategy}")
                
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating ESI normalisation configuration: {e}")

    def _validate_duplicate_detection_configuration(self) -> None:
        """Validate duplicate detection configuration"""
        check_name = "duplicate_detection_configuration"
        
        try:
            duplicate_config = self.config.get_duplicate_detection_config()
            column_mapping = self.config.get_light_transform_column_mapping()
            
            # Check if enabled
            if not duplicate_config.get('enabled', True):
                self._add_result('passed', check_name, "Duplicate detection is disabled")
                return
            
            # Validate duplicate check columns exist in column mapping
            check_columns = duplicate_config.get('duplicate_check_columns', [])
            
            if not check_columns:
                self._add_result('warnings', check_name, "No duplicate check columns specified")
                return
            
            missing_columns = [col for col in check_columns if col not in column_mapping]
            if missing_columns:
                self._add_result('errors', check_name, f"Duplicate check columns not found in column mapping: {missing_columns}")
            else:
                self._add_result('passed', check_name, f"All {len(check_columns)} duplicate check columns are valid")
                
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating duplicate removal configuration: {e}")

    def _validate_null_handling_configuration(self) -> None:
        """Validate NULL handling configuration"""
        check_name = "null_handling_configuration"
        
        try:
            null_config = self.config.get_null_handling_config()
            column_mapping = self.config.get_light_transform_column_mapping()
            
            # Validate strategy
            strategy = null_config.get('strategy', 'fail')
            valid_strategies = ['fail', 'skip', 'default', 'interpolate']
            
            if strategy not in valid_strategies:
                self._add_result('errors', check_name, f"Invalid NULL handling strategy: {strategy}. Must be one of: {valid_strategies}")
            else:
                self._add_result('passed', check_name, f"NULL handling strategy is valid: {strategy}")
            
            # If strategy is 'default', validate default values
            if strategy == 'default':
                default_values = null_config.get('default_values', {})
                
                if not default_values:
                    self._add_result('warnings', check_name, "NULL handling strategy is 'default' but no default values specified")
                else:
                    # Check that default value columns exist in column mapping
                    invalid_defaults = [col for col in default_values.keys() if col not in column_mapping]
                    if invalid_defaults:
                        self._add_result('errors', check_name, f"Default value columns not found in column mapping: {invalid_defaults}")
                    else:
                        self._add_result('passed', check_name, f"All {len(default_values)} default value columns are valid")
            
            # Validate critical fields never null
            critical_never_null = null_config.get('critical_fields_never_null', [])
            missing_critical = [col for col in critical_never_null if col not in column_mapping]
            
            if missing_critical:
                self._add_result('errors', check_name, f"Critical never-null fields not found in column mapping: {missing_critical}")
            else:
                self._add_result('passed', check_name, f"All {len(critical_never_null)} critical never-null fields are valid")
                
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating NULL handling configuration: {e}")

    def _validate_light_transform_validation_configuration(self) -> None:
        """Validate light transform validation configuration"""
        check_name = "light_transform_validation_configuration"
        
        try:
            validation_config = self.config.get_light_transform_validation_config()
            
            # Check if enabled
            if not validation_config.get('enabled', True):
                self._add_result('warnings', check_name, "Light transform validation is disabled")
                return
            
            # Validate checks
            checks = validation_config.get('checks', [])
            valid_checks = [
                'check_required_columns',
                'check_data_types',
                'check_value_ranges',
                'check_null_constraints',
                'check_duplicate_constraints'
            ]
            
            invalid_checks = [check for check in checks if check not in valid_checks]
            if invalid_checks:
                self._add_result('errors', check_name, f"Invalid validation checks: {invalid_checks}. Valid checks: {valid_checks}")
            else:
                self._add_result('passed', check_name, f"All {len(checks)} validation checks are valid")
            
            # Validate performance limits
            max_time = validation_config.get('max_processing_time_seconds', 30)
            max_memory = validation_config.get('max_memory_usage_mb', 500)
            
            if not isinstance(max_time, (int, float)) or max_time <= 0:
                self._add_result('warnings', check_name, f"Invalid max processing time: {max_time}")
            else:
                self._add_result('passed', check_name, f"Max processing time is valid: {max_time}s")
            
            if not isinstance(max_memory, (int, float)) or max_memory <= 0:
                self._add_result('warnings', check_name, f"Invalid max memory usage: {max_memory}")
            else:
                self._add_result('passed', check_name, f"Max memory usage is valid: {max_memory}MB")
                
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating light transform validation configuration: {e}")

    def _validate_cross_dependencies_light_transform(self) -> None:
        """Validate cross-dependencies between light transform and other configuration sections"""
        check_name = "cross_dependencies_light_transform"
        
        try:
            column_mapping = self.config.get_light_transform_column_mapping()
            critical_columns = self.config.get_critical_columns()
            validation_rules = self.config.get_validation_rules()
            statistical_config = self.config.get_statistical_methods_config()
            
            # Check that critical columns reference valid mapped columns
            invalid_critical = [col for col in critical_columns if col not in column_mapping]
            if invalid_critical:
                self._add_result('errors', check_name, f"Critical columns reference unmapped columns: {invalid_critical}")
            else:
                self._add_result('passed', check_name, "All critical columns reference valid mapped columns")
            
            # Check that validation rules reference valid mapped columns
            invalid_validation = [col for col in validation_rules.keys() if col not in column_mapping]
            if invalid_validation:
                self._add_result('errors', check_name, f"Validation rules reference unmapped columns: {invalid_validation}")
            else:
                self._add_result('passed', check_name, "All validation rule columns reference valid mapped columns")
            
            # Check that statistical methods fields reference valid mapped columns
            if statistical_config.get('enabled', True):
                fields_to_analyse = statistical_config.get('fields_to_analyse', [])
                invalid_statistical = [col for col in fields_to_analyse if col not in column_mapping]
                
                if invalid_statistical:
                    self._add_result('errors', check_name, f"Statistical methods fields reference unmapped columns: {invalid_statistical}")
                else:
                    self._add_result('passed', check_name, "All statistical analysis fields reference valid mapped columns")
            
            # Validate consistency between duplicate removal and critical columns
            duplicate_config = self.config.get_duplicate_detection_config()
            if duplicate_config.get('enabled', True):
                duplicate_columns = duplicate_config.get('duplicate_check_columns', [])
                
                # Check if critical columns are included in duplicate checking
                critical_for_duplicates = ['name', 'esi_field']  # Key fields for duplicate detection
                missing_from_duplicate_check = [col for col in critical_for_duplicates if col not in duplicate_columns]
                
                if missing_from_duplicate_check:
                    self._add_result('warnings', check_name, f"Critical fields not included in duplicate checking: {missing_from_duplicate_check}")
                else:
                    self._add_result('passed', check_name, "Key critical fields are included in duplicate checking")
            
            # Check validation rule coverage
            validation_rule_coverage = len([col for col in critical_columns if col in validation_rules])
            coverage_percent = (validation_rule_coverage / len(critical_columns)) * 100 if critical_columns else 0
            
            if coverage_percent < 100:
                self._add_result('warnings', check_name, f"Validation rules only cover {coverage_percent:.0f}% of critical columns")
            else:
                self._add_result('passed', check_name, "Validation rules cover all critical columns")
                
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating cross-dependencies for light transform: {e}")

    def _validate_statistical_methods_configuration(self) -> None:
        """Validate statistical methods configuration - updated for light transform compatibility"""
        check_name = "statistical_methods_configuration"
        
        try:
            statistical_config = self.config.get_statistical_methods_config()
            
            if not statistical_config.get('enabled', True):
                self._add_result('passed', check_name, "Statistical methods are disabled")
                return
            
            # Check fields to analyse exist in light transform column mapping
            column_mapping = self.config.get_light_transform_column_mapping()
            fields_to_analyse = statistical_config.get('fields_to_analyse', [])
            
            if not fields_to_analyse:
                self._add_result('warnings', check_name, "No fields specified for statistical analysis")
                return
            
            missing_fields = [field for field in fields_to_analyse if field not in column_mapping]
            if missing_fields:
                self._add_result('errors', check_name, f"Statistical analysis fields not found in light_transform.column_mapping: {missing_fields}")
            else:
                self._add_result('passed', check_name, f"All {len(fields_to_analyse)} statistical analysis fields are valid")
            
            # Validate z_threshold
            z_threshold = statistical_config.get('z_threshold', 3.0)
            if not isinstance(z_threshold, (int, float)) or z_threshold <= 0:
                self._add_result('errors', check_name, f"Invalid z_threshold: {z_threshold}. Must be positive number")
            else:
                self._add_result('passed', check_name, f"Z-threshold is valid: {z_threshold}")
            
            # Validate methods
            methods = statistical_config.get('methods', [])
            valid_methods = ['iqr', 'zscore']
            invalid_methods = [method for method in methods if method not in valid_methods]
            
            if invalid_methods:
                self._add_result('errors', check_name, f"Invalid statistical methods: {invalid_methods}. Valid methods: {valid_methods}")
            else:
                self._add_result('passed', check_name, f"Statistical methods are valid: {methods}")
                
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating statistical methods configuration: {e}")
    
    def _validate_comparison_configuration(self) -> None:
        """Validate comparison configuration - updated for light transform compatibility"""
        check_name = "comparison_configuration"
        
        try:
            comparison_config = self.config.get_comparison_config()
            column_mapping = self.config.get_light_transform_column_mapping()
            
            # Validate comparison columns exist in light transform mapping
            comparison_columns = comparison_config.get('comparison_columns', [])
            missing_columns = []
            
            for col in comparison_columns:
                if col not in column_mapping:
                    missing_columns.append(col)
            
            if missing_columns:
                self._add_result('errors', check_name, f"Comparison columns not in light_transform.column_mapping: {missing_columns}")
            else:
                self._add_result('passed', check_name, f"All {len(comparison_columns)} comparison columns are mapped in light_transform")
            
            # Validate float tolerance
            float_tolerance = comparison_config.get('float_tolerance', 0.001)
            if not isinstance(float_tolerance, (int, float)) or float_tolerance < 0:
                self._add_result('warnings', check_name, f"Invalid float tolerance: {float_tolerance}")
            else:
                self._add_result('passed', check_name, f"Float tolerance is valid: {float_tolerance}")
            
            # Check statistical comparisons configuration
            include_statistical = comparison_config.get('include_statistical_comparisons', True)
            statistical_enabled = self.config.get_statistical_methods_config().get('enabled', True)
            
            if include_statistical and not statistical_enabled:
                self._add_result('warnings', check_name, "Statistical comparisons enabled but statistical methods are disabled")
            elif include_statistical and statistical_enabled:
                self._add_result('passed', check_name, "Statistical comparisons configuration is consistent")
            else:
                self._add_result('passed', check_name, "Statistical comparisons are disabled")
                
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

    def _validate_html_generation_configuration(self) -> None:
        """Validate HTML generation configuration"""
        check_name = "html_generation_configuration"
        
        try:
            html_config = self.config.get_html_generation_config()
            
            if not html_config.get('enabled', False):
                self._add_result('passed', check_name, "HTML generation is disabled")
                return
            
            # Check HTML config file exists
            html_config_path = html_config.get('config_path', 'html_generator_config.yaml')
            
            if not Path(html_config_path).exists():
                self._add_result('errors', check_name, f"HTML config file not found: {html_config_path}")
                return
            
            # Load and validate HTML config
            try:
                html_config_data = self.config.load_html_config(html_config_path)
                
                # Validate required sections
                required_sections = ['html_generation', 'templates', 'output']
                missing_sections = []
                
                for section in required_sections:
                    if section not in html_config_data:
                        missing_sections.append(section)
                
                if missing_sections:
                    self._add_result('errors', check_name, f"HTML config missing sections: {missing_sections}")
                else:
                    self._add_result('passed', check_name, "HTML configuration is valid")
                
            except Exception as e:
                self._add_result('errors', check_name, f"Invalid HTML configuration: {e}")
                
        except Exception as e:
            self._add_result('errors', check_name, f"Error validating HTML configuration: {e}")
        
    def get_recommendations(self) -> List[str]:
        """
        Get recommendations for improving configuration - updated for light transform
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Analyse validation results for recommendations
        warnings = self.validation_results.get('warnings', [])
        
        # Check for common light transform issues and suggest improvements
        if any('light_transform' in w['message'].lower() for w in warnings):
            recommendations.append("Review light transform configuration warnings to ensure optimal data processing")
        
        if any('legacy' in w['message'].lower() for w in warnings):
            recommendations.append("Consider migrating legacy configuration sections to the new light_transform structure")
        
        if any('column_mapping_variants' in w['message'] for w in warnings):
            recommendations.append("Remove deprecated column_mapping_variants from configuration - this feature is no longer supported")
        
        if any('validation rules' in w['message'] for w in warnings):
            recommendations.append("Add validation rules for all critical columns in light_transform to improve data quality checking")
        
        if any('esi' in w['message'].lower() for w in warnings):
            recommendations.append("Ensure all expected ESI field mappings are present for comprehensive field normalisation")
        
        if any('duplicate' in w['message'].lower() for w in warnings):
            recommendations.append("Include key identifying fields (name, esi_field) in duplicate check columns")
        
        # Statistical configuration recommendations
        statistical_config = self.config.get_statistical_methods_config()
        if statistical_config.get('enabled', True):
            fields_count = len(statistical_config.get('fields_to_analyse', []))
            if fields_count < 3:
                recommendations.append("Consider analysing more numeric fields in statistical methods for comprehensive outlier detection")
        
        # Performance recommendations
        light_transform_config = self.config.get_light_transform_config()
        validation_config = light_transform_config.get('validation', {})
        max_time = validation_config.get('max_processing_time_seconds', 30)
        
        if max_time > 60:
            recommendations.append("Consider reducing max processing time threshold for faster pipeline failure detection")
        
        # General configuration recommendations
        sheets_count = len(self.config.get_sheets_to_process())
        if sheets_count > 2:
            recommendations.append("With multiple sheets, ensure light transform configuration covers all expected data variations")
        
        column_mappings_count = len(self.config.get_light_transform_column_mapping())
        if column_mappings_count < 5:
            recommendations.append("Ensure all expected columns are mapped in light_transform.column_mapping")
        
        return recommendations
    
    def generate_validation_report(self) -> str:
        """
        Generate a human-readable validation report - updated for light transform
        
        Returns:
            Formatted validation report string
        """
        results = self.validate_all()
        
        report = []
        report.append("=" * 70)
        report.append("CONFIGURATION VALIDATION REPORT - LIGHT TRANSFORM")
        report.append("=" * 70)
        report.append("")
        
        # Summary
        if results['is_valid']:
            report.append("Configuration is VALID for light transform pipeline")
        else:
            report.append("Configuration has ERRORS - light transform pipeline may fail")
        
        report.append(f"   Total checks: {results['total_checks']}")
        report.append(f"   Passed: {results['passed']}")
        report.append(f"   Warnings: {results['warnings']}")
        report.append(f"   Errors: {results['errors']}")
        report.append("")
        
        # Errors
        if results['details']['errors']:
            report.append("ERRORS (must be fixed):")
            for error in results['details']['errors']:
                report.append(f"   - {error['check']}: {error['message']}")
            report.append("")
        
        # Warnings
        if results['details']['warnings']:
            report.append("WARNINGS (should be reviewed):")
            for warning in results['details']['warnings']:
                report.append(f"   - {warning['check']}: {warning['message']}")
            report.append("")
        
        # Light transform specific summary
        config_summary = self.config.get_config_summary()
        report.append("LIGHT TRANSFORM CONFIGURATION SUMMARY:")
        report.append(f"   Column mappings: {config_summary.get('column_mappings_count', 0)}")
        report.append(f"   ESI normalisation: {'Enabled' if config_summary.get('esi_normalisation_enabled', False) else 'Disabled'}")
        report.append(f"   Duplicate removal: {'Enabled' if config_summary.get('duplicate_removal_enabled', False) else 'Disabled'}")
        report.append(f"   Statistical methods: {'Enabled' if config_summary.get('statistical_methods_enabled', False) else 'Disabled'}")
        report.append("")
        
        # Recommendations
        recommendations = self.get_recommendations()
        if recommendations:
            report.append("RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                report.append(f"   {i}. {rec}")
            report.append("")
        
        report.append("=" * 70)
        
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