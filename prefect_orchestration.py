"""
Prefect Orchestration Wrapper for Dataset Comparison Pipeline
Provides workflow management, retry logic, monitoring and scalability
Compatible with Prefect 3.x

python prefect_orchestration.py --config config/comparison_config.yaml --run
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import time
from datetime import datetime, timedelta
import traceback

# Prefect imports (compatible with Prefect 3.x)
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash

# Add current directory to path for proper package imports
sys.path.insert(0, str(Path(__file__).parent))

# Configuration and validation
from src.config.config_manager import ConfigManager
from src.config.config_validator import ConfigValidator
    
# NEW: Import specific extractors instead of PipelineOrchestrator
from src.extract.excel_extractor import ExcelDataExtractor
from src.extract.json_extractor import JSONDataExtractor
from src.extract.light_transform import DataNormaliser
    
# Transform and load components
from src.load.load_duckdb import DataLoader
from src.transform.clean_duckdb_tables import DataCleaner
from src.transform.compare_datasets import DataComparator
    
# Utilities
from src.utils.exceptions import (
    ConfigurationError, ExtractionError, NormalisationError, 
    DatabaseError, DataQualityError, ComparisonError, PipelineError, HtmlGenerationError
)
from src.utils.logging_config import setup_logging, get_logger

try:
    # Import the HTML generation flow
    from src.html_generator.prefect_html_orchestration import unified_html_generation_flow
    HTML_GENERATION_AVAILABLE = True
except ImportError as e:
    print(f"HTML generation not available: {e}")
    HTML_GENERATION_AVAILABLE = False

class PrefectPipelineOrchestrator:
    """
    Prefect wrapper for the dataset comparison pipeline
    Provides enterprise-grade workflow orchestration
    """
    
    def __init__(self, config_path: str):
        """
        Initialise Prefect orchestrator
        
        Args:
            config_path: Path to pipeline configuration file
        """
        self.config_path = config_path
        self.config = None
        self.logger = None
        
        # Load and validate configuration
        self._initialise_configuration()
        
    def _initialise_configuration(self) -> None:
        """Initialise and validate configuration"""
        try:
            self.config = ConfigManager(self.config_path)
            
            # Setup logging from config
            logging_config = self.config.get_logging_config()
            setup_logging(logging_config)
            self.logger = get_logger('prefect_orchestrator')
            
            # Validate configuration
            validator = ConfigValidator(self.config)
            validation_results = validator.validate_all()
            
            if not validation_results['is_valid']:
                error_count = validation_results['errors']
                raise ConfigurationError(f"Configuration validation failed with {error_count} errors")
                
            self.logger.info("Prefect orchestrator initialised successfully")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialise Prefect orchestrator: {e}")


# ===================================================================
# PREFECT TASKS - Individual pipeline steps wrapped as Prefect tasks
# ===================================================================

@task(
    name="validate_configuration",
    description="Validate pipeline configuration and environment",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    retries=0  # Config validation should not retry
)
def validate_pipeline_configuration(config_path: str) -> Dict[str, Any]:
    """
    Prefect task to validate pipeline configuration
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validation results dictionary
    """
    logger = get_run_logger()
    logger.info(f"Validating pipeline configuration: {config_path}")
    
    try:
        config = ConfigManager(config_path)
        validator = ConfigValidator(config)
        validation_results = validator.validate_all()
        
        if not validation_results['is_valid']:
            error_details = validation_results['details']['errors']
            raise ConfigurationError(f"Configuration validation failed: {error_details}")
        
        logger.info("✅ Configuration validation passed")
        return {
            'status': 'valid',
            'config_summary': config.get_config_summary(),
            'validation_results': validation_results
        }
        
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        raise


@task(
    name="extract_datasets",
    description="Extract data from Excel files in input folders using new extraction module",
    retries=2,
    retry_delay_seconds=30
)
def extract_datasets_task(config_path: str) -> Dict[str, Any]:
    """Prefect task for data extraction with proper timing"""
    logger = get_run_logger()
    logger.info("🔄 Starting data extraction with refactored modules")
    
    # 🔧 FIX: Add timing at task level
    task_start_time = time.time()
    
    try:
        config = ConfigManager(config_path)
        extractor = ExcelDataExtractor(config)
        
        dataset_1_config = config.get_data_source_config('dataset_1')
        dataset_2_config = config.get_data_source_config('dataset_2')
        
        logger.info(f"Extracting from dataset 1: {dataset_1_config['folder']}")
        extracted_1 = extractor.extract_files(
            dataset_1_config['folder'], 
            dataset_1_config['period_name']
        )
        
        logger.info(f"Extracting from dataset 2: {dataset_2_config['folder']}")
        extracted_2 = extractor.extract_files(
            dataset_2_config['folder'], 
            dataset_2_config['period_name']
        )
        
        # Combine results
        all_extracted_data = {**extracted_1, **extracted_2}
        
        # 🔧 FIX: Calculate task-level timing
        task_duration = time.time() - task_start_time
        
        # Calculate summary statistics
        total_datasets = len(all_extracted_data)
        total_rows = sum(data['metadata'].row_count for data in all_extracted_data.values())
        
        extraction_results = {
            'extracted_datasets': total_datasets,
            'total_rows': total_rows,
            'extracted_data': all_extracted_data,
            'performance_metrics': {
                'extraction_duration': task_duration  # 🔧 FIX: Use task timing, not internal timing
            }
        }
        
        logger.info(f"✅ Extraction completed: {total_datasets} datasets, {total_rows:,} rows in {task_duration:.2f}s")
        return extraction_results
        
    except Exception as e:
        logger.error(f"❌ Data extraction failed: {e}")
        raise ExtractionError(f"Data extraction failed: {e}")

@task(
    name="light_transform_dataframes",
    description="Light transform: column normalisation, ESI field standardisation, and data validation",
    retries=1,
    retry_delay_seconds=15
)
def light_transform_dataframes_task(config_path: str, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    UPDATED Prefect task for comprehensive light transformation
    Handles: column mapping + ESI field normalisation + validation
    """
    logger = get_run_logger()
    logger.info("🔄 Starting light transformation: column mapping + ESI field normalisation")
    
    task_start_time = time.time()
    
    try:
        config = ConfigManager(config_path)
        normaliser = DataNormaliser(config)  # Now from light_transform module
        
        # Apply all transformations: column mapping + ESI normalisation + validation
        transformed_data = normaliser.normalise_datasets(
            extraction_results['extracted_data']
        )
        
        # Get comprehensive transformation summary including ESI stats
        esi_summary = normaliser.get_normalisation_summary()
        
        task_duration = time.time() - task_start_time
        
        # Enhanced results with ESI normalisation details
        transformation_results = {
            'transformed_datasets': len(transformed_data),
            'transformed_data': transformed_data,  # Renamed from normalised_data for clarity
            'esi_normalisation_summary': esi_summary,
            'performance_metrics': {
                'transformation_duration': task_duration  # Renamed from normalisation_duration
            }
        }
        
        # Enhanced logging with ESI details
        logger.info(f"✅ Light transformation completed: {transformation_results['transformed_datasets']} datasets in {task_duration:.2f}s")
        logger.info(f"🔧 ESI fields normalised: {esi_summary['fields_normalised']} across {esi_summary['dataframes_processed']} DataFrames")
        logger.info(f"📋 Canonical ESI fields available: {esi_summary['total_canonical_fields']}")
        
        if esi_summary['normalisation_errors'] > 0:
            logger.warning(f"⚠️ ESI normalisation errors: {esi_summary['normalisation_errors']}")
        
        return transformation_results
        
    except Exception as e:
        logger.error(f"❌ Light transformation failed: {e}")
        raise NormalisationError(f"Light transformation failed: {e}")

@task(
    name="load_to_database",
    description="Load transformed data into DuckDB database", 
    retries=2,
    retry_delay_seconds=30
)
def load_to_database_task(config_path: str, transformation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: Load transformed data into DuckDB database
    """
    logger = get_run_logger()
    logger.info("🔄 Starting database loading with transformed data")
    
    task_start_time = time.time()
    
    try:
        config = ConfigManager(config_path)
        loader = DataLoader(config)
        
        # FIXED: Use the correct key name from transformation results
        # Check what key is actually available
        if 'normalised_data' in transformation_results:
            data_to_load = transformation_results['normalised_data']
            logger.debug("Using 'normalised_data' key for loading")
        elif 'transformed_data' in transformation_results:
            data_to_load = transformation_results['transformed_data'] 
            logger.debug("Using 'transformed_data' key for loading")
        else:
            available_keys = list(transformation_results.keys())
            raise DatabaseError(f"Neither 'normalised_data' nor 'transformed_data' found in transformation results. Available keys: {available_keys}")
        
        # Load all transformed data
        loaded_tables = loader.load_datasets(data_to_load)
        
        # Validate loaded data
        validation_results = loader.validate_loaded_data(loaded_tables)
        
        if not validation_results['is_valid']:
            raise DatabaseError(f"Data validation failed after loading: {validation_results['errors']}")
        
        task_duration = time.time() - task_start_time
        
        loading_results = {
            'loaded_tables': loaded_tables,
            'tables_count': len(loaded_tables),
            'total_rows_loaded': validation_results['total_rows_validated'],
            'performance_metrics': {
                'loading_duration': task_duration
            }
        }
        
        logger.info(f"✅ Database loading completed: {loading_results['tables_count']} tables, {loading_results['total_rows_loaded']:,} rows in {task_duration:.2f}s")
        return loading_results
        
    except Exception as e:
        logger.error(f"❌ Database loading failed: {e}")
        raise DatabaseError(f"Database loading failed: {e}")

@task(
    name="clean_data",
    description="Clean and validate data quality in database tables using refactored modules",
    retries=1,
    retry_delay_seconds=20
)
def clean_data_task(config_path: str, loading_results: Dict[str, Any]) -> Dict[str, Any]:
    """Prefect task for data cleaning with proper timing"""
    logger = get_run_logger()
    logger.info("🔄 Starting data cleaning with refactored modules")
    
    # 🔧 FIX: Add timing at task level
    task_start_time = time.time()
    
    try:
        config = ConfigManager(config_path)
        cleaner = DataCleaner(config)
        
        # Clean all loaded tables
        cleaned_tables = cleaner.clean_all_tables(loading_results['loaded_tables'])
        
        # Get cleaning summary
        cleaning_summary = cleaner.get_cleaning_summary(cleaned_tables)
        
        # 🔧 FIX: Calculate task-level timing
        task_duration = time.time() - task_start_time
        
        cleaning_results = {
            'cleaned_tables': cleaned_tables,
            'tables_count': cleaning_summary['tables_processed'],
            'total_rows': cleaning_summary['total_rows'],
            'performance_metrics': {
                'cleaning_duration': task_duration  # 🔧 FIX: Use task timing
            }
        }
        
        logger.info(f"✅ Data cleaning completed: {cleaning_results['tables_count']} tables, {cleaning_results['total_rows']:,} rows in {task_duration:.2f}s")
        return cleaning_results
        
    except Exception as e:
        logger.error(f"❌ Data cleaning failed: {e}")
        raise DataQualityError(f"Data cleaning failed: {e}")

@task(
    name="compare_datasets",
    description="Compare datasets and generate reports using refactored modules",
    retries=1,
    retry_delay_seconds=30
)
def compare_datasets_task(config_path: str, cleaning_results: Dict[str, Any]) -> Dict[str, Any]:
    """Prefect task for dataset comparison with proper timing"""
    logger = get_run_logger()
    logger.info("🔄 Starting dataset comparison with refactored modules")
    
    # 🔧 FIX: Add timing at task level
    task_start_time = time.time()
    
    try:
        config = ConfigManager(config_path)
        comparator = DataComparator(config)
        
        # Compare all matching tables
        comparison_results = comparator.compare_all_matching_tables(
            cleaning_results['cleaned_tables']
        )
        
        # Save comparison reports
        saved_files = comparator.save_comparison_reports(comparison_results)
        
        # 🔧 FIX: Calculate task-level timing
        task_duration = time.time() - task_start_time
        
        final_results = {
            'comparison_results': comparison_results,
            'successful_comparisons': comparison_results['summary']['successful_comparisons'],
            'failed_comparisons': comparison_results['summary']['failed_comparisons'],
            'saved_report_files': saved_files,
            'performance_metrics': {
                'comparison_duration': task_duration  # 🔧 FIX: Use task timing
            }
        }
        
        logger.info(f"✅ Dataset comparison completed: {final_results['successful_comparisons']} successful, {final_results['failed_comparisons']} failed in {task_duration:.2f}s")
        logger.info(f"📁 Reports saved: {len(saved_files)} files")
        
        return final_results
        
    except Exception as e:
        logger.error(f"❌ Dataset comparison failed: {e}")
        raise ComparisonError(f"Dataset comparison failed: {e}")

@task(
    name="generate_enhanced_pipeline_summary",
    description="Generate comprehensive pipeline execution summary with HTML generation",
    retries=0
)
def generate_enhanced_pipeline_summary(config_path: str, 
                                     extraction_results: Dict[str, Any],
                                     normalisation_results: Dict[str, Any],
                                     loading_results: Dict[str, Any],
                                     cleaning_results: Dict[str, Any],
                                     comparison_results: Dict[str, Any],
                                     json_extraction_results: Optional[Dict[str, Any]] = None,
                                     html_generation_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate comprehensive pipeline execution summary including HTML generation
    """
    logger = get_run_logger()
    logger.info("📊 Generating enhanced pipeline execution summary")
    
    try:
        # Calculate total duration
        total_duration = (
            extraction_results['performance_metrics'].get('extraction_duration', 0) +
            normalisation_results['performance_metrics'].get('normalisation_duration', 0) +
            loading_results['performance_metrics'].get('loading_duration', 0) +
            cleaning_results['performance_metrics'].get('cleaning_duration', 0) +
            comparison_results['performance_metrics'].get('comparison_duration', 0)
        )
        
        # Add JSON extraction duration if present
        if json_extraction_results:
            total_duration += json_extraction_results['performance_metrics'].get('json_extraction_duration', 0)
        
        # Add HTML generation duration if present
        if html_generation_results:
            html_duration = html_generation_results.get('performance_metrics', {}).get('html_task_duration', 0)
            total_duration += html_duration
        
        summary = {
            'pipeline_status': 'SUCCESS',
            'execution_timestamp': datetime.now().isoformat(),
            'config_file': config_path,
            'total_execution_time_seconds': total_duration,
            'step_performance': {
                'extraction': extraction_results['performance_metrics'].get('extraction_duration', 0),
                'normalisation': normalisation_results['performance_metrics'].get('normalisation_duration', 0),
                'loading': loading_results['performance_metrics'].get('loading_duration', 0),
                'cleaning': cleaning_results['performance_metrics'].get('cleaning_duration', 0),
                'comparison': comparison_results['performance_metrics'].get('comparison_duration', 0)
            },
            'data_statistics': {
                'total_datasets_extracted': extraction_results['extracted_datasets'],
                'total_rows_extracted': extraction_results['total_rows'],
                'tables_loaded': loading_results['tables_count'],
                'tables_cleaned': cleaning_results['tables_count'],
                'successful_comparisons': comparison_results['successful_comparisons'],
                'failed_comparisons': comparison_results['failed_comparisons'],
                'comparison_reports_generated': len(comparison_results.get('saved_report_files', {}))
            },
            'step_completion': {
                'extraction': True,
                'normalisation': True,
                'loading': True,
                'cleaning': True,
                'comparison': True,
                'json_extraction': json_extraction_results is not None,
                'html_generation': html_generation_results is not None  # NEW
            }
        }
        
        # Add JSON extraction statistics if present
        if json_extraction_results:
            summary['step_performance']['json_extraction'] = json_extraction_results['performance_metrics'].get('json_extraction_duration', 0)
            summary['data_statistics']['json_reports_extracted'] = json_extraction_results['extracted_reports']
            summary['data_statistics']['total_researcher_records'] = json_extraction_results['total_researcher_records']
        
        # Add HTML generation statistics if present
        if html_generation_results:
            summary['step_performance']['html_generation'] = html_generation_results.get('performance_metrics', {}).get('html_task_duration', 0)
            
            if html_generation_results.get('pipeline_status') == 'SUCCESS':
                summary['data_statistics']['html_dashboards_generated'] = html_generation_results.get('dashboards_generated', 0)
                summary['data_statistics']['html_failed_generations'] = html_generation_results.get('failed_generations', 0)
                
                # Add output summary
                output_summary = html_generation_results.get('output_summary', {})
                if output_summary:
                    summary['html_output'] = {
                        'files_generated': output_summary.get('files_generated', 0),
                        'total_size_mb': output_summary.get('total_size_mb', 0),
                        'output_directory': output_summary.get('output_directory', 'Unknown')
                    }
            else:
                summary['data_statistics']['html_dashboards_generated'] = 0
                summary['html_generation_error'] = html_generation_results.get('error', 'Unknown error')
        
        logger.info("✅ Enhanced pipeline execution summary generated")
        logger.info(f"📈 Total execution time: {total_duration:.2f} seconds")
        
        if html_generation_results:
            html_status = html_generation_results.get('pipeline_status', 'Unknown')
            dashboards_generated = summary['data_statistics'].get('html_dashboards_generated', 0)
            logger.info(f"🎨 HTML Generation: {html_status} - {dashboards_generated} dashboards")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Failed to generate enhanced pipeline summary: {e}")
        return {
            'pipeline_status': 'PARTIAL_SUCCESS',
            'error': str(e),
            'execution_timestamp': datetime.now().isoformat()
        }

@task(
    name="extract_comparison_reports",
    description="Extract JSON comparison reports for HTML generation",
    retries=1,
    retry_delay_seconds=15
)
def extract_comparison_reports_task(config_path: str, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract JSON comparison reports with proper timing"""
    logger = get_run_logger()
    logger.info("🔄 Starting JSON comparison report extraction")
    
    # 🔧 FIX: Add timing at task level
    task_start_time = time.time()
    
    try:
        config = ConfigManager(config_path)
        json_extractor = JSONDataExtractor(config)
        
        # Extract from comparison_reports directory
        output_config = config.get_output_config()
        reports_folder = output_config['reports_folder']
        
        logger.info(f"Extracting JSON reports from: {reports_folder}")
        extracted_reports = json_extractor.extract_files(reports_folder, "current")
        
        # 🔧 FIX: Calculate task-level timing
        task_duration = time.time() - task_start_time
        
        extraction_results = {
            'extracted_reports': len(extracted_reports),
            'total_researcher_records': sum(
                data['metadata'].row_count for data in extracted_reports.values()
            ),
            'extracted_data': extracted_reports,
            'performance_metrics': {
                'json_extraction_duration': task_duration  # 🔧 FIX: Use task timing
            }
        }
        
        logger.info(f"✅ JSON extraction completed: {extraction_results['extracted_reports']} reports in {task_duration:.2f}s")
        logger.info(f"📊 Total researcher records: {extraction_results['total_researcher_records']:,}")
        
        return extraction_results
        
    except Exception as e:
        logger.error(f"❌ JSON extraction failed: {e}")
        raise ExtractionError(f"JSON extraction failed: {e}")

@task(
    name="generate_html_dashboards_integrated",
    description="Generate HTML dashboards using dedicated HTML generation flow",
    retries=1,
    retry_delay_seconds=30
)
def generate_html_dashboards_integrated_task(config_path: str, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate HTML dashboards by calling the dedicated HTML generation flow
    
    Args:
        config_path: Path to configuration file
        comparison_results: Results from comparison task
        
    Returns:
        HTML generation results
    """
    logger = get_run_logger()
    logger.info("🎨 Starting integrated HTML dashboard generation")
    
    # Check if HTML generation is available
    if not HTML_GENERATION_AVAILABLE:
        logger.warning("HTML generation module not available - skipping")
        return {
            'pipeline_status': 'SKIPPED',     # ← Fixed: consistent field name
            'reason': 'HTML generation module not available',
            'dashboards_generated': 0,        # ← Fixed: consistent field name
            'failed_generations': 0,
            'performance_metrics': {'html_task_duration': 0}  # ← Fixed: consistent field name
        }
    
    task_start_time = time.time()
    
    try:
        # Check if HTML generation is enabled in config
        config = ConfigManager(config_path)
        html_config = config.get_html_generation_config()
        
        if not html_config.get('enabled', False):
            logger.info("HTML generation is disabled in configuration - skipping")
            return {
                'pipeline_status': 'SKIPPED',
                'reason': 'HTML generation disabled in config',
                'dashboards_generated': 0,
                'failed_generations': 0,
                'performance_metrics': {'html_task_duration': 0}
            }
        
        # Ensure comparison reports were generated successfully
        if comparison_results.get('successful_comparisons', 0) == 0:
            logger.warning("No successful comparisons found - skipping HTML generation")
            return {
                'pipeline_status': 'SKIPPED',
                'reason': 'No successful comparisons to process',
                'dashboards_generated': 0,
                'failed_generations': 0,
                'performance_metrics': {'html_task_duration': 0}
            }
        
        logger.info(f"Running HTML generation flow for {comparison_results['successful_comparisons']} comparison reports")
        
        # 🔧 FIX: Call the correct unified HTML generation flow
        html_results = unified_html_generation_flow(config_path)
        
        # Calculate task-level timing
        task_duration = time.time() - task_start_time
        
        # Enhance results with task timing
        if 'performance_metrics' not in html_results:
            html_results['performance_metrics'] = {}
        html_results['performance_metrics']['html_task_duration'] = task_duration
        
        # Log results
        if html_results.get('pipeline_status') == 'SUCCESS':
            dashboards_generated = html_results.get('dashboards_generated', 0)
            failed_generations = html_results.get('failed_generations', 0)
            
            logger.info(f"✅ HTML generation completed: {dashboards_generated} dashboards generated, {failed_generations} failed")
            logger.info(f"📁 Output directory: {html_results.get('output_summary', {}).get('output_directory', 'Unknown')}")
            
            if html_results.get('output_summary', {}).get('files_generated', 0) > 0:
                total_size = html_results['output_summary'].get('total_size_mb', 0)
                logger.info(f"📊 Total size: {total_size:.2f} MB")
        else:
            logger.error(f"❌ HTML generation failed: {html_results.get('error', 'Unknown error')}")
        
        return html_results
        
    except Exception as e:
        task_duration = time.time() - task_start_time
        logger.error(f"❌ HTML generation task failed: {e}")
        
        # 🔧 FIX: Return consistent error schema
        return {
            'pipeline_status': 'FAILED',
            'error': str(e),
            'dashboards_generated': 0,
            'failed_generations': 0,
            'performance_metrics': {'html_task_duration': task_duration},
            'output_summary': {
                'files_generated': 0,
                'total_size_mb': 0,
                'output_directory': 'html_dashboards'
            }
        }

# ====================================================
# PREFECT FLOWS - Main workflow orchestration
# ====================================================

@flow(
    name="dataset-comparison-pipeline-with-html",
    description="Complete EtLT pipeline with optional HTML dashboard generation and integrated ESI normalisation",
    version="2.2.0",  # Increment version to reflect ESI integration
    timeout_seconds=7200,  # Keep existing timeout
    log_prints=True
)
def dataset_comparison_flow(config_path: str, 
                          include_json_extraction: bool = False,
                          generate_html: bool = False) -> Dict[str, Any]:
    """
    Main Prefect flow with optional HTML generation and integrated ESI field normalisation
    
    Args:
        config_path: Path to configuration file
        include_json_extraction: Whether to include JSON extraction (legacy option)
        generate_html: Whether to generate HTML dashboards
    """
    logger = get_run_logger()
    logger.info("🚀 Starting Dataset Comparison Pipeline with HTML Generation")
    logger.info(f"📁 Configuration: {config_path}")
    logger.info(f"🔧 Enhanced with integrated ESI field normalisation")
    logger.info(f"🎨 HTML Generation: {'Enabled' if generate_html else 'Disabled'}")
    
    try:
        # Original pipeline steps - keeping your exact structure
        validation_results = validate_pipeline_configuration(config_path)
        extraction_results = extract_datasets_task(config_path)
        
        # Now includes ESI field normalisation alongside column mapping
        normalisation_results = light_transform_dataframes_task(config_path, extraction_results)
        
        loading_results = load_to_database_task(config_path, normalisation_results)
        cleaning_results = clean_data_task(config_path, loading_results)
        comparison_results = compare_datasets_task(config_path, cleaning_results)
        
        # Legacy JSON extraction (conditional) - keeping your exact logic
        json_extraction_results = None
        if include_json_extraction:
            logger.info("🔄 Including legacy JSON extraction")
            json_extraction_results = extract_comparison_reports_task(config_path, comparison_results)
        
        # HTML Dashboard Generation (conditional) - keeping your exact logic
        html_generation_results = None
        if generate_html:
            logger.info("🎨 Including HTML dashboard generation")
            html_generation_results = generate_html_dashboards_integrated_task(config_path, comparison_results)
        
        # Enhanced summary generation - keeping your exact structure
        pipeline_summary = generate_enhanced_pipeline_summary(
            config_path,
            extraction_results,
            normalisation_results,  # Now contains ESI normalisation statistics
            loading_results,
            cleaning_results,
            comparison_results,
            json_extraction_results,
            html_generation_results
        )
        
        # Enhanced completion logging
        logger.info("🎉 Complete pipeline finished successfully!")
        
        # NEW: Log ESI normalisation summary if available
        if 'esi_normalisation_details' in pipeline_summary:
            esi_details = pipeline_summary['esi_normalisation_details']
            logger.info(f"🔧 ESI Normalisation completed: {esi_details.get('fields_normalised', 0)} fields normalised across {esi_details.get('dataframes_processed', 0)} DataFrames")
        
        return pipeline_summary
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        
        return {
            'pipeline_status': 'FAILED',
            'error': str(e),
            'execution_timestamp': datetime.now().isoformat(),
            'config_file': config_path
        }
    
@flow(
    name="dataset-comparison-validation-only",
    description="Configuration validation flow for testing",
    version="1.0.0",
    timeout_seconds=300
)
def validation_only_flow(config_path: str) -> Dict[str, Any]:
    """
    Validation-only flow for configuration testing
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validation results
    """
    logger = get_run_logger()
    logger.info("🔍 Running configuration validation only")
    
    return validate_pipeline_configuration(config_path)


# ====================================================
# DEPLOYMENT MANAGEMENT (Prefect 3.x compatible)
# ====================================================

def create_deployment_file(config_path: str, schedule_cron: str = None) -> str:
    """
    Create a prefect.yaml deployment file for the pipeline
    
    Args:
        config_path: Path to configuration file
        schedule_cron: Optional cron expression for scheduling
        
    Returns:
        Path to created deployment file
    """
    deployment_content = f"""
# Prefect deployment configuration
name: dataset-comparison-pipeline
version: 1.0.0
description: Dataset comparison pipeline with EtLT orchestration

# Flow configuration
flow_name: dataset-comparison-pipeline
entrypoint: prefect_orchestration.py:dataset_comparison_flow

# Parameters
parameters:
  config_path: "{config_path}"

# Work pool configuration
work_pool:
  name: default-agent-pool
  job_variables:
    env:
      PREFECT_LOGGING_LEVEL: INFO

# Build configuration
build: null
push: null
pull:
  - prefect.deployments.steps.set_working_directory:
      directory: {Path.cwd()}

# Scheduling (optional)
"""
    
    if schedule_cron:
        deployment_content += f"""
schedule:
  cron: "{schedule_cron}"
  timezone: "Europe/London"
"""
    else:
        deployment_content += """
schedule: null
"""
    
    # Write deployment file
    deployment_file = Path("prefect.yaml")
    with open(deployment_file, "w") as f:
        f.write(deployment_content)
    
    return str(deployment_file)


def serve_flows():
    """
    Serve flows for immediate execution
    This replaces the old deployment system for development
    """
    print("🚀 Starting Prefect flow server...")
    print("📊 Dashboard available at: http://127.0.0.1:4200")
    print("🎛️  You can trigger flows from the dashboard or via CLI")
    
    # Serve the main flow
    dataset_comparison_flow.serve(
        name="dataset-comparison-service",
        tags=["production", "etlt"],
        description="Dataset comparison pipeline service",
        version="1.0.0"
    )


# ====================================================
# COMMAND LINE INTERFACE
# ====================================================

@flow(
    name="dataset-comparison-pipeline-with-html",
    description="Complete EtLT pipeline with optional HTML dashboard generation and integrated ESI normalisation",
    version="2.2.0",  # Increment version to reflect ESI integration
    timeout_seconds=7200,  # Keep existing timeout
    log_prints=True
)
def dataset_comparison_flow(config_path: str, 
                          include_json_extraction: bool = False,
                          generate_html: bool = False) -> Dict[str, Any]:
    """
    Main Prefect flow with optional HTML generation and integrated ESI field normalisation
    
    Args:
        config_path: Path to configuration file
        include_json_extraction: Whether to include JSON extraction (legacy option)
        generate_html: Whether to generate HTML dashboards
    """
    logger = get_run_logger()
    logger.info("🚀 Starting Dataset Comparison Pipeline with HTML Generation")
    logger.info(f"📁 Configuration: {config_path}")
    logger.info(f"🔧 Enhanced with integrated ESI field normalisation")
    logger.info(f"🎨 HTML Generation: {'Enabled' if generate_html else 'Disabled'}")
    
    try:
        # Original pipeline steps - keeping your exact structure
        validation_results = validate_pipeline_configuration(config_path)
        extraction_results = extract_datasets_task(config_path)
        
        # ENHANCED: Now includes ESI field normalisation alongside column mapping
        normalisation_results = light_transform_dataframes_task(config_path, extraction_results)
        
        loading_results = load_to_database_task(config_path, normalisation_results)
        cleaning_results = clean_data_task(config_path, loading_results)
        comparison_results = compare_datasets_task(config_path, cleaning_results)
        
        # Legacy JSON extraction (conditional) - keeping your exact logic
        json_extraction_results = None
        if include_json_extraction:
            logger.info("🔄 Including legacy JSON extraction")
            json_extraction_results = extract_comparison_reports_task(config_path, comparison_results)
        
        # HTML Dashboard Generation (conditional) - keeping your exact logic
        html_generation_results = None
        if generate_html:
            logger.info("🎨 Including HTML dashboard generation")
            html_generation_results = generate_html_dashboards_integrated_task(config_path, comparison_results)
        
        # Enhanced summary generation - keeping your exact structure
        pipeline_summary = generate_enhanced_pipeline_summary(
            config_path,
            extraction_results,
            normalisation_results,  # Now contains ESI normalisation statistics
            loading_results,
            cleaning_results,
            comparison_results,
            json_extraction_results,
            html_generation_results
        )
        
        # Enhanced completion logging
        logger.info("🎉 Complete pipeline finished successfully!")
        
        # NEW: Log ESI normalisation summary if available
        if 'esi_normalisation_details' in pipeline_summary:
            esi_details = pipeline_summary['esi_normalisation_details']
            logger.info(f"🔧 ESI Normalisation completed: {esi_details.get('fields_normalised', 0)} fields normalised across {esi_details.get('dataframes_processed', 0)} DataFrames")
        
        return pipeline_summary
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        
        return {
            'pipeline_status': 'FAILED',
            'error': str(e),
            'execution_timestamp': datetime.now().isoformat(),
            'config_file': config_path
        }


# Keep your existing CLI main() function logic but update the success logging
def main():
    """Updated main execution function with ESI normalisation details"""
    parser = argparse.ArgumentParser(
        description="Prefect Orchestration for Dataset Comparison Pipeline with HTML Generation and ESI Normalisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline without HTML generation (now includes ESI normalisation)
  python prefect_orchestration.py --config config/comparison_config.yaml --run

  # Run full pipeline with HTML dashboard generation (now includes ESI normalisation)
  python prefect_orchestration.py --config config/comparison_config.yaml --run --generate-html

  # Run with both JSON extraction and HTML generation (now includes ESI normalisation)
  python prefect_orchestration.py --config config/comparison_config.yaml --run --include-json --generate-html
        """
    )
    
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument("--run", action="store_true", help="Run the full pipeline immediately")
    parser.add_argument("--include-json", action="store_true", help="Include JSON extraction (legacy)")
    parser.add_argument("--generate-html", action="store_true", help="Generate HTML dashboards")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration")
    parser.add_argument("--serve", action="store_true", help="Serve flows for development")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Config file is required for most operations
    if not args.config and not args.serve:
        print("❌ --config is required unless using --serve")
        parser.print_help()
        return 1
    
    try:
        if args.serve:
            print("🚀 Starting Prefect flow server...")
            serve_flows()
            return 0
            
        elif args.validate_only:
            print("🔍 Running configuration validation...")
            result = validation_only_flow(args.config)
            print("✅ Configuration validation completed!")
            return 0
            
        elif args.run:
            print("🚀 Running full pipeline...")
            print("🔧 Enhanced with integrated ESI field normalisation")
            if args.generate_html:
                print("🎨 HTML dashboard generation enabled")
            
            # Run your existing flow with ESI enhancement
            result = dataset_comparison_flow(
                args.config, 
                include_json_extraction=args.include_json,
                generate_html=args.generate_html
            )
            
            print("\n" + "=" * 70)
            print("📊 PIPELINE EXECUTION SUMMARY")
            print("=" * 70)
            print(f"Status: {result.get('pipeline_status', 'UNKNOWN')}")
            
            if result.get('pipeline_status') == 'SUCCESS':
                print(f"✅ Total Duration: {result.get('total_execution_time_seconds', 0):.2f} seconds")
                print(f"📈 Datasets Processed: {result.get('data_statistics', {}).get('total_datasets_extracted', 0)}")
                print(f"🔄 Comparisons Completed: {result.get('data_statistics', {}).get('successful_comparisons', 0)}")
                
                # NEW: Show ESI normalisation results
                esi_fields_normalised = result.get('data_statistics', {}).get('esi_fields_normalised', 0)
                if esi_fields_normalised > 0:
                    print(f"🔧 ESI Fields Normalised: {esi_fields_normalised}")
                
                # Show HTML generation results (keeping your existing logic)
                if result.get('step_completion', {}).get('html_generation', False):
                    html_dashboards = result.get('data_statistics', {}).get('html_dashboards_generated', 0)
                    html_size = result.get('html_output', {}).get('total_size_mb', 0)
                    html_dir = result.get('html_output', {}).get('output_directory', 'Unknown')
                    
                    print(f"🎨 HTML Dashboards Generated: {html_dashboards}")
                    print(f"📁 HTML Output Directory: {html_dir}")
                    print(f"📊 Total HTML Size: {html_size:.2f} MB")
                
                print("\n📊 Step Performance:")
                step_perf = result.get('step_performance', {})
                for step, duration in step_perf.items():
                    if isinstance(duration, (int, float)):
                        # Enhanced step names for clarity
                        step_display = step.title()
                        if step == 'normalisation':
                            step_display = "Normalisation (inc. ESI)"
                        print(f"  {step_display}: {duration:.2f}s")
                
                # NEW: Show ESI normalisation details if available
                if 'esi_normalisation_details' in result:
                    esi_details = result['esi_normalisation_details']
                    print(f"\n🔧 ESI Normalisation Details:")
                    print(f"  DataFrames Processed: {esi_details.get('dataframes_processed', 0)}")
                    print(f"  Fields Normalised: {esi_details.get('fields_normalised', 0)}")
                    print(f"  Canonical Fields Available: {esi_details.get('total_canonical_fields', 0)}")
                    
                    if esi_details.get('normalisation_errors', 0) > 0:
                        print(f"  ⚠️ Normalisation Errors: {esi_details['normalisation_errors']}")
                
                print("=" * 70)
                return 0
            else:
                print(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")
                return 1
                
        else:
            parser.print_help()
            return 1
            
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        if args.verbose:
            print(f"📋 Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
