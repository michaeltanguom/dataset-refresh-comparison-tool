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
from src.extract.extract_light_transform import DataNormaliser
    
# Transform and load components
from src.load.load_duckdb import DataLoader
from src.transform.clean_duckdb_tables import DataCleaner
from src.transform.compare_datasets import DataComparator
    
# Utilities
from src.utils.exceptions import (
    ConfigurationError, ExtractionError, NormalisationError, 
    DatabaseError, DataQualityError, ComparisonError, PipelineError
)
from src.utils.logging_config import setup_logging, get_logger

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
    name="normalise_datasets",
    description="Normalise column names and validate data mappings using refactored modules",
    retries=1,
    retry_delay_seconds=15
)
def normalise_datasets_task(config_path: str, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
    """Prefect task for data normalisation with proper timing"""
    logger = get_run_logger()
    logger.info("🔄 Starting data normalisation with refactored modules")
    
    # 🔧 FIX: Add timing at task level
    task_start_time = time.time()
    
    try:
        config = ConfigManager(config_path)
        normaliser = DataNormaliser(config)
        
        # Normalise all extracted data
        normalised_data = normaliser.normalise_datasets(
            extraction_results['extracted_data']
        )
        
        # 🔧 FIX: Calculate task-level timing
        task_duration = time.time() - task_start_time
        
        normalisation_results = {
            'normalised_datasets': len(normalised_data),
            'normalised_data': normalised_data,
            'performance_metrics': {
                'normalisation_duration': task_duration  # 🔧 FIX: Use task timing
            }
        }
        
        logger.info(f"✅ Normalisation completed: {normalisation_results['normalised_datasets']} datasets in {task_duration:.2f}s")
        return normalisation_results
        
    except Exception as e:
        logger.error(f"❌ Data normalisation failed: {e}")
        raise NormalisationError(f"Data normalisation failed: {e}")

@task(
    name="load_to_database",
    description="Load normalised data into DuckDB database using refactored modules",
    retries=2,
    retry_delay_seconds=30
)
def load_to_database_task(config_path: str, normalisation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Prefect task for database loading with proper timing"""
    logger = get_run_logger()
    logger.info("🔄 Starting database loading with refactored modules")
    
    # 🔧 FIX: Add timing at task level
    task_start_time = time.time()
    
    try:
        config = ConfigManager(config_path)
        loader = DataLoader(config)
        
        # Load all normalised data
        loaded_tables = loader.load_datasets(normalisation_results['normalised_data'])
        
        # Validate loaded data
        validation_results = loader.validate_loaded_data(loaded_tables)
        
        if not validation_results['is_valid']:
            raise DatabaseError(f"Data validation failed after loading: {validation_results['errors']}")
        
        # 🔧 FIX: Calculate task-level timing
        task_duration = time.time() - task_start_time
        
        loading_results = {
            'loaded_tables': loaded_tables,
            'tables_count': len(loaded_tables),
            'total_rows_loaded': validation_results['total_rows_validated'],
            'performance_metrics': {
                'loading_duration': task_duration  # 🔧 FIX: Use task timing
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

#@task(
    name="extract_comparison_reports",
    description="Extract JSON comparison reports for HTML generation",
    retries=1,
    retry_delay_seconds=15
#)
#def extract_comparison_reports_task(config_path: str, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
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
    name="generate_pipeline_summary",
    description="Generate comprehensive pipeline execution summary",
    retries=0
)
def generate_pipeline_summary(config_path: str, 
                            extraction_results: Dict[str, Any],
                            normalisation_results: Dict[str, Any],
                            loading_results: Dict[str, Any],
                            cleaning_results: Dict[str, Any],
                            comparison_results: Dict[str, Any],
                            json_extraction_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate comprehensive pipeline execution summary
    USING YOUR ORIGINAL WORKING TIMING METHOD
    """
    logger = get_run_logger()
    logger.info("📊 Generating pipeline execution summary")
    
    try:
        # 🔧 Sum the individual step durations
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
        
        summary = {
            'pipeline_status': 'SUCCESS',
            'execution_timestamp': datetime.now().isoformat(),
            'config_file': config_path,
            'total_execution_time_seconds': total_duration,  # 🔧 YOUR ORIGINAL FIELD NAME
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
                'failed_comparisons': comparison_results['failed_comparisons']
            },
            'step_completion': {
                'extraction': True,
                'normalisation': True,
                'loading': True,
                'cleaning': True,
                'comparison': True,
                'json_extraction': json_extraction_results is not None
            }
        }
        
        # Add JSON extraction statistics if present
        if json_extraction_results:
            summary['step_performance']['json_extraction'] = json_extraction_results['performance_metrics'].get('json_extraction_duration', 0)
            summary['data_statistics']['json_reports_extracted'] = json_extraction_results['extracted_reports']
            summary['data_statistics']['total_researcher_records'] = json_extraction_results['total_researcher_records']
        
        logger.info("✅ Pipeline execution summary generated")
        logger.info(f"📈 Total execution time: {total_duration:.2f} seconds")
        logger.info(f"📊 Datasets processed: {summary['data_statistics']['total_datasets_extracted']}")
        logger.info(f"🔄 Comparisons completed: {summary['data_statistics']['successful_comparisons']}")
        
        if json_extraction_results:
            logger.info(f"📄 JSON reports extracted: {summary['data_statistics']['json_reports_extracted']}")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Failed to generate pipeline summary: {e}")
        return {
            'pipeline_status': 'PARTIAL_SUCCESS',
            'error': str(e),
            'execution_timestamp': datetime.now().isoformat()
        }

# ====================================================
# PREFECT FLOWS - Main workflow orchestration
# ====================================================

@flow(
    name="dataset-comparison-pipeline-refactored",
    description="Complete EtLT pipeline using refactored extraction modules",
    version="2.0.0",
    timeout_seconds=3600,
    log_prints=True
)
def dataset_comparison_flow(config_path: str, include_json_extraction: bool = False) -> Dict[str, Any]:
    """
    Main Prefect flow using YOUR ORIGINAL TIMING APPROACH
    """
    logger = get_run_logger()
    logger.info("🚀 Starting Dataset Comparison Pipeline (Refactored)")
    logger.info(f"📁 Configuration: {config_path}")
    
    try:
        # Step 1: Validate Configuration
        validation_results = validate_pipeline_configuration(config_path)
        
        # Step 2: Extract Data (using new ExcelDataExtractor)
        extraction_results = extract_datasets_task(config_path)
        
        # Step 3: Normalise Data (using DataNormaliser directly)
        normalisation_results = normalise_datasets_task(config_path, extraction_results)
        
        # Step 4: Load to Database (using DataLoader directly)
        loading_results = load_to_database_task(config_path, normalisation_results)
        
        # Step 5: Clean Data (using DataCleaner directly)
        cleaning_results = clean_data_task(config_path, loading_results)
        
        # Step 6: Compare Datasets (using DataComparator directly)
        comparison_results = compare_datasets_task(config_path, cleaning_results)
        
        # Step 7: Extract JSON Reports (NEW - conditional)
        #json_extraction_results = None
        #if include_json_extraction:
            #logger.info("🔄 Including JSON extraction for HTML generation")
            #json_extraction_results = extract_comparison_reports_task(config_path, comparison_results)
        
        # Step 8: Generate Summary (using YOUR ORIGINAL @task approach)
        pipeline_summary = generate_pipeline_summary(
            config_path,
            extraction_results,
            normalisation_results,
            loading_results,
            cleaning_results,
            comparison_results,
            #json_extraction_results  # NEW parameter
        )
        
        logger.info("🎉 Pipeline completed successfully!")
        return pipeline_summary
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed with error: {e}")
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

def main():
    """Main execution function with enhanced options for refactored architecture"""
    parser = argparse.ArgumentParser(
        description="Prefect Orchestration for Dataset Comparison Pipeline (Refactored)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with refactored modules
  python prefect_orchestration.py --config config/comparison_config.yaml --run

  # Run with JSON extraction for HTML generation
  python prefect_orchestration.py --config config/comparison_config.yaml --run --include-json

  # Validate configuration only
  python prefect_orchestration.py --config config/comparison_config.yaml --validate-only
        """
    )
    
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument("--run", action="store_true", help="Run the full pipeline immediately")
    parser.add_argument("--include-json", action="store_true", help="Include JSON extraction for HTML generation")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration")
    parser.add_argument("--serve", action="store_true", help="Serve flows for development and testing")
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
            print(f"📊 Result: {result}")
            return 0
            
        elif args.run:
            print("🚀 Running full pipeline with refactored modules...")
            result = dataset_comparison_flow(args.config, include_json_extraction=args.include_json)
            
            print("\n" + "=" * 60)
            print("📊 PIPELINE EXECUTION SUMMARY")
            print("=" * 60)
            print(f"Status: {result.get('pipeline_status', 'UNKNOWN')}")
            
            if result.get('pipeline_status') == 'SUCCESS':
                # 🔧 USE YOUR ORIGINAL FIELD NAME
                print(f"✅ Total Duration: {result.get('total_execution_time_seconds', 0):.2f} seconds")
                print(f"📈 Datasets Processed: {result.get('data_statistics', {}).get('total_datasets_extracted', 0)}")
                print(f"🔄 Comparisons Completed: {result.get('data_statistics', {}).get('successful_comparisons', 0)}")
                
                print("\n📊 Step Performance:")
                step_perf = result.get('step_performance', {})
                for step, duration in step_perf.items():
                    if isinstance(duration, (int, float)):  # Skip nested dicts
                        print(f"  {step.title()}: {duration:.2f}s")
                
                if result.get('step_completion', {}).get('json_extraction', False):
                    print(f"  JSON Reports Extracted: {result.get('data_statistics', {}).get('json_reports_extracted', 0)}")
                
                print("=" * 60)
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
