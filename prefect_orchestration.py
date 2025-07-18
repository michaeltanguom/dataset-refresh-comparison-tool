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
from datetime import datetime, timedelta
import traceback

# Prefect imports (compatible with Prefect 3.x)
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash

# Add current directory to path for proper package imports
sys.path.insert(0, str(Path(__file__).parent))

# Import your existing pipeline components (now as a proper package)
from src.pipeline_orchestrator import PipelineOrchestrator
from src.config.config_manager import ConfigManager
from src.config.config_validator import ConfigValidator
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
    description="Extract data from Excel files in input folders",
    retries=2,
    retry_delay_seconds=30
)
def extract_datasets_task(config_path: str) -> Dict[str, Any]:
    """
    Prefect task for data extraction
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Extraction results
    """
    logger = get_run_logger()
    logger.info("🔄 Starting data extraction")
    
    try:
        # Create orchestrator and run extraction only
        orchestrator = PipelineOrchestrator(config_path)
        orchestrator._execute_extraction()
        
        # Return extraction results
        extraction_results = {
            'extracted_datasets': len(orchestrator.results['extracted_data']),
            'total_rows': sum(data['metadata'].row_count for data in orchestrator.results['extracted_data'].values()),
            'performance_metrics': orchestrator.results['performance_metrics'],
            'extracted_data': orchestrator.results['extracted_data']  # Pass data to next task
        }
        
        logger.info(f"✅ Extraction completed: {extraction_results['extracted_datasets']} datasets, "
                   f"{extraction_results['total_rows']:,} rows")
        
        return extraction_results
        
    except Exception as e:
        logger.error(f"❌ Data extraction failed: {e}")
        raise


@task(
    name="normalise_datasets",
    description="Normalise column names and validate data mappings",
    retries=1,
    retry_delay_seconds=15
)
def normalise_datasets_task(config_path: str, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefect task for data normalisation
    
    Args:
        config_path: Path to configuration file
        extraction_results: Results from extraction task
        
    Returns:
        Normalisation results
    """
    logger = get_run_logger()
    logger.info("🔄 Starting data normalisation")
    
    try:
        # Create orchestrator and inject extraction results
        orchestrator = PipelineOrchestrator(config_path)
        orchestrator.results['extracted_data'] = extraction_results['extracted_data']
        orchestrator.pipeline_state['extraction_completed'] = True
        
        # Run normalisation
        orchestrator._execute_normalisation()
        
        normalisation_results = {
            'normalised_datasets': len(orchestrator.results['normalised_data']),
            'performance_metrics': orchestrator.results['performance_metrics'],
            'normalised_data': orchestrator.results['normalised_data']  # Pass data to next task
        }
        
        logger.info(f"✅ Normalisation completed: {normalisation_results['normalised_datasets']} datasets")
        
        return normalisation_results
        
    except Exception as e:
        logger.error(f"❌ Data normalisation failed: {e}")
        raise


@task(
    name="load_to_database",
    description="Load normalised data into DuckDB database",
    retries=2,
    retry_delay_seconds=30
)
def load_to_database_task(config_path: str, normalisation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefect task for database loading
    
    Args:
        config_path: Path to configuration file
        normalisation_results: Results from normalisation task
        
    Returns:
        Loading results
    """
    logger = get_run_logger()
    logger.info("🔄 Starting database loading")
    
    try:
        # Create orchestrator and inject previous results
        orchestrator = PipelineOrchestrator(config_path)
        orchestrator.results['normalised_data'] = normalisation_results['normalised_data']
        orchestrator.pipeline_state['extraction_completed'] = True
        orchestrator.pipeline_state['normalisation_completed'] = True
        
        # Run loading
        orchestrator._execute_loading()
        
        loading_results = {
            'loaded_tables': orchestrator.results['loaded_tables'],
            'tables_count': len(orchestrator.results['loaded_tables']),
            'performance_metrics': orchestrator.results['performance_metrics']
        }
        
        logger.info(f"✅ Database loading completed: {loading_results['tables_count']} tables loaded")
        
        return loading_results
        
    except Exception as e:
        logger.error(f"❌ Database loading failed: {e}")
        raise


@task(
    name="clean_data",
    description="Clean and validate data quality in database tables",
    retries=1,
    retry_delay_seconds=20
)
def clean_data_task(config_path: str, loading_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefect task for data cleaning
    
    Args:
        config_path: Path to configuration file
        loading_results: Results from loading task
        
    Returns:
        Cleaning results
    """
    logger = get_run_logger()
    logger.info("🔄 Starting data cleaning")
    
    try:
        # Create orchestrator and inject previous results
        orchestrator = PipelineOrchestrator(config_path)
        orchestrator.results['loaded_tables'] = loading_results['loaded_tables']
        orchestrator.pipeline_state['extraction_completed'] = True
        orchestrator.pipeline_state['normalisation_completed'] = True
        orchestrator.pipeline_state['loading_completed'] = True
        
        # Run cleaning
        orchestrator._execute_cleaning()
        
        cleaning_results = {
            'cleaned_tables': orchestrator.results['cleaned_tables'],
            'tables_count': len(orchestrator.results['cleaned_tables']),
            'performance_metrics': orchestrator.results['performance_metrics']
        }
        
        logger.info(f"✅ Data cleaning completed: {cleaning_results['tables_count']} tables cleaned")
        
        return cleaning_results
        
    except Exception as e:
        logger.error(f"❌ Data cleaning failed: {e}")
        raise


@task(
    name="compare_datasets",
    description="Compare datasets and generate reports",
    retries=1,
    retry_delay_seconds=30
)
def compare_datasets_task(config_path: str, cleaning_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefect task for dataset comparison
    
    Args:
        config_path: Path to configuration file
        cleaning_results: Results from cleaning task
        
    Returns:
        Comparison results
    """
    logger = get_run_logger()
    logger.info("🔄 Starting dataset comparison")
    
    try:
        # Create orchestrator and inject previous results
        orchestrator = PipelineOrchestrator(config_path)
        orchestrator.results['cleaned_tables'] = cleaning_results['cleaned_tables']
        orchestrator.pipeline_state['extraction_completed'] = True
        orchestrator.pipeline_state['normalisation_completed'] = True
        orchestrator.pipeline_state['loading_completed'] = True
        orchestrator.pipeline_state['cleaning_completed'] = True
        
        # Run comparison
        orchestrator._execute_comparison()
        
        comparison_results = {
            'comparison_results': orchestrator.results['comparison_results'],
            'successful_comparisons': orchestrator.results['comparison_results']['summary']['successful_comparisons'],
            'failed_comparisons': orchestrator.results['comparison_results']['summary']['failed_comparisons'],
            'performance_metrics': orchestrator.results['performance_metrics']
        }
        
        logger.info(f"✅ Dataset comparison completed: {comparison_results['successful_comparisons']} successful, "
                   f"{comparison_results['failed_comparisons']} failed")
        
        return comparison_results
        
    except Exception as e:
        logger.error(f"❌ Dataset comparison failed: {e}")
        raise


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
                            comparison_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comprehensive pipeline execution summary
    
    Args:
        config_path: Path to configuration file
        extraction_results: Extraction task results
        normalisation_results: Normalisation task results
        loading_results: Loading task results
        cleaning_results: Cleaning task results
        comparison_results: Comparison task results
        
    Returns:
        Pipeline execution summary
    """
    logger = get_run_logger()
    logger.info("📊 Generating pipeline execution summary")
    
    try:
        # Aggregate all performance metrics
        total_duration = (
            extraction_results['performance_metrics'].get('extraction_duration', 0) +
            normalisation_results['performance_metrics'].get('normalisation_duration', 0) +
            loading_results['performance_metrics'].get('loading_duration', 0) +
            cleaning_results['performance_metrics'].get('cleaning_duration', 0) +
            comparison_results['performance_metrics'].get('comparison_duration', 0)
        )
        
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
                'failed_comparisons': comparison_results['failed_comparisons']
            },
            'step_completion': {
                'extraction': True,
                'normalisation': True,
                'loading': True,
                'cleaning': True,
                'comparison': True
            }
        }
        
        logger.info("✅ Pipeline execution summary generated")
        logger.info(f"📈 Total execution time: {total_duration:.2f} seconds")
        logger.info(f"📊 Datasets processed: {summary['data_statistics']['total_datasets_extracted']}")
        logger.info(f"🔄 Comparisons completed: {summary['data_statistics']['successful_comparisons']}")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Failed to generate pipeline summary: {e}")
        # Return partial summary even if generation fails
        return {
            'pipeline_status': 'PARTIAL_SUCCESS',
            'error': str(e),
            'execution_timestamp': datetime.now().isoformat()
        }


# ====================================================
# PREFECT FLOWS - Main workflow orchestration
# ====================================================

@flow(
    name="dataset-comparison-pipeline",
    description="Complete EtLT pipeline for dataset comparison with monitoring and retry logic",
    version="1.0.0",
    timeout_seconds=3600,  # 1 hour timeout
    log_prints=True
)
def dataset_comparison_flow(config_path: str) -> Dict[str, Any]:
    """
    Main Prefect flow for dataset comparison pipeline
    Orchestrates all pipeline steps with proper dependency management
    
    Args:
        config_path: Path to pipeline configuration file
        
    Returns:
        Pipeline execution results
    """
    logger = get_run_logger()
    logger.info("🚀 Starting Dataset Comparison Pipeline")
    logger.info(f"📁 Configuration: {config_path}")
    
    try:
        # Step 1: Validate Configuration
        validation_results = validate_pipeline_configuration(config_path)
        
        # Step 2: Extract Data
        extraction_results = extract_datasets_task(config_path)
        
        # Step 3: Normalise Data (depends on extraction)
        normalisation_results = normalise_datasets_task(config_path, extraction_results)
        
        # Step 4: Load to Database (depends on normalisation)
        loading_results = load_to_database_task(config_path, normalisation_results)
        
        # Step 5: Clean Data (depends on loading)
        cleaning_results = clean_data_task(config_path, loading_results)
        
        # Step 6: Compare Datasets (depends on cleaning)
        comparison_results = compare_datasets_task(config_path, cleaning_results)
        
        # Step 7: Generate Summary (depends on all previous steps)
        pipeline_summary = generate_pipeline_summary(
            config_path,
            extraction_results,
            normalisation_results,
            loading_results,
            cleaning_results,
            comparison_results
        )
        
        logger.info("🎉 Pipeline completed successfully!")
        return pipeline_summary
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed with error: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        
        # Return failure summary
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
    """Main execution function with enhanced options for Prefect 3.x"""
    parser = argparse.ArgumentParser(
        description="Prefect Orchestration for Dataset Comparison Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with Prefect orchestration
  python prefect_orchestration.py --config config/comparison_config.yaml --run

  # Validate configuration only
  python prefect_orchestration.py --config config/comparison_config.yaml --validate-only

  # Serve flows for development (replaces deployments in Prefect 3.x)
  python prefect_orchestration.py --serve

  # Create deployment file for scheduled execution
  python prefect_orchestration.py --config config/comparison_config.yaml --create-deployment

  # Create deployment file with custom schedule (daily at 2 AM)
  python prefect_orchestration.py --config config/comparison_config.yaml --create-deployment --schedule "0 2 * * *"
        """
    )
    
    parser.add_argument(
        "--config", 
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--run", 
        action="store_true",
        help="Run the full pipeline immediately"
    )
    
    parser.add_argument(
        "--validate-only", 
        action="store_true",
        help="Only validate configuration, don't run pipeline"
    )
    
    parser.add_argument(
        "--serve", 
        action="store_true",
        help="Serve flows for development and testing"
    )
    
    parser.add_argument(
        "--create-deployment", 
        action="store_true",
        help="Create a prefect.yaml deployment file"
    )
    
    parser.add_argument(
        "--schedule", 
        default=None,
        help="Cron expression for scheduled deployment (e.g., '0 6 * * MON' for Mondays at 6 AM)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
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
            
        elif args.create_deployment:
            print("📄 Creating deployment file...")
            deployment_file = create_deployment_file(args.config, args.schedule)
            print(f"✅ Deployment file created: {deployment_file}")
            if args.schedule:
                print(f"📅 Schedule: {args.schedule}")
            else:
                print("📅 No schedule set (manual execution only)")
            print("\nTo deploy:")
            print(f"  prefect deploy --name dataset-comparison-pipeline")
            return 0
            
        elif args.run:
            print("🚀 Running full pipeline with Prefect orchestration...")
            result = dataset_comparison_flow(args.config)
            
            print("\n" + "=" * 60)
            print("📊 PIPELINE EXECUTION SUMMARY")
            print("=" * 60)
            print(f"Status: {result.get('pipeline_status', 'UNKNOWN')}")
            
            if result.get('pipeline_status') == 'SUCCESS':
                print(f"✅ Total Duration: {result.get('total_execution_time_seconds', 0):.2f} seconds")
                print(f"📈 Datasets Processed: {result.get('data_statistics', {}).get('total_datasets_extracted', 0)}")
                print(f"🔄 Comparisons Completed: {result.get('data_statistics', {}).get('successful_comparisons', 0)}")
                
                print("\n📊 Step Performance:")
                step_perf = result.get('step_performance', {})
                for step, duration in step_perf.items():
                    print(f"  {step.title()}: {duration:.2f}s")
                
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
