"""
Light Transform Orchestration - Individual Prefect Tasks
Coordinates all light transformation steps with individual task control
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from prefect import flow, task, get_run_logger

# Import individual transformation functions
from .column_mapping import column_mapping_transform
from .esi_normalisation import esi_normalisation_transform
from .duplicate_detection import duplicate_detection_transform
from .null_handling import null_handling_transform
from .validation import light_transform_validation

from ...utils.exceptions import NormalisationError, ValidationError


@task(
    name="column_mapping_transformation",
    description="Map Excel column names to standardised column names",
    retries=1,
    retry_delay_seconds=10
)
def column_mapping_task(config_path: str, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefect task wrapper for column mapping transformation
    
    Args:
        config_path: Path to configuration file
        extraction_results: Results from extraction task
        
    Returns:
        Column mapping results
    """
    logger = get_run_logger()
    logger.info("Starting column mapping transformation")
    
    task_start_time = time.time()
    
    try:
        # Call the transformation function
        results = column_mapping_transform(config_path, extraction_results)
        
        task_duration = time.time() - task_start_time
        
        # Enhance results with task-level timing
        if 'performance_metrics' not in results:
            results['performance_metrics'] = {}
        results['performance_metrics']['task_duration'] = task_duration
        
        logger.info(f"Column mapping completed: {results['transformation_summary']['datasets_processed']} datasets in {task_duration:.2f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"Column mapping transformation failed: {e}")
        raise NormalisationError(f"Column mapping transformation failed: {e}")


@task(
    name="esi_normalisation_transformation", 
    description="Standardise ESI field names to canonical format",
    retries=1,
    retry_delay_seconds=10
)
def esi_normalisation_task(config_path: str, column_mapping_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefect task wrapper for ESI field normalisation
    
    Args:
        config_path: Path to configuration file
        column_mapping_results: Results from column mapping task
        
    Returns:
        ESI normalisation results
    """
    logger = get_run_logger()
    logger.info("Starting ESI field normalisation transformation")
    
    task_start_time = time.time()
    
    try:
        # Call the transformation function
        results = esi_normalisation_transform(config_path, column_mapping_results)
        
        task_duration = time.time() - task_start_time
        
        # Enhance results with task-level timing
        if 'performance_metrics' not in results:
            results['performance_metrics'] = {}
        results['performance_metrics']['task_duration'] = task_duration
        
        logger.info(f"ESI normalisation completed: {results['transformation_summary']['records_changed']} fields normalised across {results['transformation_summary']['datasets_processed']} datasets in {task_duration:.2f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"ESI normalisation transformation failed: {e}")
        raise NormalisationError(f"ESI normalisation transformation failed: {e}")


@task(
    name="duplicate_detection_transformation",
    description="Flag duplicate records for manual review", 
    retries=1,
    retry_delay_seconds=10
)
def duplicate_detection_task(config_path: str, esi_normalisation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefect task wrapper for duplicate detection
    
    Args:
        config_path: Path to configuration file
        esi_normalisation_results: Results from ESI normalisation task
        
    Returns:
        Duplicate detection results
    """
    logger = get_run_logger()
    logger.info("Starting duplicate detection transformation")
    
    task_start_time = time.time()
    
    try:
        # Import here to avoid circular imports
        from .duplicate_detection import duplicate_detection_transform
        
        # Call the transformation function
        results = duplicate_detection_transform(config_path, esi_normalisation_results)
        
        task_duration = time.time() - task_start_time
        
        # Enhance results with task-level timing
        if 'performance_metrics' not in results:
            results['performance_metrics'] = {}
        results['performance_metrics']['task_duration'] = task_duration
        
        # Log results
        duplicates_flagged = results['transformation_summary'].get('duplicates_flagged', 0)
        duplicate_groups = results['transformation_summary'].get('duplicate_groups', 0)
        
        if duplicates_flagged > 0:
            logger.warning(f"DUPLICATES DETECTED: {duplicates_flagged} records across {duplicate_groups} groups")
            logger.warning("Review duplicates in the generated dashboard before production use")
        else:
            logger.info(f"Duplicate detection completed: No duplicates found in {results['transformation_summary']['datasets_processed']} datasets in {task_duration:.2f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"Duplicate detection transformation failed: {e}")
        raise NormalisationError(f"Duplicate detection transformation failed: {e}")

@task(
    name="null_handling_transformation",
    description="Handle NULL values based on configured strategy",
    retries=1, 
    retry_delay_seconds=10
)
def null_handling_task(config_path: str, duplicate_detection_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefect task wrapper for NULL handling
    
    Args:
        config_path: Path to configuration file
        duplicate_detetction_results: Results from duplicate detection task
        
    Returns:
        NULL handling results
    """
    logger = get_run_logger()
    logger.info("Starting NULL handling transformation")
    
    task_start_time = time.time()
    
    try:
        # Call the transformation function
        results = null_handling_transform(config_path, duplicate_detection_results)
        
        task_duration = time.time() - task_start_time
        
        # Enhance results with task-level timing
        if 'performance_metrics' not in results:
            results['performance_metrics'] = {}
        results['performance_metrics']['task_duration'] = task_duration
        
        logger.info(f"NULL handling completed: {results['transformation_summary']['records_changed']} NULL values processed across {results['transformation_summary']['datasets_processed']} datasets in {task_duration:.2f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"NULL handling transformation failed: {e}")
        raise NormalisationError(f"NULL handling transformation failed: {e}")


@task(
    name="light_transform_validation_final",
    description="Final validation of all light transform steps",
    retries=0  # No retries for validation - if it fails, data is invalid
)
def light_transform_validation_task(config_path: str, null_handling_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefect task wrapper for final light transform validation
    
    Args:
        config_path: Path to configuration file
        null_handling_results: Results from NULL handling task
        
    Returns:
        Validation results with final transformed data
    """
    logger = get_run_logger()
    logger.info("Starting light transform validation")
    
    task_start_time = time.time()
    
    try:
        # Call the validation function
        results = light_transform_validation(config_path, null_handling_results)
        
        task_duration = time.time() - task_start_time
        
        # Enhance results with task-level timing
        if 'performance_metrics' not in results:
            results['performance_metrics'] = {}
        results['performance_metrics']['task_duration'] = task_duration
        
        validation_summary = results['transformation_summary']
        if validation_summary['failed_records']:
            logger.warning(f"Light transform validation completed with {len(validation_summary['failed_records'])} validation warnings")
        else:
            logger.info(f"Light transform validation passed: {validation_summary['datasets_processed']} datasets validated in {task_duration:.2f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"Light transform validation failed: {e}")
        raise ValidationError(f"Light transform validation failed: {e}")


@flow(
    name="light-transform-orchestrated-pipeline",
    description="Orchestrated light transformation pipeline with individual task control",
    version="2.1.0",  # Updated version for duplicate detection changes
    timeout_seconds=1800,
    log_prints=True
)
def light_transform_orchestrated_flow(config_path: str, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrated light transform flow with individual task control
    Updated to use duplicate detection instead of removal
    
    Args:
        config_path: Path to configuration file
        extraction_results: Results from extraction task
        
    Returns:
        Final transformation results ready for database loading
    """
    logger = get_run_logger()
    logger.info("Starting orchestrated light transformation pipeline")
    logger.info(f"Processing {extraction_results.get('extracted_datasets', 0)} datasets with {extraction_results.get('total_rows', 0):,} total rows")
    
    pipeline_start_time = time.time()
    
    try:
        # Step 1: Column mapping transformation
        column_results = column_mapping_task(config_path, extraction_results)
        
        # Step 2: ESI field normalisation
        esi_results = esi_normalisation_task(config_path, column_results)
        
        # Step 3: Duplicate detection (updated from duplicate_removal_task)
        duplicate_detection_results = duplicate_detection_task(config_path, esi_results)
        
        # Step 4: NULL handling (updated variable name)
        null_results = null_handling_task(config_path, duplicate_detection_results)
        
        # Step 5: Final validation
        validation_results = light_transform_validation_task(config_path, null_results)
        
        # Calculate total pipeline duration
        total_duration = time.time() - pipeline_start_time
        
        # Prepare summary matching the original light_transform_dataframes_task structure
        summary = {
            'transformed_datasets': validation_results['transformation_summary']['datasets_processed'],
            'transformed_data': validation_results['transformed_data'],  # Final validated data
            'transformation_summary': {
                'total_records_processed': sum(
                    step_results['transformation_summary']['records_processed']
                    for step_results in [column_results, esi_results, duplicate_detection_results, null_results, validation_results]
                ) // 5,  # Divide by 5 as same records processed in each step
                'pipeline_steps_completed': 5,
                'column_mapping_changes': column_results['transformation_summary']['records_changed'],
                'esi_fields_normalised': esi_results['transformation_summary']['records_changed'],
                'duplicates_flagged': duplicate_detection_results['transformation_summary']['duplicates_flagged'],  # Updated key
                'duplicate_groups_found': duplicate_detection_results['transformation_summary']['duplicate_groups'],  # New key
                'null_values_processed': null_results['transformation_summary']['records_changed'],
                'validation_warnings': len(validation_results['transformation_summary']['failed_records']),
                'datasets_processed': validation_results['transformation_summary']['datasets_processed']
            },
            'performance_metrics': {
                'transformation_duration': total_duration,  # Total orchestrated pipeline duration
                'individual_step_durations': {
                    'column_mapping': column_results['performance_metrics']['task_duration'],
                    'esi_normalisation': esi_results['performance_metrics']['task_duration'],
                    'duplicate_detection': duplicate_detection_results['performance_metrics']['task_duration'],  # Updated key
                    'null_handling': null_results['performance_metrics']['task_duration'],
                    'validation': validation_results['performance_metrics']['task_duration']
                }
            }
        }
        
        logger.info("Orchestrated light transformation pipeline completed successfully")
        logger.info(f"Total pipeline duration: {total_duration:.2f}s")
        logger.info(f"Final datasets ready for loading: {summary['transformed_datasets']}")
        
        # Enhanced logging for duplicate detection
        duplicates_flagged = summary['transformation_summary']['duplicates_flagged']
        duplicate_groups = summary['transformation_summary']['duplicate_groups_found']
        
        if duplicates_flagged > 0:
            logger.warning(f"Data Quality Alert: {duplicates_flagged} duplicate records flagged across {duplicate_groups} groups")
            logger.warning("Review duplicate status in generated dashboards before production use")
        
        # Log step-by-step performance breakdown
        step_durations = summary['performance_metrics']['individual_step_durations']
        for step, duration in step_durations.items():
            logger.info(f"  {step.replace('_', ' ').title()}: {duration:.2f}s")
        
        return summary
        
    except Exception as e:
        pipeline_duration = time.time() - pipeline_start_time
        logger.error(f"Orchestrated light transformation pipeline failed after {pipeline_duration:.2f}s: {e}")
        
        # Return error result matching expected structure
        return {
            'pipeline_status': 'FAILED',
            'error': str(e),
            'execution_timestamp': datetime.now().isoformat(),
            'transformed_datasets': 0,
            'transformed_data': {},
            'performance_metrics': {'transformation_duration': pipeline_duration}
        }


# CLI interface for standalone testing
def main():
    """CLI for standalone light transform orchestration testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Light Transform Orchestrated Pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--test", action="store_true", help="Run test with mock data")
    
    args = parser.parse_args()
    
    if args.test:
        # Create mock extraction results for testing
        mock_extraction = {
            'extracted_datasets': 2,
            'total_rows': 1000,
            'extracted_data': {
                'test_table_1': {
                    'dataframe': None,  # Would contain actual DataFrame in real usage
                    'metadata': {'row_count': 500}
                },
                'test_table_2': {
                    'dataframe': None,
                    'metadata': {'row_count': 500}
                }
            }
        }
        
        print("Running light transform orchestration test...")
        result = light_transform_orchestrated_flow(args.config, mock_extraction)
        
        if result.get('pipeline_status') != 'FAILED':
            print(f"Test completed: {result['transformed_datasets']} datasets processed")
            print(f"Duration: {result['performance_metrics']['transformation_duration']:.2f}s")
            return 0
        else:
            print(f"Test failed: {result.get('error', 'Unknown error')}")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
