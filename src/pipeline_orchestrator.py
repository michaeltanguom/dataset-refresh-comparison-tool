"""
Pipeline Orchestrator for Dataset Comparison System
Coordinates the complete EtLT pipeline: Extract → Light Transform → Load → Transform (Clean + Compare)

Run as a module without Prefect integration
python -m src.pipeline_orchestrator --config config/comparison_config.yaml
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import argparse

# Add src to path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    # Try relative imports first (when run as part of src package)
    from .config.config_manager import ConfigManager
    from .config.config_validator import ConfigValidator
    from .extract.extract_light_transform import ExcelDataExtractor, DataNormaliser
    from .load.load_duckdb import DataLoader
    from .transform.clean_duckdb_tables import DataCleaner
    from .transform.compare_datasets import DataComparator
    from .utils.exceptions import (
        ConfigurationError, ExtractionError, NormalisationError, 
        DatabaseError, DataQualityError, ComparisonError, PipelineError
    )
    from .utils.logging_config import setup_logging, get_logger
    from .utils.common import format_number_with_commas
except ImportError:
    # Fall back to absolute imports (when run directly)
    from config.config_manager import ConfigManager
    from config.config_validator import ConfigValidator
    from extract.extract_light_transform import ExcelDataExtractor, DataNormaliser
    from load.load_duckdb import DataLoader
    from transform.clean_duckdb_tables import DataCleaner
    from transform.compare_datasets import DataComparator
    from utils.exceptions import (
        ConfigurationError, ExtractionError, NormalisationError, 
        DatabaseError, DataQualityError, ComparisonError, PipelineError
    )
    from utils.logging_config import setup_logging, get_logger
    from utils.common import format_number_with_commas

class PipelineOrchestrator:
    """
    Main pipeline orchestrator for dataset comparison
    Single responsibility: Coordinate the complete EtLT pipeline
    """
    
    def __init__(self, config_path: str):
        """
        Initialise pipeline orchestrator
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = None
        self.logger = None
        self.pipeline_start_time = None
        
        # Pipeline state tracking
        self.pipeline_state = {
            'config_loaded': False,
            'extraction_completed': False,
            'normalisation_completed': False,
            'loading_completed': False,
            'cleaning_completed': False,
            'comparison_completed': False
        }
        
        # Results storage
        self.results = {
            'extracted_data': {},
            'normalised_data': {},
            'loaded_tables': [],
            'cleaned_tables': [],
            'comparison_results': {},
            'validation_issues': [],
            'performance_metrics': {}
        }
        
        # Initialise configuration and logging
        self._initialise_configuration()
        self._initialise_logging()
        
    def _initialise_configuration(self) -> None:
        """Initialise and validate configuration"""
        try:
            # Load configuration
            self.config = ConfigManager(self.config_path)
            self.pipeline_state['config_loaded'] = True
            
            # Validate configuration
            validator = ConfigValidator(self.config)
            validation_results = validator.validate_all()
            
            if not validation_results['is_valid']:
                error_count = validation_results['errors']
                raise ConfigurationError(f"Configuration validation failed with {error_count} errors")
                
            # Store validation results
            self.results['validation_issues'] = validation_results['details']['warnings']
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialise configuration: {e}")
    
    def _initialise_logging(self) -> None:
        """Initialise logging system"""
        try:
            logging_config = self.config.get_logging_config()
            setup_logging(logging_config)
            self.logger = get_logger('orchestrator')
            self.logger.info("Pipeline orchestrator initialised successfully")
            
        except Exception as e:
            # Fallback to basic logging
            import logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger('orchestrator')
            self.logger.warning(f"Failed to setup custom logging, using basic logging: {e}")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete EtLT pipeline
        
        Returns:
            Dictionary with pipeline results and performance metrics
        """
        self.pipeline_start_time = time.time()
        
        try:
            self.logger.info("=" * 80)
            self.logger.info("STARTING DATASET COMPARISON PIPELINE")
            self.logger.info("=" * 80)
            
            # Log configuration summary
            config_summary = self.config.get_config_summary()
            self.logger.info("Pipeline Configuration:")
            for key, value in config_summary.items():
                self.logger.info(f"  {key}: {value}")
            self.logger.info("")
            
            # Execute pipeline steps
            self._execute_extraction()
            self._execute_normalisation() 
            self._execute_loading()
            self._execute_cleaning()
            self._execute_comparison()
            
            # Generate final results
            final_results = self._generate_final_results()
            
            # Log completion
            total_duration = time.time() - self.pipeline_start_time
            self.logger.info("=" * 80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total execution time: {total_duration:.2f} seconds")
            self.logger.info("=" * 80)
            
            return final_results
            
        except Exception as e:
            self._handle_pipeline_error(e)
            raise
    
    def _execute_extraction(self) -> None:
        """Execute data extraction step"""
        step_start_time = time.time()
        
        try:
            self.logger.info("STEP 1: DATA EXTRACTION")
            self.logger.info("-" * 40)
            
            # Initialise extractor
            extractor = ExcelDataExtractor(self.config)
            
            # Extract from both datasets
            dataset_1_config = self.config.get_data_source_config('dataset_1')
            dataset_2_config = self.config.get_data_source_config('dataset_2')
            
            self.logger.info(f"Extracting from dataset 1: {dataset_1_config['folder']}")
            extracted_1 = extractor.extract_files(
                dataset_1_config['folder'], 
                dataset_1_config['period_name']
            )
            
            self.logger.info(f"Extracting from dataset 2: {dataset_2_config['folder']}")
            extracted_2 = extractor.extract_files(
                dataset_2_config['folder'], 
                dataset_2_config['period_name']
            )
            
            # Combine results
            self.results['extracted_data'] = {**extracted_1, **extracted_2}
            
            # Log summary
            total_datasets = len(self.results['extracted_data'])
            total_rows = sum(data['metadata'].row_count for data in self.results['extracted_data'].values())
            
            step_duration = time.time() - step_start_time
            self.results['performance_metrics']['extraction_duration'] = step_duration
            
            self.logger.info(f"Extraction completed: {total_datasets} datasets, {format_number_with_commas(total_rows)} total rows")
            self.logger.info(f"Extraction time: {step_duration:.2f} seconds")
            self.logger.info("")
            
            self.pipeline_state['extraction_completed'] = True
            
        except Exception as e:
            raise ExtractionError(f"Data extraction failed: {e}")
    
    def _execute_normalisation(self) -> None:
        """Execute data normalisation step"""
        step_start_time = time.time()
        
        try:
            self.logger.info("STEP 2: DATA NORMALISATION")
            self.logger.info("-" * 40)
            
            # Initialise normaliser
            normaliser = DataNormaliser(self.config)
            
            # Normalise all extracted data
            self.results['normalised_data'] = normaliser.normalise_datasets(
                self.results['extracted_data']
            )
            
            # Log summary
            total_datasets = len(self.results['normalised_data'])
            
            step_duration = time.time() - step_start_time
            self.results['performance_metrics']['normalisation_duration'] = step_duration
            
            self.logger.info(f"Normalisation completed: {total_datasets} datasets processed")
            self.logger.info(f"Normalisation time: {step_duration:.2f} seconds")
            self.logger.info("")
            
            self.pipeline_state['normalisation_completed'] = True
            
        except Exception as e:
            raise NormalisationError(f"Data normalisation failed: {e}")
    
    def _execute_loading(self) -> None:
        """Execute data loading step"""
        step_start_time = time.time()
        
        try:
            self.logger.info("STEP 3: DATA LOADING")
            self.logger.info("-" * 40)
            
            # Initialise loader
            loader = DataLoader(self.config)
            
            # Load all normalised data
            self.results['loaded_tables'] = loader.load_datasets(
                self.results['normalised_data']
            )
            
            # Validate loaded data
            validation_results = loader.validate_loaded_data(self.results['loaded_tables'])
            
            if not validation_results['is_valid']:
                raise DatabaseError(f"Data validation failed after loading: {validation_results['errors']}")
            
            # Log summary
            total_tables = len(self.results['loaded_tables'])
            total_rows = validation_results['total_rows_validated']
            
            step_duration = time.time() - step_start_time
            self.results['performance_metrics']['loading_duration'] = step_duration
            
            self.logger.info(f"Loading completed: {total_tables} tables, {format_number_with_commas(total_rows)} total rows")
            self.logger.info(f"Loading time: {step_duration:.2f} seconds")
            self.logger.info("")
            
            self.pipeline_state['loading_completed'] = True
            
        except Exception as e:
            raise DatabaseError(f"Data loading failed: {e}")
    
    def _execute_cleaning(self) -> None:
        """Execute data cleaning step"""
        step_start_time = time.time()
        
        try:
            self.logger.info("STEP 4: DATA CLEANING")
            self.logger.info("-" * 40)
            
            # Initialise cleaner
            cleaner = DataCleaner(self.config)
            
            # Clean all loaded tables
            self.results['cleaned_tables'] = cleaner.clean_all_tables(
                self.results['loaded_tables']
            )
            
            # Get cleaning summary
            cleaning_summary = cleaner.get_cleaning_summary(self.results['cleaned_tables'])
            
            # Log summary
            total_tables = cleaning_summary['tables_processed']
            total_rows = cleaning_summary['total_rows']
            
            step_duration = time.time() - step_start_time
            self.results['performance_metrics']['cleaning_duration'] = step_duration
            
            self.logger.info(f"Cleaning completed: {total_tables} tables, {format_number_with_commas(total_rows)} total rows")
            self.logger.info(f"Cleaning time: {step_duration:.2f} seconds")
            self.logger.info("")
            
            self.pipeline_state['cleaning_completed'] = True
            
        except Exception as e:
            raise DataQualityError(f"Data cleaning failed: {e}")
    
    def _execute_comparison(self) -> None:
        """Execute dataset comparison step"""
        step_start_time = time.time()
        
        try:
            self.logger.info("STEP 5: DATASET COMPARISON")
            self.logger.info("-" * 40)
            
            # Initialise comparator
            comparator = DataComparator(self.config)
            
            # Compare all matching tables
            self.results['comparison_results'] = comparator.compare_all_matching_tables(
                self.results['cleaned_tables']
            )
            
            # Save comparison reports
            saved_files = comparator.save_comparison_reports(self.results['comparison_results'])
            
            # Log summary
            summary = self.results['comparison_results']['summary']
            total_comparisons = summary['successful_comparisons']
            failed_comparisons = summary['failed_comparisons']
            
            step_duration = time.time() - step_start_time
            self.results['performance_metrics']['comparison_duration'] = step_duration
            
            self.logger.info(f"Comparison completed: {total_comparisons} successful, {failed_comparisons} failed")
            self.logger.info(f"Reports saved: {len(saved_files)} files")
            self.logger.info(f"Comparison time: {step_duration:.2f} seconds")
            self.logger.info("")
            
            # Log report locations
            self.logger.info("Generated reports:")
            for comparison_key, file_path in saved_files.items():
                self.logger.info(f"  {comparison_key}: {file_path}")
            
            self.pipeline_state['comparison_completed'] = True
            
        except Exception as e:
            raise ComparisonError(f"Dataset comparison failed: {e}")
    
    def _generate_final_results(self) -> Dict[str, Any]:
        """Generate comprehensive final results"""
        total_duration = time.time() - self.pipeline_start_time
        
        # Calculate aggregate statistics
        aggregate_stats = {
            'total_datasets_processed': len(self.results['extracted_data']),
            'total_tables_loaded': len(self.results['loaded_tables']),
            'total_tables_cleaned': len(self.results['cleaned_tables']),
            'total_comparisons_completed': self.results['comparison_results']['summary']['successful_comparisons'],
            'total_pipeline_duration_seconds': total_duration,
            'average_time_per_step_seconds': total_duration / 5  # 5 steps
        }
        
        # Performance breakdown
        performance_breakdown = self.results['performance_metrics'].copy()
        performance_breakdown['total_duration'] = total_duration
        
        # Calculate percentages
        if total_duration > 0:
            for step, duration in performance_breakdown.items():
                if step != 'total_duration':
                    percentage = (duration / total_duration) * 100
                    performance_breakdown[f"{step}_percentage"] = percentage
        
        # Pipeline health check
        all_steps_completed = all(self.pipeline_state.values())
        
        return {
            'pipeline_status': 'SUCCESS' if all_steps_completed else 'PARTIAL',
            'pipeline_state': self.pipeline_state,
            'aggregate_statistics': aggregate_stats,
            'performance_metrics': performance_breakdown,
            'comparison_summary': self.results['comparison_results']['summary'],
            'validation_issues_count': len(self.results['validation_issues']),
            'config_summary': self.config.get_config_summary(),
            'execution_timestamp': time.time()
        }
    
    def _handle_pipeline_error(self, error: Exception) -> None:
        """Handle pipeline errors with comprehensive logging"""
        total_duration = time.time() - self.pipeline_start_time if self.pipeline_start_time else 0
        
        self.logger.error("=" * 80)
        self.logger.error("PIPELINE FAILED")
        self.logger.error("=" * 80)
        self.logger.error(f"Error: {error}")
        self.logger.error(f"Error Type: {type(error).__name__}")
        self.logger.error(f"Execution time before failure: {total_duration:.2f} seconds")
        self.logger.error("")
        
        # Log pipeline state at time of failure
        self.logger.error("Pipeline state at failure:")
        for step, completed in self.pipeline_state.items():
            status = "✓" if completed else "✗"
            self.logger.error(f"  {status} {step}")
        
        self.logger.error("=" * 80)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'state': self.pipeline_state,
            'results_summary': {
                'extracted_datasets': len(self.results['extracted_data']),
                'normalised_datasets': len(self.results['normalised_data']),
                'loaded_tables': len(self.results['loaded_tables']),
                'cleaned_tables': len(self.results['cleaned_tables']),
                'comparison_results': bool(self.results['comparison_results'])
            }
        }


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Dataset Comparison Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_orchestrator.py --config config/comparison_config.yaml
  python pipeline_orchestrator.py --config config/comparison_config.yaml --validate-only
        """
    )
    
    parser.add_argument(
        "--config", 
        required=True, 
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--validate-only", 
        action="store_true",
        help="Only validate configuration, don't run pipeline"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    try:
        if args.validate_only:
            # Configuration validation only
            print("Validating configuration...")
            config = ConfigManager(args.config)
            validator = ConfigValidator(config)
            validation_report = validator.generate_validation_report()
            print(validation_report)
            
            validation_results = validator.validate_all()
            if validation_results['is_valid']:
                print("\n✅ Configuration is valid!")
                return 0
            else:
                print(f"\n❌ Configuration has {validation_results['errors']} errors")
                return 1
        else:
            # Full pipeline execution
            orchestrator = PipelineOrchestrator(args.config)
            results = orchestrator.run_full_pipeline()
            
            # Print summary
            print("\n" + "=" * 60)
            print("PIPELINE EXECUTION SUMMARY")
            print("=" * 60)
            print(f"Status: {results['pipeline_status']}")
            print(f"Total Duration: {results['performance_metrics']['total_duration']:.2f} seconds")
            print(f"Datasets Processed: {results['aggregate_statistics']['total_datasets_processed']}")
            print(f"Comparisons Completed: {results['aggregate_statistics']['total_comparisons_completed']}")
            
            if results['validation_issues_count'] > 0:
                print(f"Validation Issues: {results['validation_issues_count']}")
            
            print("\nPerformance Breakdown:")
            for step, duration in results['performance_metrics'].items():
                if step.endswith('_duration') and not step.endswith('_percentage'):
                    percentage = results['performance_metrics'].get(f"{step}_percentage", 0)
                    step_name = step.replace('_duration', '').replace('_', ' ').title()
                    print(f"  {step_name}: {duration:.2f}s ({percentage:.1f}%)")
            
            print("=" * 60)
            return 0
            
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
