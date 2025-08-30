"""
HTML Generation Orchestration - Updated for Unified Dashboard
Generates a single unified HTML dashboard from all JSON reports (replicating html_report_generator.py logic)

python prefect_html_orchestration.py --config /Users/work/Documents/GitHub/dataset-refresh-comparison-tool/config/comparison_config.yaml --run
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any
import argparse
from datetime import datetime

# Fix the path - go up two levels to reach project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from prefect import flow, task, get_run_logger

# Now the imports should work
from src.config.config_manager import ConfigManager
from src.extract.json_extractor import JSONDataExtractor
from src.html_generator.templates.template_factory import TemplateFactory
from src.html_generator.renderers.html_renderer import HtmlRenderer
from src.utils.exceptions import HtmlGenerationError

@task(
    name="extract_and_aggregate_json_reports",
    description="Extract and aggregate all JSON reports for unified dashboard generation",
    retries=1
)
def extract_and_aggregate_json_reports_task(config_path: str) -> Dict[str, Any]:
    """Extract and aggregate all JSON comparison reports for unified dashboard"""
    logger = get_run_logger()
    logger.info("Extracting and aggregating JSON comparison reports for unified dashboard")
    
    task_start_time = time.time()
    
    try:
        # Load main config
        config = ConfigManager(config_path)
        
        # Get HTML config path
        html_config_info = config.get_html_generation_config()
        html_config_path = html_config_info.get('config_path', 'html_generator_config.yaml')
        
        # Convert relative path to absolute if needed
        #if not Path(html_config_path).is_absolute():
        #    main_config_dir = Path(config_path).parent
        #    html_config_path = str(main_config_dir / html_config_path)
        
        logger.info(f"Loading HTML config from: {html_config_path}")
        
        # Verify the HTML config file exists
        if not Path(html_config_path).exists():
            raise HtmlGenerationError(f"HTML config file not found: {html_config_path}")
        
        # Load HTML config
        html_config = config.load_html_config(html_config_path)
        
        # Use JSONDataExtractor with the main config
        extractor = JSONDataExtractor(config)
        
        # Extract from comparison_reports directory
        input_source = html_config['html_generation']['input_source']
        
        # Make input_source path absolute if needed
        if not Path(input_source).is_absolute():
            project_root = Path(config_path).parent.parent if 'config/' in config_path else Path(config_path).parent
            input_source = str(project_root / input_source)
        
        logger.info(f"Extracting JSON reports from: {input_source}")
        extracted_reports = extractor.extract_files(input_source, "current")
        
        # Group reports by dataset type (highly_cited_only vs incites_researchers)
        grouped_reports = _group_reports_by_dataset_type(extracted_reports)
        
        task_duration = time.time() - task_start_time
        
        results = {
            'grouped_reports': grouped_reports,  # Dict with keys like 'highly_cited_only', 'incites_researchers'
            'total_reports_extracted': len(extracted_reports),
            'html_config': html_config,
            'performance_metrics': {
                'json_extraction_duration': task_duration
            }
        }
        
        logger.info(f"Extracted {results['total_reports_extracted']} JSON reports and grouped into {len(grouped_reports)} dataset types in {task_duration:.2f}s")
        for dataset_type, reports in grouped_reports.items():
            logger.info(f"  - {dataset_type}: {len(reports)} subject reports")
        
        return results
        
    except Exception as e:
        logger.error(f"JSON extraction and aggregation failed: {e}")
        raise HtmlGenerationError(f"JSON extraction and aggregation failed: {e}")


def _group_reports_by_dataset_type(extracted_reports: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Group extracted reports by dataset type (highly_cited_only vs incites_researchers)
    This replicates the folder-based grouping from the original html_report_generator.py
    """
    grouped = {}
    
    for report_key, report_data in extracted_reports.items():
        # Determine dataset type from comparison_id or report key
        comparison_id = report_data.get('json_data', {}).get('comparison_id', report_key)
        
        if 'highly_cited_only' in comparison_id.lower():
            dataset_type = 'highly_cited_only'
        elif 'incites_researchers' in comparison_id.lower():
            dataset_type = 'incites_researchers'
        else:
            # Default grouping - could be customised based on your data structure
            dataset_type = 'general_research'
        
        if dataset_type not in grouped:
            grouped[dataset_type] = {}
        
        # Extract subject name from comparison_id for grouping
        subject_name = _extract_subject_name(comparison_id)
        grouped[dataset_type][subject_name] = report_data['json_data']
    
    return grouped


def _extract_subject_name(comparison_id: str) -> str:
    """Extract clean subject name from comparison ID"""
    # Remove common suffixes
    clean_id = comparison_id.replace('_highly_cited_only', '').replace('_incites_researchers', '')
    return clean_id.replace('_', ' ').title()


@task(
    name="generate_unified_html_dashboards",
    description="Generate unified HTML dashboards from aggregated JSON reports",
    retries=1
)
def generate_unified_html_dashboards_task(config_path: str, aggregation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate unified HTML dashboards from aggregated JSON reports"""
    logger = get_run_logger()
    logger.info("Generating unified HTML dashboards")
    
    task_start_time = time.time()
    
    try:
        # Get HTML config from aggregation results
        html_config = aggregation_results['html_config']
        
        # Initialise components
        template_factory = TemplateFactory()
        html_renderer = HtmlRenderer(html_config)
        
        generated_files = {}
        successful_generations = 0
        failed_generations = 0
        
        # Process each dataset type (e.g., highly_cited_only, incites_researchers)
        for dataset_type, reports_data in aggregation_results['grouped_reports'].items():
            try:
                logger.info(f"Processing unified dashboard for dataset type: {dataset_type}")
                logger.info(f"  - Aggregating {len(reports_data)} subject reports")
                
                # Determine template for this dataset type
                template_name = _determine_unified_template(dataset_type, html_config)
                template_config = html_config['templates'][template_name]['config']
                
                # Create template instance
                template = template_factory.create_template(template_name, template_config)
                
                # Prepare unified data structure (replicating html_report_generator.py approach)
                unified_data = {
                    'reports_data': reports_data  # All subject reports for this dataset type
                }
                
                # Generate unified HTML dashboard
                html_content = template.generate_html(unified_data)
                
                # Generate descriptive filename
                dashboard_title = _generate_dashboard_title(dataset_type, len(reports_data))
                
                # Save HTML file with descriptive name
                file_path = html_renderer.save_html(html_content, dashboard_title, template_name)
                generated_files[dataset_type] = file_path
                
                successful_generations += 1
                logger.info(f"Generated unified dashboard for {dataset_type}")
                logger.info(f"  - File: {file_path}")
                logger.info(f"  - Subjects covered: {len(reports_data)}")
                
            except Exception as e:
                logger.error(f"Failed to generate unified dashboard for {dataset_type}: {e}")
                failed_generations += 1
                continue
        
        task_duration = time.time() - task_start_time
        
        results = {
            'generated_dashboards': successful_generations,
            'failed_generations': failed_generations,
            'generated_files': generated_files,
            'output_summary': html_renderer.get_output_summary(generated_files),
            'performance_metrics': {
                'html_generation_duration': task_duration
            }
        }
        
        logger.info(f"Generated {successful_generations} unified dashboards in {task_duration:.2f}s")
        return results
        
    except Exception as e:
        logger.error(f"Unified HTML generation failed: {e}")
        raise HtmlGenerationError(f"Unified HTML generation failed: {e}")


def _determine_unified_template(dataset_type: str, html_config: Dict[str, Any]) -> str:
    """Determine which template to use for a unified dataset type"""
    template_mapping = html_config['html_generation']['template_mapping']
    default_template = html_config['html_generation']['default_template']
    
    # Check if dataset type has a specific template mapping
    for sheet_type, template_name in template_mapping.items():
        if sheet_type in dataset_type:
            return template_name
    
    return default_template


def _generate_dashboard_title(dataset_type: str, subject_count: int) -> str:
    """Generate descriptive title for the unified dashboard"""
    dataset_name = dataset_type.replace('_', ' ').title()
    return f"unified_{dataset_type}_{subject_count}_subjects"


@flow(
    name="unified-html-dashboard-generation", 
    description="Generate unified HTML dashboards from JSON comparison reports",
    version="2.0.0",
    timeout_seconds=1800
)
def unified_html_generation_flow(config_path: str) -> Dict[str, Any]:
    """Unified HTML generation flow - replicating html_report_generator.py logic"""
    logger = get_run_logger()
    logger.info("Starting Unified HTML Dashboard Generation")
    
    try:
        # Step 1: Extract and aggregate JSON reports by dataset type
        aggregation_results = extract_and_aggregate_json_reports_task(config_path)
        
        # Step 2: Generate unified HTML dashboards
        generation_results = generate_unified_html_dashboards_task(config_path, aggregation_results)
        
        # Summary
        total_duration = (
            aggregation_results['performance_metrics']['json_extraction_duration'] +
            generation_results['performance_metrics']['html_generation_duration']
        )
        
        # Return schema that matches main orchestration expectations
        summary = {
            'pipeline_status': 'SUCCESS',
            'execution_timestamp': datetime.now().isoformat(),
            'total_execution_time_seconds': total_duration,
            
            # MAIN Prefect ORCHESTRATION EXPECTED FIELDS:
            'dashboards_generated': generation_results['generated_dashboards'],  # ← Changed key name
            'failed_generations': generation_results['failed_generations'],
            'output_summary': generation_results['output_summary'],
            'performance_metrics': {
                'html_task_duration': total_duration  # ← Added expected structure
            },
            
            # ADDITIONAL METADATA for generated summary:
            'individual_reports_processed': aggregation_results['total_reports_extracted'],
            'dashboard_coverage': {
                dataset_type: len(reports)
                for dataset_type, reports in aggregation_results['grouped_reports'].items()
            }
        }
        
        logger.info("Unified HTML generation completed successfully!")
        logger.info(f"Generated {summary['dashboards_generated']} unified dashboards")
        
        return summary
        
    except Exception as e:
        logger.error(f"Unified HTML generation failed: {e}")
        return {
            'pipeline_status': 'FAILED',
            'error': str(e),
            'execution_timestamp': datetime.now().isoformat(),
            'dashboards_generated': 0,  # ← Consistent schema on failure
            'failed_generations': 0,
            'performance_metrics': {'html_task_duration': 0}
        }

def main():
    """CLI for unified HTML generation"""
    parser = argparse.ArgumentParser(description="Unified HTML Dashboard Generation")
    parser.add_argument("--config", required=True, help="Path to main configuration file")
    parser.add_argument("--run", action="store_true", help="Run unified HTML generation")
    
    args = parser.parse_args()
    
    if args.run:
        result = unified_html_generation_flow(args.config)
        
        if result['pipeline_status'] == 'SUCCESS':
            print(f"Generated {result['dashboards_generated']} unified dashboards")  # ← Updated key
            print(f"Processed {result['individual_reports_processed']} individual reports")
            print(f"Duration: {result['total_execution_time_seconds']:.2f}s")
            return 0
        else:
            print(f"Failed: {result.get('error', 'Unknown error')}")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())