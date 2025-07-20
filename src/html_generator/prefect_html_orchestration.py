"""
HTML Generation Orchestration
Standalone Prefect script for generating HTML dashboards from JSON reports
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
    name="extract_json_reports",
    description="Extract JSON comparison reports for HTML generation",
    retries=1
)
def extract_json_reports_task(config_path: str) -> Dict[str, Any]:
    """Extract JSON comparison reports"""
    logger = get_run_logger()
    logger.info("🔄 Extracting JSON comparison reports")
    
    task_start_time = time.time()
    
    try:
        # Load main config
        config = ConfigManager(config_path)
        
        # Get HTML config path - but handle both relative and absolute paths
        html_config_info = config.get_html_generation_config()
        html_config_path = html_config_info.get('config_path', 'config/html_generator_config.yaml')
        
        # 🔧 FIX: Convert relative path to absolute if needed
        if not Path(html_config_path).is_absolute():
            # If it's relative, make it relative to the main config file's directory
            main_config_dir = Path(config_path).parent
            html_config_path = str(main_config_dir / html_config_path)
        
        logger.info(f"Loading HTML config from: {html_config_path}")
        
        # Verify the HTML config file exists
        if not Path(html_config_path).exists():
            raise HtmlGenerationError(f"HTML config file not found: {html_config_path}")
        
        # Load HTML config
        html_config = config.load_html_config(html_config_path)
        
        # Use JSONDataExtractor with the main config (it has the data source info)
        extractor = JSONDataExtractor(config)
        
        # Extract from comparison_reports directory
        input_source = html_config['html_generation']['input_source']
        
        # 🔧 FIX: Make input_source path absolute if needed
        if not Path(input_source).is_absolute():
            project_root = Path(config_path).parent.parent if 'config/' in config_path else Path(config_path).parent
            input_source = str(project_root / input_source)
        
        logger.info(f"Extracting JSON reports from: {input_source}")
        extracted_reports = extractor.extract_files(input_source, "current")
        
        task_duration = time.time() - task_start_time
        
        results = {
            'extracted_reports': len(extracted_reports),
            'extracted_data': extracted_reports,
            'html_config': html_config,  # 🔧 ADD: Pass config to next task
            'performance_metrics': {
                'json_extraction_duration': task_duration
            }
        }
        
        logger.info(f"✅ Extracted {results['extracted_reports']} JSON reports in {task_duration:.2f}s")
        return results
        
    except Exception as e:
        logger.error(f"❌ JSON extraction failed: {e}")
        raise HtmlGenerationError(f"JSON extraction failed: {e}")


@task(
    name="generate_html_dashboards",
    description="Generate HTML dashboards from JSON reports",
    retries=1
)
def generate_html_dashboards_task(config_path: str, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate HTML dashboards from extracted JSON reports"""
    logger = get_run_logger()
    logger.info("🔄 Generating HTML dashboards")
    
    task_start_time = time.time()
    
    try:
        # Get HTML config from extraction results (already loaded and validated)
        html_config = extraction_results['html_config']
        
        # Initialize components
        template_factory = TemplateFactory()
        html_renderer = HtmlRenderer(html_config)
        
        generated_files = {}
        successful_generations = 0
        failed_generations = 0
        
        # Process each extracted report
        for dataset_key, report_data in extraction_results['extracted_data'].items():
            try:
                logger.info(f"Processing dataset: {dataset_key}")
                
                # Determine template for this dataset
                template_name = _determine_template(dataset_key, html_config)
                template_config = html_config['templates'][template_name]['config']
                
                # Create template instance
                template = template_factory.create_template(template_name, template_config)
                
                # Generate HTML
                json_data = report_data['json_data']
                html_content = template.generate_html(json_data)
                
                # Save HTML file
                file_path = html_renderer.save_html(html_content, dataset_key, template_name)
                generated_files[dataset_key] = file_path
                
                successful_generations += 1
                logger.info(f"✅ Generated dashboard for {dataset_key}")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate dashboard for {dataset_key}: {e}")
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
        
        logger.info(f"✅ Generated {successful_generations} dashboards in {task_duration:.2f}s")
        return results
        
    except Exception as e:
        logger.error(f"❌ HTML generation failed: {e}")
        raise HtmlGenerationError(f"HTML generation failed: {e}")


def _determine_template(dataset_key: str, html_config: Dict[str, Any]) -> str:
    """Determine which template to use for a dataset"""
    template_mapping = html_config['html_generation']['template_mapping']
    default_template = html_config['html_generation']['default_template']
    
    # Check if dataset key contains known sheet types
    for sheet_type, template_name in template_mapping.items():
        if sheet_type in dataset_key:
            return template_name
    
    return default_template


@flow(
    name="html-dashboard-generation",
    description="Generate HTML dashboards from JSON comparison reports",
    version="1.0.0",
    timeout_seconds=1800
)
def html_generation_flow(config_path: str) -> Dict[str, Any]:
    """HTML generation flow"""
    logger = get_run_logger()
    logger.info("🚀 Starting HTML Dashboard Generation")
    
    try:
        # Step 1: Extract JSON reports
        extraction_results = extract_json_reports_task(config_path)
        
        # Step 2: Generate HTML dashboards
        generation_results = generate_html_dashboards_task(config_path, extraction_results)
        
        # Summary
        total_duration = (
            extraction_results['performance_metrics']['json_extraction_duration'] +
            generation_results['performance_metrics']['html_generation_duration']
        )
        
        summary = {
            'pipeline_status': 'SUCCESS',
            'execution_timestamp': datetime.now().isoformat(),
            'total_execution_time_seconds': total_duration,
            'reports_processed': extraction_results['extracted_reports'],
            'dashboards_generated': generation_results['generated_dashboards'],
            'failed_generations': generation_results['failed_generations'],
            'output_summary': generation_results['output_summary']
        }
        
        logger.info("🎉 HTML generation completed successfully!")
        return summary
        
    except Exception as e:
        logger.error(f"💥 HTML generation failed: {e}")
        return {
            'pipeline_status': 'FAILED',
            'error': str(e),
            'execution_timestamp': datetime.now().isoformat()
        }


def main():
    """CLI for HTML generation"""
    parser = argparse.ArgumentParser(description="HTML Dashboard Generation")
    parser.add_argument("--config", required=True, help="Path to main configuration file")
    parser.add_argument("--run", action="store_true", help="Run HTML generation")
    
    args = parser.parse_args()
    
    if args.run:
        result = html_generation_flow(args.config)
        
        if result['pipeline_status'] == 'SUCCESS':
            print(f"✅ Generated {result['dashboards_generated']} dashboards")
            print(f"⏰ Duration: {result['total_execution_time_seconds']:.2f}s")
            return 0
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())