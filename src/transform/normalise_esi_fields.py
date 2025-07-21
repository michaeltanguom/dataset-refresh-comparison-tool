"""
ESI Field Normaliser
Standardises ESI field names to exact canonical format
Location: src/transform/normalise_esi_field.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from ..utils.logging_config import get_logger
from ..utils.exceptions import ConfigurationError

logger = get_logger('transform.normalise_esi_field')


class ESIFieldNormaliser:
    """Simple ESI field normaliser using exact canonical field names"""
    
    # Canonical ESI field names (your exact list)
    CANONICAL_ESI_FIELDS = {
        'agricultural sciences': 'Agricultural Sciences',
        'biology_biochemistry': 'Biology_Biochemistry', 
        'chemistry': 'Chemistry',
        'clinical medicine': 'Clinical Medicine',
        'computer science': 'Computer Science',
        'economics and business': 'Economics and Business',
        'engineering': 'Engineering',
        'environment_ecology': 'Environment_Ecology',
        'geosciences': 'GeoSciences',
        'immunology': 'Immunology',
        'materials science': 'Materials Science',
        'microbiology': 'Microbiology',
        'molecular biology and genetics': 'Molecular Biology and Genetics',
        'neuroscience and behaviour': 'Neuroscience and Behaviour',
        'pharmacology and toxicology': 'Pharmacology and Toxicology',
        'physics': 'Physics',
        'plant and animal science': 'Plant and Animal Science',
        'psychiatry_psychology': 'Psychiatry_Psychology',
        'social sciences': 'Social Sciences',
        'space science': 'Space Science'
    }
    
    @classmethod
    def normalise_esi_field(cls, field_name: str) -> str:
        """
        Normalise ESI field to canonical format
        
        Args:
            field_name: Raw ESI field name
            
        Returns:
            Canonical ESI field name
        """
        if not field_name:
            return ""
        
        # Clean and normalise input
        clean_field = field_name.strip().lower()
        
        # Direct lookup
        if clean_field in cls.CANONICAL_ESI_FIELDS:
            return cls.CANONICAL_ESI_FIELDS[clean_field]
        
        # If no exact match, return original with warning
        logger.warning(f"Unknown ESI field: '{field_name}' - keeping original")
        return field_name.strip()


def normalise_researcher_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise ESI field in a researcher record"""
    if not record:
        return record
    
    # Make a copy to avoid modifying original
    normalised = record.copy()
    
    # Find and normalise esi_field (handle different key variations)
    esi_keys = ['esi_field', 'ESI Field', 'esi field', 'ESI_Field']
    
    for key in esi_keys:
        if key in normalised:
            original_field = normalised[key]
            normalised_field = ESIFieldNormaliser.normalise_esi_field(original_field)
            
            # Use standard key name
            normalised['esi_field'] = normalised_field
            
            # Remove non-standard keys
            if key != 'esi_field':
                del normalised[key]
            
            break
    
    return normalised


def normalise_comparison_report(file_path: Path) -> bool:
    """
    Normalise ESI fields in a single comparison report file
    
    Args:
        file_path: Path to comparison report JSON
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Normalising ESI fields in: {file_path}")
        
        # Load report
        with open(file_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # Track changes
        fields_normalised = 0
        
        # Sections containing researcher records
        sections = [
            'researcher_changes',
            'researchers_unchanged', 
            'researchers_only_in_dataset_1',
            'researchers_only_in_dataset_2'
        ]
        
        # Normalise each section
        for section in sections:
            if section in report and isinstance(report[section], list):
                original_records = report[section]
                normalised_records = []
                
                for record in original_records:
                    normalised_record = normalise_researcher_record(record)
                    normalised_records.append(normalised_record)
                    
                    # Check if field was changed
                    original_field = record.get('esi_field', record.get('ESI Field', ''))
                    normalised_field = normalised_record.get('esi_field', '')
                    
                    if original_field != normalised_field:
                        fields_normalised += 1
                
                report[section] = normalised_records
        
        # Add metadata
        report['esi_normalisation_metadata'] = {
            'normalised_at': datetime.now().isoformat(),
            'fields_normalised': fields_normalised,
            'normaliser_version': '1.0.0'
        }
        
        # Save normalised report
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Normalised {fields_normalised} ESI fields in {file_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to normalise {file_path}: {e}")
        return False


def normalise_esi_fields_in_directory(directory_path: str) -> Dict[str, Any]:
    """
    Normalise ESI fields in all comparison reports in a directory
    
    Args:
        directory_path: Path to directory with comparison reports
        
    Returns:
        Summary of normalisation results
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        raise ConfigurationError(f"Directory not found: {directory_path}")
    
    logger.info(f"🔄 Starting ESI field normalisation in: {directory_path}")
    
    # Find JSON files (exclude backups)
    json_files = [f for f in directory.glob("*.json") if not f.name.endswith('.backup')]
    
    if not json_files:
        logger.warning(f"No JSON files found in {directory_path}")
        return {
            'files_processed': 0,
            'files_succeeded': 0,
            'files_failed': 0,
            'total_files_found': 0
        }
    
    logger.info(f"📁 Found {len(json_files)} comparison report files")
    
    # Process each file
    succeeded = 0
    failed = 0
    
    for file_path in json_files:
        if normalise_comparison_report(file_path):
            succeeded += 1
        else:
            failed += 1
    
    # Summary
    summary = {
        'files_processed': succeeded + failed,
        'files_succeeded': succeeded,
        'files_failed': failed,
        'total_files_found': len(json_files),
        'success_rate': round((succeeded / len(json_files)) * 100, 1) if json_files else 0,
        'processed_at': datetime.now().isoformat()
    }
    
    logger.info(f"✅ ESI normalisation complete: {succeeded}/{len(json_files)} files processed successfully")
    
    return summary


# CLI interface for standalone usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Normalise ESI fields in comparison reports")
    parser.add_argument("directory", help="Directory containing comparison report JSON files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        summary = normalise_esi_fields_in_directory(args.directory)
        
        print(f"\n✅ ESI Field Normalisation Complete")
        print(f"📁 Directory: {args.directory}")
        print(f"📊 Files found: {summary['total_files_found']}")
        print(f"✅ Succeeded: {summary['files_succeeded']}")
        print(f"❌ Failed: {summary['files_failed']}")
        print(f"📈 Success rate: {summary['success_rate']}%")
        
    except Exception as e:
        print(f"❌ Normalisation failed: {e}")
        exit(1)