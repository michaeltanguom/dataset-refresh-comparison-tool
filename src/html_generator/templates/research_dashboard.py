"""
Research Dashboard Template - Unified View
Generates comprehensive interactive dashboards for research performance data
Replicates the logic from html_report_generator.py for unified multi-subject analysis
"""

from typing import Dict, List, Any, Set
from .base_template import BaseTemplate
from ...utils.common import format_number_with_commas

class ResearchDashboardTemplate(BaseTemplate):
    """
    Unified research dashboard template
    Aggregates data from multiple JSON comparison reports into a single dashboard
    """
    
    def get_required_data_fields(self) -> List[str]:
        """Required data fields for unified research dashboard"""
        return [
            # This will now accept a dictionary of multiple JSON reports
            'reports_data'  # Dict[subject_name, report_data]
        ]
    
    def generate_html(self, data: Dict[str, Any]) -> str:
        """
        Generate unified research dashboard HTML from multiple reports
        
        Args:
            data: Dictionary containing 'reports_data' with all JSON comparison reports
            
        Returns:
            Complete HTML dashboard aggregating all subjects
        """
        self.logger.info("Generating unified research dashboard HTML")
        
        # Validate input data
        if 'reports_data' not in data:
            raise ValueError("Missing 'reports_data' in input data")
        
        reports_data = data['reports_data']
        self.logger.info(f"Processing {len(reports_data)} subject reports for unified dashboard")
        
        # Determine dataset type from first report
        first_report = next(iter(reports_data.values()))
        dataset_type = self._determine_dataset_type(reports_data)
        
        # Generate metadata
        metadata = self._generate_metadata(dataset_type)
        
        # Process aggregated data for dashboard (replicating original logic)
        processed_data = self._process_unified_dashboard_data(reports_data)
        
        # Generate HTML sections
        html_content = f"""<!DOCTYPE html>
<html lang="en-GB">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{metadata['title']}</title>
    <meta name="generated-at" content="{metadata['generated_at']}">
    <meta name="dataset-type" content="{metadata['dataset_type']}">
    <meta name="subjects-count" content="{len(reports_data)}">
    {self._generate_css()}
</head>
<body>
    {self._generate_header(metadata, len(reports_data))}
    {self._generate_navigation()}
    {self._generate_summary_section(processed_data)}
    {self._generate_researchers_section(processed_data)}
    {self._generate_cross_field_section(processed_data)}
    {self._generate_significant_changes_section(processed_data)}
    {self._generate_subject_analysis_section(processed_data)}
    {self._generate_detailed_data_section(processed_data)}
    {self._generate_javascript()}
</body>
</html>"""
        
        self.logger.info(f"Generated unified HTML dashboard with {len(html_content)} characters covering {len(reports_data)} subjects")
        return html_content
    
    def _determine_dataset_type(self, reports_data: Dict[str, Any]) -> str:
        """Determine dataset type from reports data"""
        # Look for common patterns in the comparison IDs
        comparison_ids = []
        for report in reports_data.values():
            comparison_id = report.get('comparison_id', '')
            comparison_ids.append(comparison_id)
        
        # Determine if this is highly_cited_only or incites_researchers
        if any('highly_cited_only' in cid for cid in comparison_ids):
            return 'highly_cited_only'
        elif any('incites_researchers' in cid for cid in comparison_ids):
            return 'incites_researchers'
        else:
            return 'research_performance'
    
    def _process_unified_dashboard_data(self, reports_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process aggregated data from all reports (replicating original html_report_generator.py logic)
        """
        self.logger.info("Processing unified dashboard data from multiple subject reports")
        
        # Extract summary statistics (aggregated across all subjects)
        summary_stats = self._extract_summary_stats(reports_data)
        
        # Get individual researchers from all subjects
        individual_researchers = self._get_individual_researchers(reports_data)
        
        # Get significant changes
        increases, decreases, cross_field_changes = self._get_significant_changes(reports_data)
        
        # Get subject analysis
        subject_analysis = self._get_subject_analysis(reports_data)
        
        # Get all researchers with cross-field aggregation
        all_researchers, cross_field_all = self._get_all_researchers(reports_data)
        
        # Combine cross-field data and remove duplicates
        all_cross_field = {}
        for cf_list in [cross_field_changes, cross_field_all]:
            for cf in cf_list:
                name = cf['name']
                if name not in all_cross_field or cf['field_count'] > all_cross_field[name]['field_count']:
                    all_cross_field[name] = cf
        
        cross_field_researchers = sorted(all_cross_field.values(), key=lambda x: x['field_count'], reverse=True)
        
        # Categorise cross-field researchers
        true_cross_field = [cf for cf in cross_field_researchers if cf.get('is_cross_field', False)]
        near_cross_field = [cf for cf in cross_field_researchers if cf.get('is_near_cross_field', False)]
        approaching_cross_field = [cf for cf in cross_field_researchers if cf.get('is_approaching_cross_field', False)]
        
        return {
            'summary_stats': summary_stats,
            'individual_researchers': individual_researchers,
            'increases': increases,
            'decreases': decreases,
            'cross_field_researchers': cross_field_researchers,
            'true_cross_field': true_cross_field,
            'near_cross_field': near_cross_field,
            'approaching_cross_field': approaching_cross_field,
            'subject_analysis': subject_analysis,
            'all_researchers': all_researchers,
            'subjects': self._get_unique_subjects(reports_data)
        }
    
    def _extract_summary_stats(self, reports_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary statistics from aggregated data (replicating original logic)"""
        stats = {
            'total_researchers_july': 0,
            'retention_rate': 0,
            'change_rate': 0,
            'total_subjects': 0,
            'sheet_type': 'Research Report'
        }
        
        # Calculate totals from all reports
        total_july = 0
        total_feb = 0
        total_matched = 0
        total_changes = 0
        subjects = set()
        
        for report_name, report_data in reports_data.items():
            if 'summary_statistics' in report_data:
                summary = report_data['summary_statistics']
                total_july += summary.get('dataset_2_total', 0)
                total_feb += summary.get('dataset_1_total', 0)
                total_matched += summary.get('total_researchers_compared', 0)
                total_changes += summary.get('researchers_with_changes', 0)
            
            # Extract subject from comparison_id or use report_name
            subject = self._extract_subject_from_report(report_data, report_name)
            if subject:
                subjects.add(subject)
        
        stats['total_researchers_july'] = total_july
        stats['total_subjects'] = len(subjects)
        if total_matched > 0:
            stats['change_rate'] = round((total_changes / total_matched) * 100, 1)
        if total_feb > 0:
            stats['retention_rate'] = round((total_matched / total_feb) * 100, 1)
        
        return stats
    
    def _get_individual_researchers(self, reports_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get individual researcher data by ESI field (replicating original logic)"""
        researchers = []
        
        for report_name, report_data in reports_data.items():
            subject = self._extract_subject_from_report(report_data, report_name)
            
            # Get existing researchers with changes
            if 'researcher_changes' in report_data:
                for change in report_data['researcher_changes']:
                    researcher = {
                        'name': change.get('name', ''),
                        'subject': change.get('esi_field', subject),
                        'feb_papers': change.get('feb_values', {}).get('highly_cited_papers', 0),
                        'july_papers': change.get('july_values', {}).get('highly_cited_papers', 0),
                        'paper_change': change.get('highly_cited_papers_change', 0),
                        'feb_score': round(change.get('feb_values', {}).get('indicative_cross_field_score', 0), 2),
                        'july_score': round(change.get('july_values', {}).get('indicative_cross_field_score', 0), 2),
                        'score_change': round(change.get('cross_field_score_change', 0), 2),
                        'hot_papers': change.get('july_values', {}).get('hot_papers', 0),
                        'times_cited': change.get('july_values', {}).get('times_cited', 0)
                    }
                    if researcher['name']:
                        researchers.append(researcher)
            
            # Get researchers with no changes
            if 'researchers_unchanged' in report_data:
                for unchanged in report_data['researchers_unchanged']:
                    researcher = {
                        'name': unchanged.get('name', ''),
                        'subject': unchanged.get('esi_field', subject),
                        'feb_papers': unchanged.get('highly_cited_papers', 0),
                        'july_papers': unchanged.get('highly_cited_papers', 0),
                        'paper_change': 0,
                        'feb_score': round(unchanged.get('indicative_cross_field_score', 0), 2),
                        'july_score': round(unchanged.get('indicative_cross_field_score', 0), 2),
                        'score_change': 0.0,
                        'hot_papers': unchanged.get('hot_papers', 0),
                        'times_cited': unchanged.get('times_cited', 0)
                    }
                    if researcher['name']:
                        researchers.append(researcher)
            
            # Get new researchers (only in July/dataset_2)
            if 'researchers_only_in_dataset_2' in report_data:
                for new_researcher in report_data['researchers_only_in_dataset_2']:
                    researcher = {
                        'name': new_researcher.get('name', ''),
                        'subject': new_researcher.get('esi_field', subject),
                        'feb_papers': 0,
                        'july_papers': new_researcher.get('highly_cited_papers', 0),
                        'paper_change': new_researcher.get('highly_cited_papers', 0),
                        'feb_score': 0.0,
                        'july_score': round(new_researcher.get('indicative_cross_field_score', 0), 2),
                        'score_change': round(new_researcher.get('indicative_cross_field_score', 0), 2),
                        'hot_papers': new_researcher.get('hot_papers', 0),
                        'times_cited': new_researcher.get('times_cited', 0)
                    }
                    if researcher['name'] and researcher['july_papers'] > 0:
                        researchers.append(researcher)
        
        return sorted(researchers, key=lambda x: x['paper_change'], reverse=True)
    
    def _get_significant_changes(self, reports_data: Dict[str, Any]) -> tuple:
        """Extract significant changes for increases/decreases (replicating original logic)"""
        all_changes = []
        
        for report_name, report_data in reports_data.items():
            subject = self._extract_subject_from_report(report_data, report_name)
            
            # Extract from researcher_changes section
            if 'researcher_changes' in report_data:
                for change in report_data['researcher_changes']:
                    change_data = {
                        'name': change.get('name', ''),
                        'subject': change.get('esi_field', subject),
                        'feb_papers': change.get('feb_values', {}).get('highly_cited_papers', 0),
                        'july_papers': change.get('july_values', {}).get('highly_cited_papers', 0),
                        'paper_change': change.get('highly_cited_papers_change', 0),
                        'feb_score': round(change.get('feb_values', {}).get('indicative_cross_field_score', 0), 2),
                        'july_score': round(change.get('july_values', {}).get('indicative_cross_field_score', 0), 2),
                        'score_change': round(change.get('cross_field_score_change', 0), 2)
                    }
                    if change_data['name'] and change_data['paper_change'] != 0:
                        all_changes.append(change_data)
        
        # Use improved deduplication
        unique_changes, cross_field_changes = self._remove_duplicates_keep_multifield(all_changes)
        
        # Separate increases and decreases
        increases = [c for c in unique_changes if c['paper_change'] > 0]
        decreases = [c for c in unique_changes if c['paper_change'] < 0]
        
        increases.sort(key=lambda x: x['paper_change'], reverse=True)
        decreases.sort(key=lambda x: x['paper_change'])
        
        return increases[:15], decreases[:15], cross_field_changes
    
    def _remove_duplicates_keep_multifield(self, researchers: List[Dict]) -> tuple:
        """
        Remove true duplicates while keeping researchers who appear in multiple fields
        """
        # Group by researcher name to identify cross-field researchers
        researcher_groups = {}
        for r in researchers:
            name = r['name'].strip()
            if name:
                if name not in researcher_groups:
                    researcher_groups[name] = []
                researcher_groups[name].append(r)
        
        unique_researchers = []
        cross_field_researchers = []
        
        for name, researcher_list in researcher_groups.items():
            if len(researcher_list) > 1:
                # This researcher appears in multiple fields
                unique_fields = set(r['subject'] for r in researcher_list)
                if len(unique_fields) > 1:
                    # Truly cross-field (different subjects)
                    total_cross_field_score = sum(r['july_score'] for r in researcher_list)
                    cross_field_info = {
                        'name': name,
                        'fields': list(unique_fields),
                        'total_papers': sum(r['july_papers'] for r in researcher_list),
                        'total_change': sum(r['paper_change'] for r in researcher_list),
                        'total_score': round(total_cross_field_score, 2),
                        'field_count': len(unique_fields),
                        'details': researcher_list,
                        'is_cross_field': total_cross_field_score >= 1.0,
                        'is_near_cross_field': 0.85 <= total_cross_field_score < 1.0,
                        'is_approaching_cross_field': 0.70 <= total_cross_field_score < 0.85
                    }
                    cross_field_researchers.append(cross_field_info)
                    
                    # Add all their field entries to unique list
                    unique_researchers.extend(researcher_list)
                else:
                    # Same field, different entries - keep the best one
                    best_researcher = max(researcher_list, key=lambda x: x['paper_change'])
                    unique_researchers.append(best_researcher)
            else:
                # Single entry, keep it
                unique_researchers.append(researcher_list[0])
        
        return unique_researchers, cross_field_researchers
    
    def _get_subject_analysis(self, reports_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract subject analysis data (replicating original logic)"""
        subjects = []
        
        for report_name, report_data in reports_data.items():
            if 'summary_statistics' in report_data:
                summary = report_data['summary_statistics']
                subject = self._extract_subject_from_report(report_data, report_name)
                
                # Calculate growth rate
                feb_total = summary.get('dataset_1_total', 0)
                july_total = summary.get('dataset_2_total', 0)
                
                if feb_total > 0:
                    growth_rate = ((july_total - feb_total) / feb_total) * 100
                    retention_rate = (summary.get('total_researchers_compared', 0) / feb_total) * 100
                else:
                    growth_rate = 0
                    retention_rate = 0
                
                subject_data = {
                    'subject': subject,
                    'feb_total': feb_total,
                    'july_total': july_total,
                    'retention_rate': round(retention_rate, 1),
                    'growth_rate': round(growth_rate, 1),
                    'researchers_with_changes': summary.get('researchers_with_changes', 0),
                    'status': self._get_subject_status(growth_rate)
                }
                subjects.append(subject_data)
        
        # Remove duplicates by subject name
        seen = set()
        unique_subjects = []
        for s in subjects:
            if s['subject'] not in seen:
                seen.add(s['subject'])
                unique_subjects.append(s)
        
        return sorted(unique_subjects, key=lambda x: x['growth_rate'], reverse=True)
    
    def _get_all_researchers(self, reports_data: Dict[str, Any]) -> tuple:
        """Get detailed researcher data with cross-field aggregation (replicating original logic)"""
        researchers = []
        
        # First, collect all individual researcher entries
        for report_name, report_data in reports_data.items():
            subject = self._extract_subject_from_report(report_data, report_name)
            
            # Get existing researchers with changes
            if 'researcher_changes' in report_data:
                for change in report_data['researcher_changes']:
                    researcher = {
                        'name': change.get('name', ''),
                        'subject': change.get('esi_field', subject),
                        'feb_papers': change.get('feb_values', {}).get('highly_cited_papers', 0),
                        'july_papers': change.get('july_values', {}).get('highly_cited_papers', 0),
                        'paper_change': change.get('highly_cited_papers_change', 0),
                        'feb_score': round(change.get('feb_values', {}).get('indicative_cross_field_score', 0), 2),
                        'july_score': round(change.get('july_values', {}).get('indicative_cross_field_score', 0), 2),
                        'score_change': round(change.get('cross_field_score_change', 0), 2),
                        'hot_papers': change.get('july_values', {}).get('hot_papers', 0),
                        'times_cited': change.get('july_values', {}).get('times_cited', 0)
                    }
                    if researcher['name']:
                        researchers.append(researcher)
            
            # Get researchers with no changes
            if 'researchers_unchanged' in report_data:
                for unchanged in report_data['researchers_unchanged']:
                    researcher = {
                        'name': unchanged.get('name', ''),
                        'subject': unchanged.get('esi_field', subject),
                        'feb_papers': unchanged.get('highly_cited_papers', 0),
                        'july_papers': unchanged.get('highly_cited_papers', 0),
                        'paper_change': 0,
                        'feb_score': round(unchanged.get('indicative_cross_field_score', 0), 2),
                        'july_score': round(unchanged.get('indicative_cross_field_score', 0), 2),
                        'score_change': 0.0,
                        'hot_papers': unchanged.get('hot_papers', 0),
                        'times_cited': unchanged.get('times_cited', 0)
                    }
                    if researcher['name']:
                        researchers.append(researcher)
            
            # Get new researchers (only in July/dataset_2)
            if 'researchers_only_in_dataset_2' in report_data:
                for new_researcher in report_data['researchers_only_in_dataset_2']:
                    researcher = {
                        'name': new_researcher.get('name', ''),
                        'subject': new_researcher.get('esi_field', subject),
                        'feb_papers': 0,
                        'july_papers': new_researcher.get('highly_cited_papers', 0),
                        'paper_change': new_researcher.get('highly_cited_papers', 0),
                        'feb_score': 0.0,
                        'july_score': round(new_researcher.get('indicative_cross_field_score', 0), 2),
                        'score_change': round(new_researcher.get('indicative_cross_field_score', 0), 2),
                        'hot_papers': new_researcher.get('hot_papers', 0),
                        'times_cited': new_researcher.get('times_cited', 0)
                    }
                    if researcher['name'] and researcher['july_papers'] > 0:
                        researchers.append(researcher)
        
        # Aggregate researchers by name for detailed data tab
        aggregated_researchers = {}
        cross_field_researchers = []
        
        for researcher in researchers:
            name = researcher['name'].strip()
            if not name:
                continue
                
            if name not in aggregated_researchers:
                aggregated_researchers[name] = {
                    'name': name,
                    'subjects': [],
                    'feb_papers': 0,
                    'july_papers': 0,
                    'paper_change': 0,
                    'feb_score': 0.0,
                    'july_score': 0.0,
                    'score_change': 0.0,
                    'hot_papers': 0,
                    'times_cited': 0,
                    'field_count': 0,
                    'individual_entries': []
                }
            
            # Add subject if not already included
            if researcher['subject'] not in aggregated_researchers[name]['subjects']:
                aggregated_researchers[name]['subjects'].append(researcher['subject'])
            
            # Track individual entries for proper citation calculation
            aggregated_researchers[name]['individual_entries'].append(researcher)
            
            # Aggregate values (summing papers, scores, hot papers)
            aggregated_researchers[name]['feb_papers'] += researcher['feb_papers']
            aggregated_researchers[name]['july_papers'] += researcher['july_papers']
            aggregated_researchers[name]['paper_change'] += researcher['paper_change']
            aggregated_researchers[name]['feb_score'] += researcher['feb_score']
            aggregated_researchers[name]['july_score'] += researcher['july_score']
            aggregated_researchers[name]['score_change'] += researcher['score_change']
            aggregated_researchers[name]['hot_papers'] += researcher['hot_papers']
        
        # Convert to list and format
        final_researchers = []
        for name, data in aggregated_researchers.items():
            data['field_count'] = len(data['subjects'])
            data['subject'] = ', '.join(sorted(data['subjects']))
            data['feb_score'] = round(data['feb_score'], 2)
            data['july_score'] = round(data['july_score'], 2)
            data['score_change'] = round(data['score_change'], 2)
            
            # Use maximum citation count instead of sum
            citation_counts = [entry['times_cited'] for entry in data['individual_entries'] if entry['times_cited'] > 0]
            data['times_cited'] = max(citation_counts) if citation_counts else 0
            
            # Remove individual_entries as no longer needed
            del data['individual_entries']
            
            # Determine category based on field count and total score
            if data['field_count'] > 1:
                if data['july_score'] >= 1.0:
                    data['category'] = "Cross-Field"
                    data['category_data'] = "crossfield"
                elif 0.85 <= data['july_score'] < 1.0:
                    data['category'] = "Near Cross-Field"
                    data['category_data'] = "nearcrossfield"
                elif 0.70 <= data['july_score'] < 0.85:
                    data['category'] = "Approaching Cross-Field"
                    data['category_data'] = "approachingcrossfield"
                else:
                    data['category'] = "Multidisciplinary"
                    data['category_data'] = "multidisciplinary"
            else:
                data['category'] = "Single Field"
                data['category_data'] = "singlefield"
            
            # Identify cross-field researchers for separate tracking
            if data['field_count'] > 1:
                cross_field_info = {
                    'name': name,
                    'fields': data['subjects'],
                    'total_papers': data['july_papers'],
                    'total_change': data['paper_change'],
                    'total_score': data['july_score'],
                    'field_count': data['field_count'],
                    'is_cross_field': data['july_score'] >= 1.0,
                    'is_near_cross_field': 0.85 <= data['july_score'] <= 0.99,
                    'is_approaching_cross_field': 0.70 <= data['july_score'] <= 0.84
                }
                cross_field_researchers.append(cross_field_info)
            
            final_researchers.append(data)
        
        return sorted(final_researchers, key=lambda x: x['paper_change'], reverse=True), cross_field_researchers
    
    def _extract_subject_from_report(self, report_data: Dict[str, Any], report_name: str) -> str:
        """Extract subject name from report data or report name"""
        # Try to get subject from comparison_id first
        comparison_id = report_data.get('comparison_id', '')
        if comparison_id:
            # Remove common suffixes
            subject = comparison_id.replace('_highly_cited_only', '').replace('_incites_researchers', '')
            return subject.replace('_', ' ').title()
        
        # Fallback to report name
        return report_name.replace('_', ' ').title()
    
    def _get_unique_subjects(self, reports_data: Dict[str, Any]) -> List[str]:
        """Get unique subject names for filtering"""
        subjects = set()
        for report_name, report_data in reports_data.items():
            subject = self._extract_subject_from_report(report_data, report_name)
            subjects.add(subject)
        return sorted(list(subjects))
    
    def _get_subject_status(self, growth_rate: float) -> str:
        """Get status emoji and text based on growth rate"""
        if growth_rate >= 30:
            return "High Growth"
        elif growth_rate >= 15:
            return "Growing"
        elif growth_rate >= 5:
            return "Expanding"
        elif growth_rate >= 0:
            return "Moderate"
        elif growth_rate >= -10:
            return "Declining"
        else:
            return "Major Decline"

    def _format_change_class(self, value: float) -> str:
        """Return CSS class for change values"""
        if value > 0:
            return "positive"
        elif value < 0:
            return "negative"
        else:
            return "neutral"

    def _generate_css(self) -> str:
        """Generate CSS styles for the research dashboard"""
        return """<style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f7fa;
                color: #333;
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            
            .header h1 {
                margin: 0;
                font-size: 28px;
                font-weight: 300;
            }
            
            .header p {
                margin: 10px 0 0 0;
                opacity: 0.9;
                font-size: 16px;
            }
            
            .tabs {
                display: flex;
                background: white;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            
            .tab {
                flex: 1;
                padding: 15px 20px;
                background: #f8f9fa;
                border: none;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.3s ease;
                border-right: 1px solid #e9ecef;
            }
            
            .tab:last-child {
                border-right: none;
            }
            
            .tab.active {
                background: #667eea;
                color: white;
            }
            
            .tab:hover:not(.active) {
                background: #e9ecef;
            }
            
            .sheet {
                display: none;
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            .sheet.active {
                display: block;
            }
            
            .sheet h2 {
                margin: 0 0 20px 0;
                color: #667eea;
                font-size: 24px;
                font-weight: 600;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }
            
            th {
                background: #667eea;
                color: white;
                padding: 12px 15px;
                text-align: left;
                font-weight: 600;
                font-size: 14px;
            }
            
            td {
                padding: 12px 15px;
                border-bottom: 1px solid #f1f3f4;
                font-size: 13px;
            }
            
            tr:hover {
                background-color: #f8f9ff;
            }
            
            .metric-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                margin: 10px;
                flex: 1;
            }
            
            .metric-value {
                font-size: 28px;
                font-weight: 700;
                color: #667eea;
                margin-bottom: 5px;
            }
            
            .metric-label {
                color: #6c757d;
                font-size: 14px;
                font-weight: 500;
            }
            
            .metrics-grid {
                display: flex;
                gap: 20px;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }
            
            .positive { color: #28a745; font-weight: 600; }
            .negative { color: #dc3545; font-weight: 600; }
            .neutral { color: #6c757d; }
            
            .field-badge {
                background: #28a745;
                color: white;
                padding: 2px 6px;
                border-radius: 8px;
                font-size: 10px;
                font-weight: 600;
                margin: 0 2px;
            }
            
            .section-title {
                background: #f8f9fa;
                padding: 15px 20px;
                margin: 20px -20px;
                border-left: 4px solid #667eea;
                font-weight: 600;
                color: #495057;
            }
            
            .highlight-row {
                background-color: #fff3cd !important;
            }
            
            .cross-field-row {
                background-color: #d1ecf1 !important;
            }
            
            .filter-controls {
                margin-bottom: 20px;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 6px;
            }
            
            .filter-controls select, .filter-controls input {
                padding: 8px 12px;
                margin: 0 10px 10px 0;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
            }
            
            .no-data {
                text-align: center;
                padding: 40px;
                color: #6c757d;
                font-style: italic;
            }
            
            .sortable {
                cursor: pointer;
                position: relative;
                user-select: none;
            }
            
            .sortable:hover {
                background-color: #5a6bb8 !important;
            }
            
            .sortable:after {
                content: ' ↕';
                font-size: 12px;
                opacity: 0.5;
            }
            
            .sortable.sort-asc:after {
                content: ' ↑';
                opacity: 1;
            }
            
            .sortable.sort-desc:after {
                content: ' ↓';
                opacity: 1;
            }
            
            @media (max-width: 768px) {
                .tabs {
                    flex-direction: column;
                }
                
                .metrics-grid {
                    flex-direction: column;
                }
                
                table {
                    font-size: 12px;
                }
                
                th, td {
                    padding: 8px 10px;
                }
            }
        </style>"""
    
    def _generate_javascript(self) -> str:
        """Generate JavaScript functionality for the dashboard"""
        return """<script>
            function showSheet(sheetId) {
                const sheets = document.querySelectorAll('.sheet');
                sheets.forEach(sheet => sheet.classList.remove('active'));
                
                const tabs = document.querySelectorAll('.tab');
                tabs.forEach(tab => tab.classList.remove('active'));
                
                document.getElementById(sheetId).classList.add('active');
                event.target.classList.add('active');
            }
            
            function sortTable(header, columnIndex, dataType) {
                const table = header.closest('table');
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                
                let isAsc = true;
                if (header.classList.contains('sort-asc')) {
                    isAsc = false;
                    header.classList.remove('sort-asc');
                    header.classList.add('sort-desc');
                } else {
                    table.querySelectorAll('.sortable').forEach(h => {
                        h.classList.remove('sort-asc', 'sort-desc');
                    });
                    header.classList.add('sort-asc');
                }
                
                rows.sort((a, b) => {
                    let aVal = a.cells[columnIndex].textContent.trim();
                    let bVal = b.cells[columnIndex].textContent.trim();
                    
                    if (dataType === 'number') {
                        aVal = parseFloat(aVal.replace(/[+,]/g, '')) || 0;
                        bVal = parseFloat(bVal.replace(/[+,]/g, '')) || 0;
                    } else {
                        aVal = aVal.toLowerCase();
                        bVal = bVal.toLowerCase();
                    }
                    
                    if (aVal < bVal) return isAsc ? -1 : 1;
                    if (aVal > bVal) return isAsc ? 1 : -1;
                    return 0;
                });
                
                rows.forEach(row => tbody.appendChild(row));
            }
            
            function searchIndividualResearchers() {
                const input = document.getElementById('researcherSearchInput');
                const table = document.getElementById('individualResearchersTable');
                
                if (!input || !table) return;
                
                const filter = input.value.toLowerCase();
                const rows = table.getElementsByTagName('tbody')[0].getElementsByTagName('tr');
                
                for (let row of rows) {
                    const name = row.getAttribute('data-name').toLowerCase();
                    if (name.includes(filter)) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                }
            }
            
            function filterIndividualResearchers() {
                const subjectFilter = document.getElementById('researcherSubjectFilter');
                const changeFilter = document.getElementById('researcherChangeFilter');
                const table = document.getElementById('individualResearchersTable');
                
                if (!subjectFilter || !changeFilter || !table) return;
                
                const subjectValue = subjectFilter.value;
                const changeValue = changeFilter.value;
                const rows = table.getElementsByTagName('tbody')[0].getElementsByTagName('tr');
                
                for (let row of rows) {
                    let showRow = true;
                    const subject = row.getAttribute('data-subject');
                    const change = parseInt(row.getAttribute('data-change'));
                    
                    if (subjectValue !== 'all' && subject !== subjectValue) {
                        showRow = false;
                    }
                    
                    if (changeValue === 'increase' && change <= 0) {
                        showRow = false;
                    } else if (changeValue === 'decrease' && change >= 0) {
                        showRow = false;
                    } else if (changeValue === 'major' && Math.abs(change) < 5) {
                        showRow = false;
                    }
                    
                    row.style.display = showRow ? '' : 'none';
                }
            }
            
            function filterByCategory() {
                const categoryFilter = document.getElementById('categoryFilter');
                const table = document.getElementById('crossFieldTable');
                
                if (!categoryFilter || !table) return;
                
                const categoryValue = categoryFilter.value;
                const rows = table.getElementsByTagName('tbody')[0].getElementsByTagName('tr');
                
                for (let row of rows) {
                    let showRow = true;
                    const category = row.getAttribute('data-category');
                    
                    if (categoryValue === 'crossfield' && category !== 'crossfield') {
                        showRow = false;
                    } else if (categoryValue === 'nearcrossfield' && category !== 'nearcrossfield') {
                        showRow = false;
                    } else if (categoryValue === 'approachingcrossfield' && category !== 'approachingcrossfield') {
                        showRow = false;
                    } else if (categoryValue === 'multidisciplinary' && category !== 'multidisciplinary') {
                        showRow = false;
                    }
                    
                    row.style.display = showRow ? '' : 'none';
                }
            }
            
            function searchResearchers() {
                const input = document.getElementById('searchInput');
                const table = document.getElementById('detailedTable');
                
                if (!input || !table) return;
                
                const filter = input.value.toLowerCase();
                const rows = table.getElementsByTagName('tbody')[0].getElementsByTagName('tr');
                
                for (let row of rows) {
                    const name = row.cells[0].textContent.toLowerCase();
                    if (name.includes(filter)) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                }
            }
            
            function filterResearchers() {
                const subjectFilter = document.getElementById('subjectFilter');
                const changeFilter = document.getElementById('changeFilter');
                const table = document.getElementById('detailedTable');
                
                if (!subjectFilter || !changeFilter || !table) return;
                
                const subjectValue = subjectFilter.value;
                const changeValue = changeFilter.value;
                const rows = table.getElementsByTagName('tbody')[0].getElementsByTagName('tr');
                
                for (let row of rows) {
                    let showRow = true;
                    const subject = row.getAttribute('data-subject');
                    const change = parseInt(row.getAttribute('data-change'));
                    const category = row.getAttribute('data-category');
                    
                    if (subjectValue !== 'all' && !subject.includes(subjectValue)) {
                        showRow = false;
                    }
                    
                    if (changeValue === 'increase' && change <= 0) {
                        showRow = false;
                    } else if (changeValue === 'decrease' && change >= 0) {
                        showRow = false;
                    } else if (changeValue === 'major' && Math.abs(change) < 5) {
                        showRow = false;
                    } else if (changeValue === 'crossfield' && category !== 'crossfield') {
                        showRow = false;
                    } else if (changeValue === 'nearcrossfield' && category !== 'nearcrossfield') {
                        showRow = false;
                    } else if (changeValue === 'approachingcrossfield' && category !== 'approachingcrossfield') {
                        showRow = false;
                    } else if (changeValue === 'multidisciplinary' && category !== 'multidisciplinary') {
                        showRow = false;
                    } else if (changeValue === 'singlefield' && category !== 'singlefield') {
                        showRow = false;
                    }
                    
                    row.style.display = showRow ? '' : 'none';
                }
            }
            
            // Initialize with summary sheet
            document.addEventListener('DOMContentLoaded', function() {
                showSheet('summary');
            });
        </script>"""
    
    def _generate_header(self, metadata: Dict[str, str], subject_count: int) -> str:
        """Generate header section with subject count"""
        return f"""
        <div class="header">
            <h1>{metadata['title']}</h1>
            <p>Comparative Analysis: February 2025 vs July 2025 - {subject_count} ESI Subject Areas</p>
        </div>"""

    def _generate_navigation(self) -> str:
        """Generate navigation tabs"""
        return """
        <div class="tabs">
            <button class="tab active" onclick="showSheet('summary')">Executive Summary</button>
            <button class="tab" onclick="showSheet('researchers')">Researchers</button>
            <button class="tab" onclick="showSheet('cross-field')">Cross-Field Researchers</button>
            <button class="tab" onclick="showSheet('significant-changes')">Significant Changes</button>
            <button class="tab" onclick="showSheet('subject-analysis')">Subject Analysis</button>
            <button class="tab" onclick="showSheet('detailed-data')">Detailed Data</button>
        </div>"""

    def _generate_summary_section(self, data: Dict[str, Any]) -> str:
        """Generate executive summary section"""
        stats = data['summary_stats']
        increases = data['increases']
        cross_field_count = len(data['true_cross_field'])
        
        # Generate metrics cards
        metrics_html = f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{stats['total_researchers_july']:,}</div>
                <div class="metric-label">Total Researchers (July)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['retention_rate']:.1f}%</div>
                <div class="metric-label">Highly Cited Paper Overlap Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['change_rate']:.1f}%</div>
                <div class="metric-label">Change Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{cross_field_count}</div>
                <div class="metric-label">Cross-Field Researchers</div>
            </div>
        </div>"""
        
        # Generate top changes table
        changes_html = ""
        if increases:
            changes_html = """
            <div class="section-title">Largest Changes in Researcher's Highly Cited Papers Count</div>
            <table>
                <thead>
                    <tr>
                        <th class="sortable" onclick="sortTable(this, 0, 'string')">Researcher</th>
                        <th class="sortable" onclick="sortTable(this, 1, 'string')">ESI Field</th>
                        <th class="sortable" onclick="sortTable(this, 2, 'number')">Highly Cited Papers Change</th>
                        <th class="sortable" onclick="sortTable(this, 3, 'number')">Cross-Field Fractional Score Change</th>
                        <th class="sortable" onclick="sortTable(this, 4, 'string')">Cross-Field</th>
                    </tr>
                </thead>
                <tbody>"""
            
            for i, increase in enumerate(increases[:10]):
                is_cross_field = any(cf['name'] == increase['name'] for cf in data['cross_field_researchers'])
                row_class = "highlight-row" if i == 0 else ("cross-field-row" if is_cross_field else "")
                cross_field_indicator = "Cross-Field" if is_cross_field else ""
                
                changes_html += f"""
                    <tr class="{row_class}">
                        <td><strong>{increase['name']}</strong></td>
                        <td>{increase['subject']}</td>
                        <td class="positive">+{increase['paper_change']}</td>
                        <td class="positive">+{increase['score_change']:.2f}</td>
                        <td>{cross_field_indicator}</td>
                    </tr>"""
            
            changes_html += """
                </tbody>
            </table>"""
        else:
            changes_html = """
            <div class="section-title">Researcher Changes</div>
            <div class="no-data">No researcher data available in the current dataset.</div>"""
        
        return f"""
        <div id="summary" class="sheet active">
            <h2>Executive Summary</h2>
            {metrics_html}
            {changes_html}
        </div>"""
    
    def _generate_researchers_section(self, data: Dict[str, Any]) -> str:
        """Generate researchers section with filtering"""
        researchers = data['individual_researchers']
        subjects = data['subjects']
        
        if not researchers:
            return """
            <div id="researchers" class="sheet">
                <h2>Researchers</h2>
                <div class="no-data">No researcher data available in the current dataset.</div>
            </div>"""
        
        # Generate filter controls
        filter_html = f"""
        <div class="filter-controls">
            <input type="text" id="researcherSearchInput" placeholder="Search researcher name..." onkeyup="searchIndividualResearchers()">
            <select id="researcherSubjectFilter" onchange="filterIndividualResearchers()">
                <option value="all">All Subjects</option>"""
        
        for subject in subjects:
            filter_html += f'<option value="{subject}">{subject}</option>'
        
        filter_html += """
            </select>
            <select id="researcherChangeFilter" onchange="filterIndividualResearchers()">
                <option value="all">All Changes</option>
                <option value="increase">Increases Only</option>
                <option value="decrease">Decreases Only</option>
                <option value="major">Major Changes (±5+)</option>
            </select>
        </div>"""
        
        # Generate researchers table
        table_html = """
        <div class="section-title">Researchers by Individual ESI Field</div>
        <table id="individualResearchersTable">
            <thead>
                <tr>
                    <th class="sortable" onclick="sortTable(this, 0, 'string')">Researcher</th>
                    <th class="sortable" onclick="sortTable(this, 1, 'string')">ESI Field</th>
                    <th class="sortable" onclick="sortTable(this, 2, 'number')">Feb Highly Cited Papers</th>
                    <th class="sortable" onclick="sortTable(this, 3, 'number')">July Highly Cited Papers</th>
                    <th class="sortable" onclick="sortTable(this, 4, 'number')">Highly Cited Papers Change</th>
                    <th class="sortable" onclick="sortTable(this, 5, 'number')">Feb Cross-Field Fractional Score</th>
                    <th class="sortable" onclick="sortTable(this, 6, 'number')">July Cross-Field Fractional Score</th>
                    <th class="sortable" onclick="sortTable(this, 7, 'number')">Cross-Field Fractional Score Change</th>
                    <th class="sortable" onclick="sortTable(this, 8, 'number')">Hot Papers</th>
                    <th class="sortable" onclick="sortTable(this, 9, 'number')">Citation Count</th>
                </tr>
            </thead>
            <tbody>"""
        
        for researcher in researchers:
            paper_change_sign = "+" if researcher['paper_change'] > 0 else ""
            score_change_sign = "+" if researcher['score_change'] > 0 else ""
            
            table_html += f"""
                <tr data-subject="{researcher['subject']}" data-change="{researcher['paper_change']}" data-name="{researcher['name']}">
                    <td><strong>{researcher['name']}</strong></td>
                    <td>{researcher['subject']}</td>
                    <td>{researcher['feb_papers']}</td>
                    <td>{researcher['july_papers']}</td>
                    <td class="{self._format_change_class(researcher['paper_change'])}">{paper_change_sign}{researcher['paper_change']}</td>
                    <td>{researcher['feb_score']:.2f}</td>
                    <td>{researcher['july_score']:.2f}</td>
                    <td class="{self._format_change_class(researcher['score_change'])}">{score_change_sign}{researcher['score_change']:.2f}</td>
                    <td>{researcher['hot_papers']}</td>
                    <td>{researcher['times_cited']:,}</td>
                </tr>"""
        
        table_html += """
            </tbody>
        </table>"""
        
        return f"""
        <div id="researchers" class="sheet">
            <h2>Researchers</h2>
            {filter_html}
            {table_html}
        </div>"""
    
    def _generate_cross_field_section(self, data: Dict[str, Any]) -> str:
        """Generate cross-field researchers section"""
        cross_field_researchers = data['cross_field_researchers']
        true_cross_field = data['true_cross_field']
        near_cross_field = data['near_cross_field']
        approaching_cross_field = data['approaching_cross_field']
        
        if not cross_field_researchers:
            return """
            <div id="cross-field" class="sheet">
                <h2>Cross-Field Researchers</h2>
                <div class="no-data">No cross-field researchers found in the current dataset.</div>
            </div>"""
        
        filter_html = """
        <div class="filter-controls">
            <select id="categoryFilter" onchange="filterByCategory()">
                <option value="all">All Categories</option>
                <option value="crossfield">Cross-Field (≥1.0)</option>
                <option value="nearcrossfield">Near Cross-Field (0.85-0.99)</option>
                <option value="approachingcrossfield">Approaching Cross-Field (0.70-0.84)</option>
                <option value="multidisciplinary">Multidisciplinary (&lt;0.70)</option>
            </select>
        </div>"""
        
        breakdown_html = f"""
        <div class="section-title">Cross-Field Status Breakdown<br>
        Cross-Field ≥1.0: {len(true_cross_field)} | Near Cross-Field 0.85-0.99: {len(near_cross_field)} | Approaching Cross-Field 0.70-0.84: {len(approaching_cross_field)}</div>"""
        
        table_html = """
        <table id="crossFieldTable">
            <thead>
                <tr>
                    <th class="sortable" onclick="sortTable(this, 0, 'string')">Researcher</th>
                    <th class="sortable" onclick="sortTable(this, 1, 'string')">ESI Fields</th>
                    <th class="sortable" onclick="sortTable(this, 2, 'number')">ESI Field Count</th>
                    <th class="sortable" onclick="sortTable(this, 3, 'number')">Total Highly Cited Papers</th>
                    <th class="sortable" onclick="sortTable(this, 4, 'number')">Total Change</th>
                    <th class="sortable" onclick="sortTable(this, 5, 'number')">Total Cross-Field Fractional Score</th>
                    <th class="sortable" onclick="sortTable(this, 6, 'string')">Category</th>
                </tr>
            </thead>
            <tbody>"""
        
        for cf in cross_field_researchers:
            # Create field badges
            field_badges = ""
            if isinstance(cf['fields'], list):
                for field in cf['fields']:
                    field_badges += f'<span class="field-badge">{field}</span>'
            else:
                # Handle case where fields might be a string
                field_badges = f'<span class="field-badge">{cf["fields"]}</span>'
            
            total_change_sign = "+" if cf['total_change'] > 0 else ""
            
            # Determine category and data attribute
            if cf.get('is_cross_field', False):
                category = "Cross-Field"
                category_data = "crossfield"
                row_class = "cross-field-row"  # Only highlight true cross-field
            elif cf.get('is_near_cross_field', False):
                category = "Near Cross-Field"
                category_data = "nearcrossfield"
                row_class = ""  # No highlighting
            elif cf.get('is_approaching_cross_field', False):
                category = "Approaching Cross-Field"
                category_data = "approachingcrossfield"
                row_class = ""  # No highlighting
            else:
                category = "Multidisciplinary"
                category_data = "multidisciplinary"
                row_class = ""  # No highlighting

            table_html += f"""
                <tr class="{row_class}" data-category="{category_data}">
                    <td><strong>{cf['name']}</strong></td>
                    <td>{field_badges}</td>
                    <td>{cf['field_count']}</td>
                    <td>{cf['total_papers']}</td>
                    <td class="{self._format_change_class(cf['total_change'])}">{total_change_sign}{cf['total_change']}</td>
                    <td><strong>{cf['total_score']:.2f}</strong></td>
                    <td>{category}</td>
                </tr>"""
        
        table_html += """
            </tbody>
        </table>"""
        
        return f"""
        <div id="cross-field" class="sheet">
            <h2>Cross-Field Researchers</h2>
            <p>Researchers who appear in multiple ESI subject fields. Cross-Field researchers have total scores ≥1.0, near Cross-Field have scores 0.85-0.99, approaching Cross-Field have scores 0.70-0.84, while multidisciplinary researchers have scores &lt;0.70.</p>
            {filter_html}
            {breakdown_html}
            {table_html}
        </div>"""

    def _generate_significant_changes_section(self, data: Dict[str, Any]) -> str:
        """Generate significant changes section with total cross-field scores for clarity"""
        increases = data['increases']
        decreases = data['decreases']
        cross_field_researchers = data['cross_field_researchers']
        
        # Summary statistics
        total_increases = len(increases)
        total_decreases = len(decreases)
        top_increase = increases[0] if increases else None
        top_decrease = decreases[0] if decreases else None
        
        summary_html = f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{total_increases}</div>
                <div class="metric-label">Researchers with Increases</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_decreases}</div>
                <div class="metric-label">Researchers with Decreases</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{top_increase['paper_change'] if top_increase else 0}</div>
                <div class="metric-label">Largest Increase</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{abs(top_decrease['paper_change']) if top_decrease else 0}</div>
                <div class="metric-label">Largest Decrease</div>
            </div>
        </div>"""
        
        # Top 15 increases table with total cross-field score
        increases_html = ""
        if increases:
            increases_html = """
            <div class="section-title">Top 15 Increases</div>
            <table>
                <thead>
                    <tr>
                        <th>Researcher</th>
                        <th>ESI Field</th>
                        <th>Feb Papers</th>
                        <th>July Papers</th>
                        <th>Highly Cited Papers Change</th>
                        <th>Feb Cross-Field Fractional Score</th>
                        <th>July Cross-Field Fractional Score</th>
                        <th>Cross-Field Fractional Score Change</th>
                        <th>Total July Cross-Field Fractional Score</th>
                        <th>Cross-Field Status</th>
                    </tr>
                </thead>
                <tbody>"""
            
            for i, increase in enumerate(increases[:15]):  # Top 15 increases
                # Find the researcher's total cross-field score
                total_july_score = increase['july_score']  # Default to individual field score
                
                for cf in cross_field_researchers:
                    if cf['name'] == increase['name']:
                        total_july_score = cf['total_score']  # Use aggregated total score
                        break
                
                # 🔧 FIX: Get correct category and highlighting
                category = self._get_researcher_category(increase['name'], cross_field_researchers)
                should_highlight = self._should_highlight_cross_field(increase['name'], cross_field_researchers)
                row_class = "highlight-row" if i == 0 else ("cross-field-row" if should_highlight else "")
                
                increases_html += f"""
                    <tr class="{row_class}">
                        <td><strong>{increase['name']}</strong></td>
                        <td>{increase['subject']}</td>
                        <td>{increase['feb_papers']}</td>
                        <td>{increase['july_papers']}</td>
                        <td class="positive">+{increase['paper_change']}</td>
                        <td>{increase['feb_score']:.2f}</td>
                        <td>{increase['july_score']:.2f}</td>
                        <td class="positive">+{increase['score_change']:.2f}</td>
                        <td><strong>{total_july_score:.2f}</strong></td>
                        <td>{category}</td>
                    </tr>"""
            
            increases_html += """
                </tbody>
            </table>"""
        else:
            increases_html = """
            <div class="section-title">Top 15 Increases</div>
            <div class="no-data">No increases found in the current dataset.</div>"""
        
        # Top 15 decreases table with total cross-field score
        decreases_html = ""
        if decreases:
            decreases_html = """
            <div class="section-title">Top 15 Decreases</div>
            <table>
                <thead>
                    <tr>
                        <th>Researcher</th>
                        <th>ESI Field</th>
                        <th>Feb Highly Cited Papers</th>
                        <th>July Highly Cited Papers</th>
                        <th>Highly Cited Papers Change</th>
                        <th>Feb Cross-Field Fractional Score</th>
                        <th>July Cross-Field Fractional Score</th>
                        <th>Cross-Field Fractional Score Change</th>
                        <th>Total July Cross-Field Fractional Score</th>
                        <th>Cross-Field Status</th>
                    </tr>
                </thead>
                <tbody>"""
            
            for decrease in decreases[:15]:  # Top 15 decreases
                # Find the researcher's total cross-field score
                total_july_score = decrease['july_score']  # Default to individual field score
                
                for cf in cross_field_researchers:
                    if cf['name'] == decrease['name']:
                        total_july_score = cf['total_score']  # Use aggregated total score
                        break
                
                # 🔧 FIX: Get correct category and highlighting
                category = self._get_researcher_category(decrease['name'], cross_field_researchers)
                should_highlight = self._should_highlight_cross_field(decrease['name'], cross_field_researchers)
                row_class = "cross-field-row" if should_highlight else ""
                
                decreases_html += f"""
                    <tr class="{row_class}">
                        <td><strong>{decrease['name']}</strong></td>
                        <td>{decrease['subject']}</td>
                        <td>{decrease['feb_papers']}</td>
                        <td>{decrease['july_papers']}</td>
                        <td class="negative">{decrease['paper_change']}</td>
                        <td>{decrease['feb_score']:.2f}</td>
                        <td>{decrease['july_score']:.2f}</td>
                        <td class="negative">{decrease['score_change']:.2f}</td>
                        <td><strong>{total_july_score:.2f}</strong></td>
                        <td>{category}</td>
                    </tr>"""
            
            decreases_html += """
                </tbody>
            </table>"""
        else:
            decreases_html = """
            <div class="section-title">Top 15 Decreases</div>
            <div class="no-data">No decreases found in the current dataset.</div>"""
        
        return f"""
        <div id="significant-changes" class="sheet">
            <h2>Significant Changes</h2>
            {summary_html}
            {increases_html}
            {decreases_html}
        </div>"""

    def _get_researcher_category(self, researcher_name: str, cross_field_researchers: List[Dict]) -> str:
        """Get the proper category for a researcher (same logic as detailed data tab)"""
        # Find the researcher in cross_field_researchers
        researcher_cross_field = None
        for cf in cross_field_researchers:
            if cf['name'] == researcher_name:
                researcher_cross_field = cf
                break
        
        if not researcher_cross_field:
            return "Single Field"
        
        # Multi-field researcher - categorize by total score
        total_score = researcher_cross_field['total_score']
        field_count = researcher_cross_field['field_count']
        
        if field_count > 1:
            if total_score >= 1.0:
                return "Cross-Field"
            elif 0.85 <= total_score < 1.0:
                return "Near Cross-Field"
            elif 0.70 <= total_score < 0.85:
                return "Approaching Cross-Field"
            else:
                return "Multidisciplinary"
        else:
            return "Single Field"

    def _should_highlight_cross_field(self, researcher_name: str, cross_field_researchers: List[Dict]) -> bool:
        """Only true cross-field researchers (≥1.0 total score) get blue highlighting"""
        for cf in cross_field_researchers:
            if cf['name'] == researcher_name:
                return (cf['field_count'] > 1 and cf['total_score'] >= 1.0)
        return False

    def _generate_subject_analysis_section(self, data: Dict[str, Any]) -> str:
        """Generate subject analysis section"""
        subject_analysis = data['subject_analysis']
        
        if not subject_analysis:
            return """
            <div id="subject-analysis" class="sheet">
                <h2>Subject Analysis</h2>
                <div class="no-data">No subject analysis data available.</div>
            </div>"""
        
        table_html = """
        <div class="section-title">Subject Performance Overview</div>
        <table>
            <thead>
                <tr>
                    <th class="sortable" onclick="sortTable(this, 0, 'string')">ESI Field</th>
                    <th class="sortable" onclick="sortTable(this, 1, 'number')">Feb Total</th>
                    <th class="sortable" onclick="sortTable(this, 2, 'number')">July Total</th>
                    <th class="sortable" onclick="sortTable(this, 3, 'number')">Highly Cited Paper Overlap Rate</th>
                    <th class="sortable" onclick="sortTable(this, 4, 'number')">Highly Cited Paper Growth Rate</th>
                    <th class="sortable" onclick="sortTable(this, 5, 'number')">Researchers with Changes</th>
                    <th class="sortable" onclick="sortTable(this, 6, 'string')">Status</th>
                </tr>
            </thead>
            <tbody>"""
        
        for subject in subject_analysis:
            growth_rate_sign = "+" if subject['growth_rate'] > 0 else ""
            table_html += f"""
                <tr>
                    <td><strong>{subject['subject']}</strong></td>
                    <td>{subject['feb_total']}</td>
                    <td>{subject['july_total']}</td>
                    <td>{subject['retention_rate']}%</td>
                    <td class="{self._format_change_class(subject['growth_rate'])}">{growth_rate_sign}{subject['growth_rate']}%</td>
                    <td>{subject['researchers_with_changes']}</td>
                    <td>{subject['status']}</td>
                </tr>"""
        
        table_html += """
            </tbody>
        </table>"""
        
        return f"""
        <div id="subject-analysis" class="sheet">
            <h2>Subject Analysis</h2>
            {table_html}
        </div>"""

    def _generate_detailed_data_section(self, data: Dict[str, Any]) -> str:
        """Generate detailed data section"""
        all_researchers = data['all_researchers']
        subjects = data['subjects']
        total_researchers = len(all_researchers)
        
        if not all_researchers:
            return """
            <div id="detailed-data" class="sheet">
                <h2>Detailed Researcher Data</h2>
                <div class="no-data">No detailed researcher data available.</div>
            </div>"""
        
        filter_html = f"""
        <div class="filter-controls">
            <input type="text" id="searchInput" placeholder="Search researcher name..." onkeyup="searchResearchers()">
            <select id="subjectFilter" onchange="filterResearchers()">
                <option value="all">All Subjects</option>"""
        
        for subject in subjects:
            filter_html += f'<option value="{subject}">{subject}</option>'
        
        filter_html += """
            </select>
            <select id="changeFilter" onchange="filterResearchers()">
                <option value="all">All Changes</option>
                <option value="increase">Increases Only</option>
                <option value="decrease">Decreases Only</option>
                <option value="major">Major Changes (±5+)</option>
                <option value="crossfield">Cross-Field Only</option>
                <option value="nearcrossfield">Near Cross-Field Only</option>
                <option value="approachingcrossfield">Approaching Cross-Field Only</option>
                <option value="multidisciplinary">Multidisciplinary Only</option>
                <option value="singlefield">Single Field Only</option>
            </select>
        </div>"""
        
        table_html = """
        <table id="detailedTable">
            <thead>
                <tr>
                    <th class="sortable" onclick="sortTable(this, 0, 'string')">Researcher</th>
                    <th class="sortable" onclick="sortTable(this, 1, 'string')">ESI Field</th>
                    <th class="sortable" onclick="sortTable(this, 2, 'number')">Feb Highly Cited Papers</th>
                    <th class="sortable" onclick="sortTable(this, 3, 'number')">July Highly Cited Papers</th>
                    <th class="sortable" onclick="sortTable(this, 4, 'number')">Highly Cited Papers Change</th>
                    <th class="sortable" onclick="sortTable(this, 5, 'number')">Total Feb Cross-Field Fractional Score</th>
                    <th class="sortable" onclick="sortTable(this, 6, 'number')">Total July Cross-Field Fractional Score</th>
                    <th class="sortable" onclick="sortTable(this, 7, 'number')">Cross-Field Fractional Score Change</th>
                    <th class="sortable" onclick="sortTable(this, 8, 'number')">Hot Papers</th>
                    <th class="sortable" onclick="sortTable(this, 9, 'number')">Citation Count</th>
                    <th class="sortable" onclick="sortTable(this, 10, 'string')">Category</th>
                </tr>
            </thead>
            <tbody>"""
        
        for researcher in all_researchers:
            # Only highlight if truly cross-field (≥1.0 total score)
            is_cross_field = (researcher.get('field_count', 1) > 1 and 
                            researcher.get('july_score', 0) >= 1.0)
            row_class = "cross-field-row" if is_cross_field else ""
            
            paper_change_sign = "+" if researcher['paper_change'] > 0 else ""
            score_change_sign = "+" if researcher['score_change'] > 0 else ""
            
            table_html += f"""
                <tr class="{row_class}" data-subject="{researcher['subject']}" data-change="{researcher['paper_change']}" data-category="{researcher.get('category_data', 'singlefield')}">
                    <td><strong>{researcher['name']}</strong></td>
                    <td>{researcher['subject']}</td>
                    <td>{researcher['feb_papers']}</td>
                    <td>{researcher['july_papers']}</td>
                    <td class="{self._format_change_class(researcher['paper_change'])}">{paper_change_sign}{researcher['paper_change']}</td>
                    <td>{researcher['feb_score']:.2f}</td>
                    <td>{researcher['july_score']:.2f}</td>
                    <td class="{self._format_change_class(researcher['score_change'])}">{score_change_sign}{researcher['score_change']:.2f}</td>
                    <td>{researcher['hot_papers']}</td>
                    <td>{researcher['times_cited']:,}</td>
                    <td>{researcher.get('category', 'Single Field')}</td>
                </tr>"""
        
        table_html += """
            </tbody>
        </table>"""
        
        return f"""
        <div id="detailed-data" class="sheet">
            <h2>Detailed Researcher Data</h2>
            <p>Total Researchers: <strong>{total_researchers:,}</strong></p>
            {filter_html}
            {table_html}
        </div>"""