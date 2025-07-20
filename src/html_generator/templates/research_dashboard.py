"""
Research Dashboard Template
Generates comprehensive interactive dashboards for research performance data
"""

from typing import Dict, List, Any
from .base_template import BaseTemplate
from ...utils.common import format_number_with_commas

class ResearchDashboardTemplate(BaseTemplate):
    """
    Comprehensive research dashboard template
    Based on your existing html_report_generator.py logic
    """
    
    def get_required_data_fields(self) -> List[str]:
        """Required data fields for research dashboard"""
        return [
            'comparison_id',
            'summary_statistics', 
            'researcher_changes',
            'researchers_only_in_dataset_1',
            'researchers_only_in_dataset_2',
            'researchers_unchanged'
        ]
    
    def generate_html(self, data: Dict[str, Any]) -> str:
        """
        Generate research dashboard HTML
        
        Args:
            data: JSON comparison report data
            
        Returns:
            Complete HTML dashboard
        """
        self.logger.info("Generating research dashboard HTML")
        
        # Validate input data
        self._validate_data(data)
        
        # Extract dataset type from comparison_id
        dataset_type = self._extract_dataset_type(data['comparison_id'])
        
        # Generate metadata
        metadata = self._generate_metadata(dataset_type)
        
        # Process data for dashboard
        processed_data = self._process_dashboard_data(data)
        
        # Generate HTML sections
        html_content = f"""<!DOCTYPE html>
<html lang="en-GB">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{metadata['title']}</title>
    <meta name="generated-at" content="{metadata['generated_at']}">
    <meta name="dataset-type" content="{metadata['dataset_type']}">
    {self._generate_css()}
</head>
<body>
    {self._generate_header(metadata)}
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
        
        self.logger.info(f"Generated HTML dashboard with {len(html_content)} characters")
        return html_content
    
    def _extract_dataset_type(self, comparison_id: str) -> str:
        """Extract dataset type from comparison ID"""
        # Parse comparison ID like "agricultural_sciences_highly_cited_only"
        parts = comparison_id.split('_')
        if len(parts) >= 3:
            # Remove the last part if it's a known sheet type
            if parts[-1] in ['only', 'researchers']:
                if parts[-2] in ['cited', 'incites']:
                    return '_'.join(parts[:-2])
            return '_'.join(parts[:-1])
        return comparison_id
    
    def _process_dashboard_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw JSON data for dashboard display"""
        summary = data.get('summary_statistics', {})
        
        # Extract summary statistics
        summary_stats = {
            'total_researchers_july': summary.get('dataset_2_total', 0),
            'retention_rate': self._calculate_retention_rate(summary),
            'change_rate': self._calculate_change_rate(summary),
            'total_subjects': 1,  # Single subject per report
            'sheet_type': 'Research Report'
        }
        
        # Process researchers data
        researchers_data = self._process_researchers_data(data)
        
        # Process cross-field data
        cross_field_data = self._process_cross_field_data(data)
        
        # Process significant changes
        significant_changes = self._process_significant_changes(data)
        
        return {
            'summary_stats': summary_stats,
            'researchers': researchers_data,
            'cross_field': cross_field_data,
            'significant_changes': significant_changes,
            'subject_analysis': self._process_subject_analysis(data),
            'metadata': data
        }
    
    def _calculate_retention_rate(self, summary: Dict[str, Any]) -> float:
        """Calculate retention rate percentage"""
        matched = summary.get('total_researchers_compared', 0)
        dataset_1_total = summary.get('dataset_1_total', 0)
        
        if dataset_1_total > 0:
            return round((matched / dataset_1_total) * 100, 1)
        return 0.0
    
    def _calculate_change_rate(self, summary: Dict[str, Any]) -> float:
        """Calculate change rate percentage"""
        changes = summary.get('researchers_with_changes', 0)
        matched = summary.get('total_researchers_compared', 0)
        
        if matched > 0:
            return round((changes / matched) * 100, 1)
        return 0.0
    
    def _process_researchers_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process researcher data for the researchers tab"""
        researchers = []
        
        # Process researchers with changes
        for change in data.get('researcher_changes', []):
            researcher = {
                'name': change.get('name', ''),
                'subject': change.get('esi_field', ''),
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
        
        # Process unchanged researchers
        for unchanged in data.get('researchers_unchanged', []):
            researcher = {
                'name': unchanged.get('name', ''),
                'subject': unchanged.get('esi_field', ''),
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
        
        # Process new researchers (only in July)
        for new_researcher in data.get('researchers_only_in_dataset_2', []):
            researcher = {
                'name': new_researcher.get('name', ''),
                'subject': new_researcher.get('esi_field', ''),
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
        
        # Sort by paper change (descending)
        return sorted(researchers, key=lambda x: x['paper_change'], reverse=True)
    
    def _process_cross_field_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process cross-field researcher data"""
        # For single-subject reports, identify researchers with high cross-field scores
        cross_field_researchers = []
        
        for researcher in self._process_researchers_data(data):
            if researcher['july_score'] >= 0.7:  # Threshold for cross-field consideration
                cross_field_info = {
                    'name': researcher['name'],
                    'fields': [researcher['subject']],  # Single field in this context
                    'total_papers': researcher['july_papers'],
                    'total_change': researcher['paper_change'],
                    'total_score': researcher['july_score'],
                    'field_count': 1,
                    'is_cross_field': researcher['july_score'] >= 1.0,
                    'is_near_cross_field': 0.85 <= researcher['july_score'] <= 0.99,
                    'is_approaching_cross_field': 0.70 <= researcher['july_score'] <= 0.84
                }
                cross_field_researchers.append(cross_field_info)
        
        return sorted(cross_field_researchers, key=lambda x: x['total_score'], reverse=True)
    
    def _process_significant_changes(self, data: Dict[str, Any]) -> Dict[str, List]:
        """Process significant changes for increases/decreases"""
        all_changes = []
        
        for change in data.get('researcher_changes', []):
            change_data = {
                'name': change.get('name', ''),
                'subject': change.get('esi_field', ''),
                'feb_papers': change.get('feb_values', {}).get('highly_cited_papers', 0),
                'july_papers': change.get('july_values', {}).get('highly_cited_papers', 0),
                'paper_change': change.get('highly_cited_papers_change', 0),
                'feb_score': round(change.get('feb_values', {}).get('indicative_cross_field_score', 0), 2),
                'july_score': round(change.get('july_values', {}).get('indicative_cross_field_score', 0), 2),
                'score_change': round(change.get('cross_field_score_change', 0), 2)
            }
            if change_data['name'] and change_data['paper_change'] != 0:
                all_changes.append(change_data)
        
        # Separate increases and decreases
        increases = [c for c in all_changes if c['paper_change'] > 0]
        decreases = [c for c in all_changes if c['paper_change'] < 0]
        
        increases.sort(key=lambda x: x['paper_change'], reverse=True)
        decreases.sort(key=lambda x: x['paper_change'])
        
        return {
            'increases': increases[:15],
            'decreases': decreases[:15]
        }
    
    def _process_subject_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process subject analysis data"""
        summary = data.get('summary_statistics', {})
        
        feb_total = summary.get('dataset_1_total', 0)
        july_total = summary.get('dataset_2_total', 0)
        
        growth_rate = 0
        if feb_total > 0:
            growth_rate = ((july_total - feb_total) / feb_total) * 100
        
        retention_rate = 0
        matched = summary.get('total_researchers_compared', 0)
        if feb_total > 0:
            retention_rate = (matched / feb_total) * 100
        
        return {
            'subject': data.get('comparison_id', 'Unknown').replace('_', ' ').title(),
            'feb_total': feb_total,
            'july_total': july_total,
            'retention_rate': round(retention_rate, 1),
            'growth_rate': round(growth_rate, 1),
            'researchers_with_changes': summary.get('researchers_with_changes', 0),
            'status': self._get_subject_status(growth_rate)
        }
    
    def _get_subject_status(self, growth_rate: float) -> str:
        """Get status emoji and text based on growth rate"""
        if growth_rate >= 30:
            return "🚀 High Growth"
        elif growth_rate >= 15:
            return "📈 Growing"
        elif growth_rate >= 5:
            return "⚙️ Expanding"
        elif growth_rate >= 0:
            return "🔬 Moderate"
        elif growth_rate >= -10:
            return "⚠️ Declining"
        else:
            return "📉 Major Decline"

    def _generate_css(self) -> str:
        """Generate CSS styles for the research dashboard"""
        css_classes = self._get_css_classes()
        
        return f"""<style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f7fa;
                color: #333;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}
            
            .header h1 {{
                margin: 0;
                font-size: 28px;
                font-weight: 300;
            }}
            
            .header p {{
                margin: 10px 0 0 0;
                opacity: 0.9;
                font-size: 16px;
            }}
            
            .tabs {{
                display: flex;
                background: white;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            
            .tab {{
                flex: 1;
                padding: 15px 20px;
                background: #f8f9fa;
                border: none;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.3s ease;
                border-right: 1px solid #e9ecef;
            }}
            
            .tab:last-child {{
                border-right: none;
            }}
            
            .tab.active {{
                background: #667eea;
                color: white;
            }}
            
            .tab:hover:not(.active) {{
                background: #e9ecef;
            }}
            
            .sheet {{
                display: none;
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            
            .sheet.active {{
                display: block;
            }}
            
            .sheet h2 {{
                margin: 0 0 20px 0;
                color: #667eea;
                font-size: 24px;
                font-weight: 600;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }}
            
            th {{
                background: #667eea;
                color: white;
                padding: 12px 15px;
                text-align: left;
                font-weight: 600;
                font-size: 14px;
            }}
            
            td {{
                padding: 12px 15px;
                border-bottom: 1px solid #f1f3f4;
                font-size: 13px;
            }}
            
            tr:hover {{
                background-color: #f8f9ff;
            }}
            
            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                margin: 10px;
                flex: 1;
            }}
            
            .metric-value {{
                font-size: 28px;
                font-weight: 700;
                color: #667eea;
                margin-bottom: 5px;
            }}
            
            .metric-label {{
                color: #6c757d;
                font-size: 14px;
                font-weight: 500;
            }}
            
            .metrics-grid {{
                display: flex;
                gap: 20px;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }}
            
            .positive {{ color: #28a745; font-weight: 600; }}
            .negative {{ color: #dc3545; font-weight: 600; }}
            .neutral {{ color: #6c757d; }}
            
            .field-badge {{
                background: #28a745;
                color: white;
                padding: 2px 6px;
                border-radius: 8px;
                font-size: 10px;
                font-weight: 600;
                margin: 0 2px;
            }}
            
            .section-title {{
                background: #f8f9fa;
                padding: 15px 20px;
                margin: 20px -20px;
                border-left: 4px solid #667eea;
                font-weight: 600;
                color: #495057;
            }}
            
            .highlight-row {{
                background-color: #fff3cd !important;
            }}
            
            .cross-field-row {{
                background-color: #d1ecf1 !important;
            }}
            
            .filter-controls {{
                margin-bottom: 20px;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 6px;
            }}
            
            .filter-controls select, .filter-controls input {{
                padding: 8px 12px;
                margin: 0 10px 10px 0;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
            }}
            
            .no-data {{
                text-align: center;
                padding: 40px;
                color: #6c757d;
                font-style: italic;
            }}
            
            .sortable {{
                cursor: pointer;
                position: relative;
                user-select: none;
            }}
            
            .sortable:hover {{
                background-color: #5a6bb8 !important;
            }}
            
            .sortable:after {{
                content: ' ↕';
                font-size: 12px;
                opacity: 0.5;
            }}
            
            .sortable.sort-asc:after {{
                content: ' ↑';
                opacity: 1;
            }}
            
            .sortable.sort-desc:after {{
                content: ' ↓';
                opacity: 1;
            }}
            
            @media (max-width: 768px) {{
                .tabs {{
                    flex-direction: column;
                }}
                
                .metrics-grid {{
                    flex-direction: column;
                }}
                
                table {{
                    font-size: 12px;
                }}
                
                th, td {{
                    padding: 8px 10px;
                }}
            }}
        </style>"""
    
    def _generate_javascript(self) -> str:
        """Generate JavaScript functionality for the dashboard"""
        
        return """<script>
            function showSheet(sheetId) {
                // Hide all sheets
                const sheets = document.querySelectorAll('.sheet');
                sheets.forEach(sheet => sheet.classList.remove('active'));
                
                // Remove active class from all tabs
                const tabs = document.querySelectorAll('.tab');
                tabs.forEach(tab => tab.classList.remove('active'));
                
                // Show selected sheet and activate tab
                document.getElementById(sheetId).classList.add('active');
                event.target.classList.add('active');
            }
            
            function sortTable(header, columnIndex, dataType) {
                const table = header.closest('table');
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                
                // Determine sort direction
                let isAsc = true;
                if (header.classList.contains('sort-asc')) {
                    isAsc = false;
                    header.classList.remove('sort-asc');
                    header.classList.add('sort-desc');
                } else {
                    // Remove sort classes from all headers in this table
                    table.querySelectorAll('.sortable').forEach(h => {
                        h.classList.remove('sort-asc', 'sort-desc');
                    });
                    header.classList.add('sort-asc');
                }
                
                // Sort rows
                rows.sort((a, b) => {
                    let aVal = a.cells[columnIndex].textContent.trim();
                    let bVal = b.cells[columnIndex].textContent.trim();
                    
                    if (dataType === 'number') {
                        // Extract numeric values, handling + and - signs
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
                
                // Re-append sorted rows
                rows.forEach(row => tbody.appendChild(row));
            }
            
            function searchResearchers() {
                const input = document.getElementById('researcherSearchInput');
                const table = document.getElementById('researchersTable');
                
                if (!input || !table) return;
                
                const filter = input.value.toLowerCase();
                const rows = table.getElementsByTagName('tbody')[0].getElementsByTagName('tr');
                
                for (let row of rows) {
                    const name = row.getAttribute('data-name') || row.cells[0].textContent;
                    if (name.toLowerCase().includes(filter)) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                }
            }
            
            function filterResearchers() {
                const subjectFilter = document.getElementById('subjectFilter');
                const changeFilter = document.getElementById('changeFilter');
                const table = document.getElementById('researchersTable');
                
                if (!subjectFilter || !changeFilter || !table) return;
                
                const subjectValue = subjectFilter.value;
                const changeValue = changeFilter.value;
                const rows = table.getElementsByTagName('tbody')[0].getElementsByTagName('tr');
                
                for (let row of rows) {
                    let showRow = true;
                    const subject = row.getAttribute('data-subject') || '';
                    const change = parseInt(row.getAttribute('data-change') || '0');
                    
                    // Subject filter
                    if (subjectValue !== 'all' && !subject.includes(subjectValue)) {
                        showRow = false;
                    }
                    
                    // Change filter
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
                    const category = row.getAttribute('data-category') || '';
                    
                    if (categoryValue !== 'all' && category !== categoryValue) {
                        showRow = false;
                    }
                    
                    row.style.display = showRow ? '' : 'none';
                }
            }
            
            // Initialize with summary sheet
            document.addEventListener('DOMContentLoaded', function() {
                const summaryTab = document.querySelector('.tab[onclick*="summary"]');
                if (summaryTab) {
                    summaryTab.click();
                }
            });
        </script>"""
    
    def _generate_header(self, metadata: Dict[str, str]) -> str:
        """Generate header section"""
        return f"""
        <div class="header">
            <h1>{metadata['title']}</h1>
            <p>Comparative Analysis: February 2025 vs July 2025 - Research Performance Report</p>
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
        significant_changes = data['significant_changes']
        
        # Generate metrics cards
        metrics_html = f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{stats['total_researchers_july']:,}</div>
                <div class="metric-label">Total Researchers (July)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['retention_rate']:.1f}%</div>
                <div class="metric-label">Retention Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['change_rate']:.1f}%</div>
                <div class="metric-label">Change Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(data['cross_field'])}</div>
                <div class="metric-label">Cross-Field Researchers</div>
            </div>
        </div>"""
        
        # Generate top changes table
        changes_html = ""
        if significant_changes['increases']:
            changes_html = """
            <div class="section-title">Largest Changes in Researcher's Highly Cited Papers Count</div>
            <table>
                <thead>
                    <tr>
                        <th class="sortable" onclick="sortTable(this, 0, 'string')">Researcher</th>
                        <th class="sortable" onclick="sortTable(this, 1, 'string')">ESI Field</th>
                        <th class="sortable" onclick="sortTable(this, 2, 'number')">Highly Cited Papers Change</th>
                        <th class="sortable" onclick="sortTable(this, 3, 'number')">Cross-Field Score Change</th>
                    </tr>
                </thead>
                <tbody>"""
            
            for i, increase in enumerate(significant_changes['increases'][:10]):
                row_class = "highlight-row" if i == 0 else ""
                paper_change_sign = "+" if increase['paper_change'] > 0 else ""
                score_change_sign = "+" if increase['score_change'] > 0 else ""
                
                changes_html += f"""
                    <tr class="{row_class}">
                        <td><strong>{increase['name']}</strong></td>
                        <td>{increase['subject']}</td>
                        <td class="positive">{paper_change_sign}{increase['paper_change']}</td>
                        <td class="positive">{score_change_sign}{increase['score_change']:.2f}</td>
                    </tr>"""
            
            changes_html += """
                </tbody>
            </table>"""
        
        return f"""
        <div id="summary" class="sheet active">
            <h2>Executive Summary</h2>
            {metrics_html}
            {changes_html}
        </div>"""

    def _generate_researchers_section(self, data: Dict[str, Any]) -> str:
        """Generate researchers section with filtering"""
        researchers = data['researchers']
        
        if not researchers:
            return """
            <div id="researchers" class="sheet">
                <h2>Researchers</h2>
                <div class="no-data">No researcher data available in the current dataset.</div>
            </div>"""
        
        # Get unique subjects for filter
        subjects = list(set(r['subject'] for r in researchers if r['subject']))
        
        filter_html = f"""
        <div class="filter-controls">
            <input type="text" id="researcherSearchInput" placeholder="Search researcher name..." onkeyup="searchResearchers()">
            <select id="subjectFilter" onchange="filterResearchers()">
                <option value="all">All Subjects</option>"""
        
        for subject in sorted(subjects):
            filter_html += f'<option value="{subject}">{subject}</option>'
        
        filter_html += """
            </select>
            <select id="changeFilter" onchange="filterResearchers()">
                <option value="all">All Changes</option>
                <option value="increase">Increases Only</option>
                <option value="decrease">Decreases Only</option>
                <option value="major">Major Changes (±5+)</option>
            </select>
        </div>"""
        
        # Generate table
        table_html = """
        <table id="researchersTable">
            <thead>
                <tr>
                    <th class="sortable" onclick="sortTable(this, 0, 'string')">Researcher</th>
                    <th class="sortable" onclick="sortTable(this, 1, 'string')">ESI Field</th>
                    <th class="sortable" onclick="sortTable(this, 2, 'number')">Feb Papers</th>
                    <th class="sortable" onclick="sortTable(this, 3, 'number')">July Papers</th>
                    <th class="sortable" onclick="sortTable(this, 4, 'number')">Paper Change</th>
                    <th class="sortable" onclick="sortTable(this, 5, 'number')">Feb Score</th>
                    <th class="sortable" onclick="sortTable(this, 6, 'number')">July Score</th>
                    <th class="sortable" onclick="sortTable(this, 7, 'number')">Score Change</th>
                    <th class="sortable" onclick="sortTable(this, 8, 'number')">Hot Papers</th>
                    <th class="sortable" onclick="sortTable(this, 9, 'number')">Citation Count</th>
                </tr>
            </thead>
            <tbody>"""
        
        for researcher in researchers:
            paper_change_sign = "+" if researcher['paper_change'] > 0 else ""
            score_change_sign = "+" if researcher['score_change'] > 0 else ""
            paper_change_class = "positive" if researcher['paper_change'] > 0 else ("negative" if researcher['paper_change'] < 0 else "neutral")
            score_change_class = "positive" if researcher['score_change'] > 0 else ("negative" if researcher['score_change'] < 0 else "neutral")
            
            table_html += f"""
                <tr data-subject="{researcher['subject']}" data-change="{researcher['paper_change']}" data-name="{researcher['name']}">
                    <td><strong>{researcher['name']}</strong></td>
                    <td>{researcher['subject']}</td>
                    <td>{researcher['feb_papers']}</td>
                    <td>{researcher['july_papers']}</td>
                    <td class="{paper_change_class}">{paper_change_sign}{researcher['paper_change']}</td>
                    <td>{researcher['feb_score']:.2f}</td>
                    <td>{researcher['july_score']:.2f}</td>
                    <td class="{score_change_class}">{score_change_sign}{researcher['score_change']:.2f}</td>
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
            <div class="section-title">Researchers by Individual ESI Field</div>
            {table_html}
        </div>"""

    def _generate_cross_field_section(self, data: Dict[str, Any]) -> str:
        """Generate cross-field researchers section"""
        cross_field_researchers = data['cross_field']
        
        if not cross_field_researchers:
            return """
            <div id="cross-field" class="sheet">
                <h2>Cross-Field Researchers</h2>
                <div class="no-data">No cross-field researchers found in the current dataset.</div>
            </div>"""
        
        # Count categories
        true_cross_field = [cf for cf in cross_field_researchers if cf['is_cross_field']]
        near_cross_field = [cf for cf in cross_field_researchers if cf['is_near_cross_field']]
        approaching_cross_field = [cf for cf in cross_field_researchers if cf['is_approaching_cross_field']]
        
        filter_html = """
        <div class="filter-controls">
            <select id="categoryFilter" onchange="filterByCategory()">
                <option value="all">All Categories</option>
                <option value="crossfield">Cross-Field (≥1.0)</option>
                <option value="nearcrossfield">Near Cross-Field (0.85-0.99)</option>
                <option value="approachingcrossfield">Approaching Cross-Field (0.70-0.84)</option>
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
                    <th class="sortable" onclick="sortTable(this, 2, 'number')">Field Count</th>
                    <th class="sortable" onclick="sortTable(this, 3, 'number')">Total Papers</th>
                    <th class="sortable" onclick="sortTable(this, 4, 'number')">Total Change</th>
                    <th class="sortable" onclick="sortTable(this, 5, 'number')">Total Score</th>
                    <th class="sortable" onclick="sortTable(this, 6, 'string')">Category</th>
                </tr>
            </thead>
            <tbody>"""
        
        for cf in cross_field_researchers:
            # Create field badges
            field_badges = ""
            for field in cf['fields']:
                field_badges += f'<span class="field-badge">{field}</span>'
            
            total_change_sign = "+" if cf['total_change'] > 0 else ""
            change_class = "positive" if cf['total_change'] > 0 else ("negative" if cf['total_change'] < 0 else "neutral")
            
            # Determine category
            if cf['is_cross_field']:
                category = "🌟 Cross-Field"
                category_data = "crossfield"
            elif cf['is_near_cross_field']:
                category = "⭐ Near Cross-Field"
                category_data = "nearcrossfield"
            elif cf['is_approaching_cross_field']:
                category = "🔄 Approaching Cross-Field"
                category_data = "approachingcrossfield"
            else:
                category = "🔬 Research Focus"
                category_data = "research"
            
            table_html += f"""
                <tr class="cross-field-row" data-category="{category_data}">
                    <td><strong>{cf['name']}</strong></td>
                    <td>{field_badges}</td>
                    <td>{cf['field_count']}</td>
                    <td>{cf['total_papers']}</td>
                    <td class="{change_class}">{total_change_sign}{cf['total_change']}</td>
                    <td><strong>{cf['total_score']:.2f}</strong></td>
                    <td>{category}</td>
                </tr>"""
        
        table_html += """
            </tbody>
        </table>"""
        
        return f"""
        <div id="cross-field" class="sheet">
            <h2>Cross-Field Researchers</h2>
            <p>Researchers with high cross-field scores indicating research spanning multiple disciplines.</p>
            {filter_html}
            {breakdown_html}
            {table_html}
        </div>"""

    def _generate_significant_changes_section(self, data: Dict[str, Any]) -> str:
        """Generate significant changes section"""
        significant_changes = data['significant_changes']
        
        increases_html = ""
        if significant_changes['increases']:
            increases_html = self._generate_changes_table(
                "Major Increases", 
                significant_changes['increases'], 
                "positive"
            )
        
        decreases_html = ""
        if significant_changes['decreases']:
            decreases_html = self._generate_changes_table(
                "Major Decreases", 
                significant_changes['decreases'], 
                "negative"
            )
        
        return f"""
        <div id="significant-changes" class="sheet">
            <h2>Significant Changes</h2>
            {increases_html}
            {decreases_html}
        </div>"""

    def _generate_changes_table(self, title: str, changes: List[Dict], change_type: str) -> str:
        """Generate a table for increases or decreases"""
        table_html = f"""
        <div class="section-title">{title}</div>
        <table>
            <thead>
                <tr>
                    <th>Researcher</th>
                    <th>ESI Field</th>
                    <th>Feb Papers</th>
                    <th>July Papers</th>
                    <th>Paper Change</th>
                    <th>Feb Score</th>
                    <th>July Score</th>
                    <th>Score Change</th>
                </tr>
            </thead>
            <tbody>"""
        
        for change in changes:
            paper_change_sign = "+" if change['paper_change'] > 0 else ""
            score_change_sign = "+" if change['score_change'] > 0 else ""
            
            table_html += f"""
                <tr>
                    <td><strong>{change['name']}</strong></td>
                    <td>{change['subject']}</td>
                    <td>{change['feb_papers']}</td>
                    <td>{change['july_papers']}</td>
                    <td class="{change_type}">{paper_change_sign}{change['paper_change']}</td>
                    <td>{change['feb_score']:.2f}</td>
                    <td>{change['july_score']:.2f}</td>
                    <td class="{change_type}">{score_change_sign}{change['score_change']:.2f}</td>
                </tr>"""
        
        table_html += """
            </tbody>
        </table>"""
        
        return table_html

    def _generate_subject_analysis_section(self, data: Dict[str, Any]) -> str:
        """Generate subject analysis section"""
        subject_analysis = data['subject_analysis']
        
        growth_rate_sign = "+" if subject_analysis['growth_rate'] > 0 else ""
        growth_rate_class = "positive" if subject_analysis['growth_rate'] > 0 else ("negative" if subject_analysis['growth_rate'] < 0 else "neutral")
        
        return f"""
        <div id="subject-analysis" class="sheet">
            <h2>Subject Analysis</h2>
            <div class="section-title">Subject Performance Overview</div>
            <table>
                <thead>
                    <tr>
                        <th>ESI Field</th>
                        <th>Feb Total</th>
                        <th>July Total</th>
                        <th>Retention Rate</th>
                        <th>Growth Rate</th>
                        <th>Researchers with Changes</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>{subject_analysis['subject']}</strong></td>
                        <td>{subject_analysis['feb_total']}</td>
                        <td>{subject_analysis['july_total']}</td>
                        <td>{subject_analysis['retention_rate']}%</td>
                        <td class="{growth_rate_class}">{growth_rate_sign}{subject_analysis['growth_rate']}%</td>
                        <td>{subject_analysis['researchers_with_changes']}</td>
                        <td>{subject_analysis['status']}</td>
                    </tr>
                </tbody>
            </table>
        </div>"""

    def _generate_detailed_data_section(self, data: Dict[str, Any]) -> str:
        """Generate detailed data section (same as researchers but with all data)"""
        researchers = data['researchers']
        total_researchers = len(researchers)
        
        return f"""
        <div id="detailed-data" class="sheet">
            <h2>Detailed Researcher Data</h2>
            <p>Total Researchers: <strong>{total_researchers:,}</strong></p>
            <div class="section-title">Complete Dataset</div>
            <p>This section contains the same data as the Researchers tab but without filtering options for export purposes.</p>
        </div>"""