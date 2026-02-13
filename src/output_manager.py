"""
Output file management and organization functions.
"""
import json
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def save_parquet_files(source_dir: Path, dest_dir: Path) -> List[Path]:
    """
    Copy parquet files from source to destination graphs directory.
    Excludes lancedb directory (GraphRAG embedding storage).
    
    Args:
        source_dir: Source directory containing parquet files
        dest_dir: Destination directory (graphs subdirectory will be created)
    
    Returns:
        List of paths to copied files
    """
    graphs_dir = dest_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True)
    
    # Only copy parquet files, exclude lancedb and other directories
    parquet_files = [f for f in source_dir.glob("*.parquet") if f.is_file()]
    copied_files = []
    
    for parquet_file in parquet_files:
        dest_file = graphs_dir / parquet_file.name
        shutil.copy2(parquet_file, dest_file)
        copied_files.append(dest_file)
    
    return copied_files


def save_community_details(graph_data: Dict, output_path: Path) -> Path:
    """
    Save community details as JSON file.
    
    Args:
        graph_data: Dictionary containing graph data with 'communities' and 'community_reports'
        output_path: Path where JSON file should be saved
    
    Returns:
        Path to saved JSON file
    """
    communities_data = {
        "communities": []
    }
    
    entities_df = graph_data.get('entities', pd.DataFrame())
    
    if 'communities' in graph_data:
        communities_df = graph_data['communities']
        reports_df = graph_data.get('community_reports', pd.DataFrame())
        
        for _, comm_row in communities_df.iterrows():
            comm_id = comm_row.get('community', comm_row.get('id'))
            
            # Find corresponding report if available
            comm_reports = reports_df[reports_df['community'] == comm_id] if not reports_df.empty else pd.DataFrame()
            
            comm_info = {
                "id": int(comm_id) if comm_id is not None else None,
                "level": int(comm_row.get('level', 0)),
                "size": int(comm_row.get('size', 0)),
                "title": comm_row.get('title', f'Community {comm_id}'),
            }
            
            if len(comm_reports) > 0:
                report = comm_reports.iloc[0]
                report_dict = report.to_dict() if hasattr(report, 'to_dict') else dict(report)
                
                comm_info.update({
                    "title": report_dict.get('title', comm_info['title']),
                    "summary": report_dict.get('summary', ''),
                    "rank": float(report_dict.get('rank', 0)) if report_dict.get('rank') is not None else None,
                    "rank_explanation": report_dict.get('rank_explanation', ''),
                })
                
                # Handle findings
                findings = report_dict.get('findings', [])
                if hasattr(findings, 'tolist'):
                    findings = findings.tolist()
                elif hasattr(findings, '__iter__') and not isinstance(findings, (str, bytes)):
                    try:
                        findings = list(findings)
                    except:
                        findings = []
                
                if isinstance(findings, list):
                    comm_info["findings"] = findings
                else:
                    comm_info["findings"] = []
            
            # Get entity IDs in this community
            if 'entity_ids' in comm_row:
                entity_ids = comm_row['entity_ids']
                if isinstance(entity_ids, list) and len(entity_ids) > 0:
                    comm_info["entity_ids"] = entity_ids
                    
                    # Get entity names if available
                    if not entities_df.empty and 'id' in entities_df.columns and 'title' in entities_df.columns:
                        entity_names = entities_df[entities_df['id'].isin(entity_ids)]['title'].tolist()
                        comm_info["entity_names"] = entity_names
            
            communities_data["communities"].append(comm_info)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(communities_data, f, indent=2, ensure_ascii=False)
    
    return output_path


def save_operational_metrics(metrics: Optional[Dict], errors: List[str], output_path: Path, timestamp: str) -> Path:
    """
    Save operational metrics as text file.
    
    Args:
        metrics: Metrics dictionary from load_api_metrics()
        errors: List of error messages
        output_path: Path where text file should be saved
        timestamp: Timestamp of the run
    
    Returns:
        Path to saved text file
    """
    from .metrics import format_metrics_text
    
    content = format_metrics_text(metrics, errors, timestamp)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return output_path


def load_graph_data(output_dir: Path) -> Dict:
    """
    Load all graph data from parquet files in output directory.
    Checks both output_dir and output_dir/graphs/ for parquet files.
    
    Args:
        output_dir: Directory containing parquet files (or graphs/ subdirectory)
    
    Returns:
        Dictionary with loaded dataframes
    """
    data = {}
    
    files = {
        'entities': 'entities.parquet',
        'relationships': 'relationships.parquet',
        'communities': 'communities.parquet',
        'community_reports': 'community_reports.parquet',
        'text_units': 'text_units.parquet',
        'documents': 'documents.parquet'
    }
    
    # Check both root and graphs/ subdirectory
    search_dirs = [output_dir]
    graphs_dir = output_dir / "graphs"
    if graphs_dir.exists():
        search_dirs.append(graphs_dir)
    
    for key, filename in files.items():
        for search_dir in search_dirs:
            filepath = search_dir / filename
            if filepath.exists():
                data[key] = pd.read_parquet(filepath)
                break  # Found it, no need to check other directories
    
    return data


def organize_outputs(
    run_dir: Path,
    temp_output_dir: Path,
    temp_logs_dir: Path,
    graph_data: Dict,
    metrics: Optional[Dict],
    errors: List[str],
    timestamp: str
) -> Dict:
    """
    Organize all outputs into the run directory structure.
    
    Args:
        run_dir: Target run directory (output/{keyword}/{timestamp}/)
        temp_output_dir: Temporary output directory from GraphRAG
        temp_logs_dir: Temporary logs directory
        graph_data: Dictionary with graph data
        metrics: Metrics dictionary
        errors: List of errors
        timestamp: Timestamp string
    
    Returns:
        Dictionary with paths to saved files
    """
    saved_files = {}
    
    # Files are already in run_dir (GraphRAG wrote directly there)
    # Parquet files should already be in graphs/ subdirectory (organized in CLI)
    # Just verify and list what we have
    graphs_dir = run_dir / "graphs"
    if graphs_dir.exists():
        parquet_files = list(graphs_dir.glob("*.parquet"))
        saved_files['parquet_files'] = parquet_files
    else:
        saved_files['parquet_files'] = []
    
    # Organize logs - keep comprehensive .log file and consolidate JSONL
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Copy comprehensive log file if it exists (from indexing step)
    # Only copy if source and destination are different
    comprehensive_log = None
    if temp_logs_dir.exists() and temp_logs_dir != logs_dir:
        comprehensive_logs = sorted(temp_logs_dir.glob("pipeline_*.log"), reverse=True)
        if comprehensive_logs:
            import shutil
            comprehensive_log = comprehensive_logs[0]
            dest_log = logs_dir / comprehensive_log.name
            if comprehensive_log != dest_log:  # Only copy if different paths
                shutil.copy2(comprehensive_log, dest_log)
                saved_files['comprehensive_log'] = dest_log
            else:
                # Already in the right place
                saved_files['comprehensive_log'] = comprehensive_log
    elif logs_dir.exists():
        # Logs are already in the right place, just find the file
        comprehensive_logs = sorted(logs_dir.glob("pipeline_*.log"), reverse=True)
        if comprehensive_logs:
            saved_files['comprehensive_log'] = comprehensive_logs[0]
    
    # Create a single consolidated JSONL file (fixed name, no timestamp)
    consolidated_log_path = logs_dir / "bedrock_api_calls_consolidated.jsonl"
    
    # Remove any old JSONL log files in this directory (we only want the consolidated one)
    for old_log in logs_dir.glob("bedrock_api_calls_*.jsonl"):
        if old_log != consolidated_log_path:
            old_log.unlink()
    
    # Collect all log entries from temp_logs_dir (current run's logs only)
    # Also check logs_dir in case logs were written directly there
    all_log_entries = []
    log_sources = []
    if temp_logs_dir.exists() and temp_logs_dir != logs_dir:
        log_sources.append(temp_logs_dir)
    if logs_dir.exists():
        log_sources.append(logs_dir)
    
    for log_source in log_sources:
        log_files = sorted(log_source.glob("bedrock_api_calls_*.jsonl"), reverse=True)
        # Take only the most recent log file (from current run)
        if log_files:
            log_file = log_files[0]  # Most recent
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            all_log_entries.append(line.strip())
            except Exception as e:
                errors.append(f"Error reading log file {log_file.name}: {str(e)}")
            break  # Only process one log file
    
    # Write all entries to the consolidated log file (ONLY ONE FILE)
    if all_log_entries:
        with open(consolidated_log_path, 'w', encoding='utf-8') as f:
            for entry in all_log_entries:
                f.write(entry + '\n')
        saved_files['log_file'] = consolidated_log_path
    else:
        # If no logs found, create empty file to indicate no API calls
        consolidated_log_path.touch()
        saved_files['log_file'] = consolidated_log_path
    
    # Save community details as JSON
    community_json_path = run_dir / "community_details.json"
    save_community_details(graph_data, community_json_path)
    saved_files['community_json'] = community_json_path
    
    # Save operational metrics
    metrics_path = run_dir / "operational_metrics.txt"
    save_operational_metrics(metrics, errors, metrics_path, timestamp)
    saved_files['metrics'] = metrics_path
    
    # Skip standalone interactive_graph.html and .graphml; view_graph.ipynb is the only viewer we generate.
    saved_files['graph_html'] = None
    saved_files['graphml'] = None

    # Generate Jupyter notebook for viewing
    try:
        from .notebook_generator import generate_viewer_notebook
        notebook_path = generate_viewer_notebook(run_dir, graph_data)
        saved_files['notebook'] = notebook_path
    except Exception as e:
        errors.append(f"Error generating viewer notebook: {str(e)}")
        saved_files['notebook'] = None
    
    return saved_files
