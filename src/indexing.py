"""
GraphRAG indexing orchestration functions.
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


def run_graphrag_indexing(
    project_root: Path,
    input_dir: Path,
    output_dir: Path,
    logs_dir: Path,
    verbose: bool = False
) -> Dict:
    """
    Run GraphRAG indexing pipeline with JSON parsing fix.
    
    Args:
        project_root: Root directory of the project
        input_dir: Directory containing input files
        output_dir: Directory for GraphRAG output (temporary, will be moved later)
        logs_dir: Directory for logs
        verbose: Enable verbose output
    
    Returns:
        Dictionary with:
            - success: bool
            - errors: List[str]
            - return_code: int
            - core_files_exist: bool
            - reports_exist: bool
    """
    errors = []
    
    # Find the GraphRAG indexer script
    indexer_script_path = project_root / "src" / "graphrag_indexer.py"
    
    if not indexer_script_path.exists():
        errors.append(f"GraphRAG indexer script not found: {indexer_script_path}")
        return {
            "success": False,
            "errors": errors,
            "return_code": -1,
            "core_files_exist": False,
            "reports_exist": False
        }
    
    # Set AWS environment variables
    os.environ['AWS_DEFAULT_REGION'] = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    # AWS_BEARER_TOKEN_BEDROCK should be set via environment variable or AWS credentials
    # Do not hardcode API keys in source code
    if 'AWS_BEARER_TOKEN_BEDROCK' not in os.environ:
        logger.warning("AWS_BEARER_TOKEN_BEDROCK not set in environment. Ensure AWS credentials are configured.")
    
    # The fix script writes logs to project_root/logs, but we want them in logs_dir
    # We'll copy them after indexing completes
    temp_logs_dir = project_root / "logs"
    temp_logs_dir.mkdir(exist_ok=True)
    
    # Ensure logs_dir exists
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create comprehensive log file
    comprehensive_log_path = logs_dir / f"pipeline_{timestamp}.log"
    comprehensive_log_file = open(comprehensive_log_path, 'w', encoding='utf-8')
    
    # Run indexing with the fix script - pass output_dir directly
    # Quote paths to handle spaces in directory names
    project_root_str = str(project_root.absolute())
    output_dir_str = str(output_dir.absolute())
    indexer_script_str = str(indexer_script_path.absolute())
    
    # Pass input directory to ensure GraphRAG only reads from keyword-specific folder
    input_dir_str = str(input_dir.absolute())
    
    # Build command to run the indexer script using the current Python environment
    # Use sys.executable to ensure we use the same Python interpreter with all dependencies
    cmd = [
        sys.executable, indexer_script_str,
        project_root_str,
        "--input", input_dir_str,
        "--output", output_dir_str
    ]
    if verbose:
        cmd.append("--verbose")
    
    try:
        # Write command to log
        comprehensive_log_file.write(f"=== GraphRAG Indexing Pipeline ===\n")
        comprehensive_log_file.write(f"Timestamp: {timestamp}\n")
        comprehensive_log_file.write(f"Command: {' '.join(cmd)}\n")
        comprehensive_log_file.write(f"Working Directory: {project_root}\n")
        comprehensive_log_file.write(f"Input Directory: {input_dir}\n")
        comprehensive_log_file.write(f"Output Directory: {output_dir}\n")
        comprehensive_log_file.write(f"{'='*80}\n\n")
        comprehensive_log_file.flush()
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )
        
        # Capture ALL output to comprehensive log file
        output_lines = []
        for line in process.stdout:
            if line:
                # Write to comprehensive log
                comprehensive_log_file.write(line)
                comprehensive_log_file.flush()
                
                line_stripped = line.strip()
                if line_stripped:
                    output_lines.append(line)
                    # Check for errors in output - catch various error patterns
                    if any(keyword in line_stripped for keyword in ["ERROR", "Exception", "Error", "Traceback", "ModuleNotFoundError", "ImportError"]):
                        errors.append(line_stripped)
        
        process.wait()
        return_code = process.returncode
        
        # Write completion status to log
        comprehensive_log_file.write(f"\n{'='*80}\n")
        comprehensive_log_file.write(f"Pipeline completed with return code: {return_code}\n")
        comprehensive_log_file.write(f"{'='*80}\n")
        comprehensive_log_file.close()
        
    except Exception as e:
        errors.append(f"Error running indexing: {str(e)}")
        return {
            "success": False,
            "errors": errors,
            "return_code": -1,
            "core_files_exist": False,
            "reports_exist": False
        }
    
    # Check results - files might be in output_dir or output_dir/graphs/
    graphs_dir = output_dir / "graphs"
    core_files = ['entities.parquet', 'relationships.parquet', 'communities.parquet']
    core_files_exist = all(
        (output_dir / f).exists() or (graphs_dir.exists() and (graphs_dir / f).exists())
        for f in core_files
    )
    reports_exist = (
        (output_dir / "community_reports.parquet").exists() or
        (graphs_dir.exists() and (graphs_dir / "community_reports.parquet").exists())
    )
    
    # Copy logs from project_root/logs to logs_dir
    # Find the latest log file
    if temp_logs_dir.exists():
        log_files = sorted(temp_logs_dir.glob("bedrock_api_calls_*.jsonl"), reverse=True)
        if log_files:
            # Copy to logs_dir
            import shutil
            for log_file in log_files:
                dest = logs_dir / log_file.name
                if not dest.exists():  # Don't overwrite if already moved
                    shutil.copy2(log_file, dest)
    
    success = return_code == 0 or core_files_exist
    
    return {
        "success": success,
        "errors": errors,
        "return_code": return_code,
        "core_files_exist": core_files_exist,
        "reports_exist": reports_exist,
        "comprehensive_log": comprehensive_log_path
    }
