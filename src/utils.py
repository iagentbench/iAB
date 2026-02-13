"""
Utility functions for path management and directory operations.
"""
from pathlib import Path
from datetime import datetime
from typing import Optional


def get_timestamp() -> str:
    """Return formatted timestamp for directory naming."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def get_input_directory(project_root: Path, keyword: Optional[str] = None) -> Path:
    """
    Get input directory based on keyword.
    
    Args:
        project_root: Root directory of the project
        keyword: Keyword for input organization. If None or empty, defaults to "local"
    
    Returns:
        Path to input directory:
        - If keyword is "local" or None/empty: input/local/
        - Otherwise: input/keywords/{keyword}/
    """
    if not keyword or keyword.strip() == "" or keyword.lower() == "local":
        keyword = "local"
        input_dir = project_root / "input" / keyword
    else:
        # For non-local keywords, use keywords subfolder
        input_dir = project_root / "input" / "keywords" / keyword
    
    return input_dir


def create_output_directory(project_root: Path, keyword: Optional[str] = None) -> Path:
    """
    Create output directory structure: output/{keyword}/{timestamp}/
    
    Args:
        project_root: Root directory of the project
        keyword: Keyword for output organization. If None or empty, defaults to "local"
    
    Returns:
        Path to the created run directory
    """
    if not keyword or keyword.strip() == "":
        keyword = "local"
    
    timestamp = get_timestamp()
    run_dir = project_root / "output" / keyword / timestamp
    
    # Create directory structure
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "graphs").mkdir(exist_ok=True)
    
    return run_dir
