"""
Metrics extraction and formatting functions.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional


def load_api_metrics(logs_dir: Path, timestamp: Optional[str] = None) -> Optional[Dict]:
    """
    Load and summarize API call metrics from JSONL log files.
    
    Args:
        logs_dir: Directory containing JSONL log files
        timestamp: Optional timestamp (ignored, kept for compatibility)
    
    Returns:
        Dictionary with metrics or None if no logs found
    """
    # Always use the consolidated log file (fixed name)
    consolidated_log = logs_dir / "bedrock_api_calls_consolidated.jsonl"
    
    if consolidated_log.exists():
        latest_log = consolidated_log
    else:
        # Fallback to any log file if consolidated doesn't exist
        jsonl_files = sorted(logs_dir.glob("bedrock_api_calls_*.jsonl"), reverse=True)
        if not jsonl_files:
            return None
        latest_log = jsonl_files[0]
    
    chat_calls = []
    embedding_calls = []
    
    with open(latest_log, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    call_type = entry.get('call_type', '')
                    if call_type == 'chat':
                        chat_calls.append(entry)
                    elif call_type == 'embedding':
                        embedding_calls.append(entry)
                except:
                    continue
    
    # Extract token usage - handle both nested 'usage' dict and direct fields
    total_input_tokens = 0
    total_output_tokens = 0
    
    for call in chat_calls + embedding_calls:
        usage = call.get('usage', {})
        if isinstance(usage, dict):
            # Try multiple field names
            input_tokens = (
                usage.get('input_tokens') or 
                usage.get('prompt_tokens') or 
                0
            )
            output_tokens = (
                usage.get('output_tokens') or 
                usage.get('completion_tokens') or 
                0
            )
            # Only add if we got actual values (not 0 from missing data)
            if input_tokens:
                total_input_tokens += int(input_tokens)
            if output_tokens:
                total_output_tokens += int(output_tokens)
        else:
            # Fallback if usage is not a dict
            input_tokens = call.get('input_tokens', 0) or 0
            output_tokens = call.get('output_tokens', 0) or 0
            if input_tokens:
                total_input_tokens += int(input_tokens)
            if output_tokens:
                total_output_tokens += int(output_tokens)
    
    return {
        'log_file': latest_log.name,
        'chat_calls': len(chat_calls),
        'embedding_calls': len(embedding_calls),
        'total_calls': len(chat_calls) + len(embedding_calls),
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'total_tokens': total_input_tokens + total_output_tokens
    }


def format_metrics_text(metrics: Optional[Dict], errors: List[str], timestamp: str) -> str:
    """
    Format metrics and errors as readable text.
    
    Args:
        metrics: Metrics dictionary from load_api_metrics()
        errors: List of error messages
        timestamp: Timestamp of the run
    
    Returns:
        Formatted text string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("OPERATIONAL METRICS")
    lines.append("=" * 60)
    lines.append(f"Run Timestamp: {timestamp}")
    lines.append("")
    
    if metrics:
        lines.append("API CALLS:")
        lines.append(f"  Chat:         {metrics['chat_calls']:>4} calls")
        lines.append(f"  Embedding:    {metrics['embedding_calls']:>4} calls")
        lines.append(f"  Total:        {metrics['total_calls']:>4} calls")
        lines.append("")
        lines.append("TOKEN USAGE:")
        lines.append(f"  Input:        {metrics['input_tokens']:>10,} tokens")
        lines.append(f"  Output:       {metrics['output_tokens']:>10,} tokens")
        lines.append(f"  Total:        {metrics['total_tokens']:>10,} tokens")
        lines.append("")
        lines.append(f"Log File: {metrics['log_file']}")
    else:
        lines.append("⚠️  No API metrics available (log files not found)")
    
    lines.append("")
    lines.append("=" * 60)
    lines.append("ERRORS AND WARNINGS")
    lines.append("=" * 60)
    
    if errors:
        for i, error in enumerate(errors, 1):
            lines.append(f"{i}. {error}")
    else:
        lines.append("✓ No errors encountered")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
