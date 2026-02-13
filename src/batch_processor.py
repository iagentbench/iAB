"""
Batch processing module for processing multiple queries from a JSONL file.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src.utils import get_input_directory, create_output_directory, get_timestamp
from src.html_converter import convert_html_files_in_directory
from src.indexing import run_graphrag_indexing
from src.metrics import load_api_metrics
from src.output_manager import load_graph_data, organize_outputs


class AdaptiveThrottler:
    """Adaptive throttling that increases delay based on errors/rate limits."""
    
    def __init__(self, initial_delay: float = 1.0, max_delay: float = 60.0, backoff_factor: float = 1.5):
        """
        Initialize adaptive throttler.
        
        Args:
            initial_delay: Initial delay in seconds (reduced default since processing takes minutes)
            max_delay: Maximum delay in seconds
            backoff_factor: Multiplier for delay when errors occur
        """
        self.current_delay = initial_delay
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.consecutive_errors = 0
        self.consecutive_successes = 0
    
    def wait(self):
        """Wait for the current delay period."""
        if self.current_delay > 0:
            time.sleep(self.current_delay)
    
    def on_success(self):
        """Called when a query succeeds - gradually reduce delay."""
        self.consecutive_errors = 0
        self.consecutive_successes += 1
        
        # Gradually reduce delay after multiple successes
        if self.consecutive_successes >= 3:
            self.current_delay = max(self.initial_delay, self.current_delay / self.backoff_factor)
            self.consecutive_successes = 0
    
    def on_error(self, is_rate_limit: bool = False, is_daily_limit: bool = False):
        """Called when a query fails - increase delay."""
        self.consecutive_errors += 1
        self.consecutive_successes = 0
        
        # Daily token limits require much longer waits (hours) - don't increase delay here
        # The user should pause and resume later, or we should skip processing
        if is_daily_limit:
            # Don't increase delay - daily limits need manual intervention or very long waits
            # Just log that we hit the limit
            return
        
        # Increase delay more aggressively for rate limits
        if is_rate_limit:
            self.current_delay = min(self.max_delay, self.current_delay * (self.backoff_factor * 2))
        else:
            self.current_delay = min(self.max_delay, self.current_delay * self.backoff_factor)
    
    def reset(self):
        """Reset to initial delay."""
        self.current_delay = self.initial_delay
        self.consecutive_errors = 0
        self.consecutive_successes = 0


def is_rate_limit_error(error_msg: str) -> bool:
    """Check if error message indicates a rate limit."""
    error_lower = error_msg.lower()
    rate_limit_indicators = [
        'rate limit',
        'rate_limit',
        'too many requests',
        '429',
        'throttle',
        'quota exceeded',
        'request limit'
    ]
    return any(indicator in error_lower for indicator in rate_limit_indicators)


def is_daily_token_limit_error(error_msg: str) -> bool:
    """Check if error message indicates a daily token limit (requires much longer wait)."""
    error_lower = error_msg.lower()
    daily_limit_indicators = [
        'too many tokens per day',
        'tokens per day',
        'daily limit',
        'daily quota',
        'quota exceeded for the day'
    ]
    return any(indicator in error_lower for indicator in daily_limit_indicators)


def check_output_exists(project_root: Path, query: str, batch_name: str) -> bool:
    """
    Check if output already exists for a query (for resume functionality).
    
    Checks for definitive indicators of successful completion:
    - Core parquet files (entities.parquet, relationships.parquet)
    - community_details.json (indicates successful community generation)
    - view_graph.ipynb (indicates successful output organization)
    
    Args:
        project_root: Root directory of the project
        query: Query string
        batch_name: Batch name (e.g., "2025_seeds")
    
    Returns:
        True if output directory exists and contains successful completion indicators
    """
    output_base = project_root / "output" / batch_name / query
    if not output_base.exists():
        return False
    
    # Check if any timestamped subdirectory exists with successful completion indicators
    for timestamp_dir in output_base.iterdir():
        if not timestamp_dir.is_dir():
            continue
        
        # Check for core graph files (minimum requirement)
        graphs_dir = timestamp_dir / "graphs"
        core_files = ['entities.parquet', 'relationships.parquet']
        has_core_files = False
        
        if graphs_dir.exists():
            has_core_files = all((graphs_dir / f).exists() for f in core_files)
        
        # If no graphs directory, check root for parquet files (fallback)
        if not has_core_files:
            parquet_files = list(timestamp_dir.glob("*.parquet"))
            has_core_files = len(parquet_files) >= 2
        
        if not has_core_files:
            continue
        
        # Check for definitive success indicators
        # community_details.json indicates successful community generation
        community_json = timestamp_dir / "community_details.json"
        # view_graph.ipynb indicates successful output organization
        notebook = timestamp_dir / "view_graph.ipynb"
        # interactive_graph.html indicates successful visualization
        graph_html = timestamp_dir / "interactive_graph.html"
        
        # Consider successful if we have core files AND at least one definitive indicator
        if community_json.exists() or notebook.exists() or graph_html.exists():
            return True
        
        # If we have core files and communities parquet, that's also a good sign
        if graphs_dir.exists() and (graphs_dir / "communities.parquet").exists():
            return True
    
    return False


def process_single_query(
    project_root: Path,
    query: str,
    batch_name: str,
    searxng_url: str,
    verbose: bool = False,
    skip_web_fetch: bool = True
) -> Tuple[bool, Optional[str], Optional[Path]]:
    """
    Process a single query through the pipeline.
    
    Args:
        project_root: Root directory of the project
        query: Query string to process
        batch_name: Batch name for output organization
        searxng_url: SearXNG URL (not used if skip_web_fetch=True)
        verbose: Enable verbose output
        skip_web_fetch: Skip web fetching (assume files exist)
    
    Returns:
        Tuple of (success: bool, error_message: Optional[str], output_dir: Optional[Path])
    """
    try:
        # For batch processing, input is in input/{batch_name}/keywords/{query}/
        # But utils.get_input_directory expects input/keywords/{query}/
        # So we need to handle this specially
        if batch_name:
            input_dir = project_root / "input" / batch_name / "keywords" / query
        else:
            input_dir = project_root / "input" / "keywords" / query
        
        if not input_dir.exists():
            return False, f"Input directory does not exist: {input_dir}", None
        
        # Check if input directory has files
        if not any(input_dir.iterdir()):
            return False, f"Input directory is empty: {input_dir}", None
        
        # Create output directory: output/{batch_name}/{query}/{timestamp}/
        output_base = project_root / "output" / batch_name / query
        output_base.mkdir(parents=True, exist_ok=True)
        timestamp = get_timestamp()
        run_dir = output_base / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "graphs").mkdir(exist_ok=True)
        
        # Step 1: Convert HTML files to text
        try:
            created_files = convert_html_files_in_directory(input_dir)
        except Exception as e:
            # Continue anyway - text files might already exist
            pass
        
        # Step 2: Run GraphRAG indexing
        temp_graphrag_output = run_dir / "temp_graphrag_output"
        temp_graphrag_output.mkdir(exist_ok=True)
        
        indexing_result = run_graphrag_indexing(
            project_root=project_root,
            input_dir=input_dir,
            output_dir=temp_graphrag_output,
            logs_dir=run_dir / "logs",
            verbose=verbose
        )
        
        if not indexing_result['core_files_exist']:
            errors = indexing_result.get('errors', [])
            if errors:
                error_msg = ', '.join(errors)
            else:
                # If no specific errors but core files don't exist, check return code
                return_code = indexing_result.get('return_code', -1)
                if return_code != 0:
                    error_msg = f"Indexing process exited with code {return_code}. Check logs for details."
                else:
                    error_msg = "Core parquet files were not generated. Check logs for details."
            return False, f"Indexing failed: {error_msg}", run_dir
        
        # Step 3: Load graph data and organize files
        graphs_dir = run_dir / "graphs"
        graphs_dir.mkdir(exist_ok=True)
        
        # Move parquet files to graphs/
        parquet_files = list(temp_graphrag_output.glob("*.parquet"))
        
        # Handle fallback locations (same logic as main.py)
        if not parquet_files:
            query_parts = query.split()
            if query_parts:
                fallback_dir = project_root / "output" / query_parts[0]
                if fallback_dir.exists():
                    fallback_parquet = list(fallback_dir.glob("*.parquet"))
                    if fallback_parquet:
                        for parquet_file in fallback_parquet:
                            dest = graphs_dir / parquet_file.name
                            shutil.move(parquet_file, dest)
                        parquet_files = fallback_parquet
        
        # Move parquet files from temp location
        for parquet_file in parquet_files:
            if parquet_file.exists():
                dest = graphs_dir / parquet_file.name
                shutil.move(parquet_file, dest)
        
        # Move JSON files
        for json_file in temp_graphrag_output.glob("*.json"):
            if json_file.name in ["context.json", "stats.json"]:
                dest = run_dir / json_file.name
                if not dest.exists():
                    shutil.move(json_file, dest)
        
        # Move lancedb directory
        lancedb_dir = temp_graphrag_output / "lancedb"
        if lancedb_dir.exists():
            dest_lancedb = run_dir / "lancedb"
            if dest_lancedb.exists():
                shutil.rmtree(dest_lancedb)
            shutil.move(lancedb_dir, dest_lancedb)
        
        # Load graph data
        graph_data = load_graph_data(graphs_dir)
        
        # Clean up temp directory
        try:
            if temp_graphrag_output.exists():
                shutil.rmtree(temp_graphrag_output)
        except:
            pass
        
        # Step 4: Load metrics
        metrics = load_api_metrics(run_dir / "logs", timestamp=timestamp)
        
        # Step 5: Organize outputs
        organize_outputs(
            run_dir=run_dir,
            temp_output_dir=graphs_dir,
            temp_logs_dir=run_dir / "logs",
            graph_data=graph_data,
            metrics=metrics,
            errors=indexing_result.get('errors', []),
            timestamp=timestamp
        )
        
        return True, None, run_dir
        
    except Exception as e:
        import traceback
        error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg, None


def process_batch(
    project_root: Path,
    jsonl_path: Path,
    batch_name: str,
    searxng_url: str = "http://localhost:8080",
    verbose: bool = False,
    resume: bool = True,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    max_workers: int = 4
) -> Dict:
    """
    Process a batch of queries from a JSONL file.
    
    Args:
        project_root: Root directory of the project
        jsonl_path: Path to JSONL file with query entries
        batch_name: Name for batch (used in output path: output/{batch_name}/)
        searxng_url: SearXNG URL (not used if files already exist)
        verbose: Enable verbose output
        resume: Skip queries that already have successful outputs
        initial_delay: Initial throttling delay in seconds
        max_delay: Maximum throttling delay in seconds
        max_workers: Maximum number of parallel workers (default: 4)
    
    Returns:
        Dictionary with processing statistics
    """
    # Read JSONL file
    queries = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                query = entry.get('query')
                if query:
                    queries.append(entry)
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping invalid JSON line: {e}")
                continue
    
    total_queries = len(queries)
    print(f"📋 Loaded {total_queries} queries from {jsonl_path}")
    print()
    
    # Create progress log file
    output_base = project_root / "output" / batch_name
    output_base.mkdir(parents=True, exist_ok=True)
    progress_log_path = output_base / "batch_progress.log"
    
    # Initialize throttler
    throttler = AdaptiveThrottler(initial_delay=initial_delay, max_delay=max_delay)
    
    # Statistics
    stats = {
        'total': total_queries,
        'processed': 0,
        'skipped': 0,
        'succeeded': 0,
        'failed': 0,
        'start_time': datetime.now(),
        'errors': []
    }
    
    # Write header to progress log
    with open(progress_log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Batch Processing Log\n")
        log_file.write(f"{'='*80}\n")
        log_file.write(f"Batch Name: {batch_name}\n")
        log_file.write(f"JSONL File: {jsonl_path}\n")
        log_file.write(f"Total Queries: {total_queries}\n")
        log_file.write(f"Start Time: {stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'='*80}\n\n")
    
    print("=" * 80)
    print(f"Starting batch processing: {batch_name}")
    print(f"Parallel workers: {max_workers}")
    print("=" * 80)
    print()
    
    # Thread-safe progress tracking
    progress_lock = threading.Lock()
    
    def process_query_with_tracking(entry: Dict, idx: int) -> Tuple[int, bool, Optional[str], Optional[Path], float, bool]:
        """
        Process a single query and return results with timing.
        
        Returns:
            Tuple of (idx, success, error_msg, output_dir, query_minutes, was_skipped)
        """
        query = entry.get('query', '')
        display_query = entry.get('display_query', query)
        
        query_start_time = datetime.now()
        
        # Check if should skip (resume mode)
        # Note: We check BEFORE processing, so if outputs exist from a previous run, skip
        # Use a lock to prevent race conditions in parallel mode
        should_skip = False
        if resume:
            with progress_lock:
                should_skip = check_output_exists(project_root, query, batch_name)
                if should_skip:
                    stats['skipped'] += 1
                    with open(progress_log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"[{idx}/{total_queries}] SKIPPED: {query}\n")
                        log_file.write(f"  Reason: Output already exists\n")
                        log_file.write(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        log_file.flush()  # Ensure it's written immediately
                    return idx, True, None, None, 0.0, True  # was_skipped=True, True  # was_skipped=True
        
        # Process query
        success, error_msg, output_dir = process_single_query(
            project_root=project_root,
            query=query,
            batch_name=batch_name,
            searxng_url=searxng_url,
            verbose=verbose,
            skip_web_fetch=True
        )
        
        query_duration = (datetime.now() - query_start_time).total_seconds()
        query_minutes = query_duration / 60
        
        return idx, success, error_msg, output_dir, query_minutes, False  # was_skipped=False
    
    # Process queries in parallel
    if max_workers > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(process_query_with_tracking, entry, idx): idx
                for idx, entry in enumerate(queries, 1)
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_idx):
                idx, success, error_msg, output_dir, query_minutes, was_skipped = future.result()
                entry = queries[idx - 1]
                query = entry.get('query', '')
                display_query = entry.get('display_query', query)
                
                with progress_lock:
                    if was_skipped:
                        # Skipped - already logged in process_query_with_tracking
                        print(f"[{idx}/{total_queries}] ⏭️  Skipped: {display_query}")
                        continue
                    
                    stats['processed'] += 1
                    
                    if success:
                        stats['succeeded'] += 1
                        throttler.on_success()
                        print(f"[{idx}/{total_queries}] ✅ Success ({query_minutes:.1f} min): {display_query}")
                        with open(progress_log_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"[{idx}/{total_queries}] SUCCESS: {query}\n")
                            log_file.write(f"  Output: {output_dir}\n")
                            log_file.write(f"  Duration: {query_minutes:.2f} minutes\n")
                            log_file.write(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                            log_file.flush()  # Ensure it's written immediately
                    else:
                        stats['failed'] += 1
                        is_rate_limit = is_rate_limit_error(error_msg or '')
                        is_daily_limit = is_daily_token_limit_error(error_msg or '')
                        throttler.on_error(is_rate_limit=is_rate_limit, is_daily_limit=is_daily_limit)
                        
                        error_info = {
                            'query': query,
                            'error': error_msg,
                            'is_rate_limit': is_rate_limit,
                            'is_daily_limit': is_daily_limit,
                            'duration': query_minutes
                        }
                        stats['errors'].append(error_info)
                        
                        print(f"[{idx}/{total_queries}] ❌ Failed ({query_minutes:.1f} min): {display_query}")
                        if is_daily_limit:
                            print(f"   ⚠️  ⚠️  DAILY TOKEN LIMIT HIT - AWS Bedrock quota exceeded!")
                            print(f"   ⚠️  Processing will be very slow. Consider pausing until quota resets.")
                        elif is_rate_limit:
                            print(f"   ⚠️  Rate limit detected - increasing delay to {throttler.current_delay:.1f}s")
                        
                        with open(progress_log_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"[{idx}/{total_queries}] FAILED: {query}\n")
                            log_file.write(f"  Error: {error_msg}\n")
                            log_file.write(f"  Duration: {query_minutes:.2f} minutes\n")
                            log_file.write(f"  Rate Limit: {is_rate_limit}\n")
                            log_file.write(f"  Daily Token Limit: {is_daily_limit}\n")
                            log_file.write(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                            log_file.flush()  # Ensure it's written immediately
                    
                    # Print progress summary
                    remaining = total_queries - stats['processed'] - stats['skipped']
                    completed = stats['processed']
                    if completed > 0:
                        elapsed = (datetime.now() - stats['start_time']).total_seconds()
                        avg_time = elapsed / completed
                        eta_seconds = avg_time * remaining
                        eta_minutes = eta_seconds / 60
                        eta_hours = eta_minutes / 60
                        
                        if eta_hours >= 1:
                            eta_str = f"{eta_hours:.1f} hours ({eta_minutes:.0f} min)"
                        else:
                            eta_str = f"{eta_minutes:.1f} min"
                        
                        rate_per_hour = (completed / elapsed) * 3600 if elapsed > 0 else 0
                        
                        print(f"   Progress: {stats['processed'] + stats['skipped']}/{total_queries} | "
                              f"Succeeded: {stats['succeeded']} | "
                              f"Failed: {stats['failed']} | "
                              f"Skipped: {stats['skipped']} | "
                              f"Rate: {rate_per_hour:.1f}/hr | "
                              f"ETA: {eta_str}")
                    print()
    else:
        # Sequential processing (original code)
        for idx, entry in enumerate(queries, 1):
            query = entry.get('query', '')
            display_query = entry.get('display_query', query)
            
            query_start_time = datetime.now()
            print(f"[{idx}/{total_queries}] Processing: {display_query}")
            
            # Check if should skip (resume mode)
            if resume and check_output_exists(project_root, query, batch_name):
                print(f"   ⏭️  Skipping (output already exists)")
                stats['skipped'] += 1
                with open(progress_log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"[{idx}/{total_queries}] SKIPPED: {query}\n")
                    log_file.write(f"  Reason: Output already exists\n\n")
                continue
            
            # Throttle before processing (only if we've had recent errors)
            if throttler.current_delay > throttler.initial_delay:
                throttler.wait()
            
            # Process query
            success, error_msg, output_dir = process_single_query(
                project_root=project_root,
                query=query,
                batch_name=batch_name,
                searxng_url=searxng_url,
                verbose=verbose,
                skip_web_fetch=True
            )
            
            query_duration = (datetime.now() - query_start_time).total_seconds()
            query_minutes = query_duration / 60
            
            stats['processed'] += 1
            
            # Update throttler based on result
            if success:
                stats['succeeded'] += 1
                throttler.on_success()
                print(f"   ✅ Success ({query_minutes:.1f} min): {output_dir}")
                with open(progress_log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"[{idx}/{total_queries}] SUCCESS: {query}\n")
                    log_file.write(f"  Output: {output_dir}\n")
                    log_file.write(f"  Duration: {query_minutes:.2f} minutes\n")
                    log_file.write(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            else:
                stats['failed'] += 1
                is_rate_limit = is_rate_limit_error(error_msg or '')
                is_daily_limit = is_daily_token_limit_error(error_msg or '')
                throttler.on_error(is_rate_limit=is_rate_limit, is_daily_limit=is_daily_limit)
                
                error_info = {
                    'query': query,
                    'error': error_msg,
                    'is_rate_limit': is_rate_limit,
                    'is_daily_limit': is_daily_limit,
                    'duration': query_minutes
                }
                stats['errors'].append(error_info)
                
                print(f"   ❌ Failed ({query_minutes:.1f} min): {error_msg[:100]}...")
                if is_daily_limit:
                    print(f"   ⚠️  ⚠️  DAILY TOKEN LIMIT HIT - AWS Bedrock quota exceeded!")
                    print(f"   ⚠️  Processing will be very slow. Consider pausing until quota resets.")
                elif is_rate_limit:
                    print(f"   ⚠️  Rate limit detected - increasing delay to {throttler.current_delay:.1f}s")
                
                with open(progress_log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"[{idx}/{total_queries}] FAILED: {query}\n")
                    log_file.write(f"  Error: {error_msg}\n")
                    log_file.write(f"  Duration: {query_minutes:.2f} minutes\n")
                    log_file.write(f"  Rate Limit: {is_rate_limit}\n")
                    log_file.write(f"  Daily Token Limit: {is_daily_limit}\n")
                    log_file.write(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Print progress summary
            remaining = total_queries - idx
            completed = idx - stats['skipped']
            if completed > 0:
                elapsed = (datetime.now() - stats['start_time']).total_seconds()
                avg_time = elapsed / completed
                eta_seconds = avg_time * remaining
                eta_minutes = eta_seconds / 60
                eta_hours = eta_minutes / 60
                
                if eta_hours >= 1:
                    eta_str = f"{eta_hours:.1f} hours ({eta_minutes:.0f} min)"
                else:
                    eta_str = f"{eta_minutes:.1f} min"
                
                rate_per_hour = (completed / elapsed) * 3600 if elapsed > 0 else 0
                
                print(f"   Progress: {idx}/{total_queries} | "
                      f"Succeeded: {stats['succeeded']} | "
                      f"Failed: {stats['failed']} | "
                      f"Skipped: {stats['skipped']} | "
                      f"Rate: {rate_per_hour:.1f}/hr | "
                      f"ETA: {eta_str}")
            print()
    
    # Write summary to progress log
    stats['end_time'] = datetime.now()
    duration = (stats['end_time'] - stats['start_time']).total_seconds()
    
    with open(progress_log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"{'='*80}\n")
        log_file.write(f"Batch Processing Summary\n")
        log_file.write(f"{'='*80}\n")
        log_file.write(f"Total Queries: {stats['total']}\n")
        log_file.write(f"Processed: {stats['processed']}\n")
        log_file.write(f"Skipped: {stats['skipped']}\n")
        log_file.write(f"Succeeded: {stats['succeeded']}\n")
        log_file.write(f"Failed: {stats['failed']}\n")
        log_file.write(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)\n")
        log_file.write(f"End Time: {stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if stats['errors']:
            log_file.write(f"\nErrors:\n")
            for error_info in stats['errors']:
                log_file.write(f"  - {error_info['query']}: {error_info['error'][:200]}\n")
    
    print("=" * 80)
    print("Batch Processing Complete")
    print("=" * 80)
    print(f"Total: {stats['total']} | "
          f"Processed: {stats['processed']} | "
          f"Skipped: {stats['skipped']} | "
          f"Succeeded: {stats['succeeded']} | "
          f"Failed: {stats['failed']}")
    print(f"Progress log: {progress_log_path}")
    print("=" * 80)
    
    return stats
