#!/usr/bin/env python3
"""
iAgentBench Pipeline CLI

Main entry point for running the GraphRAG indexing pipeline with organized output.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_input_directory, create_output_directory, get_timestamp
from src.html_converter import convert_html_files_in_directory
from src.indexing import run_graphrag_indexing
from src.metrics import load_api_metrics
from src.output_manager import load_graph_data, organize_outputs
from src.web_fetcher import fetch_keyword_pages
from src.batch_processor import process_batch


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='iAgentBench Pipeline CLI - Process documents and generate knowledge graphs'
    )
    parser.add_argument(
        'keyword',
        nargs='?',
        default=None,
        help='Keyword for input/output organization (default: "local")'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--project-root',
        type=str,
        default=None,
        help='Path to project root (default: current directory)'
    )
    parser.add_argument(
        '--searxng-url',
        type=str,
        default='http://localhost:8080',
        help='SearXNG instance URL (default: http://localhost:8080)'
    )
    parser.add_argument(
        '--batch',
        type=str,
        default=None,
        help='Path to JSONL file for batch processing. Each line should contain a JSON object with a "query" field.'
    )
    parser.add_argument(
        '--batch-name',
        type=str,
        default=None,
        help='Name for batch (used in output path: output/{batch_name}/). Defaults to parent directory of JSONL file.'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable resume functionality (reprocess queries even if output exists)'
    )
    parser.add_argument(
        '--initial-delay',
        type=float,
        default=1.0,
        help='Initial throttling delay in seconds (default: 1.0, only used after errors)'
    )
    parser.add_argument(
        '--max-delay',
        type=float,
        default=60.0,
        help='Maximum throttling delay in seconds (default: 60.0)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers for batch processing (default: 4, set to 1 for sequential)'
    )
    
    args = parser.parse_args()
    
    # Determine project root
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        project_root = Path(__file__).parent.parent.resolve()
    
    # Handle batch processing mode
    if args.batch:
        jsonl_path = Path(args.batch).resolve()
        if not jsonl_path.exists():
            print(f"❌ JSONL file not found: {jsonl_path}")
            return 1
        
        # Determine batch name
        if args.batch_name:
            batch_name = args.batch_name
        else:
            # Try to infer from parent directory (e.g., input/2025_seeds/ -> 2025_seeds)
            parent = jsonl_path.parent.name
            if parent in ['input', 'data']:
                # Go up one more level
                batch_name = jsonl_path.parent.parent.name
            else:
                batch_name = parent
        
        print("=" * 60)
        print("iAgentBench Pipeline CLI - Batch Mode")
        print("=" * 60)
        print(f"Project Root: {project_root}")
        print(f"JSONL File: {jsonl_path}")
        print(f"Batch Name: {batch_name}")
        print("=" * 60)
        print()
        
        # Process batch
        stats = process_batch(
            project_root=project_root,
            jsonl_path=jsonl_path,
            batch_name=batch_name,
            searxng_url=args.searxng_url,
            verbose=args.verbose,
            resume=not args.no_resume,
            initial_delay=args.initial_delay,
            max_delay=args.max_delay,
            max_workers=args.max_workers
        )
        
        return 0 if stats['failed'] == 0 else 1
    
    # Use keyword or default to "local"
    keyword = args.keyword or "local"
    
    print("=" * 60)
    print("iAgentBench Pipeline CLI")
    print("=" * 60)
    print(f"Project Root: {project_root}")
    print(f"Keyword: {keyword}")
    print("=" * 60)
    print()
    
    # Create output directory structure
    run_dir = create_output_directory(project_root, keyword)
    timestamp = get_timestamp()
    
    print(f"📁 Output Directory: {run_dir}")
    print()
    
    # Step 0: Fetch web pages (if keyword provided and not "local")
    if keyword and keyword.lower() != "local":
        print("Step 0: Fetching web pages...")
        try:
            # Check if keyword folder already exists and has files
            keyword_input_dir = get_input_directory(project_root, keyword)
            existing_files = list(keyword_input_dir.glob("*.html")) if keyword_input_dir.exists() else []
            
            if existing_files:
                print(f"   ℹ️  Keyword folder already exists with {len(existing_files)} HTML file(s)")
                print(f"   ✓ Skipping web fetch, using existing files in: {keyword_input_dir}")
            else:
                # Fetch pages using SearXNG
                keyword_input_dir.mkdir(parents=True, exist_ok=True)
                downloaded_files = fetch_keyword_pages(
                    keyword=keyword,
                    output_dir=keyword_input_dir,
                    searxng_url=args.searxng_url,
                    max_pages=10
                )
                print(f"   ✓ Fetched and saved {len(downloaded_files)} HTML page(s) to: {keyword_input_dir}")
        except Exception as e:
            print(f"   ❌ Error fetching web pages: {e}")
            return 1
        
        print()
    
    # Get input directory (after potentially fetching files)
    input_dir = get_input_directory(project_root, keyword)
    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        if keyword and keyword.lower() != "local":
            print(f"   Web fetching may have failed. Check errors above.")
        else:
            print(f"   Please create the directory and add input files.")
        return 1
    
    # Check if input directory has any files
    has_files = any(input_dir.iterdir())
    if not has_files:
        print(f"❌ Input directory is empty: {input_dir}")
        if keyword and keyword.lower() != "local":
            print(f"   Web fetching may have failed. Check errors above.")
        else:
            print(f"   Please add input files to the directory.")
        return 1
    
    print(f"📁 Input Directory: {input_dir}")
    print()
    
    # Step 1: Convert HTML files to text
    print("Step 1: Converting HTML files to text...")
    try:
        created_files = convert_html_files_in_directory(input_dir)
        if created_files:
            print(f"   ✓ Converted {len(created_files)} HTML file(s) to text")
        else:
            print("   ℹ️  No HTML files found (assuming text files already exist)")
    except Exception as e:
        print(f"   ⚠️  Error converting HTML files: {e}")
        # Continue anyway - text files might already exist
    
    print()
    
    # Step 2: Run GraphRAG indexing
    print("Step 2: Running knowledge graph indexing...")
    
    # Create graphs subdirectory for parquet files
    graphs_dir = run_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    # GraphRAG will write directly to graphs_dir (parquet files go there)
    # Other files (context.json, stats.json) will be in run_dir root
    # We'll use a temporary location for GraphRAG output, then organize
    temp_graphrag_output = run_dir / "temp_graphrag_output"
    temp_graphrag_output.mkdir(exist_ok=True)
    
    indexing_result = run_graphrag_indexing(
        project_root=project_root,
        input_dir=input_dir,
        output_dir=temp_graphrag_output,  # Write to temp location first
        logs_dir=run_dir / "logs",
        verbose=args.verbose
    )
    
    if indexing_result['success']:
        print("   ✓ Indexing completed successfully")
        if indexing_result['reports_exist']:
            print("   ✓ Community reports generated")
    elif indexing_result['core_files_exist']:
        print("   ⚠️  Indexing completed with warnings")
        print("   ✓ Core graph data generated")
        if indexing_result['reports_exist']:
            print("   ✓ Community reports generated")
        else:
            print("   ⚠️  Community reports not generated")
    else:
        print("   ❌ Indexing failed")
        if indexing_result['errors']:
            for error in indexing_result['errors']:
                print(f"      - {error}")
        return 1
    
    print()
    
    # Step 3: Load graph data and organize files
    print("Step 3: Loading graph data...")
    try:
        # Organize files from temp_graphrag_output to final locations
        graphs_dir = run_dir / "graphs"
        graphs_dir.mkdir(exist_ok=True)
        
        # Move parquet files to graphs/
        parquet_files = list(temp_graphrag_output.glob("*.parquet"))
        
        # If no parquet files in temp location, check if GraphRAG wrote to project_root/output/{keyword}
        # This can happen if --output argument wasn't properly applied
        if not parquet_files:
            # Check if files were written to project_root/output/{first_word_of_keyword}
            keyword_parts = keyword.split()
            if keyword_parts:
                # GraphRAG might have written to output/{first_word} if path was split
                fallback_dir = project_root / "output" / keyword_parts[0]
                if fallback_dir.exists():
                    fallback_parquet = list(fallback_dir.glob("*.parquet"))
                    if fallback_parquet:
                        print(f"   ⚠️  Found parquet files in fallback location: {fallback_dir}")
                        print(f"   📦 Moving {len(fallback_parquet)} file(s) to correct location...")
                        for parquet_file in fallback_parquet:
                            dest = graphs_dir / parquet_file.name
                            import shutil
                            shutil.move(parquet_file, dest)
                        parquet_files = fallback_parquet
        
        # Move parquet files from temp location
        for parquet_file in parquet_files:
            if parquet_file.exists():  # Check if still exists (might have been moved already)
                dest = graphs_dir / parquet_file.name
                import shutil
                shutil.move(parquet_file, dest)
        
        # Also check fallback location for JSON files
        if not parquet_files or len(parquet_files) == 0:
            keyword_parts = keyword.split()
            if keyword_parts:
                fallback_dir = project_root / "output" / keyword_parts[0]
                if fallback_dir.exists():
                    # Move JSON files from fallback location
                    for json_file in fallback_dir.glob("*.json"):
                        if json_file.name in ["context.json", "stats.json"]:
                            dest = run_dir / json_file.name
                            import shutil
                            if not dest.exists():
                                shutil.move(json_file, dest)
        
        # Move other useful files (context.json, stats.json) to run_dir root
        for json_file in temp_graphrag_output.glob("*.json"):
            if json_file.name in ["context.json", "stats.json"]:
                dest = run_dir / json_file.name
                import shutil
                if not dest.exists():
                    shutil.move(json_file, dest)
        
        # Move lancedb directory if it exists (keep it in run_dir root)
        lancedb_dir = temp_graphrag_output / "lancedb"
        if lancedb_dir.exists():
            dest_lancedb = run_dir / "lancedb"
            if dest_lancedb.exists():
                import shutil
                shutil.rmtree(dest_lancedb)
            import shutil
            shutil.move(lancedb_dir, dest_lancedb)
        
        # Also check fallback location for lancedb
        keyword_parts = keyword.split()
        if keyword_parts:
            fallback_dir = project_root / "output" / keyword_parts[0]
            if fallback_dir.exists():
                fallback_lancedb = fallback_dir / "lancedb"
                if fallback_lancedb.exists() and not (run_dir / "lancedb").exists():
                    dest_lancedb = run_dir / "lancedb"
                    import shutil
                    shutil.move(fallback_lancedb, dest_lancedb)
        
        # Load graph data from graphs/ directory
        graph_data = load_graph_data(graphs_dir)
        
        print(f"   ✓ Loaded {len(graph_data)} data file(s)")
        if parquet_files:
            print(f"   ✓ Organized {len(parquet_files)} parquet file(s) to graphs/")
        
        # Clean up temp directory
        try:
            import shutil
            if temp_graphrag_output.exists():
                shutil.rmtree(temp_graphrag_output)
        except:
            pass  # Ignore cleanup errors
    except Exception as e:
        print(f"   ❌ Error loading graph data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # Step 4: Load metrics
    print("Step 4: Loading API metrics...")
    metrics = load_api_metrics(run_dir / "logs", timestamp=timestamp)
    if metrics:
        print(f"   ✓ Loaded metrics from log file")
        print(f"      Chat calls: {metrics['chat_calls']}")
        print(f"      Embedding calls: {metrics['embedding_calls']}")
        print(f"      Total tokens: {metrics['total_tokens']:,}")
    else:
        print("   ⚠️  No metrics found (log files may not exist yet)")
    
    print()
    
    # Step 5: Organize outputs
    print("Step 5: Organizing outputs...")
    try:
        saved_files = organize_outputs(
            run_dir=run_dir,
            temp_output_dir=graphs_dir,  # Parquet files are in graphs/
            temp_logs_dir=run_dir / "logs",  # Logs are in run_dir/logs
            graph_data=graph_data,
            metrics=metrics,
            errors=indexing_result['errors'],
            timestamp=timestamp
        )
        
        print("   ✓ Saved outputs:")
        print(f"      - Parquet files: {len(saved_files.get('parquet_files', []))} files")
        if saved_files.get('comprehensive_log'):
            print(f"      - Comprehensive log: {saved_files['comprehensive_log']}")
        if saved_files.get('log_file'):
            print(f"      - API calls log: {saved_files['log_file']}")
        if saved_files.get('community_json'):
            print(f"      - Community details: {saved_files['community_json']}")
        if saved_files.get('metrics'):
            print(f"      - Operational metrics: {saved_files['metrics']}")
        if saved_files.get('graph_html'):
            print(f"      - Interactive graph (HTML): {saved_files['graph_html']}")
            if saved_files.get('graphml'):
                print(f"      - GraphML file (yEd/yFiles): {saved_files['graphml']}")
        else:
            print(f"      - Interactive graph: Failed to generate (check errors)")
        if saved_files.get('notebook'):
            print(f"      - Viewer notebook: {saved_files['notebook']}")
        
    except Exception as e:
        print(f"   ❌ Error organizing outputs: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    print("=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)
    print(f"📁 All outputs saved to: {run_dir}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
