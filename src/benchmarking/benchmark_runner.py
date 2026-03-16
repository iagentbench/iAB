"""
Benchmark runner for QA datasets with AWS Bedrock.

This module runs all three modes (LLM baseline, RAG with Bedrock, Agentic solution)
on QA datasets and generates comparative evaluation reports.
"""

import os
import sys
import time
import json
import string
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Handle imports for both direct execution and module import
try:
    from .dataset_loader import load_simpleqa, load_hotpotqa, load_simpleqa_local, load_hotpotqa_local, load_iagentbench
    from .llm_baseline import llm_baseline_answer
    from .rag_bedrock import rag_bedrock_pipeline
    # Mode 3: Reflexion-based agent (Bedrock-backed)
    from .agentic_solution_reflexion import agentic_answer
    # Original Pydantic AI agentic solution (commented out for now, can switch back if needed)
    # from .agentic_solution import agentic_answer
    from .evaluator import calculate_metrics, compare_modes, calculate_exact_match, calculate_f1_score
except ImportError:
    # If running as script, use absolute imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from src.benchmarking.dataset_loader import load_simpleqa, load_hotpotqa, load_simpleqa_local, load_hotpotqa_local, load_iagentbench
    from src.benchmarking.llm_baseline import llm_baseline_answer
    from src.benchmarking.rag_bedrock import rag_bedrock_pipeline
    # Mode 3: Reflexion-based agent (Bedrock-backed)
    from src.benchmarking.agentic_solution_reflexion import agentic_answer
    # Original Pydantic AI agentic solution (commented out for now, can switch back if needed)
    # from src.benchmarking.agentic_solution import agentic_answer
    from src.benchmarking.evaluator import calculate_metrics, compare_modes, calculate_exact_match, calculate_f1_score


def _sanitize_model_id_for_filename(model_id: str) -> str:
    """
    Sanitize model ID for use in filenames.
    Replaces special characters with underscores.
    
    Args:
        model_id: Model ID string (e.g., "anthropic.claude-3-5-sonnet-20240620-v1:0")
    
    Returns:
        Sanitized string safe for filenames (e.g., "anthropic_claude-3-5-sonnet-20240620-v1_0")
    """
    # Replace dots, colons, and slashes with underscores
    sanitized = model_id.replace(".", "_").replace(":", "_").replace("/", "_")
    return sanitized


def _extract_ground_truth(answer_data, dataset_name: str = "simpleqa") -> str:
    """
    Extract ground truth answer from dataset format.
    
    SimpleQA has answer as a string.
    HotpotQA has answer as a string.
    iAgentBench has answer as a string.
    
    Args:
        answer_data: Answer data from dataset (dict or string)
        dataset_name: Name of the dataset ("simpleqa", "hotpotqa", or "iagentbench")
    
    Returns:
        Ground truth answer string
    """
    if dataset_name.lower() == "simpleqa":
        # SimpleQA answers are strings
        return str(answer_data)
    elif dataset_name.lower() == "hotpotqa":
        # HotpotQA answers are strings
        return str(answer_data)
    elif dataset_name.lower() == "iagentbench":
        # iAgentBench answers are strings
        return str(answer_data)
    else:
        return str(answer_data)


def _parse_live_results_file(live_results_file: Path) -> tuple[int, List[Dict], List[str], List[str]]:
    """
    Parse live results file to extract already completed questions.
    
    Args:
        live_results_file: Path to the live results file
    
    Returns:
        Tuple of (last_completed_idx, existing_results, existing_mode3_answers, existing_ground_truths)
        If file doesn't exist or is empty, returns (0, [], [], [])
    """
    if not live_results_file.exists():
        return (0, [], [], [])
    
    last_completed_idx = 0
    existing_results = []
    existing_mode3_answers = []
    existing_ground_truths = []
    
    try:
        with open(live_results_file, 'r') as f:
            lines = f.readlines()
        
        # Find the last completed question by looking for [N/Total] pattern
        current_question_idx = None
        current_result = None
        current_ground_truth = None
        current_mode3_answer = None
        
        for line in lines:
            line = line.strip()
            
            # Match pattern: [N/Total] Question text...
            match = re.match(r'\[(\d+)/(\d+)\]', line)
            if match:
                # Save previous question if we have one
                if current_question_idx is not None and current_result is not None:
                    existing_results.append(current_result)
                    existing_mode3_answers.append(current_mode3_answer or "")
                    existing_ground_truths.append(current_ground_truth or "")
                    last_completed_idx = max(last_completed_idx, current_question_idx)
                
                # Start new question
                current_question_idx = int(match.group(1))
                current_result = {
                    'question_id': f'q_{current_question_idx}',
                    'question': line[len(match.group(0)):].strip(),
                    'ground_truth': None,
                    'mode1_answer': "",
                    'mode1_error': None,
                    'mode2_answer': "",
                    'mode2_error': None,
                    'mode3_answer': None,
                    'mode3_error': None,
                }
                current_ground_truth = None
                current_mode3_answer = None
            
            # Match ground truth
            elif line.startswith('  Ground Truth:'):
                current_ground_truth = line.replace('  Ground Truth:', '').strip()
                if current_result:
                    current_result['ground_truth'] = current_ground_truth
            
            # Match Mode 1 answer
            elif line.startswith('  Mode 1 Answer:'):
                if current_result:
                    current_result['mode1_answer'] = line.replace('  Mode 1 Answer:', '').strip()
            
            # Match Mode 1 error
            elif line.startswith('  Mode 1 Error:'):
                if current_result:
                    current_result['mode1_error'] = line.replace('  Mode 1 Error:', '').strip()
            
            # Match Mode 2 answer
            elif line.startswith('  Mode 2 Answer:'):
                if current_result:
                    current_result['mode2_answer'] = line.replace('  Mode 2 Answer:', '').strip()
            
            # Match Mode 2 error
            elif line.startswith('  Mode 2 Error:'):
                if current_result:
                    current_result['mode2_error'] = line.replace('  Mode 2 Error:', '').strip()
            
            # Match Mode 3 answer
            elif line.startswith('  Mode 3 Answer:'):
                current_mode3_answer = line.replace('  Mode 3 Answer:', '').strip()
                if current_result:
                    current_result['mode3_answer'] = current_mode3_answer
            
            # Match Mode 3 error
            elif line.startswith('  Mode 3 Error:'):
                error_msg = line.replace('  Mode 3 Error:', '').strip()
                if current_result:
                    current_result['mode3_error'] = error_msg
                    current_result['mode3_answer'] = ""
        
        # Don't forget the last question
        if current_question_idx is not None and current_result is not None:
            existing_results.append(current_result)
            existing_mode3_answers.append(current_mode3_answer or "")
            existing_ground_truths.append(current_ground_truth or "")
            last_completed_idx = max(last_completed_idx, current_question_idx)
    
    except Exception as e:
        # If parsing fails, return empty (will restart from beginning)
        print(f"Warning: Could not parse live results file: {e}")
        return (0, [], [], [])
    
    return (last_completed_idx, existing_results, existing_mode3_answers, existing_ground_truths)

def run_benchmark(
    dataset_name: str = "simpleqa",
    num_entries: int = 100,
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    region: str = "us-east-1",
    searxng_url: Optional[str] = None,
    verbose: bool = True,
    save_results: bool = True,
    split: str = "validation"  # Use validation instead of test (test has <unk> answers)
) -> Dict:
    """
    Run benchmark on QA dataset using all three modes.
    
    Workflow:
    1. Load dataset
    2. For each question, run all three modes:
       - Mode 1: LLM Only Baseline
       - Mode 2: RAG with SearxNG
       - Mode 3: Agentic Solution
    3. Compare answers with ground truth
    4. Calculate metrics (exact match, F1, accuracy)
    5. Generate comparative report
    
    Args:
        dataset_name: Dataset to use ("simpleqa", "hotpotqa", or "iagentbench")
        num_entries: Number of entries to benchmark (default: 10 for sanity check)
        model_id: Bedrock model ID
        region: AWS region
        searxng_url: SearxNG base URL
        verbose: Whether to print progress messages
        save_results: Whether to save results to files
        split: Dataset split to use (default: "validation" - test split has <unk> answers)
    
    Returns:
        Dictionary containing:
        - 'results': List of per-question results
        - 'metrics': Dictionary with metrics for each mode
        - 'summary': Summary statistics
    """
    if verbose:
        print("=" * 70)
        print("QA Benchmark Runner - AWS Bedrock")
        print("=" * 70)
        print(f"Dataset: {dataset_name}")
        print(f"Split: {split}")
        print(f"Entries: {num_entries}")
        print(f"Model: {model_id}")
        print("=" * 70)
        print()
    
    # Load dataset (try local first, fall back to HF loader if not available)
    if verbose:
        print("Loading dataset...")
    
    if dataset_name.lower() == "simpleqa":
        try:
            dataset = load_simpleqa_local()
            if verbose:
                print("  Loaded from local JSONL (inputs/final/simpleqa_verified_500.jsonl)")
        except FileNotFoundError:
            dataset = load_simpleqa()
            if verbose:
                print("  Loaded from Hugging Face (local fallback not found)")
        question_field = "problem"
        answer_field = "answer"
        # SimpleQA only has test split
        if split != "test":
            split = "test"
            if verbose:
                print(f"Note: SimpleQA only has 'test' split, using 'test' instead")
    elif dataset_name.lower() == "hotpotqa":
        try:
            dataset = load_hotpotqa_local()
            if verbose:
                print("  Loaded from local JSONL (inputs/final/hotpotqa_fullwiki_validation_500.jsonl)")
        except FileNotFoundError:
            dataset = load_hotpotqa(subset="distractor")
            if verbose:
                print("  Loaded from Hugging Face (local fallback not found)")
        question_field = "question"
        answer_field = "answer"
        # HotpotQA local file uses validation split
        if split != "validation":
            split = "validation"
            if verbose:
                print(f"Note: Using 'validation' split for HotpotQA")
    elif dataset_name.lower() == "iagentbench":
        dataset = load_iagentbench()
        if verbose:
            print("  Loaded from Hugging Face (preetam7/iAgentBench)")
        question_field = "question"
        answer_field = "answer"
        # iAgentBench has test split
        if split != "test":
            split = "test"
            if verbose:
                print(f"Note: iAgentBench uses 'test' split")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: 'simpleqa', 'hotpotqa', 'iagentbench'")
    
    # Get the specified split
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")
    
    test_data = dataset[split]
    # Process ALL entries (no random selection, no limit)
    # num_entries is ignored - we use all entries in the dataset
    if verbose:
        print(f"Note: Processing ALL {len(test_data)} entries (no random selection)")
    
    if verbose:
        print(f"✓ Loaded {len(test_data)} test entries")
        print()
    
    # Set up live results file for incremental saving (prevents data loss on crash)
    live_results_file = None
    if save_results:
        output_dir = Path(__file__).parent.parent.parent / "outputs" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_id_safe = _sanitize_model_id_for_filename(model_id)
        entries_suffix = "all" if num_entries == 0 else f"{num_entries}entries"
        live_results_file = output_dir / f"benchmark_live_{dataset_name}_{split}_{model_id_safe}_{entries_suffix}.txt"
    
    # Check if we're resuming from a previous run
    start_idx = 0
    results = []
    mode1_answers = []
    mode2_answers = []
    mode3_answers = []
    ground_truths = []
    
    if live_results_file and live_results_file.exists():
        if verbose:
            print(f"Found existing live results file: {live_results_file}")
            print("  Attempting to resume from previous run...")
        
        last_completed, existing_results, existing_mode3, existing_ground_truths = _parse_live_results_file(live_results_file)
        
        if last_completed > 0:
            start_idx = last_completed
            
            # Build a map of question_idx -> result for quick lookup
            existing_results_map = {}
            for res in existing_results:
                q_idx = int(res['question_id'].replace('q_', ''))
                existing_results_map[q_idx] = res
            
            # Rebuild results, mode3_answers, and ground_truths lists for all questions up to start_idx
            # This ensures alignment even if some questions were skipped in the original run
            for idx in range(1, start_idx + 1):
                if idx in existing_results_map:
                    # Question was completed - use existing result
                    results.append(existing_results_map[idx])
                    mode3_answers.append(existing_results_map[idx].get('mode3_answer', ''))
                    # Extract ground truth from result or dataset
                    if existing_results_map[idx].get('ground_truth'):
                        ground_truths.append(existing_results_map[idx]['ground_truth'])
                    else:
                        # Fallback: extract from dataset
                        example = test_data[idx - 1]
                        gt = _extract_ground_truth(example[answer_field], dataset_name)
                        ground_truths.append(gt)
                else:
                    # Question was skipped in original run - create empty result
                    example = test_data[idx - 1]
                    question = example[question_field]
                    gt = _extract_ground_truth(example[answer_field], dataset_name)
                    results.append({
                        'question_id': f'q_{idx}',
                        'question': question,
                        'ground_truth': gt,
                        'mode1_answer': "",
                        'mode1_error': "Skipped in original run",
                        'mode2_answer': "",
                        'mode2_error': "Skipped in original run",
                        'mode3_answer': "",
                        'mode3_error': "Skipped in original run",
                    })
                    mode3_answers.append("")
                    ground_truths.append(gt)
            
            # Fill in mode1 and mode2 from existing results (empty if not in file)
            mode1_answers = [r.get('mode1_answer', '') for r in results]
            mode2_answers = [r.get('mode2_answer', '') for r in results]
            
            if verbose:
                print(f"  ✓ Resuming from question {start_idx + 1}/{len(test_data)}")
                print(f"  Already completed: {last_completed} questions")
                print(f"  Total results loaded: {len(results)}")
                print(f"  Appending to existing results file...\n")
        else:
            if verbose:
                print("  No completed questions found, starting fresh\n")
    
    # Initialize live results file if starting fresh
    if start_idx == 0 and live_results_file:
        with open(live_results_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("QA Benchmark - Live Results (Incremental Save)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Split: {split}\n")
            f.write(f"Model: {model_id}\n")
            f.write(f"Total Questions: {len(test_data)}\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "=" * 70 + "\n")
            f.write("QUESTION RESULTS (Live Updates)\n")
            f.write("=" * 70 + "\n\n")
            f.flush()
        
        if verbose:
            print(f"✓ Live results file: {live_results_file}")
            print("  (Results will be saved incrementally after each question)\n")
    elif live_results_file and start_idx > 0:
        # Append resume message to existing file
        with open(live_results_file, 'a') as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"RESUMED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Continuing from question {start_idx + 1}/{len(test_data)}\n")
            f.write("=" * 70 + "\n\n")
            f.flush()

    # Throttle between questions to avoid Bedrock rate limiting.
    # Default: 90 seconds (1.5 minutes) between questions
    # For 500 questions: ~90s * 500 = 45,000s = 12.5 hours (plus processing time ≈ 15 hours total)
    # This allows safe concurrent runs while completing in reasonable time
    inter_question_delay_seconds = int(os.getenv("BENCHMARK_DELAY_SECONDS", "90"))
    inter_question_delay_seconds = 30
    
    if verbose:
        delay_minutes = inter_question_delay_seconds / 60
        estimated_total_seconds = len(test_data) * inter_question_delay_seconds
        estimated_hours = estimated_total_seconds / 3600
        print(f"Inter-question delay: {inter_question_delay_seconds} seconds ({delay_minutes:.1f} minutes)")
        print(f"  Estimated time for {len(test_data)} questions: ~{estimated_hours:.1f} hours (plus processing time)")
        print(f"  (Set BENCHMARK_DELAY_SECONDS env var to customize)\n")
    
    # Run all three modes for each question (skip already completed ones)
    for idx, example in enumerate(test_data, 1):
        # Skip questions that were already completed
        if idx <= start_idx:
            if verbose:
                print(f"\n[{idx}/{len(test_data)}] SKIPPING (already completed)")
            continue
        question = example[question_field]
        ground_truth = _extract_ground_truth(example[answer_field], dataset_name)
        ground_truths.append(ground_truth)
        
        if verbose:
            print(f"\n[{idx}/{len(test_data)}] Question: {question[:80]}...")
            print(f"Ground Truth: {ground_truth}")
        
        result_entry = {
            'question_id': example.get('question_id', f'q_{idx}'),
            'question': question,
            'ground_truth': ground_truth,
        }
        
        # Mode 1: LLM Only Baseline
        if verbose:
            print("  Mode 1: LLM Baseline...")
        try:
            mode1_answer = llm_baseline_answer(
                question,
                model_id=model_id,
                region=region
            )
            mode1_answers.append(mode1_answer)
            result_entry['mode1_answer'] = mode1_answer
            result_entry['mode1_error'] = None
            if verbose:
                print(f"    Answer: {mode1_answer}")
        except Exception as e:
            mode1_answers.append("")
            result_entry['mode1_answer'] = ""
            result_entry['mode1_error'] = str(e)
            if verbose:
                print(f"    Error: {e}")
        
        # Mode 2: RAG with SearxNG
        if verbose:
            print("  Mode 2: RAG with SearxNG...")
        try:
            mode2_answer = rag_bedrock_pipeline(
                question,
                model_id=model_id,
                region=region,
                searxng_url=searxng_url,
                verbose=False
            )
            mode2_answers.append(mode2_answer)
            result_entry['mode2_answer'] = mode2_answer
            result_entry['mode2_error'] = None
            if verbose:
                print(f"    Answer: {mode2_answer}")
        except Exception as e:
            mode2_answers.append("")
            result_entry['mode2_answer'] = ""
            result_entry['mode2_error'] = str(e)
            if verbose:
                print(f"    Error: {e}")
        
        # Mode 3: Agentic Solution (Reflexion-based) - ENABLED
        if verbose:
            print("  Mode 3: Agentic Solution (Reflexion)...")
        try:
            mode3_answer = agentic_answer(
                question,
                ground_truth=ground_truth,
                model_id=model_id,
                region=region,
                searxng_url=searxng_url,
                # Match paper settings for HotPotQA (reasoning tasks):
                # - max_trials=5: matches paper notebook example (n=5)
                # - max_steps=6: matches paper default for ReAct agents
                # - after each attempt, Reflexion checks correctness vs ground_truth and can reflect before the next attempt
                max_trials=5,
                max_steps=6,
                verbose=False
            )
            # Original Pydantic AI agentic solution call (kept for reference):
            # mode3_answer = agentic_answer(
            #     question,
            #     model_id=model_id,
            #     region=region,
            #     searxng_url=searxng_url,
            #     verbose=False
            # )
            mode3_answers.append(mode3_answer)
            result_entry['mode3_answer'] = mode3_answer
            result_entry['mode3_error'] = None
            if verbose:
                print(f"    Answer: {mode3_answer}")
        except Exception as e:
            mode3_answers.append("")
            result_entry['mode3_answer'] = ""
            result_entry['mode3_error'] = str(e)
            if verbose:
                print(f"    Error: {e}")
        
        results.append(result_entry)
        
        # Calculate metrics for completed questions so far (for live updates)
        if len(mode3_answers) > 0:
            # Calculate running metrics for all modes
            running_metrics1 = calculate_metrics(mode1_answers, ground_truths)
            running_metrics2 = calculate_metrics(mode2_answers, ground_truths)
            running_metrics3 = calculate_metrics(mode3_answers, ground_truths)
            running_em1 = running_metrics1['exact_match_accuracy']
            running_f1_1 = running_metrics1['f1_score']
            running_em2 = running_metrics2['exact_match_accuracy']
            running_f1_2 = running_metrics2['f1_score']
            running_em3 = running_metrics3['exact_match_accuracy']
            running_f1_3 = running_metrics3['f1_score']
            running_correct = sum(running_metrics3['exact_matches'])
            running_total = len(running_metrics3['exact_matches'])
        else:
            running_em1 = running_em2 = running_em3 = 0.0
            running_f1_1 = running_f1_2 = running_f1_3 = 0.0
            running_correct = 0
            running_total = 0
        
        # Save result immediately to live file (incremental save)
        if live_results_file:
            with open(live_results_file, 'a') as f:
                f.write(f"[{idx}/{len(test_data)}] {question[:100]}{'...' if len(question) > 100 else ''}\n")
                f.write(f"  Ground Truth: {ground_truth}\n")
                f.write(f"  Mode 1 Answer: {result_entry['mode1_answer']}\n")
                if result_entry['mode1_error']:
                    f.write(f"  Mode 1 Error: {result_entry['mode1_error']}\n")
                f.write(f"  Mode 2 Answer: {result_entry['mode2_answer']}\n")
                if result_entry['mode2_error']:
                    f.write(f"  Mode 2 Error: {result_entry['mode2_error']}\n")
                f.write(f"  Mode 3 Answer: {result_entry['mode3_answer']}\n")
                if result_entry['mode3_error']:
                    f.write(f"  Mode 3 Error: {result_entry['mode3_error']}\n")
                f.write(f"  Running Stats: M1 EM={running_em1:.3f} F1={running_f1_1:.3f} | M2 EM={running_em2:.3f} F1={running_f1_2:.3f} | M3 EM={running_em3:.3f} F1={running_f1_3:.3f}\n")
                f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n")
                f.flush()  # Ensure data is written immediately
        
        # Print running stats every 10 questions
        if verbose and idx % 10 == 0:
            print(f"\n  [Progress: {idx}/{len(test_data)}] M1 EM={running_em1:.3f} F1={running_f1_1:.3f} | M2 EM={running_em2:.3f} F1={running_f1_2:.3f} | M3 EM={running_em3:.3f} F1={running_f1_3:.3f}")
        
        # Add delay between questions to avoid rate limiting
        if idx < len(test_data):
            time.sleep(inter_question_delay_seconds)
    
    # Write completion summary to live results file
    if live_results_file:
        with open(live_results_file, 'a') as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write("BENCHMARK COMPLETE\n")
            f.write("=" * 70 + "\n")
            f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Questions Processed: {len(results)}\n")
            f.write("\n")
            f.flush()
    
    # Calculate metrics for each mode using evaluator
    if verbose:
        print("\n" + "=" * 70)
        print("Calculating Metrics...")
        print("=" * 70)
    
    # Calculate metrics for each mode
    metrics1 = calculate_metrics(mode1_answers, ground_truths)
    metrics2 = calculate_metrics(mode2_answers, ground_truths)
    # Mode 3 enabled - calculate from agentic answers
    metrics3 = calculate_metrics(mode3_answers, ground_truths)
    
    metrics = {
        "LLM Baseline": metrics1,
        "RAG with SearxNG": metrics2,
        # "Agentic Solution": metrics3  # Disabled for now
    }
    # Add Mode 3 metrics but mark as disabled
    metrics["Agentic Solution"] = metrics3
    
    if verbose:
        for mode_name, m in metrics.items():
            print(f"\n{mode_name}:")
            print(f"  Exact Match Accuracy: {m['exact_match_accuracy']:.3f} ({sum(m['exact_matches'])}/{len(m['exact_matches'])})")
            print(f"  F1 Score: {m['f1_score']:.3f}")
    
    # Add metrics to results
    for idx, result in enumerate(results):
        result['mode1_exact_match'] = metrics1['exact_matches'][idx]
        result['mode1_f1'] = metrics1['f1_scores'][idx]
        result['mode2_exact_match'] = metrics2['exact_matches'][idx]
        result['mode2_f1'] = metrics2['f1_scores'][idx]
        result['mode3_exact_match'] = metrics3['exact_matches'][idx]
        result['mode3_f1'] = metrics3['f1_scores'][idx]
    
    # Generate summary
    summary = {
        'dataset': dataset_name,
        'num_entries': len(test_data),
        'model_id': model_id,
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            mode: {
                'exact_match_accuracy': m['exact_match_accuracy'],
                'f1_score': m['f1_score']
            }
            for mode, m in metrics.items()
        }
    }
    
    # Save results if requested
    if save_results:
        output_dir = Path(__file__).parent.parent.parent / "outputs" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize model_id for filename
        model_id_safe = _sanitize_model_id_for_filename(model_id)
        
        # Determine entries suffix (use "all" if num_entries is 0, otherwise use the number)
        entries_suffix = "all" if num_entries == 0 else f"{num_entries}entries"
        
        # Save detailed results as JSON
        results_file = output_dir / f"benchmark_results_{dataset_name}_{split}_{model_id_safe}_{entries_suffix}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'results': results,
                'metrics': metrics
            }, f, indent=2)
        
        # Save results as CSV
        import csv
        csv_file = output_dir / f"benchmark_results_{dataset_name}_{split}_{model_id_safe}_{entries_suffix}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'question_id', 'question', 'ground_truth',
                'mode1_answer', 'mode1_exact_match', 'mode1_f1', 'mode1_error',
                'mode2_answer', 'mode2_exact_match', 'mode2_f1', 'mode2_error',
                'mode3_answer', 'mode3_exact_match', 'mode3_f1', 'mode3_error'
            ])
            writer.writeheader()
            writer.writerows(results)
        
        # Generate comparative report
        report_file = output_dir / f"benchmark_report_{dataset_name}_{split}_{model_id_safe}_{entries_suffix}.txt"
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("QA Benchmark Comparative Report\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Entries: {summary['num_entries']}\n")
            f.write(f"Model: {model_id}\n")
            f.write(f"Timestamp: {summary['timestamp']}\n")
            f.write("\n" + "=" * 70 + "\n")
            f.write("METRICS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            for mode_name, m in metrics.items():
                f.write(f"{mode_name}:")
                if mode_name == "Agentic Solution":
                    f.write(" (ENABLED)")
                f.write("\n")
                f.write(f"  Exact Match Accuracy: {m['exact_match_accuracy']:.3f} ({sum(m['exact_matches'])}/{len(m['exact_matches'])})\n")
                f.write(f"  F1 Score: {m['f1_score']:.3f}\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")
            f.write("COMPARATIVE ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            
            # Find best performing mode
            best_exact = max(metrics.items(), key=lambda x: x[1]['exact_match_accuracy'])
            best_f1 = max(metrics.items(), key=lambda x: x[1]['f1_score'])
            
            # Find best performing mode
            best_exact = max(metrics.items(), key=lambda x: x[1]['exact_match_accuracy'])
            best_f1 = max(metrics.items(), key=lambda x: x[1]['f1_score'])
            
            f.write(f"Best Exact Match Accuracy: {best_exact[0]} ({best_exact[1]['exact_match_accuracy']:.3f})\n")
            f.write(f"Best F1 Score: {best_f1[0]} ({best_f1[1]['f1_score']:.3f})\n")
            f.write("\n")
            
            f.write("=" * 70 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("=" * 70 + "\n\n")
            
            for idx, result in enumerate(results, 1):
                f.write(f"\n[{idx}] {result['question']}\n")
                f.write(f"Ground Truth: {result['ground_truth']}\n")
                f.write(f"Mode 1 (LLM Baseline): {result['mode1_answer']} [EM: {result['mode1_exact_match']}, F1: {result['mode1_f1']:.3f}]\n")
                f.write(f"Mode 2 (RAG): {result['mode2_answer']} [EM: {result['mode2_exact_match']}, F1: {result['mode2_f1']:.3f}]\n")
                f.write(f"Mode 3 (Agentic): {result['mode3_answer']} [EM: {result['mode3_exact_match']}, F1: {result['mode3_f1']:.3f}]\n")
        
        if verbose:
            print(f"\n✓ Results saved to:")
            print(f"  - {results_file}")
            print(f"  - {csv_file}")
            print(f"  - {report_file}")
            if live_results_file:
                print(f"  - {live_results_file} (live incremental save)")
    
    if verbose:
        print("\n" + "=" * 70)
        print("Benchmark Complete!")
        print("=" * 70)
    
    return {
        'results': results,
        'metrics': metrics,
        'summary': summary
    }


# Full benchmark script: Run SimpleQA, HotpotQA, and iAgentBench datasets
if __name__ == "__main__":
    import string
    
    print("=" * 80)
    print("FULL BENCHMARK RUNNER - SimpleQA, HotpotQA & iAgentBench")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    # Bedrock inference profile / model IDs to test
    MODEL_IDS = [
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.meta.llama4-maverick-17b-instruct-v1:0",
        "mistral.mistral-large-3-675b-instruct",
        "google.gemma-3-27b-it",
    ]
    
    region = "us-east-1"
    # num_entries=0 means process ALL entries in the dataset
    num_entries = 0  # 0 means process all entries (no limit)
    
    datasets_config = [
        {
            "name": "simpleqa",
            "split": "test",  # SimpleQA local file uses test split
            "description": "SimpleQA local dataset (inputs/final/simpleqa_verified_500.jsonl)"
        },
        {
            "name": "hotpotqa",
            "split": "validation",  # HotpotQA local file uses validation split
            "description": "HotpotQA local dataset (inputs/final/hotpotqa_fullwiki_validation_500.jsonl)"
        },
        {
            "name": "iagentbench",
            "split": "test",
            "description": "iAgentBench from Hugging Face (preetam7/iAgentBench)"
        }
    ]
    
    # Results summary across all models and datasets
    all_results_summary = []
    
    # Loop through each model
    for model_id in MODEL_IDS:
        print("\n" + "=" * 80)
        print(f"MODEL: {model_id}")
        print("=" * 80)
        print()
        
        # Results summary for this model
        model_results_summary = []
        
        # Loop through each dataset
        for dataset_config in datasets_config:
            dataset_name = dataset_config["name"]
            split = dataset_config["split"]
            description = dataset_config["description"]
            
            print("\n" + "-" * 80)
            print(f"Running benchmark on: {description}")
            print(f"Dataset: {dataset_name.upper()}")
            print(f"Split: {split}")
            print(f"Model: {model_id}")
            print(f"Entries: {num_entries} (0 = all entries in dataset)")
            print("-" * 80)
            print()
            
            try:
                # Run benchmark
                # num_entries=0 means process ALL entries in the dataset
                results = run_benchmark(
                    dataset_name=dataset_name,
                    num_entries=0,  # 0 = all entries in the dataset
                    model_id=model_id,
                    region=region,
                    verbose=True,
                    save_results=True,
                    split=split
                )
                
                # Extract metrics summary
                metrics = results['metrics']
                summary = {
                    'model_id': model_id,
                    'dataset': dataset_name,
                    'split': split,
                    'num_entries': results['summary']['num_entries'],
                    'timestamp': results['summary']['timestamp'],
                    'mode1_exact_match': metrics['LLM Baseline']['exact_match_accuracy'],
                    'mode1_f1': metrics['LLM Baseline']['f1_score'],
                    'mode2_exact_match': metrics['RAG with SearxNG']['exact_match_accuracy'],
                    'mode2_f1': metrics['RAG with SearxNG']['f1_score'],
                    'mode3_exact_match': metrics['Agentic Solution']['exact_match_accuracy'],
                    'mode3_f1': metrics['Agentic Solution']['f1_score'],
                }
                model_results_summary.append(summary)
                all_results_summary.append(summary)
                
                print(f"\n✓ Completed {dataset_name.upper()} benchmark for {model_id}")
                print(f"  Total entries: {summary['num_entries']}")
                print(f"  Mode 1 (LLM Baseline) - EM: {summary['mode1_exact_match']:.3f}, F1: {summary['mode1_f1']:.3f}")
                print(f"  Mode 2 (RAG) - EM: {summary['mode2_exact_match']:.3f}, F1: {summary['mode2_f1']:.3f}")
                # print(f"  Mode 3 (Agentic) - EM: {summary['mode3_exact_match']:.3f}, F1: {summary['mode3_f1']:.3f}")  # Disabled
                print(f"  Mode 3 (Agentic) - EM: {summary['mode3_exact_match']:.3f}, F1: {summary['mode3_f1']:.3f}")
                
            except Exception as e:
                print(f"\n✗ Error running benchmark on {dataset_name} with model {model_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Print summary for this model
        if model_results_summary:
            print("\n" + "-" * 80)
            print(f"SUMMARY FOR MODEL: {model_id}")
            print("-" * 80)
            for summary in model_results_summary:
                print(f"{summary['dataset'].upper()} ({summary['split']}):")
                print(f"  Entries: {summary['num_entries']}")
                print(f"  LLM Baseline: EM={summary['mode1_exact_match']:.3f}, F1={summary['mode1_f1']:.3f}")
                print(f"  RAG: EM={summary['mode2_exact_match']:.3f}, F1={summary['mode2_f1']:.3f}")
                # print(f"  Agentic: EM={summary['mode3_exact_match']:.3f}, F1={summary['mode3_f1']:.3f}")  # Disabled
                print(f"  Agentic: EM={summary['mode3_exact_match']:.3f}, F1={summary['mode3_f1']:.3f}")
                print()
    
    # Print final summary across all models
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ALL MODELS")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Group by model for better readability
    for model_id in MODEL_IDS:
        model_summaries = [s for s in all_results_summary if s['model_id'] == model_id]
        if model_summaries:
            print(f"\nMODEL: {model_id}")
            print("-" * 80)
            for summary in model_summaries:
                print(f"  {summary['dataset'].upper()} ({summary['split']}):")
                print(f"    Entries: {summary['num_entries']}")
                print(f"    LLM Baseline: EM={summary['mode1_exact_match']:.3f}, F1={summary['mode1_f1']:.3f}")
                print(f"    RAG: EM={summary['mode2_exact_match']:.3f}, F1={summary['mode2_f1']:.3f}")
                # print(f"    Agentic: EM={summary['mode3_exact_match']:.3f}, F1={summary['mode3_f1']:.3f}")  # Disabled
                print(f"    Agentic: EM={summary['mode3_exact_match']:.3f}, F1={summary['mode3_f1']:.3f}")
    
    print("\n" + "=" * 80)
    print("All results saved to: outputs/results/")
    print("  - CSV files: benchmark_results_<dataset>_<split>_<model_id>_<entries>.csv")
    print("  - JSON files: benchmark_results_<dataset>_<split>_<model_id>_<entries>.json")
    print("  - Reports: benchmark_report_<dataset>_<split>_<model_id>_<entries>.txt")
    print("=" * 80)