"""
LLM-based evaluation for benchmarking results.

This module provides LLM-based evaluation of benchmark results using an LLM classifier
to assess answer quality beyond exact match and F1 scores. It evaluates responses as
"correct", "incorrect", or "not attempted" based on semantic understanding.

Similar to src/evaluation/evaluator.py but designed for benchmarking results format.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import Optional
import ast

from .llm_eval_common import llm_eval_metrics as _shared_llm_eval_metrics


def llm_eval_metrics(
    answer: str,
    response: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    backend: Optional[str] = None,
    bedrock_model_id: Optional[str] = None,
    bedrock_region: Optional[str] = None,
) -> str:
    """
    Backwards-compatible wrapper around `llm_eval_common.llm_eval_metrics`.

    Historically this module defined its own Ollama-only implementation. We now
    delegate to the shared helper so that both Ollama and Bedrock backends use
    the same grading criteria.
    """
    # Question is optional for this path; many CSVs include it but older ones may not.
    question = ""
    return _shared_llm_eval_metrics(
        question=question,
        answer=answer,
        response=response,
        backend=backend,
        ollama_model=model,
        ollama_base_url=base_url,
        bedrock_model_id=bedrock_model_id,
        bedrock_region=bedrock_region,
    )


def evaluate_benchmark_results(
    csv_file: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    save_results: bool = True,
    backend: Optional[str] = None,
    bedrock_model_id: Optional[str] = None,
    bedrock_region: Optional[str] = None,
) -> pd.DataFrame:
    """
    Evaluate benchmark results using LLM-based evaluation.
    
    Reads a benchmark results CSV file, evaluates each mode's answers using LLM evaluation,
    and adds LLM evaluation columns to the dataset.
    
    Args:
        csv_file: Path to benchmark results CSV file
        model: Optional Ollama model name (defaults to OLLAMA_MODEL env var)
        base_url: Optional Ollama API base URL (defaults to OLLAMA_URL env var)
        save_results: Whether to save results to a new CSV file (default: True)
    
    Returns:
        DataFrame with added LLM evaluation columns:
        - mode1_llm_eval, mode2_llm_eval, mode3_llm_eval
    """
    # Handle both relative and absolute paths
    if not os.path.isabs(csv_file):
        # Try relative to outputs/results first
        results_dir = Path(__file__).parent.parent.parent / "outputs" / "results"
        csv_path = results_dir / csv_file
        if not csv_path.exists():
            # Try as absolute path
            csv_path = Path(csv_file)
    else:
        csv_path = Path(csv_file)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading benchmark results from: {csv_path}")
    dataset = pd.read_csv(csv_path)
    
    # Check if LLM evaluation columns already exist
    metrics_file = csv_path.parent / f"{csv_path.stem}_llm_eval.csv"
    
    if metrics_file.exists():
        try:
            metric_dataset = pd.read_csv(metrics_file)
            # Merge existing metrics into dataset if they exist
            for col in ['mode1_llm_eval', 'mode2_llm_eval', 'mode3_llm_eval']:
                if col in metric_dataset.columns:
                    dataset[col] = metric_dataset[col]
            print(f"Loaded existing LLM evaluation metrics from: {metrics_file}")
        except Exception as e:
            print(f"Warning: Could not load existing metrics file: {e}")
            print("Recalculating all metrics...")
    
    # Helper function for LLM evaluation with error handling
    def llm_eval(question: str, ground_truth: str, response: str):
        """
        Evaluate a single answer using LLM classifier.
        
        Returns "ERROR: <message>" if evaluation fails, allowing process to continue.
        """
        if pd.isna(response) or response == "":
            return "not attempted"
        try:
            return llm_eval_metrics(
                answer=str(ground_truth),
                response=str(response),
                model=model,
                base_url=base_url,
                backend=backend,
                bedrock_model_id=bedrock_model_id,
                bedrock_region=bedrock_region,
            )
        except Exception as e:
            print(f"Error evaluating: {e}")
            return f"ERROR: {str(e)}"
    
    # Calculate LLM evaluation metrics for each mode
    modes = ['mode1', 'mode2', 'mode3']
    
    for mode in modes:
        col_name = f"{mode}_llm_eval"
        answer_col = f"{mode}_answer"
        
        if col_name not in dataset.columns:
            print(f"\nCalculating LLM metrics for {mode}...")
            results = []
            total = len(dataset)
            
            for idx, row in dataset.iterrows():
                ground_truth = row['ground_truth']
                question = row['question'] if 'question' in row else ""
                answer = row[answer_col] if answer_col in row else ""
                result = llm_eval(question, ground_truth, answer)
                results.append(result)
                
                # Save progress every 50 rows to prevent data loss
                if (idx + 1) % 50 == 0 or (idx + 1) == total:
                    print(f"Progress: {idx + 1}/{total} ({100*(idx + 1)/total:.1f}%)")
                    dataset[col_name] = results + [None] * (total - len(results))
                    if save_results:
                        dataset.to_csv(metrics_file, index=False)
            
            dataset[col_name] = results
        else:
            # Resume calculation: only process rows with NaN or ERROR values
            mask = dataset[col_name].isna() | (
                dataset[col_name].astype(str).str.startswith('ERROR', na=False)
            )
            if mask.any():
                num_to_calc = mask.sum()
                print(f"\nRecalculating LLM metrics for {mode} ({num_to_calc} rows)...")
                for idx, (i, row) in enumerate(dataset[mask].iterrows(), 1):
                    ground_truth = row['ground_truth']
                    question = row['question'] if 'question' in row else ""
                    answer = row[answer_col] if answer_col in row else ""
                    result = llm_eval(question, ground_truth, answer)
                    dataset.loc[i, col_name] = result
                    
                    # Save progress every 50 rows
                    if idx % 50 == 0 or idx == num_to_calc:
                        print(f"Progress: {idx}/{num_to_calc} ({100*idx/num_to_calc:.1f}%)")
                        if save_results:
                            dataset.to_csv(metrics_file, index=False)
    
    # Save final dataset with all metrics
    if save_results:
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(metrics_file, index=False)
        print(f"\nLLM evaluation results saved to: {metrics_file}")
    
    # Calculate and print summary statistics
    print("\n" + "=" * 70)
    dataset_name = _extract_dataset_name(csv_path)
    print(f"LLM EVALUATION SUMMARY - {dataset_name.upper()}")
    print("=" * 70)
    
    for mode in modes:
        col_name = f"{mode}_llm_eval"
        mode_name = {
            'mode1': 'LLM Baseline',
            'mode2': 'RAG with SearxNG',
            'mode3': 'Agentic Solution'
        }[mode]
        
        if col_name in dataset.columns:
            valid_results = dataset[dataset[col_name].isin(['correct', 'incorrect', 'not attempted'])]
            total_valid = len(valid_results)
            
            if total_valid > 0:
                correct = len(valid_results[valid_results[col_name] == 'correct'])
                incorrect = len(valid_results[valid_results[col_name] == 'incorrect'])
                not_attempted = len(valid_results[valid_results[col_name] == 'not attempted'])
                correct_rate = correct / total_valid
                
                print(f"\n{mode_name}:")
                print(f"  Correct: {correct} ({correct_rate:.3f})")
                print(f"  Incorrect: {incorrect}")
                print(f"  Not Attempted: {not_attempted}")
                print(f"  Total Valid: {total_valid}/{len(dataset)}")
    
    print("\n" + "=" * 70)
    
    return dataset


def _extract_dataset_name(csv_path: Path) -> str:
    """
    Extract dataset name from CSV filename.
    
    Examples:
        benchmark_results_triviaqa_validation_500entries.csv -> "triviaqa"
        benchmark_results_simpleqa_test_500entries.csv -> "simpleqa"
        benchmark_results_hotpotqa_validation_500entries.csv -> "hotpotqa"
    """
    filename = csv_path.stem
    if 'triviaqa' in filename.lower():
        return 'triviaqa'
    elif 'simpleqa' in filename.lower():
        return 'simpleqa'
    elif 'hotpotqa' in filename.lower():
        return 'hotpotqa'
    else:
        # Try to extract from filename pattern
        parts = filename.split('_')
        for part in parts:
            if part in ['triviaqa', 'simpleqa', 'hotpotqa']:
                return part
        return 'unknown'


def calculate_llm_eval_metrics(csv_files: list) -> None:
    """
    Calculate LLM evaluation metrics for multiple benchmark results files.
    
    This function processes multiple CSV files, combines them into a single dataset,
    and generates a comprehensive summary report with both combined and separate metrics.
    
    Args:
        csv_files: List of paths to benchmark results CSV files (relative to outputs/results/ or absolute)
    """
    results_dir = Path(__file__).parent.parent.parent / "outputs" / "results"
    datasets = []
    csv_paths = []
    
    # Process each CSV file
    for csv_file in csv_files:
        # Handle both relative and absolute paths
        if not os.path.isabs(csv_file):
            csv_path = results_dir / csv_file
            if not csv_path.exists():
                csv_path = Path(csv_file)
        else:
            csv_path = Path(csv_file)
        
        if not csv_path.exists():
            print(f"Warning: CSV file not found: {csv_path}, skipping...")
            continue
        
        csv_paths.append(csv_path)
        print(f"\nProcessing: {csv_path.name}")
        print("=" * 70)
        
        # Evaluate this dataset
        dataset = evaluate_benchmark_results(str(csv_path))
        
        # Add dataset identifier
        dataset_name = _extract_dataset_name(csv_path)
        dataset['dataset'] = dataset_name
        
        datasets.append(dataset)
    
    if not datasets:
        raise ValueError("No valid CSV files found to process")
    
    # Combine all datasets
    print("\n" + "=" * 70)
    print("COMBINING DATASETS")
    print("=" * 70)
    
    combined_dataset = pd.concat(datasets, ignore_index=True)
    
    # Save combined results
    combined_file = results_dir / "benchmark_results_combined_llm_eval.csv"
    combined_dataset.to_csv(combined_file, index=False)
    print(f"\nCombined results saved to: {combined_file}")
    print(f"Total questions: {len(combined_dataset)}")
    print(f"  - TriviaQA: {len(combined_dataset[combined_dataset['dataset'] == 'triviaqa'])}")
    print(f"  - SimpleQA: {len(combined_dataset[combined_dataset['dataset'] == 'simpleqa'])}")
    print(f"  - HotpotQA: {len(combined_dataset[combined_dataset['dataset'] == 'hotpotqa'])}")
    
    # Generate comprehensive summary report
    report_file = results_dir / "benchmark_results_combined_llm_eval_report.txt"
    
    with open(report_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("LLM EVALUATION REPORT - COMBINED DATASETS\n")
        f.write("=" * 70 + "\n\n")
        
        # List source files
        f.write("Source Files:\n")
        for csv_path in csv_paths:
            f.write(f"  - {csv_path.name}\n")
        f.write(f"\nTotal Questions: {len(combined_dataset)}\n")
        f.write(f"  - TriviaQA: {len(combined_dataset[combined_dataset['dataset'] == 'triviaqa'])}\n")
        f.write(f"  - SimpleQA: {len(combined_dataset[combined_dataset['dataset'] == 'simpleqa'])}\n")
        f.write(f"  - HotpotQA: {len(combined_dataset[combined_dataset['dataset'] == 'hotpotqa'])}\n\n")
        
        # ===== COMBINED METRICS (ALL DATASETS TOGETHER) =====
        f.write("=" * 70 + "\n")
        f.write("COMBINED METRICS (ALL DATASETS)\n")
        f.write("=" * 70 + "\n\n")
        
        combined_mode_scores = {}
        for mode in ['mode1', 'mode2', 'mode3']:
            col_name = f"{mode}_llm_eval"
            mode_name = {
                'mode1': 'LLM Baseline',
                'mode2': 'RAG with SearxNG',
                'mode3': 'Agentic Solution'
            }[mode]
            
            if col_name in combined_dataset.columns:
                valid_results = combined_dataset[combined_dataset[col_name].isin(['correct', 'incorrect', 'not attempted'])]
                total_valid = len(valid_results)
                
                if total_valid > 0:
                    correct = len(valid_results[valid_results[col_name] == 'correct'])
                    incorrect = len(valid_results[valid_results[col_name] == 'incorrect'])
                    not_attempted = len(valid_results[valid_results[col_name] == 'not attempted'])
                    correct_rate = correct / total_valid
                    combined_mode_scores[mode] = correct_rate
                    
                    f.write(f"{mode_name}:\n")
                    f.write(f"  Correct: {correct} / {total_valid} ({correct_rate:.3f})\n")
                    f.write(f"  Incorrect: {incorrect}\n")
                    f.write(f"  Not Attempted: {not_attempted}\n")
                    f.write(f"  Total Valid: {total_valid} / {len(combined_dataset)}\n\n")
        
        # Best mode for combined
        if combined_mode_scores:
            best_mode = max(combined_mode_scores.items(), key=lambda x: x[1])
            mode_names = {
                'mode1': 'LLM Baseline',
                'mode2': 'RAG with SearxNG',
                'mode3': 'Agentic Solution'
            }
            f.write(f"Best Performing Mode (Combined): {mode_names[best_mode[0]]} ({best_mode[1]:.3f})\n\n")
        
        # ===== SEPARATE METRICS (BY DATASET) =====
        f.write("=" * 70 + "\n")
        f.write("SEPARATE METRICS (BY DATASET)\n")
        f.write("=" * 70 + "\n\n")
        
        for dataset_name in ['triviaqa', 'simpleqa', 'hotpotqa']:
            dataset_subset = combined_dataset[combined_dataset['dataset'] == dataset_name]
            
            if len(dataset_subset) == 0:
                continue
            
            f.write(f"\n{dataset_name.upper()}:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Questions: {len(dataset_subset)}\n\n")
            
            dataset_mode_scores = {}
            for mode in ['mode1', 'mode2', 'mode3']:
                col_name = f"{mode}_llm_eval"
                mode_name = {
                    'mode1': 'LLM Baseline',
                    'mode2': 'RAG with SearxNG',
                    'mode3': 'Agentic Solution'
                }[mode]
                
                if col_name in dataset_subset.columns:
                    valid_results = dataset_subset[dataset_subset[col_name].isin(['correct', 'incorrect', 'not attempted'])]
                    total_valid = len(valid_results)
                    
                    if total_valid > 0:
                        correct = len(valid_results[valid_results[col_name] == 'correct'])
                        incorrect = len(valid_results[valid_results[col_name] == 'incorrect'])
                        not_attempted = len(valid_results[valid_results[col_name] == 'not attempted'])
                        correct_rate = correct / total_valid
                        dataset_mode_scores[mode] = correct_rate
                        
                        f.write(f"  {mode_name}:\n")
                        f.write(f"    Correct: {correct} / {total_valid} ({correct_rate:.3f})\n")
                        f.write(f"    Incorrect: {incorrect}\n")
                        f.write(f"    Not Attempted: {not_attempted}\n")
                        f.write(f"    Total Valid: {total_valid} / {len(dataset_subset)}\n\n")
            
            # Best mode for this dataset
            if dataset_mode_scores:
                best_mode = max(dataset_mode_scores.items(), key=lambda x: x[1])
                mode_names = {
                    'mode1': 'LLM Baseline',
                    'mode2': 'RAG with SearxNG',
                    'mode3': 'Agentic Solution'
                }
                f.write(f"  Best Performing Mode: {mode_names[best_mode[0]]} ({best_mode[1]:.3f})\n\n")
        
        # ===== COMPARATIVE SUMMARY =====
        f.write("=" * 70 + "\n")
        f.write("COMPARATIVE SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Combined Performance:\n")
        if combined_mode_scores:
            sorted_modes = sorted(combined_mode_scores.items(), key=lambda x: x[1], reverse=True)
            mode_names = {
                'mode1': 'LLM Baseline',
                'mode2': 'RAG with SearxNG',
                'mode3': 'Agentic Solution'
            }
            for mode, score in sorted_modes:
                f.write(f"  {mode_names[mode]}: {score:.3f}\n")
    
    print(f"\nSummary report saved to: {report_file}")


# Test script: Run this file directly to test the evaluator
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_llm.py <csv_file1> [csv_file2] ...")
        print("\nExamples:")
        print("  # Single file:")
        print("  python evaluate_llm.py benchmark_results_triviaqa_validation_500entries.csv")
        print("\n  # Multiple files (combined):")
        print("  python evaluate_llm.py benchmark_results_triviaqa_validation_500entries.csv benchmark_results_simpleqa_test_500entries.csv")
        print("\nThe CSV files should be in outputs/results/ or provide absolute paths.")
        sys.exit(1)
    
    csv_files = sys.argv[1:]
    print(f"Evaluating {len(csv_files)} benchmark result file(s):")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    print("=" * 70)
    
    try:
        calculate_llm_eval_metrics(csv_files)
        print("\n✓ LLM evaluation completed successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

