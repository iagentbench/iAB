#!/usr/bin/env python3
"""
Run a minimal benchmark (few questions) and then LLM eval (Ollama) to verify pipeline.

Usage:
  BENCHMARK_DELAY_SECONDS=0 python -m src.benchmarking.run_fulltest_benchmark_and_eval

Requires:
  - AWS credentials + Bedrock for benchmark (Modes 1–3)
  - Ollama running locally for eval (or set LLM_EVAL_BACKEND=bedrock)
"""
import os
import sys
from pathlib import Path


def main():
    workspace = Path(__file__).resolve().parent.parent.parent
    results_dir = workspace / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    num_questions = int(os.getenv("FULLTEST_NUM_QUESTIONS", "2"))
    delay = int(os.getenv("BENCHMARK_DELAY_SECONDS", "0"))
    print("=" * 70)
    print("FULL TEST: Benchmark (few runs) + LLM Eval (Ollama)")
    print("=" * 70)
    print(f"Questions: {num_questions}, Delay between questions: {delay}s")
    print()

    # 1. Run benchmark
    print("Step 1: Running benchmark...")
    from src.benchmarking.benchmark_runner import run_benchmark

    results = run_benchmark(
        dataset_name="simpleqa",
        num_entries=num_questions,
        model_id=os.getenv("BENCHMARK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
        region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        verbose=True,
        save_results=True,
    )
    print("Benchmark done.")
    print()

    # 2. Find the CSV we just wrote (most recent benchmark_results_*.csv, not _llm_eval)
    csv_files = sorted(
        results_dir.glob("benchmark_results_simpleqa_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    csv_files = [f for f in csv_files if "_llm_eval" not in f.name]
    if not csv_files:
        print("No benchmark CSV found in outputs/results/")
        sys.exit(1)
    csv_path = csv_files[0]
    print(f"Step 2: LLM eval on {csv_path.name}")

    # 3. Run LLM evaluation (Ollama by default)
    os.environ.setdefault("LLM_EVAL_BACKEND", "ollama")
    from src.benchmarking.evaluate_llm import calculate_llm_eval_metrics

    calculate_llm_eval_metrics([csv_path.name])
    print()
    print("=" * 70)
    print("FULL TEST DONE: Benchmark + Eval completed.")
    print("=" * 70)
    print(f"  Benchmark CSV: {csv_path}")
    print(f"  Eval CSV:      {csv_path.parent / (csv_path.stem + '_llm_eval.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
