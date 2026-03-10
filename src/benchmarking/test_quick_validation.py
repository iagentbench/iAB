#!/usr/bin/env python3
"""
Quick validation script for the benchmarking pipeline.
Run: python -m src.benchmarking.test_quick_validation
"""
import sys


def main():
    errors = []
    
    # 1. Imports
    try:
        from src.benchmarking.dataset_loader import load_simpleqa, load_hotpotqa, load_iagentbench
        from src.benchmarking.llm_baseline import llm_baseline_answer
        from src.benchmarking.rag.retrieval import search_searxng
        from src.benchmarking.rag.pipeline import format_search_results_as_context
        from src.benchmarking.evaluator import calculate_metrics
        from src.benchmarking.build_final_results import parse_record_id
        from src.benchmarking.benchmark_runner import run_benchmark
        print("1. Imports: OK")
    except Exception as e:
        errors.append(f"Imports: {e}")
        print(f"1. Imports: FAIL - {e}")
        return 1
    
    # 2. Dataset loading
    try:
        ds = load_iagentbench()
        assert "test" in ds and len(ds["test"]) > 0
        print("2. Dataset (iAgentBench): OK")
    except Exception as e:
        errors.append(f"Dataset: {e}")
        print(f"2. Dataset: FAIL - {e}")
        return 1
    
    # 3. Evaluator
    try:
        m = calculate_metrics(["Paris", "London"], ["Paris", "Berlin"])
        assert m["exact_match_accuracy"] == 0.5
        print("3. Evaluator: OK")
    except Exception as e:
        errors.append(f"Evaluator: {e}")
        print(f"3. Evaluator: FAIL - {e}")
        return 1
    
    # 4. RAG pipeline
    try:
        ctx = format_search_results_as_context({"results": [{"title": "T", "url": "U", "content": "C"}]})
        assert "T" in ctx and "C" in ctx
        print("4. RAG pipeline: OK")
    except Exception as e:
        errors.append(f"RAG: {e}")
        print(f"4. RAG pipeline: FAIL - {e}")
        return 1
    
    # 5. build_final_results iagentbench -> qa_iab
    try:
        _, d, _ = parse_record_id("claude_iagentbench_mode1")
        assert d == "qa_iab"
        print("5. build_final_results (iagentbench->qa_iab): OK")
    except Exception as e:
        errors.append(f"build_final_results: {e}")
        print(f"5. build_final_results: FAIL - {e}")
        return 1
    
    # 6. Mode 1 (Bedrock) - optional, requires AWS
    try:
        ans = llm_baseline_answer("What is 2+2?", model_id="anthropic.claude-3-haiku-20240307-v1:0", region="us-east-1")
        assert ans and "4" in str(ans)
        print("6. Mode 1 (Bedrock): OK")
    except Exception as e:
        print(f"6. Mode 1 (Bedrock): SKIP - {e}")
    
    print()
    print("=== Quick validation passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
