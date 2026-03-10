"""
Benchmarking module for QA datasets with AWS Bedrock.

This module provides functionality to benchmark QA datasets using AWS Bedrock
endpoints across three modes: LLM baseline, RAG with SearxNG, and agentic solutions.
"""

from .bedrock_client import (
    ping_bedrock_model,
    bedrock_generate,
    _get_bedrock_client,
    AWS_REGION
)
from .llm_baseline import llm_baseline_answer
from .rag_bedrock import (
    rag_bedrock_pipeline,
    bedrock_query_rewriter,
    bedrock_rag_generation
)
from .agentic_solution import (
    agentic_answer,
    create_agentic_agent,
    QAAnswer
)
from .evaluator import (
    calculate_metrics,
    compare_modes,
    calculate_exact_match,
    calculate_f1_score,
    calculate_accuracy,
    normalize
)

__all__ = [
    "ping_bedrock_model",
    "bedrock_generate",
    "_get_bedrock_client",
    "AWS_REGION",
    "llm_baseline_answer",
    "rag_bedrock_pipeline",
    "bedrock_query_rewriter",
    "bedrock_rag_generation",
    "agentic_answer",
    "create_agentic_agent",
    "QAAnswer",
    "calculate_metrics",
    "compare_modes",
    "calculate_exact_match",
    "calculate_f1_score",
    "calculate_accuracy",
    "normalize"
]

