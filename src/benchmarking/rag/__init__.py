"""
RAG module for QA benchmarking.

Provides retrieval (SearxNG search) and pipeline (context formatting) utilities.
"""

from .retrieval import search_searxng
from .pipeline import format_search_results_as_context

__all__ = ["search_searxng", "format_search_results_as_context"]
