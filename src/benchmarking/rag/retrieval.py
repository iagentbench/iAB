"""
Retrieval utilities for RAG pipeline.

Re-exports search_searxng from web_fetcher with max_results support.
"""

import os
import sys
from typing import Dict, Optional

# Ensure src is on path for web_fetcher import
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.web_fetcher import search_searxng as _search_searxng_raw


def search_searxng(
    query: str,
    base_url: str = "http://localhost:8080",
    max_results: Optional[int] = None,
    format: str = "json",
    engines: Optional[str] = None,
    language: str = "en-US",
    pageno: int = 1,
    time_range: Optional[str] = None,
    safesearch: int = 0,
) -> Dict:
    """
    Perform a search using SearXNG API.

    Args:
        query: Search query string
        base_url: SearXNG instance URL (default: http://localhost:8080)
        max_results: Maximum number of results to return (None = all)
        format: Response format ('json', 'csv', or 'rss')
        engines: Comma-separated list of engines (optional)
        language: Language code (default: 'en-US')
        pageno: Page number (default: 1)
        time_range: Time range filter ('day', 'week', 'month', 'year')
        safesearch: SafeSearch level (0=off, 1=moderate, 2=strict)

    Returns:
        Dictionary containing search results:
        - 'results': List of result dictionaries (sliced to max_results if set)
        - 'query': Original query
        - 'number_of_results': Estimated total
        - 'error': Error message if request failed
    """
    result = _search_searxng_raw(
        query=query,
        base_url=base_url,
        format=format,
        engines=engines,
        language=language,
        pageno=pageno,
        time_range=time_range,
        safesearch=safesearch,
    )
    if max_results is not None and "results" in result and result["results"]:
        result = {**result, "results": result["results"][:max_results]}
    return result
