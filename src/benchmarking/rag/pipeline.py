"""
Pipeline utilities for RAG - context formatting.
"""

from typing import Dict, List


def format_search_results_as_context(search_results: Dict) -> str:
    """
    Format SearxNG search results into a context string for the LLM.

    Args:
        search_results: Dict from search_searxng with 'results' key containing
            list of dicts with 'title', 'url', 'content' (or 'snippet').

    Returns:
        Formatted context string suitable for RAG prompt.
    """
    if "error" in search_results:
        return f"Search error: {search_results.get('error', 'Unknown error')}"

    results = search_results.get("results", [])
    if not results:
        return "No search results found."

    parts: List[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("url", "No URL")
        content = r.get("content") or r.get("snippet", "No content")
        parts.append(f"[{i}] {title}\nURL: {url}\n{content}")

    return "\n\n".join(parts)
