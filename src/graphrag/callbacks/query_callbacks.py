# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Query Callbacks."""

from typing import Any

from graphrag.callbacks.llm_callbacks import BaseLLMCallback
# SearchResult removed - query module not used in indexing
# from graphrag.query.structured_search.base import SearchResult
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type stub for removed module
    class SearchResult:
        pass


class QueryCallbacks(BaseLLMCallback):
    """Callbacks used during query execution."""

    def on_context(self, context: Any) -> None:
        """Handle when context data is constructed."""

    def on_map_response_start(self, map_response_contexts: list[str]) -> None:
        """Handle the start of map operation."""

    def on_map_response_end(self, map_response_outputs: list[SearchResult]) -> None:
        """Handle the end of map operation."""

    def on_reduce_response_start(
        self, reduce_response_context: str | dict[str, Any]
    ) -> None:
        """Handle the start of reduce operation."""

    def on_reduce_response_end(self, reduce_response_output: str) -> None:
        """Handle the end of reduce operation."""

    def on_llm_new_token(self, token) -> None:
        """Handle when a new token is generated."""
