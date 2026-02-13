# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Factory functions for creating a cache."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from graphrag.cache.json_pipeline_cache import JsonPipelineCache
from graphrag.cache.memory_pipeline_cache import InMemoryCache
from graphrag.cache.noop_pipeline_cache import NoopPipelineCache
from graphrag.config.enums import CacheType
# Azure storage imports are lazy - only imported when needed
from graphrag.storage.file_pipeline_storage import FilePipelineStorage

if TYPE_CHECKING:
    from collections.abc import Callable

    from graphrag.cache.pipeline_cache import PipelineCache


class CacheFactory:
    """A factory class for cache implementations.

    Includes a method for users to register a custom cache implementation.

    Configuration arguments are passed to each cache implementation as kwargs
    for individual enforcement of required/optional arguments.
    """

    _registry: ClassVar[dict[str, Callable[..., PipelineCache]]] = {}

    @classmethod
    def register(cls, cache_type: str, creator: Callable[..., PipelineCache]) -> None:
        """Register a custom cache implementation.

        Args:
            cache_type: The type identifier for the cache.
            creator: A class or callable that creates an instance of PipelineCache.
        """
        cls._registry[cache_type] = creator

    @classmethod
    def create_cache(cls, cache_type: str, kwargs: dict) -> PipelineCache:
        """Create a cache object from the provided type.

        Args:
            cache_type: The type of cache to create.
            root_dir: The root directory for file-based caches.
            kwargs: Additional keyword arguments for the cache constructor.

        Returns
        -------
            A PipelineCache instance.

        Raises
        ------
            ValueError: If the cache type is not registered.
        """
        if cache_type not in cls._registry:
            msg = f"Unknown cache type: {cache_type}"
            raise ValueError(msg)

        return cls._registry[cache_type](**kwargs)

    @classmethod
    def get_cache_types(cls) -> list[str]:
        """Get the registered cache implementations."""
        return list(cls._registry.keys())

    @classmethod
    def is_supported_type(cls, cache_type: str) -> bool:
        """Check if the given cache type is supported."""
        return cache_type in cls._registry


# --- register built-in cache implementations ---
def create_file_cache(root_dir: str, base_dir: str, **kwargs) -> PipelineCache:
    """Create a file-based cache implementation."""
    # Create storage with base_dir in kwargs since FilePipelineStorage expects it there
    storage_kwargs = {"base_dir": root_dir, **kwargs}
    storage = FilePipelineStorage(**storage_kwargs).child(base_dir)
    return JsonPipelineCache(storage)


def create_blob_cache(**kwargs) -> PipelineCache:
    """Create a blob storage-based cache implementation."""
    # Azure blob storage - import lazily to avoid requiring azure packages when not used
    try:
        from graphrag.storage.blob_pipeline_storage import BlobPipelineStorage
        storage = BlobPipelineStorage(**kwargs)
        return JsonPipelineCache(storage)
    except ImportError:
        raise ImportError(
            "Azure blob storage requires azure-storage-blob package. "
            "Install with: pip install azure-storage-blob"
        )


def create_cosmosdb_cache(**kwargs) -> PipelineCache:
    """Create a CosmosDB-based cache implementation."""
    # Azure CosmosDB storage - import lazily to avoid requiring azure packages when not used
    try:
        from graphrag.storage.cosmosdb_pipeline_storage import CosmosDBPipelineStorage
        storage = CosmosDBPipelineStorage(**kwargs)
        return JsonPipelineCache(storage)
    except ImportError:
        raise ImportError(
            "Azure CosmosDB storage requires azure-cosmos package. "
            "Install with: pip install azure-cosmos"
        )


def create_noop_cache(**_kwargs) -> PipelineCache:
    """Create a no-op cache implementation."""
    return NoopPipelineCache()


def create_memory_cache(**kwargs) -> PipelineCache:
    """Create a memory cache implementation."""
    return InMemoryCache(**kwargs)


# --- register built-in cache implementations ---
CacheFactory.register(CacheType.none.value, create_noop_cache)
CacheFactory.register(CacheType.memory.value, create_memory_cache)
CacheFactory.register(CacheType.file.value, create_file_cache)
# Azure cache types - only registered if Azure storage is available
# They will fail with helpful error if used without Azure packages
CacheFactory.register(CacheType.blob.value, create_blob_cache)
CacheFactory.register(CacheType.cosmosdb.value, create_cosmosdb_cache)
