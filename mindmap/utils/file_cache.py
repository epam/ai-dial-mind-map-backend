"""Cache of the files read from the DIAL file storage, bounded by memory.

The cached values are parsed json documents: the graph files, the source
documents and the rag index files. An index file embeds the original document
together with its chunks, and the chunks of a pdf carry the base64 page
images, so a single entry can be tens of megabytes. Bounding such a cache by
the number of entries does not bound its memory at all, which is why the
capacity here is counted in bytes.
"""

import sys
from typing import Any

from cachetools import LRUCache

from dial_rag.utils import format_size, size_env_var
from mindmap.utils.logger_config import logger

FILE_CACHE_CAPACITY: int = size_env_var("FILE_CACHE_CAPACITY", "256MiB")


def json_size(value: Any) -> int:
    """Approximate memory footprint of a parsed json document."""
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(
            json_size(key) + json_size(item) for key, item in value.items()
        )
    elif isinstance(value, (list, tuple)):
        size += sum(json_size(item) for item in value)
    return size


class FileCache:
    """LRU cache of the storage files, evicting by memory footprint."""

    def __init__(self, capacity: int = FILE_CACHE_CAPACITY) -> None:
        self._capacity = capacity
        self._cache: LRUCache = LRUCache(maxsize=capacity, getsizeof=json_size)

    def get(self, key: str, default: Any = None) -> Any:
        return self._cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        try:
            self._cache[key] = value
        except ValueError:
            # cachetools rejects a value which alone exceeds the capacity,
            # such an entry is simply not cached
            logger.debug(
                f"File {key} is too large to cache "
                f"({format_size(json_size(value))} > "
                f"{format_size(self._capacity)})"
            )

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
