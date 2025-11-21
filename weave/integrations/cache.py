"""Universal cache for LLM integrations.

This module provides a disk-based cache for LLM API responses to avoid
duplicate API calls with identical parameters.
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# Global cache instance
_global_cache: Optional[LLMCache] = None

# Context variable to track if caching is disabled
_cache_disabled: ContextVar[bool] = ContextVar("cache_disabled", default=False)

# Context variable to track if current call was a cache hit
_cache_hit: ContextVar[bool] = ContextVar("cache_hit", default=False)


class LLMCache:
    """Disk cache for LLM responses with unlimited size support."""

    def __init__(
        self,
        directory: str | Path,
        size_limit: Optional[int] = None,
        ttl: Optional[float] = None,
        deterministic_only: bool = False,
    ):
        """
        Initialize cache.

        Args:
            directory: Path to cache directory
            size_limit: Optional size limit in bytes (default: unlimited ~4.6 exabytes)
            ttl: Optional time-to-live in seconds (default: no expiration)
            deterministic_only: If True, only cache deterministic requests (temperature=0).
                              If False (default), cache all requests.
        """
        try:
            import diskcache as dc
        except ImportError:
            raise ImportError(
                "diskcache is required for caching. Install with: pip install diskcache"
            )

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        # Set to very large cap if unlimited (2^62 bytes ~ 4.6 exabytes)
        cap = size_limit if size_limit is not None else (1 << 62)
        self._cache = dc.Cache(str(self.directory), size_limit=cap)
        self._ttl = ttl
        self._deterministic_only = deterministic_only

        logger.info(f"Initialized LLM cache at {self.directory}")

    def _make_cache_key(
        self,
        integration: str,
        kwargs: dict[str, Any],
    ) -> str:
        """
        Create cache key from call parameters.

        Args:
            integration: Name of integration (e.g., "openai", "anthropic")
            kwargs: Call parameters

        Returns:
            SHA256 hash as cache key
        """
        # Extract only parameters that affect the response
        cache_params = {
            "integration": integration,
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages"),
            "temperature": kwargs.get("temperature", 1.0),
            "max_tokens": kwargs.get("max_tokens"),
            "top_p": kwargs.get("top_p", 1.0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "tools": kwargs.get("tools"),
            "tool_choice": kwargs.get("tool_choice"),
            "functions": kwargs.get("functions"),
            "function_call": kwargs.get("function_call"),
            "response_format": kwargs.get("response_format"),
            "n": kwargs.get("n"),
            "stop": kwargs.get("stop"),
            "logprobs": kwargs.get("logprobs"),
            "top_logprobs": kwargs.get("top_logprobs"),
            "seed": kwargs.get("seed"),
        }

        # Remove None values
        cache_params = {k: v for k, v in cache_params.items() if v is not None}

        # Create stable hash
        key_str = json.dumps(cache_params, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def should_cache(self, kwargs: dict[str, Any]) -> bool:
        """
        Determine if this request should be cached.

        Args:
            kwargs: Call parameters

        Returns:
            True if request should be cached
        """
        # Don't cache streaming requests
        if kwargs.get("stream"):
            return False

        # Allow force cache override
        if kwargs.get("_weave_force_cache"):
            return True

        # Allow skip cache override
        if kwargs.get("_weave_skip_cache"):
            return False

        # If deterministic_only mode, check temperature
        if self._deterministic_only:
            temperature = kwargs.get("temperature", 1.0)
            if temperature > 0:
                return False

        return True

    def get(self, integration: str, kwargs: dict[str, Any]) -> Any | None:
        """
        Get cached response if available.

        Args:
            integration: Name of integration
            kwargs: Call parameters

        Returns:
            Cached response or None if not found
        """
        if not self.should_cache(kwargs):
            return None

        key = self._make_cache_key(integration, kwargs)
        cached_data = self._cache.get(key)

        if cached_data is not None:
            logger.debug(f"Cache hit for {integration} (key: {key[:8]}...)")
            # Extract response from metadata tuple
            if isinstance(cached_data, tuple) and len(cached_data) == 2:
                _, response = cached_data
                return response
            # Backwards compatibility: return as-is if not tuple
            return cached_data
        else:
            logger.debug(f"Cache miss for {integration} (key: {key[:8]}...)")

        return None

    def _serialize_response(self, response: Any) -> Any:
        """
        Serialize response for caching.
        Converts pydantic models and other non-picklable objects to dicts.

        Args:
            response: Response object to serialize

        Returns:
            Serializable version of response
        """
        import pickle

        # Try dict conversion FIRST (safer, more compatible)
        # Try pydantic model_dump first
        if hasattr(response, 'model_dump'):
            try:
                serialized = response.model_dump()
                # Verify it's picklable
                pickle.dumps(serialized)
                logger.debug(f"Serialized response using model_dump()")
                return serialized
            except Exception as e:
                logger.debug(f"model_dump() failed: {e}")

        # Try pydantic dict() for older versions
        if hasattr(response, 'dict'):
            try:
                serialized = response.dict()
                pickle.dumps(serialized)
                logger.debug(f"Serialized response using dict()")
                return serialized
            except Exception as e:
                logger.debug(f"dict() failed: {e}")

        # Fallback: Try to pickle as-is
        try:
            pickle.dumps(response)
            # If successful, return as-is (no conversion needed)
            logger.debug(f"Response pickled as-is without conversion")
            return response
        except (TypeError, AttributeError, pickle.PicklingError):
            # Can't pickle - already tried dict conversion above
            logger.debug(f"Response not directly picklable and dict conversion failed")

        # Last resort: return as-is and let the cache.set() try-catch handle it
        logger.debug(f"Could not serialize response, returning as-is")
        return response

    def _deserialize_response(self, cached_data: Any, response_type: type | None = None) -> Any:
        """
        Deserialize cached response.

        Args:
            cached_data: Cached data to deserialize
            response_type: Optional type to reconstruct

        Returns:
            Deserialized response
        """
        # For now, just return the dict - consumers can handle it
        # In the future, we could reconstruct the original type if needed
        return cached_data

    def set(
        self,
        integration: str,
        kwargs: dict[str, Any],
        response: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Cache response.

        Args:
            integration: Name of integration
            kwargs: Call parameters
            response: Response to cache
            ttl: Optional TTL override (uses instance TTL if None)
        """
        if not self.should_cache(kwargs):
            return

        key = self._make_cache_key(integration, kwargs)
        expire = ttl if ttl is not None else self._ttl
        expire_time = None if expire is None else float(expire)

        # Serialize response for caching
        serialized_response = self._serialize_response(response)

        # Store metadata with response for filtering support
        metadata = {
            "provider": integration,
            "model": kwargs.get("model"),
        }
        cached_data = (metadata, serialized_response)

        try:
            self._cache.set(key, cached_data, expire=expire_time)
            logger.debug(f"Cached {integration} response (key: {key[:8]}...)")
        except Exception as e:
            # If caching fails (e.g., unpicklable object), just skip caching
            logger.warning(f"Failed to cache {integration} response: {e}. Skipping cache for this response.")
            return

    def clear(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ) -> int:
        """
        Clear cache entries, optionally filtered by provider or model.

        Args:
            provider: Optional provider name to filter by (e.g., "openai", "anthropic")
            model: Optional model pattern to filter by (supports wildcards like "gpt-4*")

        Returns:
            Number of entries cleared
        """
        if provider is None and model is None:
            # Clear everything
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared all {count} LLM cache entries")
            return count

        # Filter by provider and/or model
        keys_to_delete = []

        for key in list(self._cache.iterkeys()):
            try:
                cached_data = self._cache.get(key)
                if cached_data is None:
                    continue

                # Extract metadata
                metadata = None
                if isinstance(cached_data, tuple) and len(cached_data) == 2:
                    metadata, _ = cached_data

                if metadata is None:
                    # Old cache format without metadata - skip
                    continue

                # Check filters
                should_delete = True

                if provider is not None:
                    if metadata.get("provider") != provider:
                        should_delete = False

                if model is not None:
                    cached_model = metadata.get("model")
                    if cached_model is None or not fnmatch(cached_model, model):
                        should_delete = False

                if should_delete:
                    keys_to_delete.append(key)

            except Exception as e:
                logger.warning(f"Error checking cache key {key}: {e}")
                continue

        # Delete filtered keys
        for key in keys_to_delete:
            try:
                del self._cache[key]
            except Exception as e:
                logger.warning(f"Error deleting cache key {key}: {e}")

        logger.info(
            f"Cleared {len(keys_to_delete)} LLM cache entries "
            f"(provider={provider}, model={model})"
        )
        return len(keys_to_delete)

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            dict with size_limit, current_size, item_count, and percent_full
        """
        size_limit = self._cache.size_limit
        volume = self._cache.volume()  # Current size in bytes
        count = len(self._cache)  # Number of items

        return {
            "directory": str(self.directory),
            "size_limit": size_limit,
            "current_size": volume,
            "item_count": count,
            "percent_full": (volume / size_limit) * 100 if size_limit > 0 else 0.0,
        }

    def print_stats(self) -> None:
        """Print human-readable cache statistics."""
        stats = self.get_stats()

        def human_size(bytes_val: float) -> str:
            """Convert bytes to human readable format."""
            for unit in ["B", "KB", "MB", "GB", "TB", "PB", "EB"]:
                if bytes_val < 1024.0:
                    return f"{bytes_val:.2f} {unit}"
                bytes_val /= 1024.0
            return f"{bytes_val:.2f} EB"

        print("=" * 60)
        print("LLM CACHE STATISTICS")
        print("=" * 60)
        print(f"  Directory:      {stats['directory']}")
        print(f"  Size limit:     {human_size(stats['size_limit'])}")
        print(f"  Current size:   {human_size(stats['current_size'])}")
        print(f"  Items cached:   {stats['item_count']}")
        print(f"  % full:         {stats['percent_full']:.6f}%")
        print("=" * 60)


def get_global_cache() -> Optional[LLMCache]:
    """Get the global cache instance."""
    return _global_cache


def set_global_cache(cache: Optional[LLMCache]) -> None:
    """Set the global cache instance."""
    global _global_cache
    _global_cache = cache


def enable_cache() -> None:
    """
    Programmatically enable LLM caching globally.

    Usage:
        weave.cache.enable()
        # All subsequent LLM calls will be cached
    """
    # Initialize cache if not already initialized
    if get_global_cache() is None:
        from weave.trace import settings as settings_module
        from pathlib import Path

        cache = LLMCache(
            directory=settings_module.cache_dir(),
            size_limit=settings_module.cache_size_limit(),
            ttl=settings_module.cache_ttl(),
            deterministic_only=settings_module.cache_deterministic_only(),
        )
        set_global_cache(cache)
        logger.info(f"Initialized and enabled LLM cache at {settings_module.cache_dir()}")

    _cache_disabled.set(False)
    logger.info("LLM cache enabled globally")


def disable_cache_globally() -> None:
    """
    Programmatically disable LLM caching globally.

    Usage:
        weave.cache.disable()
        # All subsequent LLM calls will NOT be cached
    """
    _cache_disabled.set(True)
    logger.info("LLM cache disabled globally")


@contextmanager
def disable_cache() -> Iterator[None]:
    """
    Context manager to temporarily disable LLM caching.

    Usage:
        with weave.disable_cache():
            # Caching is disabled within this block
            result = model.generate(...)

    Yields:
        None
    """
    token = _cache_disabled.set(True)
    try:
        yield
    finally:
        _cache_disabled.reset(token)


def with_llm_cache(integration_name: str, unwrap_fn: Optional[Any] = None, wrap_fn: Optional[Any] = None):
    """
    Decorator to add caching to any LLM integration function.

    Usage:
        @with_llm_cache("anthropic")
        def wrapper(fn):
            def _wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return _wrapper

    Args:
        integration_name: Name of the integration (e.g., "openai", "anthropic")
        unwrap_fn: Optional function to unwrap responses before caching.
                   Called as unwrap_fn(response) -> unwrapped_response
        wrap_fn: Optional function to wrap cached responses when returning from cache.
                 Called as wrap_fn(cached_dict) -> wrapped_response

    Returns:
        Decorator function that adds caching
    """
    from functools import wraps

    def decorator(fn):
        @wraps(fn)
        def cached_wrapper(self, *args, **kwargs):
            # Reset cache hit flag for this call
            _cache_hit.set(False)

            # Check if caching is disabled via context manager
            if _cache_disabled.get():
                logger.debug(f"Cache disabled for {integration_name}")
                return fn(self, *args, **kwargs)

            # Check cache first
            cache = get_global_cache()
            if cache is not None:
                cached_response = cache.get(integration_name, kwargs)
                if cached_response is not None:
                    logger.debug(f"Cache hit for {integration_name}")
                    # Set flag that this is a cache hit
                    _cache_hit.set(True)

                    # Also set it directly on the Call object so it persists
                    try:
                        from weave.trace.context.call_context import get_current_call
                        current_call = get_current_call()
                        if current_call:
                            if current_call.summary is None:
                                current_call.summary = {}
                            current_call.summary["weave.cache_hit"] = True

                            # Print simple cache hit alert
                            from weave.trace import settings as trace_settings
                            if trace_settings.should_print_call_link():
                                logger.info(f"💾 cache hit for call {current_call.id}")
                    except Exception:
                        pass

                    # Re-wrap cached response if wrap function provided
                    if wrap_fn is not None:
                        try:
                            cached_response = wrap_fn(cached_response)
                        except Exception as e:
                            logger.debug(f"Response wrap failed: {e}, returning as-is")

                    return cached_response

            # Call original function
            response = fn(self, *args, **kwargs)

            # Cache the response
            if cache is not None:
                response_to_cache = response

                # Unwrap response if unwrap function provided
                if unwrap_fn is not None:
                    try:
                        response_to_cache = unwrap_fn(response)
                    except Exception as e:
                        logger.debug(f"Response unwrap failed: {e}, caching as-is")

                cache.set(integration_name, kwargs, response_to_cache)

            return response

        return cached_wrapper

    return decorator
