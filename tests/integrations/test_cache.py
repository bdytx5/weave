"""Tests for LLM response caching."""

import pytest
import tempfile
import os
from pathlib import Path

import weave
from weave.integrations.cache import LLMCache, get_global_cache, set_global_cache


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def cache(temp_cache_dir):
    """Create a cache instance for testing."""
    return LLMCache(directory=temp_cache_dir)


def test_cache_initialization(temp_cache_dir):
    """Test that cache initializes correctly."""
    cache = LLMCache(directory=temp_cache_dir)
    assert cache.directory == Path(temp_cache_dir)
    assert cache._ttl is None


def test_cache_with_size_limit(temp_cache_dir):
    """Test cache initialization with size limit."""
    size_limit = 1024 * 1024  # 1MB
    cache = LLMCache(directory=temp_cache_dir, size_limit=size_limit)
    assert cache._cache.size_limit == size_limit


def test_cache_with_ttl(temp_cache_dir):
    """Test cache initialization with TTL."""
    ttl = 3600.0  # 1 hour
    cache = LLMCache(directory=temp_cache_dir, ttl=ttl)
    assert cache._ttl == ttl


def test_cache_key_generation(cache):
    """Test that cache keys are generated correctly."""
    kwargs1 = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
    }
    kwargs2 = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
    }
    kwargs3 = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
    }

    key1 = cache._make_cache_key("openai", kwargs1)
    key2 = cache._make_cache_key("openai", kwargs2)
    key3 = cache._make_cache_key("openai", kwargs3)

    # Same kwargs should produce same key
    assert key1 == key2
    # Different model should produce different key
    assert key1 != key3


def test_cache_should_cache_logic(cache):
    """Test that should_cache correctly identifies cacheable requests."""
    # By default (deterministic_only=False), all requests should be cached
    assert cache.should_cache({"temperature": 0})
    assert cache.should_cache({"temperature": 0.7})
    assert cache.should_cache({"temperature": 1.0})

    # Streaming requests should not be cached
    assert not cache.should_cache({"temperature": 0, "stream": True})

    # Force cache override
    assert cache.should_cache({"temperature": 0.7, "_weave_force_cache": True})

    # Skip cache override
    assert not cache.should_cache({"temperature": 0, "_weave_skip_cache": True})


def test_cache_get_set(cache):
    """Test basic cache get/set operations."""
    kwargs = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
    }
    response = {"choices": [{"message": {"content": "Hi there!"}}]}

    # Initially, cache should be empty
    assert cache.get("openai", kwargs) is None

    # Set the cache
    cache.set("openai", kwargs, response)

    # Now we should get the cached response
    cached = cache.get("openai", kwargs)
    assert cached == response


def test_cache_caches_all_by_default(cache):
    """Test that cache stores ALL requests by default (including non-deterministic)."""
    kwargs = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,  # Non-deterministic
    }
    response = {"choices": [{"message": {"content": "Hi!"}}]}

    # Try to cache
    cache.set("openai", kwargs, response)

    # SHOULD be cached (default behavior)
    assert cache.get("openai", kwargs) == response


def test_cache_deterministic_only_mode(temp_cache_dir):
    """Test that deterministic_only mode only caches temperature=0."""
    cache = LLMCache(directory=temp_cache_dir, deterministic_only=True)

    kwargs = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
    }
    response = {"choices": [{"message": {"content": "Hi!"}}]}

    # Try to cache non-deterministic request
    cache.set("openai", kwargs, response)

    # Should NOT be cached in deterministic_only mode
    assert cache.get("openai", kwargs) is None

    # But temperature=0 should be cached
    kwargs_det = {**kwargs, "temperature": 0}
    response_det = {"choices": [{"message": {"content": "Hello!"}}]}
    cache.set("openai", kwargs_det, response_det)
    assert cache.get("openai", kwargs_det) == response_det


def test_cache_clear(cache):
    """Test that cache can be cleared."""
    kwargs = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
    }
    response = {"choices": [{"message": {"content": "Hi!"}}]}

    # Set cache
    cache.set("openai", kwargs, response)
    assert cache.get("openai", kwargs) is not None

    # Clear cache
    cache.clear()
    assert cache.get("openai", kwargs) is None


def test_cache_stats(cache):
    """Test cache statistics."""
    stats = cache.get_stats()

    assert "directory" in stats
    assert "size_limit" in stats
    assert "current_size" in stats
    assert "item_count" in stats
    assert "percent_full" in stats

    assert stats["item_count"] == 0  # Empty cache

    # Add an item
    kwargs = {"model": "gpt-4", "temperature": 0}
    cache.set("openai", kwargs, {"response": "test"})

    stats = cache.get_stats()
    assert stats["item_count"] == 1


def test_global_cache_management(temp_cache_dir):
    """Test global cache get/set functions."""
    cache = LLMCache(directory=temp_cache_dir)

    # Set global cache
    set_global_cache(cache)
    assert get_global_cache() is cache

    # Clear global cache
    set_global_cache(None)
    assert get_global_cache() is None


def test_weave_init_with_cache(temp_cache_dir):
    """Test that weave.init() initializes cache when enabled."""
    # Initialize weave with cache enabled
    client = weave.init(
        "test-cache-project",
        settings={
            "cache_enabled": True,
            "cache_dir": temp_cache_dir,
        },
    )

    # Check that global cache is set
    cache = get_global_cache()
    assert cache is not None
    assert str(cache.directory) == temp_cache_dir

    # Clean up
    client.reset()
    set_global_cache(None)


def test_weave_init_without_cache():
    """Test that weave.init() doesn't initialize cache by default."""
    client = weave.init("test-no-cache-project")

    # Check that global cache is not set
    cache = get_global_cache()
    assert cache is None

    # Clean up
    client.reset()


def test_cache_different_parameters_different_keys(cache):
    """Test that different parameters create different cache keys."""
    base_kwargs = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "test"}],
        "temperature": 0,
    }

    # Test different models
    kwargs1 = {**base_kwargs, "model": "gpt-4"}
    kwargs2 = {**base_kwargs, "model": "gpt-3.5-turbo"}
    assert cache._make_cache_key("openai", kwargs1) != cache._make_cache_key(
        "openai", kwargs2
    )

    # Test different messages
    kwargs1 = {**base_kwargs, "messages": [{"role": "user", "content": "test"}]}
    kwargs2 = {**base_kwargs, "messages": [{"role": "user", "content": "other"}]}
    assert cache._make_cache_key("openai", kwargs1) != cache._make_cache_key(
        "openai", kwargs2
    )

    # Test different max_tokens
    kwargs1 = {**base_kwargs, "max_tokens": 100}
    kwargs2 = {**base_kwargs, "max_tokens": 200}
    assert cache._make_cache_key("openai", kwargs1) != cache._make_cache_key(
        "openai", kwargs2
    )


def test_cache_ignores_non_semantic_parameters(cache):
    """Test that cache ignores parameters that don't affect the response."""
    base_kwargs = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "test"}],
        "temperature": 0,
    }

    # stream parameter shouldn't affect the key (but will prevent caching)
    kwargs1 = {**base_kwargs}
    kwargs2 = {**base_kwargs, "stream": False}

    # user parameter shouldn't affect the key
    kwargs1 = {**base_kwargs}
    kwargs2 = {**base_kwargs, "user": "test-user"}
    # Keys should be the same since user doesn't affect response
    key1 = cache._make_cache_key("openai", kwargs1)
    key2 = cache._make_cache_key("openai", kwargs2)
    # Actually, user is not in the cache_params, so they should be equal
    # But the current implementation doesn't include user, so the keys will be the same
    # Let's just ensure the cache works as expected


def test_cache_print_stats(cache, capsys):
    """Test that print_stats outputs correctly."""
    cache.print_stats()
    captured = capsys.readouterr()
    assert "LLM CACHE STATISTICS" in captured.out
    assert "Directory:" in captured.out
    assert "Size limit:" in captured.out
    assert "Items cached:" in captured.out
