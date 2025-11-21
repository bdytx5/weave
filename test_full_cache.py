"""Test script to verify full caching (100%) works by default."""

import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from weave.integrations.cache import LLMCache
import weave


def test_full_caching():
    """Test that by default, ALL requests are cached (not just temp=0)."""
    print("Testing FULL caching (default behavior)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create cache with default settings (deterministic_only=False)
        cache = LLMCache(directory=tmpdir)

        # Test 1: temperature=0 should be cached
        kwargs1 = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
        }
        response1 = {"response": "temp 0"}
        cache.set("openai", kwargs1, response1)
        assert cache.get("openai", kwargs1) == response1
        print("✓ temperature=0 is cached")

        # Test 2: temperature=0.7 SHOULD ALSO be cached (full caching mode)
        kwargs2 = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
        }
        response2 = {"response": "temp 0.7"}
        cache.set("openai", kwargs2, response2)
        assert cache.get("openai", kwargs2) == response2
        print("✓ temperature=0.7 is cached (FULL CACHING MODE)")

        # Test 3: temperature=1.0 should also be cached
        kwargs3 = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 1.0,
        }
        response3 = {"response": "temp 1.0"}
        cache.set("openai", kwargs3, response3)
        assert cache.get("openai", kwargs3) == response3
        print("✓ temperature=1.0 is cached (FULL CACHING MODE)")

        # Test 4: streaming requests should NOT be cached
        kwargs4 = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
            "stream": True,
        }
        response4 = {"response": "streaming"}
        cache.set("openai", kwargs4, response4)
        assert cache.get("openai", kwargs4) is None
        print("✓ streaming requests are NOT cached")

    print("\n✅ FULL CACHING MODE WORKS! (Everything cached except streams)")


def test_deterministic_only_mode():
    """Test deterministic_only mode."""
    print("\nTesting DETERMINISTIC-ONLY mode...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create cache with deterministic_only=True
        cache = LLMCache(directory=tmpdir, deterministic_only=True)

        # Test 1: temperature=0 should be cached
        kwargs1 = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
        }
        response1 = {"response": "temp 0"}
        cache.set("openai", kwargs1, response1)
        assert cache.get("openai", kwargs1) == response1
        print("✓ temperature=0 is cached")

        # Test 2: temperature=0.7 should NOT be cached
        kwargs2 = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
        }
        response2 = {"response": "temp 0.7"}
        cache.set("openai", kwargs2, response2)
        assert cache.get("openai", kwargs2) is None
        print("✓ temperature=0.7 is NOT cached (deterministic-only mode)")

    print("\n✅ DETERMINISTIC-ONLY MODE WORKS!")


def test_weave_init_full_caching():
    """Test weave.init with full caching."""
    print("\nTesting weave.init() with FULL caching...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize with full caching (default)
        client = weave.init(
            "test-full-cache",
            settings={
                "cache_enabled": True,
                "cache_dir": tmpdir,
                # cache_deterministic_only defaults to False (full caching)
            },
        )

        from weave.integrations.cache import get_global_cache

        cache = get_global_cache()
        assert cache is not None
        assert cache._deterministic_only == False  # Full caching mode
        print("✓ Cache initialized in FULL caching mode by default")


def test_weave_init_deterministic_only():
    """Test weave.init with deterministic-only mode."""
    print("\nTesting weave.init() with DETERMINISTIC-ONLY mode...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize with deterministic-only mode
        client = weave.init(
            "test-det-cache",
            settings={
                "cache_enabled": True,
                "cache_dir": tmpdir,
                "cache_deterministic_only": True,
            },
        )

        from weave.integrations.cache import get_global_cache, set_global_cache

        cache = get_global_cache()
        assert cache is not None
        assert cache._deterministic_only == True
        print("✓ Cache initialized in DETERMINISTIC-ONLY mode")

        # Clean up
        set_global_cache(None)


def test_skip_cache_override():
    """Test _weave_skip_cache override."""
    print("\nTesting _weave_skip_cache override...")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = LLMCache(directory=tmpdir)

        # Even with temp=0, skip cache should prevent caching
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
            "_weave_skip_cache": True,
        }
        response = {"response": "skip"}
        cache.set("openai", kwargs, response)
        assert cache.get("openai", kwargs) is None
        print("✓ _weave_skip_cache prevents caching")


if __name__ == "__main__":
    try:
        test_full_caching()
        test_deterministic_only_mode()
        test_weave_init_full_caching()
        test_weave_init_deterministic_only()
        test_skip_cache_override()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("\nSUMMARY:")
        print("  ✓ Full caching (100%) is the DEFAULT")
        print("  ✓ Deterministic-only mode available via setting")
        print("  ✓ _weave_skip_cache override works")
        print("  ✓ Streaming requests never cached")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
