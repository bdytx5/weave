"""
Test programmatic cache configuration via weave.init() settings.

Tests:
1. Enabling cache via weave.init(settings={"cache_enabled": True})
2. Filtering cache.clear() by provider (tests OpenAI + Anthropic)
3. Filtering cache.clear() by model pattern (tests gpt-4o* vs gpt-5)
4. Cache deterministic_only mode

This version uses native OpenAI and Anthropic libraries directly (no litellm).
"""

import os
import time
import weave
from openai import OpenAI
from anthropic import Anthropic

# Don't use env vars - we'll configure via settings
if "WEAVE_CACHE_ENABLED" in os.environ:
    del os.environ["WEAVE_CACHE_ENABLED"]

TEST_PROMPT = "Say hello in one word"


def measure_latency(func):
    """Measure execution time of a function"""
    start = time.time()
    result = func()
    elapsed = time.time() - start
    return result, elapsed


class TestCacheConfiguration:
    """Test cache configuration via weave.init() settings"""

    def test_cache_enable_disable(self):
        """Test weave.cache.enable() and weave.cache.disable()"""
        # Initialize without cache enabled
        weave.init("test_cache_enable_disable")

        # Create client AFTER weave.init() so weave can intercept calls
        client = OpenAI()

        # Manually enable cache
        weave.cache.enable()
        weave.cache.clear()

        def make_call():
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=10,
            )

        # First call - cache miss
        print("\n[Enable/Disable] First call (cache miss)...")
        _, latency1 = measure_latency(make_call)
        print(f"Latency: {latency1:.3f}s")

        # Second call - cache hit
        print("[Enable/Disable] Second call (cache hit)...")
        _, latency2 = measure_latency(make_call)
        print(f"Latency: {latency2:.3f}s")

        speedup = latency1 / latency2
        print(f"Speedup: {speedup:.1f}x")
        assert speedup >= 10, f"Cache should be 10x+ faster, got {speedup:.1f}x"

        # Now disable cache globally
        print("[Enable/Disable] Disabling cache globally...")
        weave.cache.disable()

        # Third call - should be slow again (no cache)
        print("[Enable/Disable] Third call (cache disabled)...")
        _, latency3 = measure_latency(make_call)
        print(f"Latency: {latency3:.3f}s")
        assert latency3 > latency2 * 5, "Should be slow when cache disabled"

        print("✅ Cache enable/disable test PASSED")
        weave.finish()

    def test_cache_enabled_via_settings(self):
        """Test enabling cache via weave.init(settings={"cache_enabled": True})"""
        # Initialize with cache enabled via settings
        weave.init(
            "test_cache_config_enabled",
            settings={"cache_enabled": True}
        )

        # Create client AFTER weave.init() so weave can intercept calls
        client = OpenAI()

        weave.cache.clear()

        def make_call():
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=10,
            )

        # First call - cache miss
        print("\n[Cache Enabled via Settings] First call (cache miss)...")
        _, latency1 = measure_latency(make_call)
        print(f"Latency: {latency1:.3f}s")

        # Second call - cache hit
        print("[Cache Enabled via Settings] Second call (cache hit)...")
        _, latency2 = measure_latency(make_call)
        print(f"Latency: {latency2:.3f}s")

        speedup = latency1 / latency2
        print(f"Speedup: {speedup:.1f}x")
        assert speedup >= 10, f"Cache should be 10x+ faster, got {speedup:.1f}x"

        print("✅ Cache enabled via settings test PASSED")
        weave.finish()

    def test_cache_clear_by_provider(self):
        """Test cache.clear(provider='...') to clear specific provider"""
        weave.init(
            "test_cache_clear_provider",
            settings={"cache_enabled": True}
        )

        # Create clients AFTER weave.init() so weave can intercept calls
        openai_client = OpenAI()
        anthropic_client = Anthropic()

        weave.cache.clear()

        def make_openai_call():
            return openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test OpenAI"}],
                max_tokens=5,
            )

        def make_anthropic_call():
            return anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                messages=[{"role": "user", "content": "Test Anthropic"}],
                max_tokens=5,
            )

        # Populate cache with OpenAI calls
        print("\n[Clear by Provider] Populating cache with OpenAI (first call)...")
        _, openai_latency1 = measure_latency(make_openai_call)
        print(f"OpenAI first call: {openai_latency1:.3f}s")

        # Populate cache with Anthropic calls
        print("[Clear by Provider] Populating cache with Anthropic (first call)...")
        _, anthropic_latency1 = measure_latency(make_anthropic_call)
        print(f"Anthropic first call: {anthropic_latency1:.3f}s")

        # Verify both are cached (fast)
        print("[Clear by Provider] Verifying both are cached...")
        _, openai_cached = measure_latency(make_openai_call)
        print(f"OpenAI cached: {openai_cached:.3f}s")
        _, anthropic_cached = measure_latency(make_anthropic_call)
        print(f"Anthropic cached: {anthropic_cached:.3f}s")

        assert openai_cached < openai_latency1 / 5, "OpenAI should be cached"
        assert anthropic_cached < anthropic_latency1 / 5, "Anthropic should be cached"

        # Clear only OpenAI cache
        print("[Clear by Provider] Clearing OpenAI cache only...")
        weave.cache.clear(provider="openai")

        # OpenAI should be slow again, Anthropic still fast
        print("[Clear by Provider] Testing after OpenAI clear...")
        _, openai_after_clear = measure_latency(make_openai_call)
        print(f"OpenAI after clear: {openai_after_clear:.3f}s")
        _, anthropic_still_cached = measure_latency(make_anthropic_call)
        print(f"Anthropic still cached: {anthropic_still_cached:.3f}s")

        assert openai_after_clear > openai_cached * 5, "OpenAI should NOT be cached"
        assert anthropic_still_cached < anthropic_latency1 / 5, "Anthropic should still be cached"

        # Clear Anthropic cache
        print("[Clear by Provider] Clearing Anthropic cache...")
        weave.cache.clear(provider="anthropic")

        # Anthropic should be slow now
        print("[Clear by Provider] Testing after Anthropic clear...")
        _, anthropic_after_clear = measure_latency(make_anthropic_call)
        print(f"Anthropic after clear: {anthropic_after_clear:.3f}s")

        assert anthropic_after_clear > anthropic_cached * 5, "Anthropic should NOT be cached"

        print("✅ Cache clear by provider test PASSED")
        weave.finish()

    def test_cache_clear_by_model(self):
        """Test cache.clear(model='...') to clear specific model pattern"""
        weave.init(
            "test_cache_clear_model",
            settings={"cache_enabled": True}
        )

        # Create client AFTER weave.init() so weave can intercept calls
        client = OpenAI()

        weave.cache.clear()

        def make_gpt4o_call():
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test GPT-4o-mini"}],
                max_tokens=5,
            )

        def make_gpt5_call():
            return client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": "Test GPT-5"}],
                max_completion_tokens=100,
            )

        # Populate cache with gpt-4o-mini
        print("\n[Clear by Model] Populating cache with gpt-4o-mini (first call)...")
        _, gpt4o_latency1 = measure_latency(make_gpt4o_call)
        print(f"gpt-4o-mini first call: {gpt4o_latency1:.3f}s")

        # Populate cache with gpt-5 (different model pattern)
        print("[Clear by Model] Populating cache with gpt-5 (first call)...")
        _, gpt5_latency1 = measure_latency(make_gpt5_call)
        print(f"gpt-5 first call: {gpt5_latency1:.3f}s")

        # Verify both are cached
        print("[Clear by Model] Verifying both are cached...")
        _, gpt4o_cached = measure_latency(make_gpt4o_call)
        print(f"gpt-4o-mini cached: {gpt4o_cached:.3f}s")
        _, gpt5_cached = measure_latency(make_gpt5_call)
        print(f"gpt-5 cached: {gpt5_cached:.3f}s")

        assert gpt4o_cached < gpt4o_latency1 / 5, "gpt-4o-mini should be cached"
        assert gpt5_cached < gpt5_latency1 / 5, "gpt-5 should be cached"

        # Clear only gpt-4o* models (should clear gpt-4o-mini, but NOT gpt-5)
        print("[Clear by Model] Clearing gpt-4o* cache...")
        weave.cache.clear(model="gpt-4o*")

        # gpt-4o-mini should be slow again, gpt-5 still fast
        print("[Clear by Model] Testing after gpt-4o* clear...")
        _, gpt4o_after_clear = measure_latency(make_gpt4o_call)
        print(f"gpt-4o-mini after clear: {gpt4o_after_clear:.3f}s")
        _, gpt5_still_cached = measure_latency(make_gpt5_call)
        print(f"gpt-5 still cached: {gpt5_still_cached:.3f}s")

        assert gpt4o_after_clear > gpt4o_cached * 5, "gpt-4o-mini should NOT be cached"
        assert gpt5_still_cached < gpt5_latency1 / 5, "gpt-5 should still be cached"

        print("✅ Cache clear by model test PASSED")
        weave.finish()

    def test_cache_deterministic_only(self):
        """Test cache_deterministic_only mode (only cache temp=0)"""
        weave.init(
            "test_cache_deterministic",
            settings={
                "cache_enabled": True,
                "cache_deterministic_only": True
            }
        )

        # Create client AFTER weave.init() so weave can intercept calls
        client = OpenAI()

        weave.cache.clear()

        def make_nondeterministic_call():
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Non-deterministic"}],
                max_tokens=5,
                temperature=0.7,
            )

        def make_deterministic_call():
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Deterministic"}],
                max_tokens=5,
                temperature=0,
            )

        # Call with temperature > 0 (should NOT be cached)
        print("\n[Deterministic Only] First call with temperature=0.7 (should NOT cache)...")
        _, nondeterministic_latency1 = measure_latency(make_nondeterministic_call)
        print(f"Non-deterministic first call: {nondeterministic_latency1:.3f}s")

        # Second call with same params - should STILL be slow (not cached)
        print("[Deterministic Only] Second call with temperature=0.7 (should NOT cache)...")
        _, nondeterministic_latency2 = measure_latency(make_nondeterministic_call)
        print(f"Non-deterministic second call: {nondeterministic_latency2:.3f}s")

        # Should NOT be significantly faster (not cached)
        assert nondeterministic_latency2 > nondeterministic_latency1 / 10, "Should NOT be cached"

        # Call with temperature = 0 (SHOULD be cached)
        print("[Deterministic Only] First call with temperature=0 (SHOULD cache)...")
        _, deterministic_latency1 = measure_latency(make_deterministic_call)
        print(f"Deterministic first call: {deterministic_latency1:.3f}s")

        # Second call - should be fast (cached)
        print("[Deterministic Only] Second call with temperature=0 (cached)...")
        _, deterministic_latency2 = measure_latency(make_deterministic_call)
        print(f"Deterministic second call: {deterministic_latency2:.3f}s")

        speedup = deterministic_latency1 / deterministic_latency2
        print(f"Speedup: {speedup:.1f}x")
        assert speedup >= 10, f"Should be cached (10x+ faster), got {speedup:.1f}x"

        print("✅ Cache deterministic_only test PASSED")
        weave.finish()


def run_all_tests():
    """Run all cache configuration tests"""
    print("=" * 80)
    print("CACHE CONFIGURATION TESTS (OpenAI)")
    print("=" * 80)

    test_class = TestCacheConfiguration()

    tests = [
        ("test_cache_enable_disable", test_class.test_cache_enable_disable),
        ("test_cache_enabled_via_settings", test_class.test_cache_enabled_via_settings),
        ("test_cache_clear_by_provider", test_class.test_cache_clear_by_provider),
        ("test_cache_clear_by_model", test_class.test_cache_clear_by_model),
        ("test_cache_deterministic_only", test_class.test_cache_deterministic_only),
    ]

    results = {"passed": [], "failed": []}

    for test_name, test_func in tests:
        print(f"\n{'=' * 80}")
        print(f"Running {test_name}")
        print(f"{'=' * 80}")

        try:
            test_func()
            results["passed"].append(test_name)
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            results["failed"].append(test_name)
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results["failed"].append(test_name)
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Passed: {len(results['passed'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    print("=" * 80)

    if results["passed"]:
        print("\nPassed tests:")
        for test in results["passed"]:
            print(f"  ✅ {test}")

    if results["failed"]:
        print("\nFailed tests:")
        for test in results["failed"]:
            print(f"  ❌ {test}")

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()