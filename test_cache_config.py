"""
Test programmatic cache configuration via weave.init() settings.

Tests:
1. Enabling cache via weave.init(settings={"cache_enabled": True})
2. Filtering cache.clear() by provider
3. Filtering cache.clear() by model pattern
4. Cache deterministic_only mode
"""

import os
import time
import weave

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
        from litellm import completion

        # Initialize without cache enabled
        weave.init("test_cache_enable_disable")

        # Manually enable cache
        weave.cache.enable()
        weave.cache.clear()

        def make_call():
            return completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=10,
            )

        # First call - cache miss
        print("\n[Enable/Disable] First call (cache miss)...")
        response1, latency1 = measure_latency(make_call)
        print(f"Latency: {latency1:.3f}s")

        # Second call - cache hit
        print("[Enable/Disable] Second call (cache hit)...")
        response2, latency2 = measure_latency(make_call)
        print(f"Latency: {latency2:.3f}s")

        speedup = latency1 / latency2
        print(f"Speedup: {speedup:.1f}x")
        assert speedup >= 10, f"Cache should be 10x+ faster, got {speedup:.1f}x"

        # Now disable cache globally
        print("[Enable/Disable] Disabling cache globally...")
        weave.cache.disable()

        # Third call - should be slow again (no cache)
        print("[Enable/Disable] Third call (cache disabled)...")
        response3, latency3 = measure_latency(make_call)
        print(f"Latency: {latency3:.3f}s")
        assert latency3 > latency2 * 5, "Should be slow when cache disabled"

        print("✅ Cache enable/disable test PASSED")
        weave.finish()

    def test_cache_enabled_via_settings(self):
        """Test enabling cache via weave.init(settings={"cache_enabled": True})"""
        from litellm import completion

        # Initialize with cache enabled via settings
        weave.init(
            "test_cache_config_enabled",
            settings={"cache_enabled": True}
        )

        weave.cache.clear()

        def make_call():
            return completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=10,
            )

        # First call - cache miss
        print("\n[Cache Enabled via Settings] First call (cache miss)...")
        response1, latency1 = measure_latency(make_call)
        print(f"Latency: {latency1:.3f}s")

        # Second call - cache hit
        print("[Cache Enabled via Settings] Second call (cache hit)...")
        response2, latency2 = measure_latency(make_call)
        print(f"Latency: {latency2:.3f}s")

        speedup = latency1 / latency2
        print(f"Speedup: {speedup:.1f}x")
        assert speedup >= 10, f"Cache should be 10x+ faster, got {speedup:.1f}x"

        print("✅ Cache enabled via settings test PASSED")
        weave.finish()

    def test_cache_clear_by_provider(self):
        """Test cache.clear(provider='...') to clear specific provider"""
        from litellm import completion

        weave.init(
            "test_cache_clear_provider",
            settings={"cache_enabled": True}
        )

        weave.cache.clear()

        def make_call():
            return completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test 1"}],
                max_tokens=5,
            )

        # Populate cache with OpenAI calls (via LiteLLM)
        print("\n[Clear by Provider] Populating cache with OpenAI (first call)...")
        _, latency1 = measure_latency(make_call)
        print(f"First call: {latency1:.3f}s")

        # Verify it's cached
        print("[Clear by Provider] Verifying cache (second call)...")
        _, latency2 = measure_latency(make_call)
        print(f"Second call (cached): {latency2:.3f}s")
        assert latency2 < latency1 / 5, "Should be cached"

        # Clear only OpenAI cache
        print("[Clear by Provider] Clearing OpenAI cache...")
        weave.cache.clear(provider="openai")

        # Should be slow again
        print("[Clear by Provider] Testing after clear...")
        _, latency3 = measure_latency(make_call)
        print(f"After clear: {latency3:.3f}s")
        assert latency3 > latency2 * 5, "Should NOT be cached"

        print("✅ Cache clear by provider test PASSED")
        weave.finish()

    def test_cache_clear_by_model(self):
        """Test cache.clear(model='...') to clear specific model pattern"""
        from litellm import completion

        weave.init(
            "test_cache_clear_model",
            settings={"cache_enabled": True}
        )

        weave.cache.clear()

        def make_call():
            return completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test GPT-4o"}],
                max_tokens=5,
            )

        # Populate cache with different models
        print("\n[Clear by Model] Populating cache with gpt-4o-mini (first call)...")
        _, latency1 = measure_latency(make_call)
        print(f"First call: {latency1:.3f}s")

        # Verify it's cached
        print("[Clear by Model] Verifying cache (second call)...")
        _, latency2 = measure_latency(make_call)
        print(f"Second call (cached): {latency2:.3f}s")
        assert latency2 < latency1 / 5, "Should be cached"

        # Clear only gpt-4o* models
        print("[Clear by Model] Clearing gpt-4o* cache...")
        weave.cache.clear(model="gpt-4o*")

        # Should be slow again
        print("[Clear by Model] Testing after clear...")
        _, latency3 = measure_latency(make_call)
        print(f"After clear: {latency3:.3f}s")
        assert latency3 > latency2 * 5, "Should NOT be cached"

        print("✅ Cache clear by model test PASSED")
        weave.finish()

    def test_cache_deterministic_only(self):
        """Test cache_deterministic_only mode (only cache temp=0)"""
        from litellm import completion

        weave.init(
            "test_cache_deterministic",
            settings={
                "cache_enabled": True,
                "cache_deterministic_only": True
            }
        )

        weave.cache.clear()

        def make_nondeterministic_call():
            return completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Non-deterministic"}],
                max_tokens=5,
                temperature=0.7,
            )

        def make_deterministic_call():
            return completion(
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
    print("CACHE CONFIGURATION TESTS")
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
