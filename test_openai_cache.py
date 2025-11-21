"""Test cache configuration with OpenAI."""
import time

import weave


def test_openai_cache_basic():
    """Test basic OpenAI caching."""
    from openai import OpenAI

    # Initialize weave with cache enabled
    weave.init(
        project_name="test-cache",
        settings={"cache_enabled": True}
    )

    client = OpenAI()

    print("\n=== Testing OpenAI Cache ===")

    # First call - should be slow (cache miss)
    print("\n1. First call (cache miss)...")
    start = time.time()
    response1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=5,
    )
    time1 = time.time() - start
    print(f"   Time: {time1:.3f}s")
    print(f"   Response: {response1.choices[0].message.content}")

    # Second call - should be fast (cache hit)
    print("\n2. Second call (cache hit)...")
    start = time.time()
    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=5,
    )
    time2 = time.time() - start
    print(f"   Time: {time2:.3f}s")
    print(f"   Response: {response2.choices[0].message.content}")
    speedup = time1/time2
    print(f"   Speedup: {speedup:.1f}x")

    # Verify responses are identical
    assert response1.choices[0].message.content == response2.choices[0].message.content
    # Verify cache speedup
    assert speedup >= 10, f"Cache should be 10x+ faster, got {speedup:.1f}x"

    # Disable cache
    print("\n3. Third call (cache disabled)...")
    weave.cache.disable()
    start = time.time()
    response3 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=5,
    )
    time3 = time.time() - start
    print(f"   Time: {time3:.3f}s")
    print(f"   Response: {response3.choices[0].message.content}")
    assert time3 > time2 * 5, "Should be slow when cache disabled"

    # Re-enable cache
    print("\n4. Fourth call (cache re-enabled, should hit)...")
    weave.cache.enable()
    start = time.time()
    response4 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=5,
    )
    time4 = time.time() - start
    print(f"   Time: {time4:.3f}s")
    print(f"   Response: {response4.choices[0].message.content}")
    speedup4 = time1/time4
    print(f"   Speedup: {speedup4:.1f}x")
    assert speedup4 >= 10, f"Cache should be 10x+ faster, got {speedup4:.1f}x"

    # Clear cache
    print("\n5. Clearing cache...")
    weave.cache.clear()

    # Fifth call - should be slow again (cache miss after clear)
    print("\n6. Fifth call (cache miss after clear)...")
    start = time.time()
    response5 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=5,
    )
    time5 = time.time() - start
    print(f"   Time: {time5:.3f}s")
    print(f"   Response: {response5.choices[0].message.content}")
    assert time5 > time2 * 5, "Should be slow after cache clear"

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_openai_cache_basic()
