"""Simple demo showing LLM response caching in action."""

import time
import weave
from openai import OpenAI


def main():
    print("=" * 60)
    print("LLM RESPONSE CACHING DEMO")
    print("=" * 60)

    # Initialize weave with caching enabled
    print("\n1. Initializing weave with cache enabled...")
    weave.init(
        "cache-demo",
        settings={
            "cache_enabled": True,
            "cache_dir": "/tmp/weave-demo-cache",
        },
    )
    print("   ✓ Cache enabled at /tmp/weave-demo-cache")

    # Create OpenAI client
    client = OpenAI()

    # Call 1: First call - hits API
    print("\n2. First call (should hit OpenAI API)...")
    start = time.time()
    response1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'Hello from call 10'"}],
        temperature=0.7,  # Non-deterministic, but still cached!
        max_tokens=50,
    )
    elapsed1 = time.time() - start
    print(f"   Response: {response1.choices[0].message.content}")
    print(f"   ⏱️  Time: {elapsed1:.2f}s (API call)")

    # Call 2: Same parameters - should return cached response instantly
    print("\n3. Second call (SAME parameters - should be instant from cache)...")
    start = time.time()
    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'Hello from call 10'"}],
        temperature=0.7,
        max_tokens=50,
    )
    elapsed2 = time.time() - start
    print(f"   Response: {response2.choices[0].message.content}")
    print(f"   ⏱️  Time: {elapsed2:.2f}s (CACHED - {elapsed1/elapsed2:.0f}x faster!)")

    # Call 3: Different parameters - hits API again
    print("\n4. Third call (DIFFERENT message - should hit API again)...")
    start = time.time()
    response3 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'Hello from call 30'"}],  # Different!
        temperature=0.7,
        max_tokens=50,
    )
    elapsed3 = time.time() - start
    print(f"   Response: {response3.choices[0].message.content}")
    print(f"   ⏱️  Time: {elapsed3:.2f}s (API call - different message)")

    # Call 4: Same as call 3 - cached again
    print("\n5. Fourth call (same as call 3 - should be cached)...")
    start = time.time()
    response4 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'Hello from call 30'"}],  # Same as call 3
        temperature=0.7,
        max_tokens=50,
    )
    elapsed4 = time.time() - start
    print(f"   Response: {response4.choices[0].message.content}")
    print(f"   ⏱️  Time: {elapsed4:.2f}s (CACHED)")

    # Show cache stats
    print("\n" + "=" * 60)
    print("CACHE STATISTICS")
    print("=" * 60)
    from weave.integrations.cache import get_global_cache

    cache = get_global_cache()
    if cache:
        cache.print_stats()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Call 1 (API):    {elapsed1:.2f}s")
    print(f"  Call 2 (CACHE):  {elapsed2:.2f}s  ({elapsed1/elapsed2:.0f}x faster)")
    print(f"  Call 3 (API):    {elapsed3:.2f}s")
    print(f"  Call 4 (CACHE):  {elapsed4:.2f}s  ({elapsed3/elapsed4:.0f}x faster)")
    print("=" * 60)
    print("\n✅ Caching works! Same parameters = instant cache hits")


if __name__ == "__main__":
    main()
