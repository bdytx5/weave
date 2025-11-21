"""
Comprehensive Demo of Weave LLM Caching System

This demo showcases all the caching features available in Weave for LLM providers:
- OpenAI
- Anthropic
- Google (Gemini)
- Cerebras
- Hugging Face

Features demonstrated:
1. Enable/disable caching via weave.init() settings
2. Enable/disable caching programmatically
3. Temporary cache disable with context manager
4. Cache statistics and monitoring
5. Cache clearing (all, by provider, by model pattern)
6. Deterministic-only caching mode
7. Per-request cache control
8. Cache performance measurement
"""

import os
import time
import weave
from openai import OpenAI
from anthropic import Anthropic

try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ============================================================================
# DEMO 1: Basic Cache Enable/Disable via Settings
# ============================================================================
def demo_basic_cache_enable():
    """Show how to enable caching via weave.init() settings"""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Cache Enable/Disable via Settings")
    print("=" * 80)

    # Enable cache via weave.init() settings
    weave.init(
        "demo_basic_cache",
        settings={
            "cache_enabled": True,
            "cache_size_limit": 1024 * 1024 * 100,  # 100 MB
            "cache_ttl": 3600,  # 1 hour TTL (optional)
        }
    )

    client = OpenAI()

    # Clear any existing cache
    weave.cache.clear()
    print("\nCache cleared. Starting fresh...\n")

    def make_request():
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}],
            max_tokens=10,
            temperature=0,  # Deterministic
        )

    # First request - cache MISS (slow)
    print("Request 1 (cache MISS - should be slow)...")
    start = time.time()
    response1 = make_request()
    latency1 = time.time() - start
    print(f"Response: {response1.choices[0].message.content}")
    print(f"Latency: {latency1:.3f}s")

    # Second request - cache HIT (fast!)
    print("\nRequest 2 (cache HIT - should be FAST!)...")
    start = time.time()
    response2 = make_request()
    latency2 = time.time() - start
    print(f"Response: {response2.choices[0].message.content}")
    print(f"Latency: {latency2:.3f}s")

    speedup = latency1 / latency2
    print(f"\nSpeedup from cache: {speedup:.1f}x faster!")
    print(f"Cache saved ~{(latency1 - latency2):.3f}s")

    # Show cache stats
    stats = weave.cache.stats()
    print(f"\nCache Statistics:")
    print(f"  Items cached: {stats['item_count']}")
    print(f"  Cache size: {stats['current_size']:,} bytes")

    weave.finish()


# ============================================================================
# DEMO 2: Programmatic Cache Enable/Disable
# ============================================================================
def demo_programmatic_control():
    """Show how to enable/disable cache programmatically"""
    print("\n" + "=" * 80)
    print("DEMO 2: Programmatic Cache Control")
    print("=" * 80)

    # Initialize WITHOUT cache enabled
    weave.init("demo_programmatic")

    client = OpenAI()

    # Manually enable cache
    print("\nManually enabling cache with weave.cache.enable()...")
    weave.cache.enable()
    weave.cache.clear()

    def make_request():
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello in one word"}],
            max_tokens=5,
            temperature=0,
        )

    # Request with cache enabled
    print("\nRequest 1 (cache enabled)...")
    start = time.time()
    make_request()
    latency1 = time.time() - start
    print(f"Latency: {latency1:.3f}s")

    print("\nRequest 2 (cache HIT)...")
    start = time.time()
    make_request()
    latency2 = time.time() - start
    print(f"Latency: {latency2:.3f}s (cached - {latency1/latency2:.1f}x faster)")

    # Disable cache globally
    print("\nDisabling cache with weave.cache.disable()...")
    weave.cache.disable()

    print("\nRequest 3 (cache disabled - should be slow again)...")
    start = time.time()
    make_request()
    latency3 = time.time() - start
    print(f"Latency: {latency3:.3f}s (no cache)")

    weave.finish()


# ============================================================================
# DEMO 3: Context Manager for Temporary Cache Disable
# ============================================================================
def demo_context_manager():
    """Show how to temporarily disable cache with context manager"""
    print("\n" + "=" * 80)
    print("DEMO 3: Temporary Cache Disable with Context Manager")
    print("=" * 80)

    weave.init("demo_context_manager", settings={"cache_enabled": True})

    client = OpenAI()
    weave.cache.clear()

    def make_request():
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=10,
            temperature=0,
        )

    # Request WITH cache
    print("\nRequest 1 (cache enabled)...")
    start = time.time()
    make_request()
    latency1 = time.time() - start
    print(f"Latency: {latency1:.3f}s")

    print("\nRequest 2 (cache HIT)...")
    start = time.time()
    make_request()
    latency2 = time.time() - start
    print(f"Latency: {latency2:.3f}s (cached)")

    # Temporarily disable cache with context manager
    print("\nUsing context manager to temporarily disable cache...")
    print("with weave.disable_cache():")
    with weave.disable_cache():
        print("  Request 3 (cache temporarily disabled)...")
        start = time.time()
        make_request()
        latency3 = time.time() - start
        print(f"  Latency: {latency3:.3f}s (no cache inside context)")

    # Cache automatically re-enabled after context
    print("\nRequest 4 (cache re-enabled after context)...")
    start = time.time()
    make_request()
    latency4 = time.time() - start
    print(f"Latency: {latency4:.3f}s (cached again)")

    weave.finish()


# ============================================================================
# DEMO 4: Multi-Provider Caching
# ============================================================================
def demo_multi_provider():
    """Show caching works across different LLM providers"""
    print("\n" + "=" * 80)
    print("DEMO 4: Multi-Provider Caching (OpenAI + Anthropic)")
    print("=" * 80)

    weave.init("demo_multi_provider", settings={"cache_enabled": True})

    openai_client = OpenAI()
    anthropic_client = Anthropic()

    weave.cache.clear()
    print("\nCache cleared. Testing multiple providers...\n")

    # Test OpenAI
    print("=" * 40)
    print("Testing OpenAI Cache")
    print("=" * 40)

    def openai_request():
        return openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'OpenAI' only"}],
            max_tokens=5,
            temperature=0,
        )

    print("\nOpenAI Request 1 (cache MISS)...")
    start = time.time()
    resp1 = openai_request()
    latency1 = time.time() - start
    print(f"Response: {resp1.choices[0].message.content}")
    print(f"Latency: {latency1:.3f}s")

    print("\nOpenAI Request 2 (cache HIT)...")
    start = time.time()
    resp2 = openai_request()
    latency2 = time.time() - start
    print(f"Response: {resp2.choices[0].message.content}")
    print(f"Latency: {latency2:.3f}s ({latency1/latency2:.1f}x faster)")

    # Test Anthropic
    print("\n" + "=" * 40)
    print("Testing Anthropic Cache")
    print("=" * 40)

    def anthropic_request():
        return anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Say 'Anthropic' only"}],
            max_tokens=5,
        )

    print("\nAnthropic Request 1 (cache MISS)...")
    start = time.time()
    resp3 = anthropic_request()
    latency3 = time.time() - start
    print(f"Response: {resp3.content[0].text}")
    print(f"Latency: {latency3:.3f}s")

    print("\nAnthropic Request 2 (cache HIT)...")
    start = time.time()
    resp4 = anthropic_request()
    latency4 = time.time() - start
    print(f"Response: {resp4.content[0].text}")
    print(f"Latency: {latency4:.3f}s ({latency3/latency4:.1f}x faster)")

    # Show cache stats
    stats = weave.cache.stats()
    print(f"\nCache now contains {stats['item_count']} items")
    print("(includes both OpenAI and Anthropic responses)")

    weave.finish()


# ============================================================================
# DEMO 5: Selective Cache Clearing
# ============================================================================
def demo_selective_clearing():
    """Show how to clear cache selectively by provider or model"""
    print("\n" + "=" * 80)
    print("DEMO 5: Selective Cache Clearing")
    print("=" * 80)

    weave.init("demo_selective_clear", settings={"cache_enabled": True})

    openai_client = OpenAI()
    anthropic_client = Anthropic()

    weave.cache.clear()
    print("\nPopulating cache with multiple providers and models...\n")

    # Populate with different providers and models
    print("Adding OpenAI gpt-4o-mini to cache...")
    openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test 1"}],
        max_tokens=5,
        temperature=0,
    )

    print("Adding OpenAI gpt-3.5-turbo to cache...")
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test 2"}],
        max_tokens=5,
        temperature=0,
    )

    print("Adding Anthropic Claude to cache...")
    anthropic_client.messages.create(
        model="claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": "Test 3"}],
        max_tokens=5,
    )

    stats = weave.cache.stats()
    print(f"\nTotal cache items: {stats['item_count']}")

    # Clear by provider
    print("\n" + "=" * 40)
    print("Clearing OpenAI cache only...")
    print("=" * 40)
    cleared = weave.cache.clear(provider="openai")
    print(f"Cleared {cleared} OpenAI entries")

    stats = weave.cache.stats()
    print(f"Remaining cache items: {stats['item_count']}")
    print("(Anthropic cache should still exist)")

    # Repopulate for model pattern demo
    print("\n" + "=" * 40)
    print("Repopulating for model pattern demo...")
    print("=" * 40)
    weave.cache.clear()

    openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "GPT-4o test"}],
        max_tokens=5,
        temperature=0,
    )

    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "GPT-3.5 test"}],
        max_tokens=5,
        temperature=0,
    )

    stats = weave.cache.stats()
    print(f"Total cache items: {stats['item_count']}")

    # Clear by model pattern (wildcard)
    print("\n" + "=" * 40)
    print("Clearing gpt-4o* models only...")
    print("=" * 40)
    cleared = weave.cache.clear(model="gpt-4o*")
    print(f"Cleared {cleared} entries matching 'gpt-4o*'")

    stats = weave.cache.stats()
    print(f"Remaining cache items: {stats['item_count']}")
    print("(gpt-3.5-turbo cache should still exist)")

    weave.finish()


# ============================================================================
# DEMO 6: Deterministic-Only Caching Mode
# ============================================================================
def demo_deterministic_only():
    """Show deterministic-only caching mode (only cache temperature=0)"""
    print("\n" + "=" * 80)
    print("DEMO 6: Deterministic-Only Caching Mode")
    print("=" * 80)

    weave.init(
        "demo_deterministic",
        settings={
            "cache_enabled": True,
            "cache_deterministic_only": True,  # Only cache deterministic requests
        }
    )

    client = OpenAI()
    weave.cache.clear()

    print("\nCache mode: deterministic_only=True")
    print("(only requests with temperature=0 will be cached)\n")

    # Non-deterministic request (temperature > 0) - should NOT cache
    print("=" * 40)
    print("Request with temperature=0.7 (non-deterministic)")
    print("=" * 40)

    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Random response"}],
        max_tokens=10,
        temperature=0.7,  # Non-deterministic
    )

    stats = weave.cache.stats()
    print(f"\nCache items: {stats['item_count']}")
    print("Expected: 0 (non-deterministic request not cached)")

    # Deterministic request (temperature = 0) - SHOULD cache
    print("\n" + "=" * 40)
    print("Request with temperature=0 (deterministic)")
    print("=" * 40)

    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Deterministic response"}],
        max_tokens=10,
        temperature=0,  # Deterministic
    )

    stats = weave.cache.stats()
    print(f"\nCache items: {stats['item_count']}")
    print("Expected: 1 (deterministic request cached)")

    weave.finish()


# ============================================================================
# DEMO 7: Per-Request Cache Control
# ============================================================================
def demo_per_request_control():
    """Show per-request cache control with special parameters"""
    print("\n" + "=" * 80)
    print("DEMO 7: Per-Request Cache Control")
    print("=" * 80)

    weave.init("demo_per_request", settings={"cache_enabled": True})

    client = OpenAI()
    weave.cache.clear()

    print("\nYou can control caching on a per-request basis using:")
    print("  - _weave_skip_cache=True  : Skip cache for this request")
    print("  - _weave_force_cache=True : Force cache even if non-deterministic\n")

    # Normal cached request
    print("=" * 40)
    print("Normal Request (will be cached)")
    print("=" * 40)

    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Cached request"}],
        max_tokens=5,
        temperature=0,
    )

    stats = weave.cache.stats()
    print(f"Cache items: {stats['item_count']} (cached)")

    # Request with _weave_skip_cache
    print("\n" + "=" * 40)
    print("Request with _weave_skip_cache=True")
    print("=" * 40)

    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Skipped request"}],
        max_tokens=5,
        temperature=0,
        _weave_skip_cache=True,  # Skip cache
    )

    stats = weave.cache.stats()
    print(f"Cache items: {stats['item_count']} (not cached due to _weave_skip_cache)")

    # Request with _weave_force_cache on non-deterministic
    print("\n" + "=" * 40)
    print("Request with _weave_force_cache=True (temperature=0.7)")
    print("=" * 40)
    print("Normally temperature=0.7 might not be cached in deterministic_only mode,")
    print("but _weave_force_cache=True overrides that.")

    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Forced cache"}],
        max_tokens=5,
        temperature=0.7,  # Non-deterministic
        _weave_force_cache=True,  # Force cache anyway
    )

    stats = weave.cache.stats()
    print(f"Cache items: {stats['item_count']} (cached due to _weave_force_cache)")

    weave.finish()


# ============================================================================
# DEMO 8: Cache Statistics and Monitoring
# ============================================================================
def demo_cache_statistics():
    """Show how to monitor cache statistics"""
    print("\n" + "=" * 80)
    print("DEMO 8: Cache Statistics and Monitoring")
    print("=" * 80)

    weave.init("demo_statistics", settings={"cache_enabled": True})

    client = OpenAI()
    weave.cache.clear()

    print("\nMonitoring cache statistics as we make requests...\n")

    def show_stats():
        stats = weave.cache.stats()
        size_mb = stats['current_size'] / (1024 * 1024)
        print(f"  Items: {stats['item_count']}")
        print(f"  Size: {size_mb:.4f} MB")
        print(f"  Directory: {stats['directory']}")

    print("Initial cache state:")
    show_stats()

    # Add some requests
    for i in range(3):
        print(f"\nAdding request {i+1}...")
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Request {i+1}"}],
            max_tokens=10,
            temperature=0,
        )
        show_stats()

    # Print detailed stats with built-in method
    print("\n" + "=" * 40)
    print("Detailed Cache Statistics")
    print("=" * 40)
    weave.cache.print_stats()

    weave.finish()


# ============================================================================
# DEMO 9: Provider Comparison
# ============================================================================
def demo_provider_comparison():
    """Compare caching behavior across different providers"""
    print("\n" + "=" * 80)
    print("DEMO 9: Provider Comparison (OpenAI, Anthropic, Google)")
    print("=" * 80)

    weave.init("demo_provider_comparison", settings={"cache_enabled": True})

    # Test with available providers
    providers = []

    # OpenAI
    try:
        providers.append(("OpenAI", OpenAI()))
    except:
        print("OpenAI not available")

    # Anthropic
    try:
        providers.append(("Anthropic", Anthropic()))
    except:
        print("Anthropic not available")

    # Google
    if genai is not None:
        try:
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            providers.append(("Google", genai))
        except:
            print("Google GenAI not available")
    else:
        print("Google GenAI not available")

    weave.cache.clear()

    for provider_name, client in providers:
        print("\n" + "=" * 40)
        print(f"Testing {provider_name}")
        print("=" * 40)

        try:
            if provider_name == "OpenAI":
                def make_request():
                    return client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=5,
                        temperature=0,
                    )

            elif provider_name == "Anthropic":
                def make_request():
                    return client.messages.create(
                        model="claude-3-5-haiku-20241022",
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=5,
                    )

            elif provider_name == "Google":
                def make_request():
                    model = client.GenerativeModel("gemini-1.5-flash")
                    return model.generate_content("Hello")

            # First request (cache MISS)
            print(f"\n{provider_name} Request 1 (cache MISS)...")
            start = time.time()
            make_request()
            latency1 = time.time() - start
            print(f"Latency: {latency1:.3f}s")

            # Second request (cache HIT)
            print(f"{provider_name} Request 2 (cache HIT)...")
            start = time.time()
            make_request()
            latency2 = time.time() - start
            print(f"Latency: {latency2:.3f}s")

            speedup = latency1 / latency2
            print(f"Speedup: {speedup:.1f}x")

        except Exception as e:
            print(f"Error testing {provider_name}: {e}")

    stats = weave.cache.stats()
    print(f"\n\nTotal items cached across all providers: {stats['item_count']}")

    weave.finish()


# ============================================================================
# Main Demo Runner
# ============================================================================
def run_all_demos():
    """Run all caching demos"""
    print("\n")
    print("=" * 80)
    print("  WEAVE LLM CACHING SYSTEM - COMPREHENSIVE DEMO")
    print("=" * 80)
    print("\nThis demo showcases all caching features in the Weave LLM integration.")
    print("Supported providers: OpenAI, Anthropic, Google, Cerebras, Hugging Face")
    print("\nNote: Some demos require API keys for different providers.")
    print("=" * 80)

    demos = [
        ("Basic Cache Enable/Disable", demo_basic_cache_enable),
        ("Programmatic Control", demo_programmatic_control),
        ("Context Manager", demo_context_manager),
        ("Multi-Provider Caching", demo_multi_provider),
        ("Selective Cache Clearing", demo_selective_clearing),
        ("Deterministic-Only Mode", demo_deterministic_only),
        ("Per-Request Control", demo_per_request_control),
        ("Cache Statistics", demo_cache_statistics),
        ("Provider Comparison", demo_provider_comparison),
    ]

    for demo_name, demo_func in demos:
        try:
            demo_func()
            time.sleep(1)  # Brief pause between demos
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user.")
            break
        except Exception as e:
            print(f"\n\nError in {demo_name}: {e}")
            import traceback
            traceback.print_exc()
            print("\nContinuing to next demo...\n")

    print("\n" + "=" * 80)
    print("  ALL DEMOS COMPLETED!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Enable caching via weave.init(settings={'cache_enabled': True})")
    print("  2. Or use weave.cache.enable() / disable() programmatically")
    print("  3. Use weave.disable_cache() context manager for temporary disable")
    print("  4. Clear cache selectively with clear(provider='...', model='...')")
    print("  5. Use cache_deterministic_only=True to only cache temp=0 requests")
    print("  6. Per-request control with _weave_skip_cache / _weave_force_cache")
    print("  7. Monitor with weave.cache.stats() and .print_stats()")
    print("=" * 80)


if __name__ == "__main__":
    run_all_demos()