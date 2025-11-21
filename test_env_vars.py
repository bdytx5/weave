"""Test that all cache environment variables work."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables BEFORE importing weave
os.environ["WEAVE_CACHE_ENABLED"] = "true"
os.environ["WEAVE_CACHE_DETERMINISTIC_ONLY"] = "false"
os.environ["WEAVE_CACHE_DIR"] = "/tmp/test-env-cache"
os.environ["WEAVE_CACHE_TTL"] = "7200"
os.environ["WEAVE_CACHE_SIZE_LIMIT"] = "500000000"

import weave
from weave.trace import settings

print("=" * 60)
print("Testing Cache Environment Variables")
print("=" * 60)

# Test each setting
print("\n1. WEAVE_CACHE_ENABLED=true")
result = settings.cache_enabled()
print(f"   Result: {result}")
assert result == True, f"Expected True, got {result}"
print("   ✅ WORKS")

print("\n2. WEAVE_CACHE_DETERMINISTIC_ONLY=false")
result = settings.cache_deterministic_only()
print(f"   Result: {result}")
assert result == False, f"Expected False, got {result}"
print("   ✅ WORKS")

print("\n3. WEAVE_CACHE_DIR=/tmp/test-env-cache")
result = settings.cache_dir()
print(f"   Result: {result}")
assert result == "/tmp/test-env-cache", f"Expected '/tmp/test-env-cache', got {result}"
print("   ✅ WORKS")

print("\n4. WEAVE_CACHE_TTL=7200")
result = settings.cache_ttl()
print(f"   Result: {result}")
assert result == 7200.0, f"Expected 7200.0, got {result}"
print("   ✅ WORKS")

print("\n5. WEAVE_CACHE_SIZE_LIMIT=500000000")
result = settings.cache_size_limit()
print(f"   Result: {result}")
assert result == 500000000, f"Expected 500000000, got {result}"
print("   ✅ WORKS")

print("\n" + "=" * 60)
print("Testing weave.init() with env vars...")
print("=" * 60)

# Initialize weave (should read from env vars)
client = weave.init("test-env-vars")

from weave.integrations.cache import get_global_cache

cache = get_global_cache()
assert cache is not None, "Cache should be initialized"
print(f"\n✅ Cache initialized from env vars!")
print(f"   Directory: {cache.directory}")
print(f"   TTL: {cache._ttl}")
print(f"   Deterministic only: {cache._deterministic_only}")

assert str(cache.directory) == "/tmp/test-env-cache"
assert cache._ttl == 7200.0
assert cache._deterministic_only == False

print("\n" + "=" * 60)
print("🎉 ALL ENVIRONMENT VARIABLES WORK!")
print("=" * 60)
