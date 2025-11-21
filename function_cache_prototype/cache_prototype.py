"""
Function caching with automatic invalidation when function code changes.

The cache key includes a hash of the function's source code, so any changes
to the function body automatically invalidate the cache.
"""

import hashlib
import inspect
import json
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional


class FunctionCache:
    """Cache that invalidates when function code changes."""

    def __init__(self):
        self._cache: Dict[str, tuple[Any, float]] = {}

    def get_function_hash(self, fn: Callable) -> str:
        """Hash the function's source code to detect changes."""
        try:
            source = inspect.getsource(fn)
            return hashlib.md5(source.encode()).hexdigest()[:8]
        except (OSError, TypeError):
            # Fallback to bytecode if source not available
            return hashlib.md5(fn.__code__.co_code).hexdigest()[:8]

    def get_inputs_hash(self, args: tuple, kwargs: dict) -> str:
        """Hash the function inputs."""
        inputs_repr = {
            "args": args,
            "kwargs": kwargs,
        }
        inputs_str = json.dumps(inputs_repr, sort_keys=True, default=str)
        return hashlib.md5(inputs_str.encode()).hexdigest()[:8]

    def create_cache_key(self, fn: Callable, args: tuple, kwargs: dict) -> str:
        """Create cache key from function hash + inputs hash."""
        fn_hash = self.get_function_hash(fn)
        inputs_hash = self.get_inputs_hash(args, kwargs)
        return f"{fn.__module__}.{fn.__name__}:{fn_hash}:{inputs_hash}"

    def get(self, key: str, ttl: Optional[float] = None) -> Optional[Any]:
        """Get value from cache if it exists and hasn't expired."""
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]

        if ttl is not None and (time.time() - timestamp) > ttl:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any) -> None:
        """Store value in cache with current timestamp."""
        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "keys": list(self._cache.keys()),
        }


# Global cache instance
_global_cache = FunctionCache()


def cached(ttl: Optional[float] = None):
    """
    Decorator to cache function results with automatic invalidation on code changes.

    Args:
        ttl: Time-to-live in seconds. If None, cache never expires.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            cache_key = _global_cache.create_cache_key(fn, args, kwargs)

            cached_result = _global_cache.get(cache_key, ttl=ttl)
            if cached_result is not None:
                print(f"[CACHE HIT] {fn.__name__}")
                return cached_result

            print(f"[CACHE MISS] {fn.__name__}")
            result = fn(*args, **kwargs)
            _global_cache.set(cache_key, result)

            return result

        wrapper.cache_clear = _global_cache.clear
        wrapper.cache_stats = _global_cache.stats

        return wrapper

    return decorator


# ============================================================================
# TESTS
# ============================================================================


def test_basic_caching():
    """Test that identical calls return cached results."""
    print("\n=== Test: Basic Caching ===")

    call_count = 0

    @cached()
    def add(x, y):
        nonlocal call_count
        call_count += 1
        return x + y

    result1 = add(1, 2)
    assert result1 == 3
    assert call_count == 1
    print(f"First call: {result1}, call_count: {call_count}")

    result2 = add(1, 2)
    assert result2 == 3
    assert call_count == 1  # Should NOT increment
    print(f"Second call (cached): {result2}, call_count: {call_count}")

    result3 = add(2, 3)
    assert result3 == 5
    assert call_count == 2
    print(f"Different inputs: {result3}, call_count: {call_count}")

    print("✓ Basic caching works")


def test_ttl_expiration():
    """Test that TTL causes cache expiration."""
    print("\n=== Test: TTL Expiration ===")

    call_count = 0

    @cached(ttl=0.1)
    def slow_function(x):
        nonlocal call_count
        call_count += 1
        return x * 2

    result1 = slow_function(5)
    assert call_count == 1
    print(f"First call: call_count = {call_count}")

    result2 = slow_function(5)
    assert call_count == 1
    print(f"Immediate call (cached): call_count = {call_count}")

    time.sleep(0.15)

    result3 = slow_function(5)
    assert call_count == 2
    print(f"After TTL expired: call_count = {call_count}")

    print("✓ TTL expiration works")


def test_kwargs_handling():
    """Test that kwargs are properly cached."""
    print("\n=== Test: Kwargs Handling ===")

    call_count = 0

    @cached()
    def greet(name, greeting="Hello"):
        nonlocal call_count
        call_count += 1
        return f"{greeting}, {name}!"

    result1 = greet("Alice")
    assert call_count == 1
    print(f"greet('Alice'): call_count = {call_count}")

    result2 = greet("Alice")
    assert call_count == 1
    print(f"greet('Alice') again (cached): call_count = {call_count}")

    result3 = greet("Alice", greeting="Hi")
    assert call_count == 2
    print(f"greet('Alice', greeting='Hi'): call_count = {call_count}")

    result4 = greet("Alice", greeting="Hi")
    assert call_count == 2
    print(f"greet('Alice', greeting='Hi') again (cached): call_count = {call_count}")

    print("✓ Kwargs handling works")


def test_manual_function_change():
    """
    THIS IS THE KEY TEST FOR FUNCTION INVALIDATION.

    Instructions to test:
    1. Run this script - note the hash below
    2. Edit the compute() function below (change x * 2 to x * 3)
    3. Run again - the hash will be DIFFERENT
    4. Old cache won't be used because function code changed!
    """
    print("\n=== Test: Function Change Detection ===")

    @cached()
    def compute(x):
        # EDIT THIS LINE to test cache invalidation:
        # Try changing to: return x * 3
        return x * 2

    result = compute(10)
    func_hash = _global_cache.get_function_hash(compute)

    print(f"Result: {result}")
    print(f"Function hash: {func_hash}")
    print("")
    print("TO TEST CACHE INVALIDATION:")
    print("1. Note the hash above")
    print("2. Edit this file - change 'return x * 2' to 'return x * 3'")
    print("3. Run again - hash will be different!")
    print("4. Old cache won't be used")


def test_cache_stats():
    """Test cache statistics."""
    print("\n=== Test: Cache Stats ===")

    @cached()
    def multiply(x, y):
        return x * y

    multiply.cache_clear()

    multiply(2, 3)
    multiply(4, 5)
    multiply(6, 7)

    stats = multiply.cache_stats()
    assert stats["size"] == 3
    print(f"Cache has {stats['size']} entries")

    multiply.cache_clear()
    stats = multiply.cache_stats()
    assert stats["size"] == 0
    print(f"After clear: {stats['size']} entries")

    print("✓ Cache stats work")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("FUNCTION CACHE PROTOTYPE - Tests")
    print("=" * 70)

    test_basic_caching()
    test_ttl_expiration()
    test_kwargs_handling()
    test_cache_stats()
    test_manual_function_change()

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
