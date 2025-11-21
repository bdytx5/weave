"""
Comprehensive test for all LLM providers with caching enabled.
Tests: OpenAI, Anthropic, Cerebras, LiteLLM, and HuggingFace.
"""

import os
import pytest
import weave

# Test prompt
TEST_PROMPT = "Write a short python function that adds two numbers"

# Enable caching via environment variable
os.environ["WEAVE_CACHE_ENABLED"] = "true"


class TestOpenAICache:
    """Test OpenAI with caching"""

    def test_openai_cache_hit(self):
        """Test that OpenAI caching works correctly"""
        try:
            import openai
        except ImportError:
            pytest.skip("openai not installed")

        # Initialize weave
        weave.init("test_openai_cache")

        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # First call - should miss cache
        response1 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=100,
        )

        # Second call with identical params - should hit cache
        response2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=100,
        )

        # Verify responses are identical
        assert response1.choices[0].message.content == response2.choices[0].message.content
        print(f"OpenAI Response: {response1.choices[0].message.content}")
        print("OpenAI cache test PASSED")


class TestAnthropicCache:
    """Test Anthropic with caching"""

    def test_anthropic_cache_hit(self):
        """Test that Anthropic caching works correctly"""
        try:
            import anthropic
        except ImportError:
            pytest.skip("anthropic not installed")

        # Initialize weave
        weave.init("test_anthropic_cache")

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        # First call - should miss cache
        response1 = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=100,
            messages=[{"role": "user", "content": TEST_PROMPT}],
        )

        # Second call with identical params - should hit cache
        response2 = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=100,
            messages=[{"role": "user", "content": TEST_PROMPT}],
        )

        # Verify responses are identical
        assert response1.content[0].text == response2.content[0].text
        print(f"Anthropic Response: {response1.content[0].text}")
        print("Anthropic cache test PASSED")


class TestCerebrasCache:
    """Test Cerebras with caching"""

    def test_cerebras_cache_hit(self):
        """Test that Cerebras caching works correctly"""
        try:
            from cerebras.cloud.sdk import Cerebras
        except ImportError:
            pytest.skip("cerebras.cloud.sdk not installed")

        # Initialize weave
        weave.init("test_cerebras_cache")

        client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

        # First call - should miss cache
        response1 = client.chat.completions.create(
            model="llama-3.3-70b",
            messages=[{"role": "user", "content": TEST_PROMPT}],
        )

        # Second call with identical params - should hit cache
        response2 = client.chat.completions.create(
            model="llama-3.3-70b",
            messages=[{"role": "user", "content": TEST_PROMPT}],
        )

        # Verify responses are identical
        assert response1.choices[0].message.content == response2.choices[0].message.content
        print(f"Cerebras Response: {response1.choices[0].message.content}")
        print("Cerebras cache test PASSED")


class TestHuggingFaceCache:
    """Test HuggingFace with caching"""

    def test_huggingface_cache_hit(self):
        """Test that HuggingFace caching works correctly"""
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            pytest.skip("huggingface_hub not installed")

        # Initialize weave
        weave.init("test_huggingface_cache")

        client = InferenceClient(token=os.environ.get("HUGGINGFACE_TOKEN"))

        # First call - should miss cache
        response1 = client.chat_completion(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=100,
        )

        # Second call with identical params - should hit cache
        response2 = client.chat_completion(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=100,
        )

        # Verify responses are identical
        assert response1.choices[0].message.content == response2.choices[0].message.content
        print(f"HuggingFace Response: {response1.choices[0].message.content}")
        print("HuggingFace cache test PASSED")

def run_all_tests():
    """Run all provider cache tests"""
    print("=" * 80)
    print("Testing all LLM providers with caching enabled")
    print("=" * 80)

    test_classes = [
        TestOpenAICache,
        TestAnthropicCache,
        TestCerebrasCache,
        TestHuggingFaceCache,
    ]

    for test_class in test_classes:
        print(f"\n{'=' * 80}")
        print(f"Running {test_class.__name__}")
        print(f"{'=' * 80}")

        try:
            test_instance = test_class()
            # Get the test method (first method starting with test_)
            test_methods = [
                method
                for method in dir(test_instance)
                if method.startswith("test_") and callable(getattr(test_instance, method))
            ]

            for method_name in test_methods:
                test_method = getattr(test_instance, method_name)
                try:
                    test_method()
                except Exception as e:
                    if "skip" in str(type(e).__name__).lower():
                        print(f"SKIPPED: {e}")
                    else:
                        print(f"FAILED: {e}")
                        import traceback

                        traceback.print_exc()
        except Exception as e:
            print(f"FAILED to initialize: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
