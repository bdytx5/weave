"""
Basic test to verify caching works for all providers.
Prints both uncached and cached responses to confirm behavior.
"""

import os
import time
import weave

# Enable caching
os.environ["WEAVE_CACHE_ENABLED"] = "true"

PROMPT = "Say hello in one word"


def test_openai():
    """Test OpenAI caching"""
    print("\n" + "="*60)
    print("TESTING OPENAI")
    print("="*60)

    try:
        from openai import OpenAI
        client = OpenAI()

        # Clear cache first
        weave.cache.clear()

        # First call - uncached
        print("\n[1] First call (UNCACHED)...")
        start = time.time()
        response1 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=5,
        )
        t1 = time.time() - start
        print(f"Response: {response1.choices[0].message.content}")
        print(f"Time: {t1:.3f}s")
        print(f"Type: {type(response1)}")

        # Second call - cached
        print("\n[2] Second call (CACHED)...")
        start = time.time()
        response2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=5,
        )
        t2 = time.time() - start
        print(f"Response: {response2.choices[0].message.content}")
        print(f"Time: {t2:.3f}s ({t1/t2:.1f}x faster)")
        print(f"Type: {type(response2)}")

        # Verify
        assert response1.choices[0].message.content == response2.choices[0].message.content
        print("\n✅ OpenAI PASSED")

    except Exception as e:
        print(f"\n❌ OpenAI FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_anthropic():
    """Test Anthropic caching"""
    print("\n" + "="*60)
    print("TESTING ANTHROPIC")
    print("="*60)

    try:
        from anthropic import Anthropic
        client = Anthropic()

        # Clear cache first
        weave.cache.clear()

        # First call - uncached
        print("\n[1] First call (UNCACHED)...")
        start = time.time()
        response1 = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=5,
            messages=[{"role": "user", "content": PROMPT}],
        )
        t1 = time.time() - start
        print(f"Response: {response1.content[0].text}")
        print(f"Time: {t1:.3f}s")
        print(f"Type: {type(response1)}")

        # Second call - cached
        print("\n[2] Second call (CACHED)...")
        start = time.time()
        response2 = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=5,
            messages=[{"role": "user", "content": PROMPT}],
        )
        t2 = time.time() - start
        print(f"Response: {response2.content[0].text}")
        print(f"Time: {t2:.3f}s ({t1/t2:.1f}x faster)")
        print(f"Type: {type(response2)}")

        # Verify
        assert response1.content[0].text == response2.content[0].text
        print("\n✅ Anthropic PASSED")

    except Exception as e:
        print(f"\n❌ Anthropic FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_cerebras():
    """Test Cerebras caching"""
    print("\n" + "="*60)
    print("TESTING CEREBRAS")
    print("="*60)

    try:
        from cerebras.cloud.sdk import Cerebras
        client = Cerebras()

        # Clear cache first
        weave.cache.clear()

        # First call - uncached
        print("\n[1] First call (UNCACHED)...")
        start = time.time()
        response1 = client.chat.completions.create(
            model="llama-3.3-70b",
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=5,
        )
        t1 = time.time() - start
        print(f"Response: {response1.choices[0].message.content}")
        print(f"Time: {t1:.3f}s")
        print(f"Type: {type(response1)}")

        # Second call - cached
        print("\n[2] Second call (CACHED)...")
        start = time.time()
        response2 = client.chat.completions.create(
            model="llama-3.3-70b",
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=5,
        )
        t2 = time.time() - start
        print(f"Response: {response2.choices[0].message.content}")
        print(f"Time: {t2:.3f}s ({t1/t2:.1f}x faster)")
        print(f"Type: {type(response2)}")

        # Verify
        assert response1.choices[0].message.content == response2.choices[0].message.content
        print("\n✅ Cerebras PASSED")

    except Exception as e:
        print(f"\n❌ Cerebras FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_huggingface():
    """Test HuggingFace caching"""
    print("\n" + "="*60)
    print("TESTING HUGGINGFACE")
    print("="*60)

    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient()

        # Clear cache first
        weave.cache.clear()

        # First call - uncached
        print("\n[1] First call (UNCACHED)...")
        start = time.time()
        response1 = client.chat_completion(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=5,
        )
        t1 = time.time() - start
        print(f"Response: {response1.choices[0].message.content}")
        print(f"Time: {t1:.3f}s")
        print(f"Type: {type(response1)}")

        # Second call - cached
        print("\n[2] Second call (CACHED)...")
        start = time.time()
        response2 = client.chat_completion(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=5,
        )
        t2 = time.time() - start
        print(f"Response: {response2.choices[0].message.content}")
        print(f"Time: {t2:.3f}s ({t1/t2:.1f}x faster)")
        print(f"Type: {type(response2)}")

        # Verify
        assert response1.choices[0].message.content == response2.choices[0].message.content
        print("\n✅ HuggingFace PASSED")

    except Exception as e:
        print(f"\n❌ HuggingFace FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_google_genai():
    """Test Google GenAI caching"""
    print("\n" + "="*60)
    print("TESTING GOOGLE GENAI")
    print("="*60)

    try:
        from google import genai

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        # Clear cache first
        weave.cache.clear()

        # First call - uncached
        print("\n[1] First call (UNCACHED)...")
        start = time.time()
        response1 = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=PROMPT,
        )
        t1 = time.time() - start
        print(f"Response: {response1.text}")
        print(f"Time: {t1:.3f}s")
        print(f"Type: {type(response1)}")

        # Second call - cached
        print("\n[2] Second call (CACHED)...")
        start = time.time()
        response2 = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=PROMPT,
        )
        t2 = time.time() - start
        print(f"Response: {response2.text}")
        print(f"Time: {t2:.3f}s ({t1/t2:.1f}x faster)")
        print(f"Type: {type(response2)}")

        # Verify
        assert response1.text == response2.text
        print("\n✅ Google GenAI PASSED")

    except Exception as e:
        print(f"\n❌ Google GenAI FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Initialize weave
    weave.init("basic_cache_test", settings={"cache_enabled": True})

    print("="*60)
    print("BASIC CACHED INFERENCE TEST")
    print("="*60)

    test_openai()
    test_anthropic()
    test_cerebras()
    test_huggingface()
    test_google_genai()


    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
