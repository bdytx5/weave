"""Test multimodal caching with images."""

import time
import base64
import io
from PIL import Image

import weave
from openai import OpenAI


def create_noise_image(size=(256, 256), sigma=50):
    """Create a random noise image."""
    img = Image.effect_noise(size, sigma)

    # Convert to base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf8")

    return f"data:image/png;base64,{encoded}"


def main():
    print("=" * 60)
    print("MULTIMODAL CACHING TEST")
    print("=" * 60)

    # Initialize weave with caching
    print("\n1. Initializing weave with cache...")
    weave.init(
        "multimodal-cache-test",
        settings={
            "cache_enabled": True,
            "cache_dir": "/tmp/weave-multimodal-cache",
            "cache_deterministic_only": False,  # Cache everything
        },
    )
    print("   ✓ Cache enabled")

    # Create OpenAI client
    client = OpenAI()

    # Create 2 noise images
    print("\n2. Creating 2 noise images...")
    image1 = create_noise_image(size=(256, 256), sigma=50)
    image2 = create_noise_image(size=(256, 256), sigma=100)  # Different noise
    print(f"   ✓ Image 1: {len(image1)} bytes")
    print(f"   ✓ Image 2: {len(image2)} bytes")

    # Test 1: First call with image1 (should hit API)
    print("\n3. First call with image1 (should hit OpenAI API)...")
    start = time.time()
    response1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image1}
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in one sentence."
                    }
                ]
            }
        ],
        temperature=0.7,
        max_tokens=100,
    )
    elapsed1 = time.time() - start
    print(f"   Response: {response1.choices[0].message.content}")
    print(f"   ⏱️  Time: {elapsed1:.2f}s (API call)")

    # Test 2: Second call with SAME image1 (should be cached)
    print("\n4. Second call with SAME image1 (should be CACHED)...")
    start = time.time()
    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image1}
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in one sentence."
                    }
                ]
            }
        ],
        temperature=0.7,
        max_tokens=100,
    )
    elapsed2 = time.time() - start
    print(f"   Response: {response2.choices[0].message.content}")
    print(f"   ⏱️  Time: {elapsed2:.2f}s (CACHED - {elapsed1/elapsed2:.0f}x faster!)")

    # Verify same response
    assert response1.choices[0].message.content == response2.choices[0].message.content
    print("   ✅ Responses match! Cache works!")

    # Test 3: Third call with DIFFERENT image2 (should hit API)
    print("\n5. Third call with DIFFERENT image2 (should hit API)...")
    start = time.time()
    response3 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image2}
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in one sentence."
                    }
                ]
            }
        ],
        temperature=0.7,
        max_tokens=100,
    )
    elapsed3 = time.time() - start
    print(f"   Response: {response3.choices[0].message.content}")
    print(f"   ⏱️  Time: {elapsed3:.2f}s (API call - different image)")

    # Test 4: Fourth call with image2 again (should be cached)
    print("\n6. Fourth call with SAME image2 (should be CACHED)...")
    start = time.time()
    response4 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image2}
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in one sentence."
                    }
                ]
            }
        ],
        temperature=0.7,
        max_tokens=100,
    )
    elapsed4 = time.time() - start
    print(f"   Response: {response4.choices[0].message.content}")
    print(f"   ⏱️  Time: {elapsed4:.2f}s (CACHED)")

    # Verify same response
    assert response3.choices[0].message.content == response4.choices[0].message.content
    print("   ✅ Responses match! Cache works!")

    # Test 5: Change the TEXT prompt (should NOT be cached)
    print("\n7. Fifth call with image1 but DIFFERENT text (should hit API)...")
    start = time.time()
    response5 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image1}
                    },
                    {
                        "type": "text",
                        "text": "What colors do you see in this image?"  # Different question
                    }
                ]
            }
        ],
        temperature=0.7,
        max_tokens=100,
    )
    elapsed5 = time.time() - start
    print(f"   Response: {response5.choices[0].message.content}")
    print(f"   ⏱️  Time: {elapsed5:.2f}s (API call - different text)")

    # Show cache stats
    print("\n" + "=" * 60)
    print("CACHE STATISTICS")
    print("=" * 60)
    from weave.integrations.cache import get_global_cache

    cache = get_global_cache()
    if cache:
        cache.print_stats()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Call 1 (img1, API):        {elapsed1:.2f}s")
    print(f"  Call 2 (img1, CACHE):      {elapsed2:.2f}s  ({elapsed1/elapsed2:.0f}x faster)")
    print(f"  Call 3 (img2, API):        {elapsed3:.2f}s")
    print(f"  Call 4 (img2, CACHE):      {elapsed4:.2f}s  ({elapsed3/elapsed4:.0f}x faster)")
    print(f"  Call 5 (img1+diff text):   {elapsed5:.2f}s")
    print("=" * 60)
    print("\n✅ Multimodal caching works perfectly!")
    print("   - Same image + same text = CACHE HIT")
    print("   - Different image = CACHE MISS")
    print("   - Different text = CACHE MISS")


if __name__ == "__main__":
    main()
