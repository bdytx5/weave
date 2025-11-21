"""Test if cache emoji appears in call links"""
import os
import sys

# Make sure we use local weave
sys.path.insert(0, '/Users/brettyoung/Desktop/dev25/wandb_work/litelogger_pr/weave')

# Enable caching
os.environ["WEAVE_CACHE_ENABLED"] = "true"

import weave

# Initialize weave
weave.init("test_cache_emoji")

# Clear cache first
weave.cache.clear()

print("\n" + "="*60)
print("Testing Cache Emoji Display")
print("="*60)

# Test with OpenAI
try:
    import openai

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print("\n[1] First call - should show: 🍩 (no cache emoji)")
    response1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10,
    )

    print("\n[2] Second call - should show: 💾 🍩 (cache emoji!)")
    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10,
    )

    # Debug: Check if responses are the same
    print(f"\nDEBUG: Response 1 content: {response1.choices[0].message.content if hasattr(response1, 'choices') else response1}")
    print(f"DEBUG: Response 2 content: {response2.choices[0].message.content if hasattr(response2, 'choices') else response2}")
    print(f"DEBUG: Responses identical? {response1 == response2}")

    # Check cache stats
    stats = weave.cache.stats()
    print(f"\nDEBUG: Cache stats: {stats}")
    print(f"DEBUG: Cache has items? {stats.get('item_count', 0) > 0}")

    print("\n" + "="*60)
    print("✅ Test complete! Check the output above for emojis.")
    print("="*60)

except ImportError:
    print("❌ OpenAI not installed. Install with: pip install openai")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
