import weave
import os
import sys

# Initialize weave FIRST
print("Initializing weave...")
weave.init("debug_google", settings={"cache_enabled": True})

# Check import hook
print(f"\nImport hooks: {[type(h).__name__ for h in sys.meta_path[:5]]}")

print("\nImporting google.genai...")
from google import genai
import google.genai.models

print("\n=== AFTER IMPORT ===")
print(f"google.genai.models.Models.generate_content: {google.genai.models.Models.generate_content}")
print(f"Is weave op? {hasattr(google.genai.models.Models.generate_content, '__wrapped__')}")
print(f"Has __name__? {hasattr(google.genai.models.Models.generate_content, '__name__')}")
if hasattr(google.genai.models.Models.generate_content, '__name__'):
    print(f"Function name: {google.genai.models.Models.generate_content.__name__}")
