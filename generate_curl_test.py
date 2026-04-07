"""
Quick script to generate a curl command for testing
"""
import base64
from pathlib import Path

# Load and encode the screenshot
screenshot_path = Path(__file__).parent.parent / "virtual_desktop" / "brave_with_page.png"

with open(screenshot_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# Generate curl command
curl_command = f'''curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "input": {{
      "prompt": "search bar",
      "image": "{image_base64[:100]}..."
    }}
  }}'
'''

print("CURL TEST COMMAND")
print("="*60)
print("\nReplace YOUR_ENDPOINT_ID and YOUR_API_KEY, then run:\n")
print(curl_command)
print("\n" + "="*60)
print(f"\nFull base64 image ({len(image_base64)} chars) saved to: base64_image.txt")

# Save full base64 for easy copy-paste
with open("base64_image.txt", "w") as f:
    f.write(image_base64)

print("\nTry these prompts:")
print("  - 'search bar'")
print("  - 'reddit logo'")
print("  - 'login button'")
print("  - 'menu icon'")
print("  - 'text input field'")
