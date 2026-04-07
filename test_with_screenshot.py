"""
Test the UI Vision RunPod endpoint with an actual screenshot
"""
import runpod
import base64
from pathlib import Path

# Configuration
RUNPOD_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your RunPod API key
ENDPOINT_ID = "YOUR_ENDPOINT_ID_HERE"  # Replace with your endpoint ID

# Load and encode the screenshot
screenshot_path = Path(__file__).parent.parent / "virtual_desktop" / "brave_with_page.png"
print(f"Loading screenshot: {screenshot_path}")

with open(screenshot_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

print(f"Image encoded: {len(image_base64)} characters")

# Create the test request
test_request = {
    "input": {
        "prompt": "search bar",  # Try: "search bar", "logo", "menu button", "login button"
        "image": image_base64
    }
}

print("\n" + "="*60)
print("TEST REQUEST")
print("="*60)
print(f"Prompt: {test_request['input']['prompt']}")
print(f"Image size: {len(image_base64)} chars")
print("\nSending to RunPod...")

# Initialize client and run
runpod.api_key = RUNPOD_API_KEY
endpoint = runpod.Endpoint(ENDPOINT_ID)

try:
    result = endpoint.run_sync(test_request, timeout=300)  # 5 minute timeout
    
    print("\n" + "="*60)
    print("RESPONSE")
    print("="*60)
    print(result)
    
    if result and "x" in result and "y" in result:
        print(f"\n✅ Found element at coordinates: ({result['x']}, {result['y']})")
        print(f"   Confidence: {result.get('confidence', 'N/A')}")
        print(f"\n💡 You can use these coordinates with SimpleDesktopController.click({result['x']}, {result['y']})")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nMake sure to:")
    print("1. Replace RUNPOD_API_KEY with your actual API key")
    print("2. Replace ENDPOINT_ID with your endpoint ID")
    print("3. Deploy the Docker image on RunPod serverless")
