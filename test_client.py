"""
Test client for UI Vision RunPod serverless endpoint.
Tests the deployed endpoint with a sample screenshot.
"""

import runpod
import base64
import sys
from pathlib import Path

# Configuration
RUNPOD_API_KEY = "YOUR_RUNPOD_API_KEY"  # Replace with your API key
ENDPOINT_ID = "YOUR_ENDPOINT_ID"        # Replace with your endpoint ID


def encode_image_to_base64(image_path):
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_ui_vision(image_path, prompt):
    """
    Test the UI vision endpoint with an image and prompt.
    
    Args:
        image_path: Path to screenshot image
        prompt: Natural language query about UI element
    """
    # Configure RunPod
    runpod.api_key = RUNPOD_API_KEY
    
    # Encode image
    print(f"📸 Loading image: {image_path}")
    image_base64 = encode_image_to_base64(image_path)
    print(f"✓ Image encoded ({len(image_base64)} bytes)")
    
    # Create endpoint
    print(f"\n🚀 Connecting to endpoint: {ENDPOINT_ID}")
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Prepare request
    request_data = {
        "input": {
            "prompt": prompt,
            "image": image_base64
        }
    }
    
    print(f"\n💬 Prompt: {prompt}")
    print("⏳ Running inference...")
    
    # Run inference
    try:
        result = endpoint.run_sync(request_data, timeout=60)
        
        # Display results
        print("\n" + "="*60)
        print("📊 UI GROUNDING RESULTS")
        print("="*60)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✓ Element found")
            print(f"\n📍 Click Coordinates:")
            print(f"   X: {result.get('x', 'N/A')}")
            print(f"   Y: {result.get('y', 'N/A')}")
            print(f"\n🎯 Confidence: {result.get('confidence', 0.0):.2%}")
        
        print("\n" + "="*60)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def main():
    """Main test function."""
    print("="*60)
    print("UI Vision RunPod Serverless - Test Client")
    print("="*60)
    
    # Check configuration
    if RUNPOD_API_KEY == "YOUR_RUNPOD_API_KEY":
        print("\n⚠️  Please set your RUNPOD_API_KEY in this file!")
        print("   Get it from: https://www.runpod.io/console/user/settings")
        sys.exit(1)
    
    if ENDPOINT_ID == "YOUR_ENDPOINT_ID":
        print("\n⚠️  Please set your ENDPOINT_ID in this file!")
        print("   Get it from your serverless endpoint page")
        sys.exit(1)
    
    # Example test
    image_path = input("\n📁 Enter path to screenshot image: ").strip()
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    prompt = input(\"💬 Enter UI element to locate (e.g., 'login button'): \").strip()
    
    if not prompt:
        prompt = \"submit button\"
    
    # Run test
    result = test_ui_vision(image_path, prompt)
    
    if result and 'x' in result and 'y' in result:
        print(\"\\n✅ UI grounding successful!\")
        print(f\"\\n🖱️  You can now click at ({result['x']}, {result['y']})\")
    else:
        print(\"\\n❌ Could not locate element!\")


if __name__ == "__main__":
    main()
