"""Create a complete test request JSON file"""
import json

# Read base64 image
with open('base64_image.txt', 'r') as f:
    base64_image = f.read().strip()

# Create the request
test_request = {
    "input": {
        "prompt": "search bar",
        "image": base64_image
    }
}

# Save to JSON file
with open('test_request.json', 'w') as f:
    json.dump(test_request, f, indent=2)

print("✅ Created test_request.json")
print(f"\nFile size: {len(json.dumps(test_request)):,} bytes")
print(f"Image length: {len(base64_image):,} characters")
print("\n" + "="*60)
print("READY TO SEND TO RUNPOD!")
print("="*60)
print("\nOption 1: Using curl")
print("-" * 60)
print("""curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d @test_request.json""")

print("\nOption 2: Using Python (test_with_screenshot.py)")
print("-" * 60)
print("1. Edit test_with_screenshot.py with your API key and endpoint ID")
print("2. Run: python test_with_screenshot.py")

print("\n" + "="*60)
print("Test prompts you can try:")
print("="*60)
print('  • "search bar" - Find Reddit search field')
print('  • "reddit logo" - Find the main logo')
print('  • "login button" - Find sign-in button')
print('  • "text input" - Find input fields')
print('  • "menu icon" - Find navigation buttons')
