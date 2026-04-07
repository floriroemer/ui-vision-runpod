# UI Grounding - RunPod Serverless

A RunPod serverless endpoint that uses UI-grounding models to locate UI elements and return precise pixel coordinates for automation.

## 🚀 Features

- **UI Grounding Model**: Uses UI-TARS-1.5-7B from ByteDance - specialized for UI element detection
- **Dual Loading Strategy**: Auto-detects processor (VLM) or tokenizer (LLM) architecture
- **Pixel Coordinates**: Returns exact click coordinates (x, y) for automation
- **Bounding Box**: Full bbox [x1, y1, x2, y2] + center point
- **Flexible Input**: Accepts screenshots as base64 encoded strings
- **Float32 Precision**: Full precision for stable inference
- **Production Ready**: Comprehensive error handling, logging, and debug output

## 📁 Project Structure

```
ui_vision_runpod/
├── handler.py          # RunPod serverless handler
├── requirements.txt    # Python dependencies
├── Dockerfile         # Container definition
├── README.md          # This file
└── test_client.py     # Example client code
```

## 🛠️ Deployment

### 1. Build Docker Image

```bash
docker build -t ui-vision-runpod .
```

### 2. Push to Container Registry

```bash
# Tag for Docker Hub
docker tag ui-vision-runpod yourusername/ui-vision-runpod:latest
docker push yourusername/ui-vision-runpod:latest
```

### 3. Deploy on RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click "New Template"
3. Configure:
   - **Template Name**: UI Vision Analysis
   - **Container Image**: `yourusername/ui-vision-runpod:latest`
   - **Container Disk**: 20 GB (for model storage)
   - **GPU**: RTX 4090 or A100 (recommended for 7B model)
4. Create Endpoint using the template
5. Copy your Endpoint ID and API Key

## 📡 API Usage

### Python Client

```python
import runpod
import base64

# Configure RunPod client
runpod.api_key = "YOUR_RUNPOD_API_KEY"

# Load and encode image
with open("screenshot.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# Create endpoint
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Run inference
result = endpoint.run_sync(
    {
        "input": {
            "prompt": "login button",
            "image": image_base64
        }
    },
    timeout=60
)

print(result)
# Output:
# {
#   "x": 450,
#   "y": 320,
#   "confidence": 0.89
# }
```

### Complete Example: Encode Image

```python
import base64
from PIL import Image
import io

def encode_image_to_base64(image_path):
    """Encode image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def encode_pil_to_base64(pil_image):
    """Encode PIL Image to base64 string."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Example usage
image_b64 = encode_image_to_base64("screenshot.png")
```

### Using `requests` Library

```python
import requests
import base64
import json

ENDPOINT_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run"
API_KEY = "YOUR_RUNPOD_API_KEY"

# Encode image
with open("screenshot.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# Prepare request
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "input": {
        "prompt": "Where is the submit button?",
        "image": imsubmit button",
        "image": image_base64
    }
}

# Send request
response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
result = response.json()

print(f"Click at: ({result['x']}, {result['y']})")
print(f"Confidence: {result['confidence']:.2%

## 📊 Input/Output Format

### Input

```json
{login button",
  "image": "<base64 encoded PNG/JPG>"
}
```

### Output (Success)

```json
{
  "x": 450,
  "y": 320,
  "confidence": 0.89
}
```

### Output (Error)

```json
{
  "error": "element not founmessage",
  "status": "failed"
}
```

## ⚙️ Configuration

### Model Selection

To use a different vision-language model, edit `handler.py`:

```python
MODEL_NAME = "microsoft/Phi-3-vision-128k-instruct"  # or any other VLM
```

### GPU Memory Optimization

For smaller GPUs, you can:

1. Use quantization (4-bit or 8-bit)
2. Use a smaller model (e.g., Phi-3-vision)
3. Adjust `max_new_tokens` in the handler

## 🧪 Local Testing

### Test with Docker

```bash
# Build
docker build -t ui-vision-runpod .

# Run locally
docker run --gpus all -p 8000:8000 ui-vision-runpod

# Test endpoint
python test_client.py
```

### Test Handler Directly

```python
from handler import handler
import base64

# Load image
with open("screenshot.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

# Simulate RunPod job
job = {
    "input": {
        "prompt": "Where is the search box?",
        "image": img_b64
    }
}

result = handler(job)
print(result)
```

## 🔍 Supported Models

This handler uses **UI-TARS-1.5-7B** from ByteDance - a specialized UI grounding model:

- ✅ **UI-TARS-1.5-7B** (current) - ByteDance's specialized UI element detection
- ✅ **Florence-2** - Microsoft's alternative vision model (see handler_florence2.py)
- ✅ **CogAgent** - THUDM's UI understanding model
- ✅ **OmniParser** - Icon and element detection

## 📈 Performance

| Model | GPU | Cold Start | Inference | VRAM |
|-------|-----|-----------|-----------|------|
| UI-TARS-1.5-7B | RTX 4090 | ~30-40s | ~0.8s | ~14-16GB |
| UI-TARS-1.5-7B | A100 | ~25-35s | ~0.5s | ~14-16GB |
| Florence-2 | RTX 4090 | ~15s | ~0.3s | ~6GB |

## 🐛 Troubleshooting

### Model not loading
- **VRAM**: UI-TARS requires 14-16GB VRAM (use A100 or RTX 4090)
- **Float32**: Model uses full precision, not bfloat16/float16
- **Dual Loading**: Handler tries AutoProcessor first, then AutoTokenizer as fallback
- Check container logs for detailed error messages

### Element not found
- **Prompt Format**: UI-TARS uses `<grounding> {your prompt}` internally
- **Be Specific**: Use "login button" not just "button"
- **Check Response**: `model_response` field shows what model returned
- Image must be clear and element visible

### Coordinates wrong
- **Multiple Ranges**: Handler auto-detects [0-1], [0-1000], or pixel coordinates
- **Returns Both**: Get center point (x, y) AND bbox [x1, y1, x2, y2]
- Verify with `bbox` field in response

### Slow cold starts
- Pre-download model in Dockerfile (uncomment line in Dockerfile)
- Use smaller model
- Consider RunPod's model caching

## 📝 License

MIT License - feel free to use in your projects!

## 🤝 Contributing

Contributions welcome! Feel free to:
- Add support for more models
- Improve coordinate parsing
- Add batch processing
- Optimize inference speed

## 🔗 Links

- [RunPod Documentation](https://docs.runpod.io/)
- [UI-TARS-1.5-7B Model](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B)
- [Florence-2 Alternative](https://huggingface.co/microsoft/Florence-2-large)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
