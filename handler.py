"""
RunPod Serverless Handler for UI Grounding with UI-TARS-1.5-7B
Returns pixel coordinates for UI elements using ByteDance UI-TARS model.
"""

import runpod
import torch
import base64
import io
import re
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model configuration - UI-TARS-1.5-7B from ByteDance
MODEL_NAME = "ByteDance-Seed/UI-TARS-1.5-7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # UI-TARS requires full precision

print(f"Loading UI-TARS UI grounding model: {MODEL_NAME}")
print(f"Device: {DEVICE}, dtype: {DTYPE}")

# Load model globally (once on startup)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()
    print("✓ UI-TARS model loaded successfully!")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    model = None
    tokenizer = None


def extract_bbox_from_response(text, image_width, image_height):
    """
    Extract bounding box from UI-TARS response.
    UI-TARS typically returns coordinates in format like: [x1, y1, x2, y2] or <box>x1,y1,x2,y2</box>
    Coordinates may be normalized [0-1] or in pixels or in [0-1000] range.
    """
    try:
        # Try to find box coordinates in various formats
        # Format 1: [x1, y1, x2, y2]
        match = re.search(r'\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]', text)
        if match:
            x1, y1, x2, y2 = map(float, match.groups())
        else:
            # Format 2: <box>x1,y1,x2,y2</box>
            match = re.search(r'<box>(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)</box>', text)
            if match:
                x1, y1, x2, y2 = map(float, match.groups())
            else:
                # Format 3: Just four numbers
                numbers = re.findall(r'\d+\.?\d*', text)
                if len(numbers) >= 4:
                    x1, y1, x2, y2 = map(float, numbers[:4])
                else:
                    return None, None, None, None
        
        # Normalize coordinates based on their range
        if max(x1, y1, x2, y2) <= 1.0:
            # Normalized [0-1]
            x1 = int(x1 * image_width)
            y1 = int(y1 * image_height)
            x2 = int(x2 * image_width)
            y2 = int(y2 * image_height)
        elif max(x1, y1, x2, y2) <= 1000:
            # Normalized [0-1000]
            x1 = int((x1 / 1000.0) * image_width)
            y1 = int((y1 / 1000.0) * image_height)
            x2 = int((x2 / 1000.0) * image_width)
            y2 = int((y2 / 1000.0) * image_height)
        else:
            # Already in pixels
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        return x1, y1, x2, y2
    except Exception as e:
        print(f"Error extracting bbox: {e}")
        return None, None, None, None


def bbox_to_center(x1, y1, x2, y2):
    """Convert bounding box to center point coordinates."""
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    # Calculate confidence based on box size
    box_area = abs(x2 - x1) * abs(y2 - y1)
    confidence = 0.90 if box_area > 0 else 0.50
    
    return center_x, center_y, confidence


def decode_base64_image(base64_string):
    """Decode base64 image string to PIL Image."""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")


def handler(job):
    """
    RunPod serverless handler for UI grounding with UI-TARS-1.5-7B.
    
    Input:
    {
        "prompt": "login button",
        "image": "<base64>"
    }
    
    Output:
    {
        "x": int,
        "y": int,
        "confidence": float
    }
    """
    try:
        if model is None or tokenizer is None:
            return {"error": "Model not loaded. Check container logs."}
        
        job_input = job.get("input", {})
        prompt = job_input.get("prompt")
        base64_image = job_input.get("image")
        
        if not prompt:
            return {"error": "Missing 'prompt' in input"}
        if not base64_image:
            return {"error": "Missing 'image' in input"}
        
        try:
            image = decode_base64_image(base64_image)
            width, height = image.size
        except Exception as e:
            return {"error": f"Failed to decode image: {str(e)}"}
        
        # Prepare prompt for UI-TARS - it expects a specific format
        # UI-TARS uses chat format or direct grounding queries
        query = f"Find the location of: {prompt}"
        
        # Process with UI-TARS
        try:
            # UI-TARS may use different input processing
            # Try standard vision-language model approach
            inputs = tokenizer(query, return_tensors="pt").to(DEVICE)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )
            
            # Decode response
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            return {"error": f"Model inference failed: {str(e)}"}
        
        # Extract bounding box from response
        x1, y1, x2, y2 = extract_bbox_from_response(response_text, width, height)
        
        if x1 is None:
            return {
                "error": "Element not found",
                "model_response": response_text
            }
        
        # Convert to center point
        x, y, confidence = bbox_to_center(x1, y1, x2, y2)
        
        return {
            "x": x,
            "y": y,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2],
            "model_response": response_text
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Start RunPod serverless handler
if __name__ == "__main__":
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
