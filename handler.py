"""
RunPod Serverless Handler - Florence-2 UI Grounding
Microsoft Florence-2 is a proven vision model with object detection and grounding capabilities.
"""

import runpod
import torch
import base64
import io
import re
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Model configuration - Florence-2-large (Microsoft, proven and reliable)
MODEL_NAME = "microsoft/Florence-2-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

print(f"Loading Florence-2 UI grounding model: {MODEL_NAME}")
print(f"Device: {DEVICE}, dtype: {DTYPE}")

# Load model globally (once on startup)
try:
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("✓ Florence-2 model loaded successfully!")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    model = None
    processor = None


def parse_florence_output(text, image_width, image_height):
    """Parse Florence-2 output and extract bounding box coordinates."""
    try:
        # Florence-2 returns format like: <loc_x1><loc_y1><loc_x2><loc_y2>
        # Coordinates are normalized to 1000
        matches = re.findall(r'<loc_(\d+)>', text)
        if len(matches) >= 4:
            x1, y1, x2, y2 = map(int, matches[:4])
            # Denormalize from [0, 999] to pixels
            x1 = int((x1 / 999.0) * image_width)
            y1 = int((y1 / 999.0) * image_height)
            x2 = int((x2 / 999.0) * image_width)
            y2 = int((y2 / 999.0) * image_height)
            return x1, y1, x2, y2
        return None, None, None, None
    except Exception as e:
        print(f"Error parsing Florence output: {e}")
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
    RunPod serverless handler for UI grounding with Florence-2.
    
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
        if model is None or processor is None:
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
        
        # Prepare task prompt for Florence-2 object detection
        task_prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {prompt}"
        
        # Prepare inputs
        inputs = processor(
            text=task_prompt,
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                num_beams=3
            )
        
        # Decode output
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Parse bounding box from output
        x1, y1, x2, y2 = parse_florence_output(generated_text, width, height)
        
        if x1 is None:
            return {"error": "Element not found"}
        
        # Convert to center point
        x, y, confidence = bbox_to_center(x1, y1, x2, y2)
        
        return {
            "x": x,
            "y": y,
            "confidence": confidence,
            "debug_output": generated_text  # For debugging
        }
        
    except Exception as e:
        return {"error": str(e)}


# Start RunPod serverless handler
if __name__ == "__main__":
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
