"""
RunPod Serverless Handler for UI Grounding with SeeClick
Returns pixel coordinates for UI elements using SeeClick UI grounding model.
"""

import runpod
import torch
import base64
import io
from PIL import Image
from transformers import AutoModel, AutoProcessor

# Model configuration - SeeClick UI grounding model
MODEL_NAME = "cckevinn/SeeClick"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

print(f"Loading SeeClick UI grounding model: {MODEL_NAME}")
print(f"Device: {DEVICE}, dtype: {DTYPE}")

# Load model globally (once on startup)
try:
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("✓ SeeClick model loaded successfully!")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    model = None
    processor = None


def extract_bbox_from_output(output, image_width, image_height):
    """
    Extract bounding box from SeeClick model output.
    SeeClick returns normalized coordinates [x1, y1, x2, y2] in range [0, 1000].
    """
    try:
        if isinstance(output, dict) and 'boxes' in output:
            boxes = output['boxes']
            if len(boxes) > 0:
                box = boxes[0]  # Take first detected box
                x1, y1, x2, y2 = box
                # Denormalize from [0, 1000] to pixel coordinates
                x1 = int((x1 / 1000.0) * image_width)
                y1 = int((y1 / 1000.0) * image_height)
                x2 = int((x2 / 1000.0) * image_width)
                y2 = int((y2 / 1000.0) * image_height)
                return x1, y1, x2, y2
        
        # Fallback: try to extract from list/tensor
        if isinstance(output, (list, tuple)) and len(output) >= 4:
            x1, y1, x2, y2 = output[:4]
            x1 = int((float(x1) / 1000.0) * image_width)
            y1 = int((float(y1) / 1000.0) * image_height)
            x2 = int((float(x2) / 1000.0) * image_width)
            y2 = int((float(y2) /):
    """Convert bounding box to center point coordinates."""
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    # Calculate confidence based on box validity
    box_width = abs(x2 - x1)
    box_height = abs(y2 - y1)
    
    if box_width > 0 and box_height > 0:
        confidence = 0.90
    else:
        confidence = 0.50
        y2 = int(y2 * image_height)
    
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    # Calculate confidence based on box size (larger boxes = more confident)
    box_area = abs(x2 - x1) * abs(y2 - y1)
    image_area = image_width * image_height
    confidence = min(0.95, 0.7 + (box_area / image_area) * 0.25)
    
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
    RunPod serverless handler for UI grounding.
    
    Input:
    { with SeeClick.
    
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
        
        # Prepare inputs for SeeClick
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract bounding box from output
        x1, y1, x2, y2 = extract_bbox_from_output(outputs, width, height)
        
        if x1 is None:
            return {"error": "element not found"}
        
        # Convert to center point
        x, y, confidence = bbox_to_center(x1, y1, x2, y2
            "y": y,
            "confidence": confidence
        }
        
    except Exception as e:
        return {"error": str(e)}


# Start RunPod serverless handler
if __name__ == "__main__":
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
