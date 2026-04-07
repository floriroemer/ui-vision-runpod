"""
RunPod Serverless Handler for UI-TARS-1.5-7B
Properly configured for ByteDance UI-TARS vision-language model.
"""

import runpod
import torch
import base64
import io
import re
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoProcessor

# Model configuration - UI-TARS-1.5-7B
MODEL_NAME = "ByteDance-Seed/UI-TARS-1.5-7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

print(f"Loading UI-TARS model: {MODEL_NAME}")
print(f"Device: {DEVICE}, dtype: {DTYPE}")

# Load model globally
try:
    # Try using AutoModel with trust_remote_code (UI-TARS may have custom code)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("✓ UI-TARS model loaded successfully!")
    print(f"✓ Model type: {type(model)}")
except Exception as e:
    print(f"✗ Failed to load with AutoModel, trying AutoModelForCausalLM: {e}")
    try:
        # Fallback to tokenizer if processor doesn't exist
        from transformers import AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            device_map="auto",
            trust_remote_code=True
        )
        processor = None
        model.eval()
        print("✓ UI-TARS model loaded with AutoModelForCausalLM!")
    except Exception as e2:
        print(f"✗ Complete failure to load model: {e2}")
        import traceback
        traceback.print_exc()
        model = None
        processor = None
        tokenizer = None


def extract_coordinates(text, image_width, image_height):
    """
    Extract coordinates from UI-TARS output.
    UI-TARS may return: [x1,y1,x2,y2], <box>x,y,w,h</box>, or other formats.
    """
    try:
        # Pattern 1: [x1, y1, x2, y2] format
        match = re.search(r'\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]', text)
        if match:
            coords = [float(x) for x in match.groups()]
        else:
            # Pattern 2: <box>x1,y1,x2,y2</box>
            match = re.search(r'<box>(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)</box>', text)
            if match:
                coords = [float(x) for x in match.groups()]
            else:
                # Pattern 3: Extract first 4 numbers
                numbers = re.findall(r'\d+\.?\d*', text)
                if len(numbers) >= 4:
                    coords = [float(x) for x in numbers[:4]]
                else:
                    return None, None, None, None
        
        x1, y1, x2, y2 = coords
        
        # Normalize based on range
        if max(x1, y1, x2, y2) <= 1.0:
            # [0, 1] normalized
            x1, y1, x2, y2 = x1 * image_width, y1 * image_height, x2 * image_width, y2 * image_height
        elif max(x1, y1, x2, y2) <= 1000:
            # [0, 1000] normalized
            x1 = (x1 / 1000.0) * image_width
            y1 = (y1 / 1000.0) * image_height
            x2 = (x2 / 1000.0) * image_width
            y2 = (y2 / 1000.0) * image_height
        
        # Return center point
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        return center_x, center_y, int(x1), int(y1), int(x2), int(y2)
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return None, None, None, None, None, None


def decode_base64_image(base64_string):
    """Decode base64 to PIL Image."""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')


def handler(job):
    """
    RunPod handler for UI-TARS grounding.
    
    Input: {"prompt": "button text", "image": "base64..."}
    Output: {"x": int, "y": int, "confidence": float}
    """
    try:
        if model is None:
            return {"error": "Model not loaded. Check container logs."}
        
        job_input = job.get("input", {})
        prompt = job_input.get("prompt", "")
        base64_image = job_input.get("image", "")
        
        if not prompt or not base64_image:
            return {"error": "Missing 'prompt' or 'image' in input"}
        
        # Decode image
        image = decode_base64_image(base64_image)
        width, height = image.size
        
        # Prepare input for UI-TARS
        # UI-TARS uses a grounding query format
        query = f"<grounding> {prompt}"
        
        # Process with model
        if processor is not None:
            # Use processor (vision-language model)
            inputs = processor(
                text=query,
                images=image,
                return_tensors="pt"
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            
            response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            # Text-only fallback (shouldn't happen for UI-TARS)
            inputs = tokenizer(query, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract coordinates
        cx, cy, x1, y1, x2, y2 = extract_coordinates(response, width, height)
        
        if cx is None:
            return {
                "error": "Could not find element",
                "model_response": response
            }
        
        # Calculate confidence
        box_area = abs(x2 - x1) * abs(y2 - y1)
        confidence = 0.95 if box_area > 0 else 0.50
        
        return {
            "x": cx,
            "y": cy,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2],
            "model_response": response
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Start RunPod handler
if __name__ == "__main__":
    print("Starting RunPod serverless handler for UI-TARS...")
    runpod.serverless.start({"handler": handler})
