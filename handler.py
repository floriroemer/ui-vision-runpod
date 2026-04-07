"""
RunPod Serverless Handler for UI Grounding
Returns pixel coordinates for UI elements using specialized UI-grounding model.
"""

import runpod
import torch
import base64
import io
import re
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Model configuration
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print(f"Loading UI grounding model: {MODEL_NAME}")
print(f"Device: {DEVICE}, dtype: {DTYPE}")

# Load model globally (once on startup)
try:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("✓ UI grounding model loaded successfully!")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    model = None
    processor = None


def parse_bbox_from_text(text):
    """
    Extract bounding box coordinates from model output.
    Looks for patterns like: <box>(x1,y1),(x2,y2)</box> or <ref>element</ref><box>coordinates</box>
    """
    # Try to find bounding box pattern
    box_pattern = r'<box>\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*,?\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*</box>'
    match = re.search(box_pattern, text)
    
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        return x1, y1, x2, y2
    
    # Try normalized coordinates pattern [x1, y1, x2, y2]
    norm_pattern = r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]'
    match = re.search(norm_pattern, text)
    
    if match:
        coords = [float(x) for x in match.groups()]
        return coords[0], coords[1], coords[2], coords[3]
    
    return None, None, None, None


def bbox_to_center(x1, y1, x2, y2, image_width, image_height):
    """Convert bounding box to center point coordinates."""
    # Handle normalized coordinates (0-1 range)
    if x1 <= 1.0 and x2 <= 1.0 and y1 <= 1.0 and y2 <= 1.0:
        x1 = int(x1 * image_width)
        x2 = int(x2 * image_width)
        y1 = int(y1 * image_height)
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
    {
        "prompt": "Click the login button",
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
        
        # Grounding prompt for bounding box detection
        grounding_prompt = f"""In this {width}x{height} screenshot, locate the UI element: {prompt}

Provide the bounding box coordinates in the format: <box>(x1,y1),(x2,y2)</box> where:
- x1,y1 is top-left corner
- x2,y2 is bottom-right corner
- Coordinates are in pixels"""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": grounding_prompt}
                ]
            }
        ]
        
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(DEVICE)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Parse bounding box
        x1, y1, x2, y2 = parse_bbox_from_text(output_text)
        
        if x1 is None:
            return {"error": "element not found"}
        
        # Convert to center point
        x, y, confidence = bbox_to_center(x1, y1, x2, y2, width, height)
        
        return {
            "x": x,
            "y": y,
            "confidence": confidence
        }
        
    except Exception as e:
        return {"error": str(e)}


# Start RunPod serverless handler
if __name__ == "__main__":
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
