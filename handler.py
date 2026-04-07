"""
RunPod Serverless Handler for UI Vision Analysis
Analyzes UI screenshots and returns coordinates of UI elements based on natural language prompts.
"""

import runpod
import torch
import base64
import io
import json
import re
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# Model configuration
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

print(f"Loading model: {MODEL_NAME}")
print(f"Device: {DEVICE}, dtype: {DTYPE}")

# Load model and processor globally (only once on container startup)
try:
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    processor = None
    model = None


def parse_coordinates(text):
    """
    Extract x, y coordinates from model output.
    Looks for patterns like: "coordinates: (x, y)" or "x: 100, y: 200" or "[100, 200]"
    """
    # Try various coordinate patterns
    patterns = [
        r'\((\d+),\s*(\d+)\)',           # (x, y)
        r'\[(\d+),\s*(\d+)\]',           # [x, y]
        r'x[:\s]*(\d+).*?y[:\s]*(\d+)',  # x: 100 y: 200
        r'(\d+)\s*,\s*(\d+)',            # 100, 200
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            return x, y
    
    # If no coordinates found, return None
    return None, None


def decode_base64_image(base64_string):
    """Decode base64 image string to PIL Image."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")


def handler(job):
    """
    RunPod serverless handler function.
    
    Expected input:
    {
        "prompt": "Where is the login button?",
        "image": "<base64 encoded image>"
    }
    
    Returns:
    {
        "x": int,
        "y": int,
        "description": str,
        "raw_response": str
    }
    """
    try:
        # Validate model is loaded
        if model is None or processor is None:
            return {
                "error": "Model not loaded. Check container logs.",
                "status": "failed"
            }
        
        # Get input from job
        job_input = job.get("input", {})
        prompt = job_input.get("prompt")
        base64_image = job_input.get("image")
        
        # Validate inputs
        if not prompt:
            return {"error": "Missing 'prompt' in input", "status": "failed"}
        if not base64_image:
            return {"error": "Missing 'image' in input", "status": "failed"}
        
        # Decode image
        try:
            image = decode_base64_image(base64_image)
            width, height = image.size
        except Exception as e:
            return {"error": f"Failed to decode image: {str(e)}", "status": "failed"}
        
        # Prepare enhanced prompt for coordinate extraction
        enhanced_prompt = f"""You are analyzing a UI screenshot that is {width}x{height} pixels.

User question: {prompt}

Please identify the UI element and provide its approximate center coordinates in the format: (x, y)
Where x is the horizontal position (0 to {width}) and y is the vertical position (0 to {height}).

Respond with the coordinates and a brief description."""
        
        # Prepare inputs for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": enhanced_prompt}
                ]
            }
        ]
        
        # Process inputs
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Parse coordinates from output
        x, y = parse_coordinates(output_text)
        
        # Prepare response
        response = {
            "x": x if x is not None else -1,
            "y": y if y is not None else -1,
            "description": output_text.strip(),
            "raw_response": output_text,
            "image_size": {"width": width, "height": height},
            "status": "success" if x is not None else "no_coordinates_found"
        }
        
        return response
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }


# Start RunPod serverless handler
if __name__ == "__main__":
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
