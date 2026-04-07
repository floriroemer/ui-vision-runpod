# Use RunPod's PyTorch base image
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY handler.py .

# Pre-download the model (optional but recommended for faster cold starts)
# Uncomment if you want to bake the model into the image
# RUN python -c "from transformers import AutoProcessor, AutoModelForVision2Seq; \
#     AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', trust_remote_code=True); \
#     AutoModelForVision2Seq.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', trust_remote_code=True)"

# Set the entrypoint
CMD ["python", "-u", "handler.py"]
